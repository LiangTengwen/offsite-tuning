import gc
import os
from copy import deepcopy
import torch
from torch import nn
from accelerate.logging import get_logger
from transformers import (
    SchedulerType,
    MODEL_MAPPING,
    OPTForCausalLM,
    GPT2LMHeadModel,
    BloomForCausalLM,
    ViTForImageClassification,
)
from offsite_tuning.models.clip_vit import CLIPViTForImageClassification
from offsite_tuning.models.eva_vit import EVAViTForImageClassification

import argparse


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, activation=nn.ReLU):
        #将传入的参数赋值给相应的实例变量。
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation()
        
        #创建了一个`nn.ModuleList()`类型的`layers`变量，用于存储模型的各个层。
        self.layers = nn.ModuleList()
        #通过`nn.Linear`函数将输入层连接到第一个隐藏层，并将其添加到`layers`中。
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        #使用循环将剩余的隐藏层添加到`layers`中。
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        #将最后一个隐藏层连接到输出层，并将其添加到`layers`中。
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    #接受输入`x`作为参数，在前向传播过程中，通过遍历`layers`列表，将输入`x`依次传递给每个线性层和激活函数进行计算。
    # 最后，将最后一层的输出作为模型的输出，并返回它。
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

#`add_prologue`和`add_epilogue`，用于在神经网络模型的前向传播过程中添加前导操作和后继操作。
#两个函数可以用于在神经网络模型的前向传播过程中添加额外的操作，例如数据预处理、数据转换或特征提取等。
def add_prologue(module, prologue):
    #首先保存了`module`的原始前向传播函数`module.forward`和传入的`prologue`函数。
    module.old_forward = module.forward
    module.prologue = prologue

    #定义了一个新的前向传播函数`new_forward`
    def new_forward(self):
        #新的匿名函数`lambda_forward`，该函数接受`*args`和`**kwargs`，
        #即任意数量的位置参数和关键字参数。在`lambda_forward`函数中，
        #首先保存了输入参数`args`和`kwargs`到`self.input_args`和`self.input_kwargs`中。
        def lambda_forward(*args, **kwargs):
            self.input_args = args
            self.input_kwargs = kwargs
            #根据`prologue`函数是否为`None`，对输入数据进行前导操作。
            #如果`prologue`不为`None`，则将输入数据的第一个元素`args[0]`传递给`prologue`函数进行操作，并将结果赋给变量`x`；
            #否则，将`args[0]`赋给变量`x`。
            if self.prologue is not None:
                x = self.prologue(args[0])
            else:
                x = args[0]
            #将`x`与`args[1:]`拼接成新的输入参数`args`
            args = (x,) + args[1:]
            #调用原始的前向传播函数`self.old_forward`，传入新的参数`args`和`kwargs`
            return self.old_forward(*args, **kwargs)
        return lambda_forward
    #将`lambda_forward`函数赋值给`module.forward`，完成对模型的前向传播函数的替换，并返回修改后的`module`。
    module.forward = new_forward(module)
    return module

def add_epilogue(module, epilogue):
    module.old_forward = module.forward
    module.epilogue = epilogue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            output = self.old_forward(*args, **kwargs)
            #判断`output`是否为元组类型，如果是，则将其中的第一个元素赋值给变量`x`；否则，将`output`赋值给`x`。
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output

            if self.epilogue is not None:
                x = self.epilogue(x)
            #如果`output`是元组类型，则将`x`与`output[1:]`拼接成新的输出元组；否则，将`x`赋值给`output`。
            if isinstance(output, tuple):
                output = (x,) + output[1:]
            else:
                output = x

            self.cached_output = x
            return output
        return lambda_forward
    module.forward = new_forward(module)
    return module

#`layers`是一个`nn.ModuleList`类型的列表，用于存储神经网络的层；
#`num_student_layers`是一个整数，表示要选择的学生模型的层数。
def uniform_choose_layers(layers: nn.ModuleList, num_student_layers=None):
    #如果`num_student_layers`为`None`，则默认选择和原始模型一样多的层数。
    if num_student_layers is None:
        num_student_layers = len(layers)
    #定义了一个空的`nn.ModuleList`类型的列表`student`，用于存储选出来的学生模型的层。
    student = nn.ModuleList()
    #计算每个学生模型层之间的步长`stride`，通过`(len(layers) - 1) / (num_student_layers - 1)`计算得到。
    stride = (len(layers) - 1) / (num_student_layers - 1)
    #在每次迭代中，通过计算得到的`stride`和迭代变量`i`，计算出应该选择的原始模型层的索引`idx`。然后，将选择的原始模型层`layers[idx]`添加到`student`列表中。同时，使用日志记录工具记录添加的层的索引。
    for i in range(num_student_layers):
        idx = round(i * stride)
        logger.info(f"Adding layer {idx} to student")
        student.append(layers[idx])
    #返回`student`列表
    return student

#在函数内部，首先使用`torch.no_grad()`装饰器将下面的代码包装起来，以确保在参数裁剪过程中不会进行梯度计算。
@torch.no_grad()
#`ratio`是一个介于0和1之间的浮点数，表示要裁剪掉的参数比例。
def magnitude_prune(model, ratio):
    for param in model.parameters():
        #对于维度为1的参数，跳过不处理。
        if param.dim() == 1:
            continue
        #计算需要裁剪的数量`num_prune`，将参数元素总数乘以裁剪比例得到。
        num_prune = int(param.numel() * ratio)
        #使用`param.abs().view(-1).kthvalue(num_prune).values.item()`计算出裁剪的阈值，该阈值是排序后第`num_prune`个最小的参数绝对值。
        threshold = param.abs().view(-1).kthvalue(num_prune).values.item()
        #根据阈值将参数进行裁剪，通过将参数`param`的绝对值与阈值进行比较，并将结果转换为与参数相同类型的`mask`。
        mask = (param.abs() >= threshold).to(param.dtype)
        #将参数与`mask`相乘，实现裁剪操作。
        param.mul_(mask)

#
@torch.no_grad()
def quantize(model, bits):
    for param in model.parameters():
        if param.dim() == 1:
            continue
        min, max = param.min(), param.max()
        #计算参数的零点`zp`，通过将最大值和最小值相加除以2得到。
        zp = (max + min) / 2
        #计算参数的缩放因子`scale`，通过将最大值和最小值的差除以(2 ** bits - 1)得到
        scale = (max - min) / (2 ** bits - 1)
        #对参数进行量化操作。首先，将参数减去零点`zp`，然后除以缩放因子`scale`，再使用`round_()`函数四舍五入。接着，将参数乘以缩放因子`scale`，再加上零点`zp`。
        param.sub_(zp).div_(scale).round_().mul_(scale).add_(zp)


def parse_args():
    #创建了一个`argparse.ArgumentParser`对象`parser`，用于处理命令行参数的解析工作。
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    #调用`parser.add_argument`方法添加命令行参数。
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        type=int,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        help='Optimizer to use. Can be adamw or sgd',
        choices=['adamw', 'sgd']
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum to use for sgd optimizer."
    )
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=88,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        '--no_save_model',
        action='store_true',
        help='Whether or not to save the model.'
    )
    parser.add_argument(
        '--kd_weight',
        type=float,
        default=0.0,
        help='Weight of the knowledge distillation loss.'
    )
    parser.add_argument(
        '--lm_weight',
        type=float,
        default=1.0,
        help='Weight of the knowledge distillation loss.'
    )
    parser.add_argument(
        '--train_tokenized_dataset',
        type=str,
        default=None,
        help='Path to the tokenized training dataset.'
    )
    parser.add_argument(
        '--val_tokenized_dataset',
        type=str,
        default=None,
        help='Path to the tokenized validation dataset.'
    )
    parser.add_argument(
        "--train_num_samples",
        type=int,
        default=None,
        help="The number of samples to use for training set.",
    )
    parser.add_argument(
        "--validation_num_samples",
        type=int,
        default=None,
        help="The number of samples to use for validation set.",
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=200,
    )

    parser.add_argument(
        '--num_student_layers',
        type=int,
        default=None,
        help='Number of layers in the student model.'
    )

    parser.add_argument(
        '--load_student',
        type=str,
        default=None,
        help='Path to the student model'
    )

    parser.add_argument(
        '--student_l_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--student_r_pad',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--student_layer_selection_strategy',
        type=str,
        default='uniform',
        help='Layer selection strategy',
        choices=['uniform', 'random', 'changes']
    )

    parser.add_argument(
        '--restart_training',
        action='store_true',
        help='Whether to restart training of all dataset.'
    )

    parser.add_argument(
        '--train_module',
        type=str,
        default='student',
        help='Part of the model to train.',
        choices=['student', 'adapter', 'all']
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Max gradient norm.'
    )

    parser.add_argument(
        '--magnitude_pruning_ratio',
        type=float,
        default=0.0,
        help='Magnitude pruning ratio.'
    )

    parser.add_argument(
        '--weight_quantization_bits',
        type=int,
        default=None,
        help='Weight quantization bits.'
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    # vit
    parser.add_argument("--train_dir", type=str, default=None,
                        help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None,
                        help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
    )

    parser.add_argument(
        '--freeze_bottom',
        action='store_true',
    )
    
    parser.add_argument(
        '--no_teacher',
        action='store_true',
    )

    parser.add_argument(
        '--classifier_lr_multiplier',
        type=float,
        default=1.0,
    )
    
    parser.add_argument(
        '--select_by_kd',
        action='store_true',
    )
    
    parser.add_argument(
        '--use_pt_imagefolder',
        action='store_true',
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=12,
    )

    parser.add_argument(
        '--train_lm_head',
        action='store_true',
    )
    parser.add_argument(
        '--save_module',
        type=str,
        default='student',
        choices=['student', 'adapter', 'all']
    )
    
    parser.add_argument(
        '--load_adapter',
        type=str,
        default=None,
        help='Path to the student model'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        default='piqa',
        help='Evaluation tasks',
    )
    
    parser.add_argument(
        '--use_adapter',
        action='store_true',
    )
    
    parser.add_argument(
        '--use_lora',
        action='store_true',
    )
    
    parser.add_argument(
        '--use_bitfit',
        action='store_true',
    )
    
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=4,
        help='Rank of the LoRA matrix',
    )

    parser.add_argument(
        '--lora_alpha',
        type=float,
        default=32,
        help='Alpha of the LoRA matrix',
    )

    parser.add_argument(
        '--adapter_size',
        type=int,
        default=64,
        help='Size of the adapter',
    )
    #
    parser.add_argument
    #`parser.parse_args()`方法会解析命令行参数，并返回一个命名空间对象`args`，其中包含了命令行参数的信息。通过点操作符可以访问这些参数的值。
    args = parser.parse_args()

    return args
#使用多个`if-elif-else`语句来确定传入的`model`属于哪种类型，并相应地获取对应的层或模块。
def get_layers(model):
    if isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, GPT2LMHeadModel):
        layers = model.transformer.h
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif isinstance(model, ViTForImageClassification):
        layers = model.vit.encoder.layer
    elif isinstance(model, CLIPViTForImageClassification):
        layers = model.vit.encoder.layers
    elif isinstance(model, EVAViTForImageClassification):
        layers = model.blocks
    else:
        raise NotImplementedError
    return layers

def set_layers(model, layers):
    if isinstance(model, OPTForCausalLM):
        model.model.decoder.layers = layers
    elif isinstance(model, GPT2LMHeadModel):
        model.transformer.h = layers
    elif isinstance(model, BloomForCausalLM):
        model.transformer.h = layers
    elif isinstance(model, ViTForImageClassification):
        model.vit.encoder.layer = layers
    elif isinstance(model, CLIPViTForImageClassification):
        model.vit.encoder.layers = layers
    elif isinstance(model, EVAViTForImageClassification):
        model.blocks = layers
    else:
        raise NotImplementedError

#设置一个用于训练的师生模型
def setup_teacher_student(model, args, accelerator):
    #在函数内部，首先通过循环将`model`中的所有参数的`requires_grad`属性设为`False`，即冻结所有参数，使其不可训练。
    for param in model.parameters():
        param.requires_grad = False
    #调用`get_layers`函数获取`model`的层或模块，并将结果保存在`layers`变量中。
    layers = get_layers(model)
    #根据`args.student_l_pad`和`args.student_r_pad`确定需要选择的层的范围，
    l, r = args.student_l_pad, len(layers) - args.student_r_pad
    #然后根据`args.load_student`是否有值选择加载已有的学生模型或者创建新的学生模型。
    #若加载已有的学生模型，则从`student_state_dict`加载参数，并将其赋值给`student`。
    
    #如果提供了 `args.load_student`，则从指定位置加载一个已存在的学生模型，并将其赋值给 `student` 变量。
    #否则，通过深拷贝指定范围内的层来创建一个新的学生模型。
    if args.load_student:
        student_state_dict = torch.load(os.path.join(
            args.load_student, 'student.pt'), map_location='cpu')
        student_layers_len = len(
            set([k.split('.')[0] for k in student_state_dict.keys()]))
        logger.info(
            f"Loading student module from {args.load_student} with {student_layers_len} layers.")
        student = deepcopy(layers[:student_layers_len])
        student.load_state_dict(student_state_dict)
    else:
        student = deepcopy(layers[l:r])
    #如果 `args.student_layer_selection_strategy` 设置为 `'uniform'`，
    #则从学生模型中选择一部分层，其分布均匀且大小由 `args.num_student_layers` 指定。
    if args.student_layer_selection_strategy == 'uniform':
        student = uniform_choose_layers(student, args.num_student_layers)
    else:
        raise NotImplementedError
    #将`student`移动到加速器设备上。
    student = student.to(accelerator.device)
    #如果`args.magnitude_pruning_ratio`大于0，则调用`magnitude_prune`函数对`student`进行剪枝操作。
    if args.magnitude_pruning_ratio > 0:
        logger.info(
            f"Pruning student module with magnitude ratio {args.magnitude_pruning_ratio}")
        magnitude_prune(student, args.magnitude_pruning_ratio)
    #如果`args.weight_quantization_bits`不为`None`，则调用`quantize`函数对`student`进行权重量化操作。
    if args.weight_quantization_bits is not None:
        logger.info(
            f"Quantizing student module with {args.weight_quantization_bits} bits")
        quantize(student, args.weight_quantization_bits)

    #如果`args.train_module`等于`student`，则将`student`中的参数设置为可训练。
    if args.train_module == 'student':
        for param in student.parameters():
            param.data = param.data.float()
            param.requires_grad = True
    #如果`args.train_module`等于`adapter`，则将`student`中的参数冻结，并根据`args.freeze_bottom`的值选择是否冻结`layers`的底部层。
    elif args.train_module == 'adapter':
        for param in student.parameters():
            param.requires_grad = False
        if not args.freeze_bottom:
            for param in layers[:l].parameters():
                param.data = param.data.float()
                param.requires_grad = True
        for param in layers[r:].parameters():
            param.data = param.data.float()
            param.requires_grad = True
    #如果`args.train_module`等于`all`，则将`student`和`layers`的所有参数设置为可训练。
    elif args.train_module == 'all':
        for param in student.parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in layers[:l].parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in layers[r:].parameters():
            param.data = param.data.float()
            param.requires_grad = True
    else:
        raise NotImplementedError

    model.student = student

    model.teacher = layers[l:r].half()
    
    model.adapter = layers[:l] + layers[r:]

    for param in model.teacher.parameters():
        param.requires_grad = False

    add_prologue(model.student[0], None)
    add_epilogue(model.student[-1], None)
    #将学生模型的第一层和最后一层分别赋值给 `model.student_l` 和 `model.student_r`。
    model.student_l = model.student[0]
    model.student_r = model.student[-1]
    #确定学生模型层数，并记录日志。
    num_student_layers = len(model.student)
    logger.info(f"Number of student layers: {num_student_layers}")
    #根据 `args.train_module` 的值，将适当的可训练模块赋值给 `model.trainable_module`。
    if args.train_module == 'student':
        model.trainable_module = model.student
    elif args.train_module == 'adapter':
        model.trainable_module = model.adapter
    elif args.train_module == 'all':
        model.trainable_module = model.student + model.adapter
    else:
        raise NotImplementedError
    #进行垃圾回收并清除 GPU 缓存。
    gc.collect()
    torch.cuda.empty_cache()
    return model

#该函数接受一个 `model` 和 `args` 作为输入。它用于将模型的一部分替换为教师模型。
#输入的 `model` 可以是不同类型的模型。要替换的具体层由 `args.student_l_pad` 和 `args.student_r_pad` 的值确定。
#在替换层之后，函数会相应地更新 `model`。
def to_teacher(model, args):
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[
            :l] + model.teacher + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.teacher + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.teacher + model.transformer.h[r:]
    elif isinstance(model, ViTForImageClassification):
        r = len(model.vit.encoder.layer) - args.student_r_pad
        model.vit.encoder.layer = model.vit.encoder.layer[:l] + \
            model.teacher + model.vit.encoder.layer[r:]
    elif isinstance(model, CLIPViTForImageClassification):
        r = len(model.vit.encoder.layers) - args.student_r_pad
        model.vit.encoder.layers = model.vit.encoder.layers[:l] + \
            model.teacher + model.vit.encoder.layers[r:]
    elif isinstance(model, EVAViTForImageClassification):
        r = len(model.blocks) - args.student_r_pad
        model.blocks = model.blocks[:l] + \
            model.teacher + model.blocks[r:]
    else:
        raise NotImplementedError

#该函数与 `to_teacher` 类似，但它将指定的层替换为学生模型而不是教师模型。
def to_student(model, args):
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[
            :l] + model.student + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.student + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.student + model.transformer.h[r:]
    elif isinstance(model, ViTForImageClassification):
        r = len(model.vit.encoder.layer) - args.student_r_pad
        model.vit.encoder.layer = model.vit.encoder.layer[:l] + \
            model.student + model.vit.encoder.layer[r:]
    elif isinstance(model, CLIPViTForImageClassification):
        r = len(model.vit.encoder.layers) - args.student_r_pad
        model.vit.encoder.layers = model.vit.encoder.layers[:l] + \
            model.student + model.vit.encoder.layers[r:]
    elif isinstance(model, EVAViTForImageClassification):
        r = len(model.blocks) - args.student_r_pad
        model.blocks = model.blocks[:l] + \
            model.student + model.blocks[r:]
    else:
        raise NotImplementedError

#该函数计算教师模型和学生模型之间的知识蒸馏损失。
def get_kd_loss(model):
    kwargs = model.student_l.input_kwargs
    args = model.student_l.input_args
    output_teacher = args[0].to(torch.float16)
    args = list(args[1:])
    for i, arg in enumerate(args):
        if torch.is_tensor(arg) and arg.dtype == torch.float32:
            args[i] = arg.to(torch.float16)
    args = tuple(args)

    for k, v in kwargs.items():
        if torch.is_tensor(v) and v.dtype == torch.float32:
            kwargs[k] = v.to(torch.float16)

    with torch.no_grad():
        model.teacher.eval()
        for teacher_layer in model.teacher:
            output_teacher = teacher_layer(output_teacher, *args, **kwargs)
            if isinstance(output_teacher, tuple):
                output_teacher = output_teacher[0]

    output_student = model.student_r.cached_output.float()
    output_teacher = output_teacher.float()

    std = output_teacher.pow(2).mean().sqrt()
    kd_loss = (output_teacher - output_student).div(std).pow(2).mean()
    return kd_loss

#用于设置可训练的分类头。根据模型的类型，该函数会将分类器层的参数设置为可训练，并将参数的数据类型转换为 float。
def setup_trainable_classification_head(model):
    # Setup trainable classification heads
    if isinstance(model, ViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    elif isinstance(model, CLIPViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    elif isinstance(model, EVAViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    else:
        raise NotImplementedError
#该函数用于加载适配器。根据模型的类型，加载适配器层的状态字典，并将其赋值给对应的模型层。加载适配器后，函数返回更新后的模型。
def load_adapter(model, adapter_state_dict, args):
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        adapter_layers = model.model.decoder.layers[:l] + model.model.decoder.layers[r:]
        adapter_layers.load_state_dict(adapter_state_dict)
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        adapter_layers = model.transformer.h[:l] + model.transformer.h[r:]
        adapter_layers.load_state_dict(adapter_state_dict)
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        adapter_layers = model.transformer.h[:l] + model.transformer.h[r:]
        adapter_layers.load_state_dict(adapter_state_dict)
    else:
        raise NotImplementedError
    return model
#该函数用于加载学生模型。
def load_student(model, student_state_dict, args):
    #根据 `args.student_l_pad` 获取学生模型中需要加载的层数。
    l = args.student_l_pad
    
    student_layers_len = len(
        set([k.split('.')[0] for k in student_state_dict.keys()]))
    logger.info(f"Loading student module from with {student_layers_len} layers.")
    #加载相应层数的学生模型参数，并将其赋值给对应的模型层。加载学生模型后，函数返回更新后的模型。
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        student_layers = model.model.decoder.layers[l:l+student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.model.decoder.layers = model.model.decoder.layers[:l] + \
            student_layers + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        student_layers = model.transformer.h[l:l+student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.transformer.h = model.transformer.h[:l] + \
            student_layers + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        student_layers = model.transformer.h[l:l+student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.transformer.h = model.transformer.h[:l] + \
            student_layers + model.transformer.h[r:]
    else:
        raise NotImplementedError
    return model
#用于保存状态字典到指定的输出目录和文件名。函数首先将状态字典中的参数转换为 float16 数据类型，并移动到 CPU 上。
#然后，使用 `torch.save` 方法将状态字典保存到指定的输出目录和文件名中。
def save_state_dict(state_dict, output_dir, filename):
    for k in state_dict:
        state_dict[k] = state_dict[k].to(torch.float16).cpu()
    torch.save(state_dict, os.path.join(output_dir, filename))