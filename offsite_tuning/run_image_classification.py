# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning any 🤗 Transformers model for image classification leveraging 🤗 Accelerate."""
import argparse
import json
import logging
import math
import os
import sys

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
#这段代码导入了一些来自torchvision.transforms模块的图像转换函数。
#这些函数可用于对图像进行预处理和数据增强，以准备输入给深度学习模型。

#下面是导入的函数及其功能的简要说明：

#`CenterCrop`: 对图像进行中心裁剪，通过指定裁剪的输出尺寸，将图像裁剪为中心区域的正方形或矩形。
# `Compose`: 将多个图像转换函数组合成一个序列，依次应用这些函数。
# `Normalize`: 对图像进行标准化，通过减去均值并除以标准差来将图像的像素值调整到特定范围。
# `RandomHorizontalFlip`: 随机水平翻转图像，以增加训练数据的变化性和多样性。
# `RandomResizedCrop`: 随机裁剪图像，并将其缩放到指定的大小。这个函数可以在裁剪和缩放的过程中引入随机性，以增加训练数据的多样性。
# `Resize`: 调整图像的大小，通过指定输出尺寸或缩放因子来改变图像的大小。
# `ToTensor`: 将图像转换为张量（tensor），方便在深度学习模型中进行处理。
# 这些转换函数可用于创建图像数据集的预处理管道，在训练过程中对图像进行预处理和数据增强，以提高模型的性能和鲁棒性。
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

#这段代码导入了一些来自`transformers`库的模块和类，这些模块和类提供了用于图像分类和视觉模型的一些功能和工具。

# 下面是导入的一些模块和类的简要说明：

# `AutoConfig`: 用于自动配置各种模型的配置类。可以根据具体的模型类名称或预训练模型的名称，自动加载相应的模型配置。
# `AutoFeatureExtractor`: 用于自动加载和配置各种特征提取器的类。可以根据具体的模型类名称或预训练模型的名称，自动加载相应的特征提取器。
# `AutoModelForImageClassification`: 用于自动加载和配置各种图像分类模型的类。可以根据具体的模型类名称或预训练模型的名称，自动加载相应的图像分类模型。
# `get_scheduler`: 用于获取优化器的学习率调度器的函数。可以根据指定的参数来获取不同类型的学习率调度器。
# `CLIPVisionConfig`: 用于配置CLIP视觉模型的配置类。CLIP（Contrastive Language-Image Pretraining）是一种联合训练语言和视觉模型的方法。
# `CLIPVisionModel`: 用于加载和使用CLIP视觉模型的类。
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    get_scheduler,
    CLIPVisionConfig,
    CLIPVisionModel,
)


from offsite_tuning.utils import (
    parse_args,
    setup_teacher_student,
    get_kd_loss,
    to_teacher,
    to_student,
    setup_trainable_classification_head
)

from offsite_tuning.models.clip_vit import CLIPViTForImageClassification
from offsite_tuning.models.eva_vit import EVAViTForImageClassification
import gc

logger = get_logger(__name__)


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    #定义了一个空的字典对象`accelerator_log_kwargs = {}`，用于保存加速器日志相关的参数。
    accelerator_log_kwargs = {}
    #将`args.report_to`的值赋给`log_with`键，将`args.output_dir`的值赋给`logging_dir`键，
    # 将这两个键值对添加到`accelerator_log_kwargs`字典中。
    # 这样做的目的是将日志输出方式（log_with）和日志保存的目录（logging_dir）传递给加速器对象进行配置。
    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["logging_dir"] = args.output_dir

    #实例化`Accelerator`类，创建了一个名为`accelerator`的加速器对象，
    # 并传递了梯度累积步数参数（gradient_accumulation_steps）和之前创建的`accelerator_log_kwargs`字典作为关键字参数。
    #加速器对象可以用于管理和加速深度学习训练过程中的计算，同时可以提供日志记录和管理的功能。
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Handle the repository creation
    #通过`accelerator.is_main_process`和`args.output_dir`不为空进行判断，
    # 如果当前进程是主进程且`output_dir`参数不为空，则调用`os.makedirs()`函数创建输出目录`args.output_dir`。
    # `os.makedirs()`函数会递归创建目录，如果目录已存在则不会抛出异常（通过`exist_ok=True`参数设置）。
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    #打印加速器状态信息
    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    #日志输出的格式设置为"% (asctime)s -% (levelname)s -% (name)s -% (message)s"，日期格式为"%m/%d/%Y %H:%M:%S"，
    # 日志级别为`logging.INFO`，日志处理程序设置为将日志同时输出到标准输出流（`StreamHandler(sys.stdout)`）和日志文件（“log.txt”，位于`args.output_dir`目录下）。
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
        ] if accelerator.is_main_process else []
    )
    logger.info(accelerator.state, main_process_only=False)
    
    #根据当前进程是否为主进程，通过条件表达式来设置日志记录的处理程序。
    # 如果当前进程是主进程，使用之前设置的日志处理程序；否则，设置一个空的列表，即不执行日志记录。
    
    #设置数据集的日志记录级别。如果当前进程是本地主进程（即非分布式训练），则将数据集和transformers模块的日志级别设置为`INFO`；否则，将它们的日志级别设置为`ERROR`，即仅记录错误信息。
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    #进入加速器的等待状态，以确保所有进程都达到同步状态。
    accelerator.wait_for_everyone()

    #使用`AutoFeatureExtractor.from_pretrained()`函数从预训练模型加载特征提取器。
    # `args.model_name_or_path`参数指定了预训练模型的名称或路径。
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name_or_path)

    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in feature_extractor.size:
        size = feature_extractor.size["shortest_edge"]
    else:
        size = (feature_extractor.size["height"],
                feature_extractor.size["width"])

    normalize = Normalize(mean=feature_extractor.image_mean,
                          std=feature_extractor.image_std)
    #`train_transforms`和`val_transforms`分别定义了训练集和验证集的转换器，
    # 在转换过程中应用了一系列的图像处理操作，如随机裁剪（RandomResizedCrop）、随机水平翻转（RandomHorizontalFlip）、调整尺寸（Resize）、中心裁剪（CenterCrop）、转换为张量（ToTensor）和归一化（Normalize），
    # 并保存到`train_transforms`和`val_transforms`变量中。
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )
    #`preprocess_train()`和`preprocess_val()`，用来在批处理过程中应用相应的图像转换器。
    # 这些函数将输入的图像批量进行转换和处理，并将处理后的结果存储在`pixel_values`字段中，作为特征输入参与后续的训练过程中。
    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(
            image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(
            image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(args.dataset_name, task="image-classification")
    elif args.use_pt_imagefolder:
        # Load a local dataset using a PyTorch Dataset.
        import torchvision.datasets as pt_datasets
        logging.info("Using PyTorch ImageFolder")
        dataset = {
            "train": pt_datasets.ImageFolder(root=args.train_dir, transform=train_transforms),
            "validation": pt_datasets.ImageFolder(root=args.validation_dir, transform=val_transforms),
        }
    else:
        data_files = {}
        if args.train_dir is not None:
            data_files["train"] = os.path.join(args.train_dir, "**")
        if args.validation_dir is not None:
            data_files["validation"] = os.path.join(args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            task="image-classification",
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder.

    # If we don't have a validation split, split off a percentage of train as validation.
    args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.

    if args.use_pt_imagefolder:
        labels = dataset["train"].classes
    else:
        labels = dataset["train"].features["labels"].names

    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Load pretrained model and feature extractor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if 'CLIP' in args.model_name_or_path:
        config = CLIPVisionConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        model = CLIPVisionModel.from_pretrained(
            args.model_name_or_path,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            torch_dtype=torch.float16
        )
        model = CLIPViTForImageClassification(config, model.vision_model)
    elif 'eva' in args.model_name_or_path:
        config = json.load(
            open(os.path.join(args.model_name_or_path, 'config.json')))
        config['num_labels'] = len(labels)
        model = EVAViTForImageClassification(**config)
        state_dict = torch.load(os.path.join(
            args.model_name_or_path, 'pytorch_model.bin'))
        model.load_state_dict(state_dict, strict=False)
    else:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            torch_dtype=torch.float16
        )

    #############################################
    # Teacher-Student model
    model = setup_teacher_student(model, args, accelerator)

    # Setup trainable classification heads
    #这个函数会对`model`进行一些操作，使得分类头部可以被训练。
    if args.train_module in ['adapter', 'all']:
        setup_trainable_classification_head(model)
    #通过以上两步的操作，教师-学生模型被设定好，并且可训练的分类头部也被设置好，准备进行训练。
    #############################################

    if args.use_pt_imagefolder:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        #`collate_fn`函数接收一个批量的样本列表，并将像素值和标签分别存储在`pixel_values`和`labels`中，最后返回一个字典，其中包含批量的像素值和标签。
        def collate_fn(examples):
            pixel_values = torch.stack([example[0] for example in examples])
            labels = torch.tensor([example[1] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}
    else:
        #使用`accelerator.main_process_first()`上下文管理器确保只有主进程执行以下代码。
        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                #对训练数据集进行洗牌并选择前`args.max_train_samples`个样本。
                dataset["train"] = dataset["train"].shuffle(
                    seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            #使用`preprocess_train`函数对训练数据集进行转换。
            train_dataset = dataset["train"].with_transform(preprocess_train)
            #如果`args.max_eval_samples`不为`None`，则对验证数据集进行洗牌并选择前`args.max_eval_samples`个样本。
            if args.max_eval_samples is not None:
                dataset["validation"] = dataset["validation"].shuffle(
                    seed=args.seed).select(range(args.max_eval_samples))
            # Set the validation transforms
            #使用`preprocess_val`函数对验证数据集进行转换。
            eval_dataset = dataset["validation"].with_transform(preprocess_val)
        #`collate_fn`函数用于将一个批次的样本转换为模型所需的格式。
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"]
                                        for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

    # DataLoaders creation:

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size, num_workers=args.num_workers
    )

    if args.load_student and not args.restart_training:
        base_results = json.load(
            open(os.path.join(args.load_student, 'all_results.json'), 'r'))
        starting_epoch = base_results['epoch']
        resume_step = base_results['step'] - \
            starting_epoch * len(train_dataloader)
    else:
        starting_epoch = 0
        resume_step = -1

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    #创建了用于优化模型参数的优化器，并对模型参数进行分组设置。
    no_decay = ["bias", "LayerNorm.weight"]
    # `optimizer_grouped_parameters`是一个列表，其中包含多个字典，每个字典用于设置不同组的参数。其中，每个字典包含两个键值对：
    # “params”：一个列表，包含了需要优化的参数。
    # “weight_decay”：一个浮点数，表示权重衰减的系数。
    optimizer_grouped_parameters = [
        {
            #设置不需要权重衰减的参数，即那些参数名称中不包含`no_decay`列表中任何一个字符串，并且不属于`classifier`模块的参数。这些参数将应用权重衰减，系数由`args.weight_decay`指定。
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": args.weight_decay,
        },
        {
            #第二个字典用于设置需要权重衰减的参数，即那些参数名称中包含`no_decay`列表中任何一个字符串，并且不属于`classifier`模块的参数。这些参数将不应用权重衰减，所以权重衰减系数设置为0.0。
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": 0.0,
        },
        {
            #第三个字典用于设置`classifier`模块中的参数，不论是否需要权重衰减。这些参数的权重衰减系数由`args.weight_decay`指定，并且学习率设置为`args.learning_rate * args.classifier_lr_multiplier`。`classifier_lr_multiplier`是一个用于调整`classifier`模块学习率的倍数。
            "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * args.classifier_lr_multiplier
        },
        {
            #根据`args.optimizer`的值选择使用`torch.optim.SGD`还是`torch.optim.AdamW`创建相应的优化器对象，并将`optimizer_grouped_parameters`作为优化器的参数。如果`args.optimizer`的值不是"sgd"或"adamw"，则抛出一个异常。
            "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate * args.classifier_lr_multiplier
        },
    ]
    #根据`args.optimizer`的值选择使用`torch.optim.SGD`还是`torch.optim.AdamW`创建相应的优化器对象，并将`optimizer_grouped_parameters`作为优化器的参数。
    #如果`args.optimizer`的值不是"sgd"或"adamw"，则抛出一个异常。
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters, lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type {args.optimizer}")

    #Scheduler and math around the number of training steps.
    #用于设置学习率调度器（learning rate scheduler）。
    overrode_max_train_steps = False
    #计算了每个训练周期（epoch）中的更新步数（num_update_steps_per_epoch）。
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    #通过判断`args.max_train_steps`是否为None来确定是否需要计算并设置`args.max_train_steps`的值。
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    #调用`get_scheduler`函数来创建学习率调度器。
    # name：学习率调度器的类型。
    # optimizer：优化器对象，用于设置学习率调度器。
    # num_warmup_steps：预热步数，即在训练开始时逐渐增加学习率的步数。计算方式是`args.num_warmup_steps`乘以`args.gradient_accumulation_steps`。
    # num_training_steps：总的训练步数，用于确定学习率调度器的变化范围。计算方式是`args.max_train_steps`乘以`args.gradient_accumulation_steps`。
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("offsite_tuning", experiment_config)

    # Get the metric function
    metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def eval_epoch():
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        return eval_metric["accuracy"]

    if args.select_by_kd:
        teacher_zero_shot_acc = student_zero_shot_acc = 0
        model = to_student(model, args)
    else:
        model = to_teacher(model, args)
        teacher_zero_shot_acc = eval_epoch()

        model = to_student(model, args)
        student_zero_shot_acc = eval_epoch()

    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    logger.info(f"Number of trainable parameters: {trainable_params}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                f"Trainable parameter: {name} with shape {param.shape} and dtype {param.dtype}")

    logger.info(
        f"Teacher zero shot accuracy: {teacher_zero_shot_acc}")
    logger.info(
        f"Student zero shot accuracy: {student_zero_shot_acc}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    def evaluator(model):
        if evaluator.eval_steps == 0:
            return

        task_loss = evaluator.interval_task_loss / evaluator.eval_steps
        kd_loss = evaluator.interval_kd_loss / evaluator.eval_steps
        evaluator.interval_task_loss = 0
        evaluator.interval_kd_loss = 0
        evaluator.eval_steps = 0

        if args.select_by_kd:
            is_best = kd_loss < evaluator.best_kd_loss
            evaluator.best_kd_loss = min(evaluator.best_kd_loss, kd_loss)
            eval_acc = plug_acc = 0
        else:
            model = to_teacher(model, args)
            plug_acc = eval_epoch()
            model = to_student(model, args)
            eval_acc = eval_epoch()
            is_best = eval_acc > evaluator.best_acc
            evaluator.best_acc = max(evaluator.best_acc, eval_acc)

        logger.info(
            f"Epoch {epoch} step {completed_steps}: eval_acc: {eval_acc:.4f} plug_acc: {plug_acc:.4f} task_loss: {task_loss:.4f} kd_loss: {kd_loss:.4f}")

        accelerator.log(
            {
                "eval_acc": eval_acc,
                "plug_acc": plug_acc,
                "acc_gap": plug_acc - eval_acc,
                "train_task_loss": task_loss,
                "train_kd_loss": kd_loss,
                "epoch": epoch,
                "step": completed_steps,
            },
            step=completed_steps,
        )
        if not args.no_save_model and is_best and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = unwrapped_model.student.state_dict()
            for k in state_dict:
                state_dict[k] = state_dict[k].to(torch.float16).cpu()
            torch.save(state_dict, os.path.join(
                args.output_dir, "student.pt"))
            gc.collect()
            torch.cuda.empty_cache()

        if is_best and accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "all_results.json"), "w+") as f:
                json.dump({"best_acc": eval_acc,
                           "plug_acc": plug_acc,
                           "teacher_zero_shot_acc": teacher_zero_shot_acc,
                           "student_zero_shot_acc": student_zero_shot_acc,
                           "train_task_loss": task_loss,
                           "train_kd_loss": kd_loss,
                           "epoch": epoch,
                           "step": completed_steps,
                           "trainable_params": trainable_params}, f)

    evaluator.best_acc = student_zero_shot_acc
    evaluator.best_kd_loss = float("inf")
    evaluator.eval_steps = 0
    evaluator.interval_task_loss = 0
    evaluator.interval_kd_loss = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_task_loss, total_kd_loss = 0, 0
        skipped_steps = 0

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.load_student and epoch == starting_epoch and step <= resume_step:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Skipping step {step} (already completed)")
                completed_steps += 1
                skipped_steps += 1
                continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                task_loss = outputs.loss

                kd_loss = get_kd_loss(model)

                loss = args.lm_weight * task_loss + args.kd_weight * \
                    kd_loss if args.kd_weight != 0 else task_loss
                progress_bar.set_description(
                    f"Epoch {epoch} - Step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - Task loss: {task_loss:.4f} - KD loss: {kd_loss:.4f}")

                total_task_loss += task_loss.item()
                total_kd_loss += kd_loss.item()

                evaluator.interval_task_loss += task_loss.item()
                evaluator.interval_kd_loss += kd_loss.item()
                evaluator.eval_steps += 1

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # end accumulate gradients

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            else:
                continue

            if completed_steps % args.eval_steps == 0:
                evaluator(model)

        evaluator(model)

    accelerator.end_training()


if __name__ == "__main__":
    main()
