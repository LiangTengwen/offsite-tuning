import os
from offsite_tuning.utils import parse_args, load_adapter, load_student, get_layers, set_layers, uniform_choose_layers
from offsite_tuning.tasks import LM_EVAL_TASK_NAME_MAPPING
import torch
from lm_eval.base import BaseLM
from lm_eval import evaluator, tasks
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate.logging import get_logger

logger = get_logger(__name__)


class LMEvalAdaptor(BaseLM):

    def __init__(self, model, tokenizer, batch_size=1):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if hasattr(self.model.config, 'n_ctx'):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        else:
            return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            out = self.model(inps)[0]
            return out  # [:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )


def main():
    #解析命令行参数：使用`parse_args`函数解析命令行参数，将结果保存在`args`对象中。
    args = parse_args()
    #加载预训练模型：使用`AutoModelForCausalLM.from_pretrained`函数加载预训练模型。在这个函数中，会根据`args.model_name_or_path`指定的模型名称或路径加载相应的模型。如果`torch_dtype`参数被指定为`torch.float16`，则将模型参数的数据类型设置为`torch.float16`。
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float16)

    #设置学生模型的层数:如果`args.num_student_layers`参数不为`None`，则通过`get_layers`函数获取模型的所有层，并使用`uniform_choose_layers`函数选择指定数量的学生模型层。最后，使用`set_layers`函数将选定的学生模型层设置到模型中。
    if args.num_student_layers is not None:
        layers = get_layers(model)
        layers = uniform_choose_layers(layers, args.num_student_layers)
        set_layers(model, layers)
    #加载适配器:如果`args.load_adapter`参数被指定，则使用`torch.load`函数加载适配器模型的状态字典，并将适配器加载到模型中。
    if args.load_adapter:
        adapter_state_dict = torch.load(args.load_adapter, map_location='cpu')
        model = load_adapter(model, adapter_state_dict, args)
    #加载学生模型：如果`args.load_student`参数被指定，则使用`torch.load`函数加载学生模型的状态字典，并将学生模型加载到模型中。
    if args.load_student:
        student_state_dict = torch.load(args.load_student, map_location='cpu')
        model = load_student(model, student_state_dict, args)

    model = model.to("cuda")
    #加载分词器：使用`AutoTokenizer.from_pretrained`函数加载预训练模型对应的分词器。这个分词器将用于将输入文本分成适合模型输入的标记。
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    #创建语言模型评估器对象：将模型和分词器传递给`LMEvalAdaptor`类的构造函数，创建语言模型评估器对象`lm_eval_model`。
    lm_eval_model = LMEvalAdaptor(model, tokenizer)
    #确定任务列表：根据命令行参数中指定的任务列表，确定要在评估中使用的任务。如果`args.tasks`参数为`None`，则使用预定义的所有任务列表`tasks.ALL_TASKS`；否则，根据逗号分隔的任务名称字符串创建任务列表。
    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    #
    task_names = [LM_EVAL_TASK_NAME_MAPPING.get(t, t) for t in task_names]
    #进行简单评估：使用评估器对象`evaluator`的`simple_evaluate`方法对模型进行评估。传递给该方法的参数包括模型`lm_eval_model`、任务列表`task_names`、批量大小`batch_size`、是否禁用缓存`no_cache`等。
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=128,
        no_cache=True,
    )
    #打印评估结果
    print(evaluator.make_table(results))
    #输出评估结果到文件：如果`args.output_dir`参数被指定，则创建相应的输出目录，将评估结果中的模型配置移除，并将结果以JSON格式保存在输出文件中。
    if args.output_dir is not None:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        del results["config"]["model"]
        with open(args.output_dir, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
