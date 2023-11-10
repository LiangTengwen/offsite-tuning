from datasets import load_dataset
import logging
from itertools import chain
from offsite_tuning.tasks import task_dict, map_dataset_name_and_config

logger = logging.getLogger(__name__)

#取数据集
def get_raw_datasets(args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset_name, dataset_config_name = map_dataset_name_and_config(args)
        raw_datasets = load_dataset(
            dataset_name, dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        elif extension == 'zst':
            extension = 'json'
        raw_datasets = load_dataset(
            extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    return raw_datasets

#分词
def get_tokenized_datasets(raw_datasets, args, accelerator, tokenizer, lm_type='clm'):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        if lm_type == 'clm':
            return tokenizer(examples[text_column_name])
        elif lm_type == 'mlm':
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
        else:
            raise ValueError(f'lm_type {lm_type} not supported')

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    return tokenized_datasets


def _get_block_size(args, tokenizer):
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)
    return block_size


def get_lm_datasets(tokenized_datasets, args, accelerator, tokenizer, lm_type='clm'):
    block_size = _get_block_size(args, tokenizer)
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        if lm_type == 'clm':
            result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    return lm_datasets


def process_text2text_datasets(raw_datasets, args, tokenizer, accelerator):
    #获取任务相关信息：从`task_dict`中获取与`args.dataset_name`对应的任务信息。
    task = task_dict[args.dataset_name]

    column_names = raw_datasets["train"].column_names

    
    def tokenize_function(examples):
        #通过任务信息获得上下文(`context`)和目标(`target`)文本。
        context = task.get_context(examples)
        target = task.get_target(examples)

        context = tokenizer(context)
        target = tokenizer(target)
        #分别对上下文和目标进行分词。如果上下文以特殊标记符(token)结尾，将其移除。如果目标以特殊标记符开头，则将其移除。
        # if context is ending with special token, remove it
        if len(context['input_ids'][0]) > 0 and context['input_ids'][0][-1] in tokenizer.all_special_ids:
            context['input_ids'] = [i[:-1] for i in context['input_ids']]
            context['attention_mask'] = [a[:-1]
                                         for a in context['attention_mask']]

        # if target is starting with special token, remove it
        if len(target['input_ids'][0]) > 0 and target['input_ids'][0][0] in tokenizer.all_special_ids:
            target['input_ids'] = [i[1:] for i in target['input_ids']]
            target['attention_mask'] = [a[1:]
                                        for a in target['attention_mask']]
        #将上下文和目标的分词结果拼接在一起，并创建相应的注意力掩码。
        out = {}
        out['input_ids'] = [i1 + i2 for i1,
                            i2 in zip(context['input_ids'], target['input_ids'])]
        out['attention_mask'] = [a1 + a2 for a1,
                                 a2 in zip(context['attention_mask'], target['attention_mask'])]
        #为上下文标记设定标签，值为-100，以便在训练时模型不会预测上下文中的标记。
        # set -100 for context tokens
        out["labels"] = [
            [-100] * len(i1) + i2 for i1, i2 in zip(context['input_ids'], target['input_ids'])]

        return out
    #并行处理和分词：使用`raw_datasets`数据集对象的`map`方法对数据集进行分词，并行处理以提高处理速度。
    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if "gpt2" in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.bos_token

    # pad all instances in lm_datasets to the max length of the dataset
    max_length = -1
    for v in tokenized_datasets.values():
        for x in v:
            max_length = max(max_length, len(x['input_ids']))

    # pad to the multiple of 8
    max_length = (max_length // 8 + 1) * 8

    block_size = _get_block_size(args, tokenizer)
    max_length = min(max_length, block_size)

    #该函数对分词后的数据进行填充，使其达到最大长度。
    def pad_function(examples):
        #首先，将`input_ids`列表中的每个实例补充到最大长度，并在末尾添加填充标记（通过`tokenizer.pad_token_id`获取）。
        examples["input_ids"] = [i + [tokenizer.pad_token_id] *
                                 (max_length - len(i)) for i in examples["input_ids"]]
        #然后，在`attention_mask`中为每个实例添加相应的注意力掩码。
        examples["attention_mask"] = [[1] * len(i) + [0] *
                                      (max_length - len(i)) for i in examples["attention_mask"]]
        #最后，将`labels`列表中的每个实例补充到最大长度，并在末尾添加填充标记-100，同时截断超过最大长度的部分。
        examples["labels"] = [i + [-100] *
                              (max_length - len(i)) for i in examples["labels"]]
        # truncate to max_length
        examples["input_ids"] = [i[:max_length] for i in examples["input_ids"]]
        examples["attention_mask"] = [a[:max_length]
                                      for a in examples["attention_mask"]]
        examples["labels"] = [l[:max_length] for l in examples["labels"]]
        return examples
    #并行填充处理：使用`tokenized_datasets`数据集对象的`map`方法对数据集进行填充处理，并行处理以提高处理速度。
    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(
            pad_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Padding dataset to max length {max_length}",
        )

    return tokenized_datasets
