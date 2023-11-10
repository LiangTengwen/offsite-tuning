#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import sys
import datasets
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from copy import deepcopy

import transformers
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel
from transformers.models.bloom.modeling_bloom import BloomForCausalLM, BloomModel

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    get_scheduler,
)
from datasets import load_from_disk
from offsite_tuning.tasks import task_dict
from offsite_tuning.data import get_raw_datasets, get_tokenized_datasets, get_lm_datasets, process_text2text_datasets
from offsite_tuning.utils import (
    MLP,
    add_epilogue,
    add_prologue,
    uniform_choose_layers,
    magnitude_prune,
    quantize,
    parse_args,
    setup_teacher_student,
    get_kd_loss,
    save_state_dict,
    to_student,
    to_teacher
)

from offsite_tuning.param_efficient import (
    use_lora,
    use_bitfit,
    use_adapter
)
import gc

logger = get_logger(__name__)


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    # also log to a file in output_dir
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
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            torch_dtype=torch.float16
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.

    # 首先，代码通过 `model.get_input_embeddings().weight.shape[0]` 
    # 获取嵌入层的大小（即词嵌入的维度），并将其赋值给变量 `embedding_size`。
    # 接下来，代码检查 tokenizer 的长度是否大于嵌入层的大小。
    # 如果 tokenizer 的长度大于嵌入层的大小，代码通过 
    # `model.resize_token_embeddings(len(tokenizer))` 调整嵌入层的大小，
    # 使其能够适应新的 tokenizer。
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    # 然后，代码检查参数 `args.dataset_name` 是否存在于 `task_dict` 中。
    # 如果存在，表示正在处理特殊案例的数据集，具体是针对e2e_nlg 数据集。
    # 在这种特殊情况下，代码通过调用 `get_raw_datasets(args)` 获取原始数据集，
    # 并通过 `process_text2text_datasets()` 对数据集进行处理，生成用于语言模型训练的数据集 `lm_datasets`。
    if args.dataset_name in task_dict:  # special case for e2e_nlg dataset
        raw_datasets = get_raw_datasets(args)
        lm_datasets = process_text2text_datasets(
            raw_datasets, args, tokenizer, accelerator)
    #如果 `args.dataset_name` 不在 `task_dict` 中，则表示处理的是其他数据集。
    # 代码接着检查参数 `args.train_tokenized_dataset` 和 `args.val_tokenized_dataset` 是否存在，
    # 如果两者都存在，则表示已经有经过分词的数据集可用。
    # 代码通过 `load_from_disk()` 加载经过分词的训练和验证数据集，并将它们存储在 `tokenized_datasets` 中。
    else:
        if args.train_tokenized_dataset and args.val_tokenized_dataset:
            tokenized_datasets = load_from_disk(args.train_tokenized_dataset)
            val_dataset = load_from_disk(args.val_tokenized_dataset)
            # 如果加载的验证数据集中包含键 `'validation'`，则将其赋值给 `tokenized_datasets["validation"]`；
            # 否则，将其赋值给 `tokenized_datasets["train"]`。
            if 'validation' in val_dataset:
                tokenized_datasets["validation"] = val_dataset['validation']
            else:
                tokenized_datasets["validation"] = val_dataset['train']
        else:
            #如果 `args.train_tokenized_dataset` 和 `args.val_tokenized_dataset` 不存在，
            # 则需要从原始数据集开始处理。代码通过调用 `get_raw_datasets(args)` 获取原始数据集，
            # 并通过 `get_tokenized_datasets()` 在加速器上对数据集进行分词处理，
            # 得到经过分词的数据集 `tokenized_datasets`。
            raw_datasets = get_raw_datasets(args)

            tokenized_datasets = get_tokenized_datasets(
                raw_datasets, args, accelerator, tokenizer)

        lm_datasets = get_lm_datasets(
            tokenized_datasets, args, accelerator, tokenizer)
    #这段代码的作用是根据给定的参数和数据集，进行数据预处理，
    # 生成用于模型训练和评估的数据集。
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    #如果设置了参数 `args.train_num_samples`，则检查训练数据集中样本的数量是否足够。
    # 如果设置的训练样本数大于训练数据集的长度，将 `args.train_num_samples` 设置为训练数据集的长度。
    # 然后，通过选择前 `args.train_num_samples` 个样本，将 `train_dataset` 调整为包含指定数量样本的数据集。
    if args.train_num_samples is not None:
        # check if we have enough samples for the training set
        if args.train_num_samples > len(train_dataset):
            args.train_num_samples = len(train_dataset)
        train_dataset = train_dataset.select(
            range(args.train_num_samples))
    #如果设置了参数 `args.validation_num_samples`，则检查验证数据集中样本的数量是否足够。
    # 如果设置的验证样本数大于验证数据集的长度，将 `args.validation_num_samples` 设置为验证数据集的长度。
    # 然后，通过选择前 `args.validation_num_samples` 个样本，将 `eval_dataset` 调整为包含指定数量样本的数据集。
    if args.validation_num_samples is not None:
        # check if we have enough samples for the validation set
        if args.validation_num_samples > len(eval_dataset):
            args.validation_num_samples = len(eval_dataset)
        eval_dataset = eval_dataset.select(
            range(args.validation_num_samples))
    #创建 `collator` 对象，用于对样本进行批次处理。然后，使用 `DataLoader` 创建训练数据集和验证数据集的数据加载器。
    # 训练数据加载器使用随机洗牌（shuffle=True），并且指定批次大小为 `args.per_device_train_batch_size`。
    # 验证数据加载器没有洗牌，批次大小为 `args.per_device_eval_batch_size`。
    collator = default_data_collator
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collator, batch_size=args.per_device_eval_batch_size
    )

    model = setup_teacher_student(model, args, accelerator)

    if args.no_teacher:
        model.teacher = None
        to_student(model, args)
        gc.collect()
        torch.cuda.empty_cache()
    #如果 `args.train_module` 在 `['adapter', 'all']` 中并且 `args.train_lm_head` 为真，
    # 则将语言模型头部中的参数设置为可训练，并将参数的数据类型转为 `float` 类型。
    if args.train_module in ['adapter', 'all'] and args.train_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = True
            param.data = param.data.float()

    if args.use_lora:
        use_lora(model.trainable_module, args.lora_rank, args.lora_alpha)

    if args.use_adapter:
        use_adapter(model.trainable_module, args.adapter_size)

    if args.use_bitfit:
        use_bitfit(model.trainable_module)
    #如果设置了参数 `args.load_student` 并且 `args.restart_training` 为假，
    # 则加载预训练的学生模型，并根据加载的模型设置起始的 epoch 和 resume step 的值。
    # 否则，将起始的 epoch 设置为 0，resume step 设置为 -1。
    if args.load_student and not args.restart_training:
        base_results = json.load(
            open(os.path.join(args.load_student, 'all_results.json'), 'r'))
        starting_epoch = base_results['epoch']
        resume_step = base_results['step'] - \
            starting_epoch * len(train_dataloader)
    else:
        starting_epoch = 0
        resume_step = -1
    #统计模型中可训练参数的数量，并记录日志。
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    logger.info(f"Number of trainable parameters: {trainable_params}")
    #遍历模型的每个参数，如果参数为可训练的，则记录参数的名称、形状和数据类型的日志。
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                f"Trainable parameter: {name} with shape {param.shape} and dtype {param.dtype}")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

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
    #通过除以梯度累积步数 `args.gradient_accumulation_steps`，将训练数据集的长度除以每个epoch中的更新步数，
    # 然后使用 `math.ceil()` 函数向上取整，得到一个整数值 `num_update_steps_per_epoch`，
    # 表示训练一个epoch所需的更新步数。
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    #接下来，如果之前覆盖了 `args.max_train_steps` 的设置，即 `overrode_max_train_steps` 为 `True`，
    # 则将 `args.max_train_steps` 重新计算为训练epoch数 `args.num_train_epochs` 乘以 `num_update_steps_per_epoch`。
    # 这是因为在之前的代码中，如果没有设置 `args.max_train_steps`，
    # 则根据 `args.num_train_epochs` 和 `num_update_steps_per_epoch` 计算出 `args.max_train_steps`，
    # 后续的代码会使用这个值。
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Afterwards we recalculate our number of training epochs
    #最后，通过将 `args.max_train_steps` 除以 `num_update_steps_per_epoch`，
    # 再使用 `math.ceil()` 函数向上取整，重新计算出训练的epoch数，赋值给 `args.num_train_epochs`。
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)
    
    #这段代码的作用是根据数据集大小和梯度累积步数计算训练一个epoch所需的更新步数，并根据已计算的更新步数和训练epoch数重新计算训练的epoch数。

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("offsite_tuning", experiment_config)

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

    #用于在评估阶段计算损失和困惑度。
    def eval_epoch():
        #首先，将模型设置为评估模式，即 `model.eval()`。
        model.eval()
        losses = []
        #然后，使用一个循环遍历评估数据集 `eval_dataloader`中的每个批次数据。
        #在循环中，通过 `torch.no_grad()` 上下文管理器关闭了梯度计算，以降低内存消耗。
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            #对于每个批次数据，通过调用模型(`model`)并传入批次数据(`**batch`)来获取模型的预测输出(`outputs`)。
            # 其中，`**batch`表示将批次数据解包成关键字参数。
            
            #然后，从预测输出中获取损失值(`loss`)，并将其添加到列表 `losses` 中。
            #这里使用了 `accelerator.gather_for_metrics()` 函数来收集损失值，以便进行度量。
            loss = outputs.loss
            #接下来，将列表 `losses` 中的损失值进行拼接，使用 `torch.cat()` 
            # 函数将它们拼接成一个一维张量(`losses`)。
            losses.append(accelerator.gather_for_metrics(
                loss.repeat(args.per_device_eval_batch_size)).cpu())
        losses = torch.cat(losses).flatten()
        # filter out nan
        #然后，过滤掉损失值中的 `nan` 值，通过索引选择不包含 `nan` 的损失值，
        # 使用 `~torch.isnan(losses)` 进行掩码操作。
        losses = losses[~torch.isnan(losses)]
        #尝试计算平均损失(`eval_loss`)和困惑度(`perplexity`)。如果遇到 `OverflowError`，
        # 则将困惑度设置为正无穷大(`float("inf")`)。
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")
        #返回计算得到的评估损失(`eval_loss`)和困惑度(`perplexity`)。
        return eval_loss, perplexity

    #如果为 `False` 表示使用教师模型。
    # 然后调用 `to_teacher(model.module, args)` 函数，该函数将学生模型转换为教师模型。
    # 接下来，调用 `eval_epoch()` 函数计算教师模型在评估数据集上的评估损失和困惑度，将得到的零射击困惑度存储在变量 `teacher_zero_shot_perplexity` 中。
    if not args.no_teacher:
        to_teacher(model.module, args)
        _, teacher_zero_shot_perplexity = eval_epoch()
        logger.info(
            f"Teacher zero shot perplexity: {teacher_zero_shot_perplexity}")
    #如果 `args.no_teacher` 为 `True`，表示不使用教师模型，
    #则将 `teacher_zero_shot_perplexity` 设置为0。
    else:
        teacher_zero_shot_perplexity = 0

    #该函数将模型从教师模型转换回学生模型。
    to_student(model.module, args)

    # for name, param in model.named_parameters():
    #     logger.info(
    #         f"Parameter: {name} with shape {param.shape}, dtype {param.dtype}, and requires_grad {param.requires_grad}")

    #调用 `eval_epoch()` 函数计算学生模型在评估数据集上的评估损失和困惑度，
    #将得到的零射击困惑度存储在变量 `student_zero_shot_perplexity` 中。
    _, student_zero_shot_perplexity = eval_epoch()
    logger.info(
        f"Student zero shot perplexity: {student_zero_shot_perplexity}")
    best_perplexity = float("inf")

    #代码中的 `best_perplexity` 被初始化为正无穷大(`float("inf")`)，
    #并且进度条(`progress_bar`)被设置为循环迭代 `args.max_train_steps` 次，
    #用于表示训练过程的进度条。最后，初始化变量 `completed_steps` 为0，
    #用于记录已完成的训练步骤的数量。
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)

    completed_steps = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_lm_loss, total_kd_loss = 0, 0
        interval_lm_loss, interval_kd_loss = 0, 0
        best_lm_loss, best_kd_loss = float("inf"), float("inf")
        skipped_steps = 0
        #迭代训练数据集中的每个批次(`batch`)
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            #如果设置了 `args.load_student` 并且当前批次的步骤数小于等于已恢复的步骤数(`resume_step`)，
            #则跳过该步骤。在跳过的步骤中，更新进度条，将已完成的训练步骤数量 `completed_steps` 
            #和被跳过的步骤数量 `skipped_steps`都增加1。然后继续下一个循环。
            if args.load_student and epoch == starting_epoch and step <= resume_step:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Skipping step {step} (already completed)")
                completed_steps += 1
                skipped_steps += 1
                continue
            
            with accelerator.accumulate(model):
                outputs = model(**batch)
                lm_loss = outputs.loss
                #如果 `args.no_teacher` 为 `False`，即使用教师模型，
                #调用 `get_kd_loss(model.module)` 函数来获取知识蒸馏损失 `kd_loss`，否则 `kd_loss` 设置为0。
                if not args.no_teacher:
                    kd_loss = get_kd_loss(model.module)
                else:
                    kd_loss = 0
                #计算总损失 `loss`，其中 `args.lm_weight` 表示语言模型损失的权重，
                #`args.kd_weight` 表示知识蒸馏损失的权重。如果 `args.kd_weight` 不等于0，
                #则使用加权求和的方式计算总损失，否则只使用语言模型损失。
                loss = args.lm_weight * lm_loss + args.kd_weight * \
                    kd_loss if args.kd_weight != 0 else lm_loss
                progress_bar.set_description(
                    f"Epoch {epoch} - Step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - LM loss: {lm_loss:.4f} - KD loss: {kd_loss:.4f}")

                #累积语言模型损失和知识蒸馏损失，分别存储在 `total_lm_loss` 和 `total_kd_loss` 变量中
                total_lm_loss += lm_loss.item()
                #更新在当前间隔(`interval`)中的语言模型损失和知识蒸馏损失，
                #分别存储在 `interval_lm_loss` 和 `interval_kd_loss` 变量中。
                interval_lm_loss += lm_loss.item()
                #此外，更新最佳的语言模型损失和知识蒸馏损失，分别存储在 `best_lm_loss` 和 `best_kd_loss` 变量中。
                best_lm_loss = min(best_lm_loss, lm_loss.item())

                if not args.no_teacher:
                    total_kd_loss += kd_loss.item()
                    interval_kd_loss += kd_loss.item()
                    best_kd_loss = min(best_kd_loss, kd_loss.item())
                
                #使用 `accelerator.backward(loss)` 计算梯度，并在需要时进行梯度裁剪(`accelerator.clip_grad_norm_()`)
                #更新模型参数(`optimizer.step()`)、更新学习率调度器(`lr_scheduler.step()`)，并将梯度清零(`optimizer.zero_grad()`)。
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
            #如果 `completed_steps` 除以 `args.eval_steps` 的余数为0，即完成了一个完整的间隔(`interval`)。
            #该条件判断确保只在完成了指定步数的训练后执行下面的操作。
            if completed_steps % args.eval_steps == 0:
                if not args.no_teacher:
                    #调用 `to_teacher(model.module, args)` 函数来将当前模型切换为教师模型
                    to_teacher(model.module, args)
                    plug_eval_loss, plug_ppl = eval_epoch()
                else:
                    plug_eval_loss, plug_ppl = 0, 0
                #调用 `to_student(model.module, args)` 函数来将当前模型切换为学生模型，
                #并传递相关参数 `args`。再次调用 `eval_epoch()` 函数对当前模型进行评估，得到评估损失 `eval_loss` 和困惑度 `perplexity`。
                to_student(model.module, args)
                eval_loss, perplexity = eval_epoch()

                #计算损失：
                #计算当前间隔(`interval`)中的平均语言模型损失 `lm_loss` 和平均知识蒸馏损失 `kd_loss`，通过除以 `args.eval_steps` 来计算。
                lm_loss = interval_lm_loss / args.eval_steps
                kd_loss = interval_kd_loss / args.eval_steps
                #将 `interval_lm_loss` 和 `interval_kd_loss` 重新设置为0，
                #以便在下一个间隔(`interval`)进行累积。
                interval_lm_loss = 0
                interval_kd_loss = 0

                #日志记录和保存模型：
                logger.info(
                    f"epoch {epoch} step {completed_steps}: student_ppl: {perplexity:.4f} plug_ppl: {plug_ppl:.4f} lm_loss: {lm_loss:.4f} kd_loss: {kd_loss:.4f}")

                accelerator.log(
                    {
                        "student_ppl": perplexity,
                        "student_eval_loss": eval_loss,
                        "plug_ppl": plug_ppl,
                        "plug_eval_loss": plug_eval_loss,
                        "ppl_gap": perplexity - plug_ppl,
                        "train_lm_loss": lm_loss,
                        "train_kd_loss": kd_loss,
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )
                
                #判断当前模型的困惑度是否是最佳性能(`is_best`)
                is_best = perplexity < best_perplexity
                best_perplexity = min(best_perplexity, perplexity)

                #如果 `args.no_save_model` 为 `False`、`is_best` 为 `True`，
                #且分布式训练中的主进程 (`accelerator.is_main_process`) 为 True，则将模型的状态字典保存到文件中。
                #根据 `args.save_module` 的值，可以选择保存学生模型、适配器模型或两者。
                if not args.no_save_model and is_best and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    if args.save_module in ["student", "all"]:
                        state_dict = unwrapped_model.student.state_dict()
                        save_state_dict(
                            state_dict, args.output_dir, "student.pt")
                    if args.save_module in ["adapter", "all"]:
                        state_dict = unwrapped_model.adapter.state_dict()
                        save_state_dict(
                            state_dict, args.output_dir, "adapter.pt")

                    gc.collect()
                    torch.cuda.empty_cache()
                #记录所有结果：
                #如果 `is_best` 为 `True`，且分布式训练中的主进程 (`accelerator.is_main_process`) 为 True，
                #则将最佳困惑度 (`best_perplexity`)、插值困惑度 (`plug_ppl`)、教师模型的零样本困惑度 (`teacher_zero_shot_perplexity`)、
                #学生模型的零样本困惑度 (`student_zero_shot_perplexity`)、语言模型损失 (`lm_loss`)、知识蒸馏损失 (`kd_loss`)、当前的训练轮数和步数、可训练参数的数量 (`trainable_params`) 写入一个JSON文件。
                #该文件用于记录所有结果的综合统计。
                if is_best and accelerator.is_main_process:
                    with open(os.path.join(args.output_dir, "all_results.json"), "w+") as f:
                        json.dump({"best_perplexity": best_perplexity,
                                   "plug_perplexity": plug_ppl,
                                   "teacher_zero_shot_perplexity": teacher_zero_shot_perplexity,
                                   "student_zero_shot_perplexity": student_zero_shot_perplexity,
                                   "train_lm_loss": lm_loss,
                                   "train_kd_loss": kd_loss,
                                   "epoch": epoch,
                                   "step": completed_steps,
                                   "trainable_params": trainable_params}, f)

    accelerator.end_training()


if __name__ == "__main__":
    main()
