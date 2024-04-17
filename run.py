from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from module.dataloader import create_data_loader,read_file
from module.add_adapter import PaperClassifier
from module.train import train

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from logger import logger

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

def main(verbose=True):
    RANDOM_SEED = 2024

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True 

    device_map = None

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        # quantization_config=GPTQConfig(
        #     bits=4, disable_exllama=True
        # )
        # if training_args.use_lora and lora_args.q_lora
        # else None
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    modules_to_save = ["wte", "lm_head"]

    lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="TOKEN_CLS",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )

    model = get_peft_model(model, lora_config)

    # Print peft trainable params
    model.print_trainable_parameters()
    if verbose:
        # 遍历模型的所有模块和参数
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                # 参数存在且可训练（requires_grad=True）
                if module.weight.requires_grad:
                    logger.info(f"Module Name: {name}, Parameter is trainable.")
                else:
                    logger.info(f"Module Name: {name}, Parameter is not trainable.")

        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"Parameter Name: {name}, Updateable: True")
            else:
                logger.info(f"Parameter Name: {name}, Updateable: False")



    tokenizer = AutoTokenizer.from_pretrained(
        'QwenBase',
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        padding_side='left',
        trust_remote_code=True
    )

    df_train = read_file("train.xlsx")
    df_valid  =  read_file("val.xlsx")
    df_test = read_file("test.xlsx")

    train_dataloader = create_data_loader(data=df_train,tokenizer=tokenizer,max_len=512,batch_size=2)
    valid_dataloader = create_data_loader(data=df_valid,tokenizer=tokenizer,max_len=512,batch_size=2)
    test_dataloader = create_data_loader(data=df_test,tokenizer=tokenizer,max_len=512,batch_size=2)

    model = PaperClassifier(model=model,tokenizer=tokenizer,n_classes=7)

    train(model,train_dataloader,valid_dataloader,test_dataloader)


main()


# 




# 
