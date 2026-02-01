# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import copy
import logging
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import json
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from safetensors.torch import load_file
from tqdm import tqdm
from math import ceil
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from functools import partial

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model
)

IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# compute_loss方法通常不需要用户显式调用, Trainer类在训练循环（如_inner_training_loop）中会自动调用compute_loss方法计算损失
# 1.若需修改损失计算逻辑（如添加正则化、调整标签处理），需通过以下方式覆盖默认方法：
class CustomTrainer(Trainer):
    # 模型调用时指定args=training_args, 因此这里self.args默认为调用training_args的参数
    # 这里的inputs是data_collator, 也就是DataCollatorForSupervisedDataset的输出
    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        # self.state是Trainer父类中定义的一个动态更新的状态字典，存储训练过程中的实时信息，包括：
        # -- 训练进度：当前epoch、全局步数（global_step）
        # -- 性能指标：训练损失、验证损失、学习率等
        # -- 控制标志：如是否应中断训练（should_training_stop）
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size      # for gpt2. 64
        gradient_accumulation_steps = self.args.gradient_accumulation_steps     # 2
        num_epochs = self.args.num_train_epochs     # 40
        dataset_size = len(self.train_dataset)

        # self.args.world_size, 表示当前分布式训练中的总进程数（如GPU数量）, world_size 由分布式训练环境自动计算
        # world_size继承自TrainingArguments, 也就是必须继承TrainingArguments的模型才能调用此参数, 但是不需要自己指定值, 可以直接调用
        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method, 这里的model是一个class, 不是model.codi
        outputs = model(**inputs)
        loss = outputs["loss"]
        #"ce_loss": ce_loss_total, "mse_loss": mse_loss_total, "ref_ce_loss": ref_ce_loss
        if step % self.args.logging_steps == 0:
            self.log({"loss": loss.item(), "ce_loss": outputs["ce_loss"], "distill_loss": outputs["distill_loss"], "ref_ce_loss": outputs["ref_ce_loss"],})
        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, training_args) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length= training_args.model_max_length, # default: 256
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    # 因为是训练，所以input_text就是本身的label
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # ne: not equal, item(): tensor-> int; 这一步计算输入有效token的长度
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = [sentence]
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ##########################
    #       Peft Model       #
    ##########################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM  
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,        # 控制lora rank大小，值越大参数量越多，拟合能力越强	
            lora_alpha=model_args.lora_alpha,   # 缩放新增参数的梯度，避免原始权重被剧烈扰动
            lora_dropout=0.1,           # 在LoRA层添加Dropout防止微调过拟合
            target_modules=target_modules,  # 指定哪些层添加LoRA（冻结其他层参数）
            init_lora_weights=True,
        )

    # 定义初始化模型
    model = CODI(model_args, training_args, lora_config)
    # infer时, padding_side='left', train时padding_side="right"; 注意一下逻辑区别
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,             # "HF token to access to private models, e.g., meta-llama"
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,    # tokenizer编码长度, 512 for gpt2
            padding_side="right",
            use_fast=False,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None: # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    def get_answer_token_position(tokens, answer_prompts, tokenizer):
        # 寻找answer_id中, answer_prompt所在的位置是多少，好奇为啥不用字符串匹配
        #answer_prompt = torch.tensor([464, 3280, 318, 25])
        try:
            match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
            answer_token_id = match_indices + len(answer_prompts[0])
            return answer_token_id
        except Exception:
            breakpoint()

    def preprocess(
        sources: Sequence[str],     # questions
        targets: Sequence[str],     # cots
        answers: Sequence[str],     # answers
        tokenizer: transformers.PreTrainedTokenizer, 
        bot_id: int,
        eot_id: int,
    ) -> Dict:
        print("Step 2.2 For data preprocess. Tokenizing inputs... This may take some time...")
        sources_id = _tokenize_fn(sources, tokenizer, training_args)["input_ids"]
        cot_id = _tokenize_fn(targets, tokenizer, training_args)["input_ids"]
        answers_id = _tokenize_fn(answers, tokenizer, training_args)["input_ids"]

        print("Step 2.3 For data preprocess. Format the inputs.")
        # add eos token to accomodate pretrained model's format
        if not training_args.remove_eos:  # false
            sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
            cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
        answers_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]

        # 如果cot_id和answer_id包含bos_token，进行剔除
        if cot_id[0][0] == tokenizer.bos_token_id:
            cot_id = [x[1:] for x in cot_id]
            answers_id = [x[1:] for x in answers_id]
        # 做到question, cot, answer的无缝拼接 (如果是组batch，会遇到padding="longest"使短文本中间拼接的问题?)
        ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_id)]
        ref_labels = []
        for x, y in zip(ref_input_ids, sources_id):
            z = x.clone()
            z[:len(y)] = -100       # 这个地方为什么是-100, 而不是pad_token_id，或者eos_token_id?
            ref_labels.append(z)
        
        # add eot to source
        sources_id = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
        # add eot and eos
        if training_args.remove_eos:    # true
            answers_id = [torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]
        else:
            answers_id = [torch.tensor([eot_id, tokenizer.eos_token_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]

        # e.g. for gpt2: answer_prompts: [tensor([ 464, 3280,  318,   25]), tensor([ 464, 1306, 2239, 1255,  318,   25])]
        answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")), torch.tensor(tokenizer.encode("The next step result is:"))]
        if answer_prompts[0][0] == tokenizer.bos_token_id: # remove the bos
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]
        
        # 获取ref_input_ids, answers_id中答案对应的开始位置
        ref_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for i, x in enumerate(ref_input_ids)]
        model_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id]
        # 获取ref_input_ids, answers_id中, eos 对应的位置
        ref_eos_position = [len(x)-1 for x in ref_input_ids]
        model_eos_position = [len(x)-1 for x in answers_id]
        # 这里做一个详细的格式说明: 
        # sources[0]: 'Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?'
        # sources_id[0]: querstion_id + bot_id, tensor([ 7975, 286, 10053, ...,  257, 7202, 30, 50258])
        # cots[0]: '<<600*30/100=180>> <<600*10/100=60>> <<180+60=240>>'
        # cot_id[0]: tensor([16791,8054,9, 1270, ..., 28, 16102, 4211])
        # answers[0]: 'The answer is: 360'
        # answers_id[0]: eot_id + answer_id + eos_id, tensor([50259, 464, 3280, 318, 25, 11470, 50256])
        # ref_input_ids: question_id + cot_id + answer_id + eos_id
        # ref_answer_position: ref_input_ids中答案对应的开始位置, top10 is [60, 34, 28, 39, 80, 106, 84, 50, 90, 43]
        # model_answer_position: answers_id中答案对应的开始位置, top10 is [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        # ref_eos_position: ref_input_ids中, eos 对应的位置,  top10 is [61, 35, 29, 40, 83, 107, 85, 51, 93, 44]
        # model_eos_position: answers_id中, eos 对应的位置, top10 is: [6, 6, 6, 6, 8, 6, 6, 6, 8, 6]
        # ref_labels: [-100*len(question_id), cot_id, answer_id, eos_id]
        return dict(encoder_input_ids=sources_id, decoder_input_ids=answers_id, ref_input_ids=ref_input_ids, labels=answers_id, \
                    ref_answer_position=ref_answer_position, model_answer_position=model_answer_position, \
                        ref_eos_position=ref_eos_position, model_eos_position=model_eos_position, ref_labels=ref_labels)

    class SupervisedDataset(Dataset):
        QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
        QUESTION_DA_PROMPT = "\nAnswer the above question. Answer the final number directly in one number.\n"
        def __init__(self, data_name, raw_data, tokenizer, bot, eot):
            super(SupervisedDataset, self).__init__()
            logging.warning("Data Preparing. Step2. Formatting inputs....")

            self.data_name = data_name
            questions, cots, answers = [], [], []
            num_ops_list = []
            operators = ["+", "-", "*", "/"]

            print("Step 2.1 For data preprocess. Load inputs...")
            token_nums = []
            for num_iter, example in enumerate(raw_data):
                # exp_mode: Use partial number of data. for debugging.
                # exp_data_num: The number of data used in exp mode.
                if training_args.exp_mode and num_iter > training_args.exp_data_num:
                    break
                question = f"{example['question']}"
                if "icot" in self.data_name and "full" in self.data_name: # icot-full (GSM8k-Aug-NL)
                    # bad data
                    if example["answer"] is None: # or example["response"] is None:
                        continue
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))
                    if token_num > training_args.max_token_num:
                        continue
 
                    cot = f"{example['cot']}".split(". ")
                    if not (training_args.include_last_cot):
                        cot = cot[:-1]

                    answer = example['answer'].split(' ')[-1]
                    if not answer[0].isdigit():
                        continue
                    answer = f"The answer is: {answer}"
                    answer = answer.replace("####", "")
                    questions.append(question)
                    
                    if cot:
                        cot = ". ".join(cot)+".\n"
                    else:
                        cot = ""
                    cots.append(cot)
                    answers.append(answer)
                elif "icot" in self.data_name: # icot (GSM8k-Aug)
                    # avoid OOM: remove very long data # 这一步很耗时间, 可以改为下一句
                    # token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))       # total: 138s
                    token_num = len(example["question"].split()) + len(example["cot"].split()) + len(example["answer"].split()) # total: 11s
                    if token_num > training_args.max_token_num/3:
                        continue
 
                    cot_list = []
                    cot = f"{example['cot']}".split(" ")
                    # include_last_cot: Include the last CoT step in the training data.
                    if not training_args.include_last_cot:  # True
                        cot = cot[:-1]
                    
                    len_cot = len(cot) 
                    for i in range(training_args.num_latent):   # 6
                        cot_list.append(" ".join(cot[:max(0, len_cot-i)]))
                    answer = example['answer'].split(' ')[-1]
                    
                    # some answers startwith the negative sign (-), bringing distillation problems for LLaMA
                    if not answer[0].isdigit():
                        continue

                    answer = f"The answer is: {answer}" 
                    answer = answer.replace("####", "")
                    questions.append(question)
                    cots.append(" ".join(cot))
                    answers.append(answer)
                elif "commonsense" in self.data_name or "strategy" in self.data_name:
                    question = example['question'].strip() + '\n'
                    cot = example['cot'].strip() + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num/3: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                elif "prontoqa" in data_args.data_name:
                    question = example['question'].strip() + '\n'
                    cot = '\n'.join(example['steps'][:-1]) + "\n"
                    answer = f"The answer is: {str(example['answer']).strip()}"
                    
                    # avoid OOM: remove very long data
                    token_num = len(tokenizer.encode(question + " " + cot + " " + answer))
                    if token_num > training_args.max_token_num/3: 
                        continue
                    questions.append(question)
                    cots.append(cot)
                    answers.append(answer)
                else:
                    raise NotImplementedError
            # exp_mode: Use partial number of data. for debugging.
            # exp_data_num: The number of data used in exp mode.
            if training_args.exp_mode:
                questions = questions[:training_args.exp_data_num]
                cots = cots[:training_args.exp_data_num]
                answers = answers[:training_args.exp_data_num]
            
            print(f"{len(cots)} data in total...")
            logging.warning("Tokenizing inputs... This may take some time...")
            # keys: [encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_answer_position,
            #        model_answer_position, ref_eos_position, model_eos_position, ref_labels]
            self.data_dict = preprocess(questions, cots, answers, tokenizer, bot, eot)
            self.keys = list(self.data_dict.keys())


        def __len__(self):
            return len(self.data_dict["encoder_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    # DataCollator数据整理器, 通常与一个自定义的 Dataset 类配合使用。在数据集中，每个样本通过 __getitem__ 返回一个包含上述键的字典
    # 对应SupervisedDataset, 输入数据为: dict = {str: torch.Tensor}, 每次仅输入一个的样本
    # dict keys: [encoder_input_ids, decoder_input_ids, ..., model_eos_position, ref_labels]
    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer

        # 模型的输入参数就一个: instance, 数据类型Sequence[Dict], 即一个字典组成的序列 (列表或元组)。每个字典代表数据集中的一个样本。
        # Sequence[Dict]表述instancede的结构, Dict[str, torch.Tensor]表示__call__最终返回一个Dict, 格式是{str: torch.Tensor}
        # ->作用: 
        # -- 1. 静态检查支持：帮助 IDE（如 PyCharm/VSCode）和类型检查工具（如 mypy）进行代码验证
        # -- 2. 代码可读性：明确告知开发者该方法的输出结构
        # -- 3. 运行时无影响：Python 解释器会忽略此注解，不影响实际执行
        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            # todo. 需要测试是否是batch数据，还是单条数据
            encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_answer_position, model_answer_position, ref_labels= \
                tuple([instance[key] for instance in instances]
                    for key in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "ref_answer_position", "model_answer_position", "ref_labels"))
        
            # pad left
            # encoder_input_ids: querstion_id + bot_id, tensor([ 7975, 286, 10053, ...,  257, 7202, 30, 50258])
            # filp(0): 交换了行的顺序，但保持了每行内的元素顺序不变。如果需要沿着列维度反转，可以使用 flip(1)
            # 这一段是对输入prompt进行left padding
            reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(1)
            
            # pad
            # ref_input_ids: question_id + cot_id + answer_id + eos_id
            # ref_labels: [-100*len(question_id), cot_id, answer_id, eos_id]
            # 这一段是对参考输入进行right填充
            ref_input_ids = torch.nn.utils.rnn.pad_sequence(ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            ref_labels = torch.nn.utils.rnn.pad_sequence(ref_labels, batch_first=True, padding_value=IGNORE_INDEX) 

            # decoder_input_ids: eot_id + answer_id + eos_id, tensor([50259, 464, 3280, 318, 25, 11470, 50256])
            # labels:            eot_id + answer_id + eos_id, tensor([50259, 464, 3280, 318, 25, 11470, 50256])
            # 这一段是对decoder_input_ids进行right填充
            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
          
            return dict(
                encoder_input_ids=encoder_input_ids,        # [padding] + querstion_id + bot_id -> longest length
                decoder_input_ids=decoder_input_ids,        # eot_id + answer_id + eos_id + [padding] -> longest length
                ref_input_ids=ref_input_ids,                # question_id + cot_id + answer_id + eos_id + [padding] -> longest length
                labels=labels,                              # eot_id + answer_id + eos_id + [-100] -> longest length
                encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),   # encoder_input_ids中非pad: [[False] + [True]] -> longest length
                ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),    # ref_input_ids中答案对应的开始位置
                model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),    # labels (eot_id + answer_id + eos_id)中答案对应的开始位置
                ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),           # ref_input_ids的非pad: [[True] + [False]] -> longest length
                ref_labels=ref_labels,                      # [-100] + cot_id + answer_id + eos_id + [-100]
            )

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        logging.warning("Data Preparing. Step1. Downloading Data.....")
        if "icot" in data_args.data_name:
            if 'full' in data_args.data_name:
                dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]
            else:
                # dataset: ['question', 'cot', 'answer'], total_num: 385620
                # 'question': 'Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?'
                # 'cot': '<<600*30/100=180>> <<600*10/100=60>> <<180+60=240>> <<600-240=360>>'
                # 'answer': '360'
                dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "strategy" in data_args.data_name:
            dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "commonsense" in data_args.data_name:
            dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        elif "prontoqa" in data_args.data_name:
            with open("/home/ubuntu/coconut/data/prontoqa_train.json") as f:
                dataset = json.load(f)
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
        else:
            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")
        
    file_name = model_args.model_name_or_path.split('/')[-1] + '_' + f"ep_{int(training_args.num_train_epochs)}" + '_' + f"lr_{training_args.learning_rate}" + f"seed_{training_args.seed}"

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        file_name,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # CustomTrainer继承Trainer，需要显式指定train_dataset, 否则会报错; 同时指定data_collator数据整理器, 表示数据如何处理;
    # CustomTrainer继承自Trainer的常用参数: 
    # - model: 必须, 待训练的模型实例 （不过仍有一个疑问, 是一个torch.nn.Module, 还是需要一个class）
    # - args: 必须, TrainingArguments对象（如学习率、批次大小等超参数）
    # - data_collator: 必须, 数据整理器（如DataCollatorForLanguageModeling）
    # - train_dataset/eval_dataset: 必须, 训练/验证数据集
    # - tokenizer: 必须, 分词器实例
    # - callbacks: 自定义回调函数列表
    # **data_module含义等同于: train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()

    # to avoid the error of saving the model
    #if "llama" in model_args.model_name_or_path:
    #    trainer.model.codi.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.model.lm_head.weight.clone())
    #if "gpt2" in model_args.model_name_or_path:
    #    trainer.model.codi.transformer.wte.weight = torch.nn.Parameter(model.codi.lm_head.weight.clone())
    #if "qwen" in model_args.model_name_or_path.lower():
    #    trainer.model.codi.base_model.model.model.embed_tokens.weight = torch.nn.Parameter(model.codi.base_model.model.lm_head.weight.clone())

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
