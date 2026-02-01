import transformers

## 1. 加载模型和tokenizer切词
model = transformers.AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=False)

# 加载vocab词表
vocab = tokenizer.get_vocab()  # token : id
vocab_index = {v:k for k,v in vocab.items()}    # id : token

# 设置pad_token, 一些模型没有预设，需要额外加载，并做好special_token的初始化
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# 初始化new_added_token_embedding的方法见文档: https://www.cs.columbia.edu/~johnhew/vocab-expansion.html
model.resize_token_embeddings(tokenizer.vocab_size + 1)
# 另一种方式: （更建议，因为更稳定）
tokenizer.pad_token = tokenizer.eos_token

## 2. 保存模型的状态并加载
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# 创建模型
model = BartForConditionalGeneration.from_pretrained("openai-community/gpt2")

# 保存模型状态
torch.save(model.state_dict(), "model_weights.pth")

# 稍后加载模型状态
model = BartForConditionalGeneration.from_pretrained("openai-community/gpt2")
state_dict = torch.load("gpt2-model_weights.pth")
model.load_state_dict(state_dict)


## 3. 查看模型的state_dict结构
for name, param in model.state_dict().items():
    print(f"{name}: {param.shape}")

## 4. model的保存两种方式对比
mode.save_pretrained() vs torch.save()
# save_pretrained的优势：
# 1. 自动保存配置文件
# 2. 保存tokenizer
# 3. 兼容from_pretrained加载
# 4. 支持推送到Hub

# load_state_dict的优势：
# 1. 更精细的控制
# 2. 可以部分加载
# 3. 可以处理架构不完全匹配的情况
# 4. 更小的存储空间（只保存权重）

# 方法1：使用transformers的save_pretrained
model.save_pretrained("./my_model")
# 保存：pytorch_model.bin + config.json + tokenizer文件

# 加载
model = BartForConditionalGeneration.from_pretrained("./my_model")

# 方法2：使用torch.save + load_state_dict
torch.save(model.state_dict(), "model_weights.pth")

# 加载
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")  # 需要先创建架构
model.load_state_dict(torch.load("model_weights.pth"))

## 5. 加载模型为bf16各种的几种方法



# 设置pad_token, 一些模型没有预设，需要额外加载，并做好special_token的初始化
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.pad_token_id = tokenizer.vocab_size
model.bot_id = tokenizer.vocab_size + 1
model.eot_id = tokenizer.vocab_size + 2
# 初始化new_added_token_embedding的方法见文档: https://www.cs.columbia.edu/~johnhew/vocab-expansion.html
model.resize_token_embeddings(tokenizer.vocab_size + 3)



from datasets import load_dataset, concatenate_datasets
from typing import Dict, Optional, Sequence
import math
import torch
import logging
import transformers


pretrained_model = "openai-community/gpt2"
batch_size = 128
model_max_length = 512
num_latent = 6  # 意图推理轮数
max_token_num = 2000 # 最大处理文档token长度
exp_mode = False     # 筛选一批数据用于快速测试
remove_eos = True   # remove_eos: Do not add <eos> as a delimiter to split QA.
include_last_cot = False    # Include the last CoT step in the training data.
exp_data_num = 20000
IGNORE_INDEX = -100

# tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, use_fast=False)
tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model,
        token=None,
        model_max_length=model_max_length,    # 512
        padding_side="left",
        use_fast=False,
    )

# 配置模型参数
ori_vocab_size = tokenizer.vocab_size
# 设置pad_token, 一些模型没有预设，需要额外加载，并做好special_token的初始化
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = ori_vocab_size
bot_id = tokenizer.vocab_size + 1
eot_id = tokenizer.vocab_size + 2

# 加载vocab词表
vocab = tokenizer.get_vocab()  # token : id
vocab_index = {v:k for k,v in vocab.items()}    # id : token

logging.warning("Data Preparing. Step1. Downloading Data.....")
# dataset: ['question', 'cot', 'answer'], total_num: 385620
# 示例: 
# 'question': 'Out of 600 employees in a company, 30% got promoted while 10% received bonus. How many employees did not get either a promotion or a bonus?'
# 'cot': '<<600*30/100=180>> <<600*10/100=60>> <<180+60=240>> <<600-240=360>>'
# 'answer': '360'
dataset = load_dataset("zen-E/GSM8k-Aug")["train"]

logging.warning("Data Preparing. Step2. Formatting inputs....")
questions, cots, answers = [], [], []
num_ops_list = []
operators = ["+", "-", "*", "/"]
token_nums = []

import time

print("Step 2.1 For data preprocess. Load inputs...")
start_time = time.time()
for num_iter, example in enumerate(dataset):
    # exp_mode: Use partial number of data. for debugging.
    # exp_data_num: The number of data used in exp mode.
    if exp_mode and num_iter > exp_data_num:
        break
    # avoid OOM: remove very long data
    question = f"{example['question']}"
    # 这一步很耗时间, 可以改为下一句
    #token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))    # total: 138s
    token_num = len(example["question"].split()) + len(example["cot"].split()) + len(example["answer"].split()) # total: 11s
    if token_num > max_token_num/3:
        continue
    cot_list = []
    cot = f"{example['cot']}".split(" ")
    # include_last_cot: Include the last CoT step in the training data.
    if not include_last_cot:
        cot = cot[:-1]
    
    len_cot = len(cot) 
    for i in range(num_latent):   # 6
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

if exp_mode:
    questions = questions[:exp_data_num]
    cots = cots[:exp_data_num]
    answers = answers[:exp_data_num]

print(f"{len(cots)} data in total...")

end_time = time.time()
print('the total time is: ', end_time - start_time)

print("Step 2.2 For data preprocess. Tokenizing inputs... This may take some time...")
start_time = time.time()
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length= model_max_length, # default: 256
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

def get_answer_token_position(tokens, answer_prompts, tokenizer):
    #answer_prompt = torch.tensor([464, 3280, 318, 25]) # as: "the answer is:"
    try:
        match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
        answer_token_id = match_indices + len(answer_prompts[0])
        return answer_token_id
    except Exception:
        breakpoint()

sources_id = _tokenize_fn(questions, tokenizer)["input_ids"]
cot_id = _tokenize_fn(cots, tokenizer)["input_ids"]
answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]

end_time = time.time()
print('the total time is: ', end_time - start_time)

print("Step 2.3 For data preprocess. Format the inputs.")
start_time = time.time()
# add eos token to accomodate pretrained model's format
if not remove_eos:  # false
    sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
    cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
answers_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]
# 如果cot_id和answer_id包含bos_token，进行剔除
if cot_id[0][0] == tokenizer.bos_token_id:
    cot_id = [x[1:] for x in cot_id]
    answers_id = [x[1:] for x in answers_id]
# 做到question, cot, answer的无缝拼接 (如果是组batch，会遇到padding="longest"使短文本中间拼接的问题?)
ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_id)]
# 实现训练数据中仅cot, answer会出现，其他值为-100
ref_labels = []
for x, y in zip(ref_input_ids, sources_id):
    z = x.clone()
    z[:len(y)] = -100   # 这个地方为什么是-100, 而不是pad_token_id，或者eos_token_id?
    ref_labels.append(z)

# add eot to source
sources_id = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
# add eot and eos
if remove_eos:  # false
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
data_dict = dict(encoder_input_ids=sources_id, decoder_input_ids=answers_id, ref_input_ids=ref_input_ids, labels=answers_id, \
                    ref_answer_position=ref_answer_position, model_answer_position=model_answer_position, \
                    ref_eos_position=ref_eos_position, model_eos_position=model_eos_position, ref_labels=ref_labels)
keys = list(data_dict.keys())

end_time = time.time()
print('the total time is: ', end_time - start_time)