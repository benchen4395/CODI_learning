##### step1: 加载模型
import os
import transformers
import torch
import torch.nn as nn
from peft import (get_peft_model, PeftModel, PeftConfig, TaskType, LoraConfig)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model = "openai-community/gpt2"
model_name = "/etc/ssd1/chenben03/LLM_learn/CODI_learning/pretrained_model/codi-gpt2" #'zen-E/CODI-gpt2'

# model configurations
bf16 = True
num_latent = 6
use_lora = True
lora_init = True
lora_r = 128
lora_alpha = 32
model_max_length = 512


# 1.1 加载模型和tokenizer切词
if lora_init: # True
    task_type = TaskType.CAUSAL_LM
    if any(name in pretrained_model for name in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif any(name in pretrained_model for name in ["phi"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif any(name in pretrained_model for name in ["gpt2"]):
        target_modules = ["c_attn", "c_proj", 'c_fc']
    else:
        raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {pretrained_model}.")
    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )
else:
    raise NotImplementedError

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for name, param in model.named_parameters():
        print('the name is: {}, the param is: {}'.format(name, param.shape))
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print('the require_grad name is: {}, and the param is: {}'.format(name, param.shape))


model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model,
            torch_dtype=(       # 注意：当使用量化时，相同参数影响的数据类型可能会被覆盖，因为量化有自己的数据类型设置, 优先级没有bnb_4bit_compute_dtype高
                torch.float16 if bf16 is False else torch.bfloat16
            ),
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,      # 表示使用4位量化加载模型。这将把模型的权重转换为4位表示，从而大幅减少内存占用（约为原始精度的1/8)
                bnb_4bit_compute_dtype=torch.bfloat16,  # 即使权重被量化为4位，但在计算（如矩阵乘法）之前需要反量化为更高精度的类型。这里设置为bfloat16，表示计算过程中使用bfloat16类型
                bnb_4bit_use_double_quant=False,    # 是否对量化使用的量化常数（quantization constants）再次进行量化（双重量化）。双重量化可以进一步减少内存占用，但可能会略微增加量化误差。这里设置为False，表示不使用双重量化
                bnb_4bit_quant_type='nf4', # 指定4位量化的数据类型。'nf4'代表4-bit NormalFloat，这是一种针对正态分布权重优化的量化数据类型，通常比标准的4位整数量化表现更好。另一种可选类型是'fp4'（4位浮点数）
            )
        )
# tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model, use_fast=False)
tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model,
        token=None,
        model_max_length=model_max_length,    # 512
        padding_side="left",
        use_fast=False,
    )

# 配置模型参数
ori_vocab_size = model.config.vocab_size    # 50257
dim = model.config.hidden_size              # 768
# 加载vocab词表
vocab = tokenizer.get_vocab()  # token : id
vocab_index = {v:k for k,v in vocab.items()}    # id : token

# 1.2 添加special token pad_token，以及latent thought thinking.
# 设置pad_token, 一些模型没有预设，需要额外加载，并做好special_token的初始化
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = ori_vocab_size
bot_id = tokenizer.vocab_size + 1
eot_id = tokenizer.vocab_size + 2
# 初始化new_added_token_embedding的方法见文档: https://www.cs.columbia.edu/~johnhew/vocab-expansion.html
model.resize_token_embeddings(tokenizer.vocab_size + 3)

if use_lora:
    model = get_peft_model(model, lora_config)

# 1.3 构建模型映射层
use_prj = True
prj_no_ln = False
prj_dim = 768
prj_dropout = 0.0 # 模型测试时为空
model_dim = model.config.hidden_size    # 768
if use_prj:
    model_prj = nn.Sequential(
        nn.Dropout(prj_dropout),      # test时为0
        nn.Linear(model_dim, prj_dim, dtype=torch.bfloat16), # test时为768, 
        nn.GELU(),
        nn.Linear(prj_dim, model_dim, dtype=torch.bfloat16), # test是为1024,
    )
    if not prj_no_ln:
        model_prj.add_module("ln", nn.LayerNorm(model_dim))

model_prj = model_prj.to(device, dtype=torch.bfloat16)

##### step2: 加载预训练好的模型
try:
    state_dict = load_file(os.path.join(model_name, "model.safetensors"))
except Exception:
    state_dict = torch.load(os.path.join(model_name, "pytorch_model.bin"))
# 将保存的模型参数（权重和偏置）加载到当前模型中, strict=True (默认): 严格模式, 要求state_dict中的键与模型完全匹配
model.load_state_dict(state_dict, strict=False) 
model.tie_weights()    # 输入/输出层权重共享的关键方法，通过减少参数冗余提升模型效率; 通常是输入和输出的嵌入层（embedding layers）共享参数


##### step3: 加载数据和切词
from datasets import load_dataset, concatenate_datasets
import torch
import math
import logging

data_name = "gsm8k"
question_name = "question"
answer_name = "answer"
# 数据集有两种格式, 
# - main: 答案以自然语言直接描述推理步骤，类似解题笔记;
# - socratic: 通过子问题引导思考（如“How many clips...”），模拟苏格拉底问答法，强制分解问题。
# main配置更常用作评估标准，而socratic可作为训练辅助
try: # 第一次加载使用
    dataset = load_dataset(data_name, "main") 
except:
    dataset = load_dataset(data_name) 
test_set = dataset['test']

# 加载question:
question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]

# 加载answer: 
answer = []
# get numerical answer, 对于gms8k这种天然包含计算过程的，就把计算过程去掉，只保留最后结果
for example in test_set:
    example = example[answer_name]
    if isinstance(example, bool):
        answer.append(example)
        continue
    if example in ["True", "False"]:
        if example == "True":
            ans = True
        else:
            ans = False
        answer.append(ans)
        continue
    if example in "ABCDE":
        answer.append(example)
        continue
    if "####" in example:
        ans = example.split('####')[-1]
    else:
        ans = example
    ans = ans.replace(',', '')  # handle numbers like 2,000
    try:
        ans = float(ans)
    except ValueError:
        ans = float("inf")
    answer.append(ans)

batch_size = 128
question_data = []

logging.warning("Tokenizing inputs...")
eval_step = math.ceil(len(question)/batch_size)
logging.warning(f"Total example: {len(question)} | eval batch size: {batch_size}"
                f"eval steps: {eval_step}")


# 加载一定batch数据进行批处理
i = 0 
remove_eos = True # remove_eos: Do not add <eos> as a delimiter to split QA.
batch = tokenizer(question[i*batch_size: (i+1)*batch_size], return_tensors="pt", padding="longest")

if remove_eos:
    bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
else:
    bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)

batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
batch['input_len'] = len(batch['input_ids'][0])
question_data.append(batch.to(device))


##### step3: 模型latent reasoning实验:
past_key_values = None      # 第一次加载时, 默认为空
outputs = model(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=batch["attention_mask"])
past_key_values = outputs.past_key_values   # 缓存历史解码的 Key/Value 矩阵，避免重复计算
latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1).to(device)  # 最后一层transformers层的输出[batch_size, hidden_size]

if use_prj:   # test时, 是做了两层线性层, [hidden_size -> prj_size -> hidden_size]
    latent_embd = model_prj(latent_embd)

# 意图推理轮数
inf_latent_iterations = 6
for i in range(inf_latent_iterations):
    # decode the latent embeddings
    outputs = model(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
    past_key_values = outputs.past_key_values
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1).to(device)
    
    if use_prj:
        latent_embd = model_prj(latent_embd)        # [batch_size, 1 hidden_size]: (128, 1, 768)

def model_get_embd(model, model_name):
    try:
        if "pythia" in model_name:
            return model.get_base_model().gpt_neox.embed_in
        elif "gpt2" in model_name:
            try:
                # get_base_model(): 当模型被封装（如使用 LoRA、Adapter 等微调技术）时，此方法剥离附加的适配层，返回原始基础模型结构
                # .transformer.wte: 获取基础模型（Base Model）的词嵌入层（Word Token Embeddings）
                print('-- the model is splited')
                return model.get_base_model().transformer.wte
            except Exception: # no lora
                print('-- the model is unified')
                return model.transformer.wte
        else:
            try:
                return model.get_base_model().model.embed_tokens
            except Exception: # no lora
                return model.model.embed_tokens
    except AttributeError:
        if "pythia" in model_name:
            return model.gpt_neox.embed_in
        raise NotImplementedError

if remove_eos: # shape (1,1,768)
    eot_emb = model_get_embd(model, 'gpt2')(torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
else: # shape: (1,2,768)
    eot_emb = model_get_embd(model, 'gpt2')(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)

eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1) 

## step4 开始线上推理
output = eot_emb  # [128,2,768]

seq_len = 0
finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")  # Track EOS for each sequence [128,]
pred_tokens = [[] for _ in range(batch_size)]
for i in range(gen_kwargs["max_new_tokens"]):       # 256
    seq_len += 1
    out = model(
            inputs_embeds=output,   # [128, 2, 768]
            output_hidden_states=False, # 不输出hidden_state
            attention_mask=None,
            use_cache=True,
            output_attentions=False,
            past_key_values=past_key_values
        )

    past_key_values = out.past_key_values
    logits = out.logits[:, -1, :model.codi.config.vocab_size-1]     # 预测为非eos得概率值 [128, 50259]

    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)       # [128,]
    # Handle EOS for each sequence
    for b in range(batch_size):
        if not finished[b]:
            pred_tokens[b].append(next_token_ids[b].item())
            if next_token_ids[b] == tokenizer.eos_token_id:
                finished[b] = True
    # Break if all sequences have finished
    if finished.all():
        break
    output = model_get_embd(model, 'gpt2')(next_token_ids).unsqueeze(1).to(device)  # (128, 1, 768)



#### step5 加载训练数据学习
    ###########################
    #       Train Stage       #
    ###########################
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
    # 'question': 'John reads twice as fast as David. David can read 50 pages in an hour. How many pages can they both read in 3 hours?'
    question = f"{example['question']}"
    # 这一步很耗时间, 可以改为下一句
    #token_num = len(tokenizer.encode(example["question"] + example["cot"] + example["answer"]))    # total: 138s
    token_num = len(example["question"].split()) + len(example["cot"].split()) + len(example["answer"].split()) # total: 11s
    if token_num > max_token_num/3:
        continue
    cot_list = []
    cot = f"{example['cot']}".split(" ")
    # include_last_cot: Include the last CoT step in the training data.
    if not include_last_cot:    # True
        cot = cot[:-1]
    
    len_cot = len(cot) 
    for i in range(num_latent):   # 6
        cot_list.append(" ".join(cot[:max(0, len_cot-i)]))
    # 'answer': '450'
    answer = example['answer'].split(' ')[-1]
    
    # some answers startwith the negative sign (-), bringing distillation problems for LLaMA
    if not answer[0].isdigit():
        continue

    # answer: 'The answer is: 450'
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

def preprocess(
    sources: Sequence[str],     # questions
    targets: Sequence[str],     # cots
    answers: Sequence[str],     # answers
    tokenizer: transformers.PreTrainedTokenizer, 
    bot_id: int,
    eot_id: int,
) -> Dict:
    print("Step 3.1 For data preprocess. Tokenizing inputs... This may take some time...")
    sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
    cot_id = _tokenize_fn(targets, tokenizer)["input_ids"]
    answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]


    print("Step 3.2 For data preprocess. Format the inputs.")
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
    ref_labels = []
    for x, y in zip(ref_input_ids, sources_id):
        z = x.clone()
        z[:len(y)] = -100       # 这个地方为什么是-100, 而不是pad_token_id，或者eos_token_id?
        ref_labels.append(z)
    
    # add eot to source
    sources_id = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
    # add eot and eos
    if remove_eos:  # True
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

logging.warning("Data Preparing. Step3. Tokenizing inputs... This may take some time...")
start_time = time.time()
data_dict = preprocess(questions, cots, answers, tokenizer, bot_id, eot_id)
keys = list(data_dict.keys())

end_time = time.time()
print('the total time is: ', end_time - start_time)


logging.warning("Data Preparing. Step4. Construct the input data format...")
start_time = time.time()
# 取一个batch数据
instances = {key: data_dict[key][:batch_size] for key in keys}
encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_answer_position, model_answer_position, ref_labels = \
    tuple(instances[key] 
        for key in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "ref_answer_position", "model_answer_position", "ref_labels"))

# pad left
# encoder_input_ids: querstion_id + bot_id, tensor([ 7975, 286, 10053, ...,  257, 7202, 30, 50258])
# filp(0): 交换了行的顺序，但保持了每行内的元素顺序不变。如果需要沿着列维度反转，可以使用 flip(1)
# 这一段是对输入prompt进行left padding
reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
encoder_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).flip(1)

# pad
# ref_input_ids: question_id + cot_id + answer_id + eos_id
# ref_labels: [-100*len(question_id), cot_id, answer_id, eos_id]
# 这一段是对参考输入进行right填充
ref_input_ids = torch.nn.utils.rnn.pad_sequence(ref_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
ref_labels = torch.nn.utils.rnn.pad_sequence(ref_labels, batch_first=True, padding_value=IGNORE_INDEX) 

# decoder_input_ids: eot_id + answer_id + eos_id, tensor([50259, 464, 3280, 318, 25, 11470, 50256])
# labels:            eot_id + answer_id + eos_id, tensor([50259, 464, 3280, 318, 25, 11470, 50256])
# 这一段是对decoder_input_ids进行right填充
decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

train_dataset = dict(
    encoder_input_ids=encoder_input_ids,        # [padding] + querstion_id + bot_id -> longest length
    decoder_input_ids=decoder_input_ids,        # eot_id + answer_id + eos_id + [padding] -> longest length
    ref_input_ids=ref_input_ids,                # question_id + cot_id + answer_id + eos_id + [padding] -> longest length
    labels=labels,                              # eot_id + answer_id + eos_id + [-100] -> longest length
    encoder_attention_mask=encoder_input_ids.ne(tokenizer.pad_token_id),         # [[False] + [True]] -> longest length
    ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),     # ref_input_ids中答案对应的开始位置
    model_answer_position=torch.tensor(model_answer_position, dtype=torch.long), # answers_id中答案对应的开始位置
    ref_attention_mask=ref_input_ids.ne(tokenizer.pad_token_id),     # [[False] + [True]] -> longest length
    ref_labels=ref_labels,                                           # [-100] + cot_id + answer_id + eos_id
)

end_time = time.time()
print('the total time is: ', end_time - start_time)




logging.warning("Data Preparing. Step5. Training Stage inference...")
#### 加载模型

encoder_input_ids=encoder_input_ids.to(device)
decoder_input_ids=decoder_input_ids.to(device)
ref_input_ids=ref_input_ids.to(device)
labels=labels.to(device)
encoder_attention_mask=encoder_input_ids.ne(tokenizer.pad_token_id).to(device)
ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long).to(device)
model_answer_position=torch.tensor(model_answer_position, dtype=torch.long).to(device)
ref_attention_mask=ref_input_ids.ne(tokenizer.pad_token_id).to(device)
ref_labels=ref_labels.to(device)
#### 模型推理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fix_attn_mask = False
if not fix_attn_mask:      # True
    ref_attention_mask = None

# Encode the question
past_key_values = None
# encoder_input_ids: [padding] + question_id + bot_id -> longest length
# encoder_input_ids的非pad_token_id: [[False] + [True]] -> longest length
outputs = model(input_ids=encoder_input_ids, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=encoder_attention_mask)
past_key_values = outputs.past_key_values
latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1) # as the next input  [128, 1, 768]
if use_prj:
    latent_embd = model_prj(latent_embd)     # [128, 1, 768]

len_pred_loss = 0
dynamic_mask = None
if fix_attn_mask:  # False
    dynamic_mask = torch.ones((encoder_attention_mask.size(0), num_latent), device=ref_labels.device)

# Iterate over the latent embeddings
distill_loss_total = 0
ce_loss_total = 0
# ref_input_ids:  question_id + cot_id + answer_id + eos_id + [padding] -> longest length
# ref_attention_mask: [[True] + [False]] -> longest length
with torch.no_grad():   # 在 torch.no_grad() 上下文中，禁用梯度计算
    ref_outputs = model(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
ref_outputs_with_grad = model(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask) 

# Formatting for deprecated exps
ref_outputs_list = [ref_outputs]        # outputs
ref_input_ids = [ref_input_ids]         # [128, 118]

# Process the position tensor
# Normalise the position definition 
if "llama" in pretrained_model.lower() or "qwen" in pretrained_model.lower(): # there is one more token standing for " " 
    model_answer_position = model_answer_position + 1
    ref_answer_position = ref_answer_position + 1

# Losses
print_loss = True
# The multiplier that scales the teacher's cross entropy loss in the total loss calculation.
ref_loss_factor = 1.0

# Cross Entropy Loss
loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 

# Distillation Loss
# distill_loss_div_std: false or true. Divide the distillation loss by a std for normallisation.
distill_loss_div_std = False
# distill_loss_type: Specify the distillation loss. Use smoothL1 by default.
distill_loss_type = "smooth_l1"
# distill_loss_factor: A multiplier of the distillation loss.
distill_loss_factor = 1.0
if distill_loss_type == "smooth_l1":
    distill_loss_fct = nn.SmoothL1Loss()
elif distill_loss_type == "l2":
    distill_loss_fct = nn.MSELoss()
else:
    raise NotImplementedError


for i in range(num_latent):
    # Implicit CoT generation
    outputs = model(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
    past_key_values = outputs.past_key_values
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)      # [128, 1, 768]
    if use_prj:
        latent_embd = model_prj(latent_embd)     # [128, 1, 768]
    
    # Calculate the distillation loss
    if i == num_latent - 1:
        # Decode the final answer in natural language
        # decoder_input_ids: eot_id + answer_id + eos_id + [padding] -> longest length
        embds = model_get_embd(model, pretrained_model)(decoder_input_ids)      # [128, 10, 768]

        if dynamic_mask is not None: # Prevent attending the paddings
            decoder_mask = torch.ones((embds.size(0), embds.size(1)), dtype=torch.bool).to(dynamic_mask)
            dynamic_mask = torch.cat((encoder_attention_mask, dynamic_mask, decoder_mask), dim=1)
            dynamic_mask = dynamic_mask.bool()
        # Student task's output
        outputs = model(inputs_embeds=embds, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=dynamic_mask) 
        # Teacher task's output
        ref_outputs = ref_outputs_list[0]

        distill_loss = 0
        # Calculate distillation loss between the teacher's logits and the student's logits for every layer
        # out.shape: [128, 10 ,768]; ref_out: [128, 118 ,768]
        # ref_selected: [128, 1, 768]
        # ref_selected: [128, 1, 768]
        for j, (out, ref_out) in enumerate(zip(outputs.hidden_states, ref_outputs.hidden_states)):
            ref_selected = ref_out.gather(1, ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1)))
            out_selected = out.gather(1, model_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1)))

            distill_loss_tmp = distill_loss_fct(out_selected, ref_selected.detach())
            
            if self.distill_loss_div_std:
                if self.distill_loss_type == 'l2':
                    distill_loss_tmp /= ref_selected.std()  # 因为是l2，所以除两遍
                distill_loss_tmp /= ref_selected.std()
            distill_loss += distill_loss_tmp
        
        distill_loss /= len(outputs.hidden_states)
        
        if print_loss: # True
            print(f'latent{i}: distill_loss={distill_loss}')

        distill_loss_total += distill_loss

    # Calculate the CE loss for the student task
    if i == num_latent - 1:
        logits = outputs.logits
        effective_logits = logits[:, :-1, :]
        effective_logits = effective_logits.reshape(-1, logits.size(-1))
        target_ids = labels[:, 1:].reshape(-1)                        
        ce_loss = loss_fct(effective_logits, target_ids)
        ce_loss_total += ce_loss

# Calculate the CE loss for the teacher task
ref_ce_loss = 0
ref_logits = ref_outputs_with_grad.logits
effective_ref_logits = ref_logits[:, :-1, :]
effective_ref_logits = effective_ref_logits.reshape(-1, ref_logits.size(-1))
ref_target_ids = ref_labels[:, 1:].reshape(-1)
ref_ce_loss = loss_fct(effective_ref_logits, ref_target_ids)
ref_ce_loss *= ref_loss_factor 

# Weigh the distillation loss
distill_loss *= distill_loss_factor
distill_loss_total *= distill_loss_factor

if print_loss:
    print(f'loss={ce_loss+distill_loss}, ce_loss={ce_loss}, distill_loss={distill_loss}, ce_loss_total={ce_loss_total}, distill_loss_total={distill_loss_total}, ref_ce_loss={ref_ce_loss}')

loss = ce_loss_total + distill_loss_total + ref_ce_loss