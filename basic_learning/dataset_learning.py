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
        nn.Linear(model_dim, prj_dim), # test时为768, 
        nn.GELU(),
        nn.Linear(prj_dim, model_dim), # test是为1024,
    )
    if not prj_no_ln:
        model_prj.add_module("ln", nn.LayerNorm(model_dim))

model_prj = model_prj.to(device)

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
remove_eos = False # remove_eos: Do not add <eos> as a delimiter to split QA.
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
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    
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
                return model.get_base_model().transformer.wte
            except Exception: # no lora
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

