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

