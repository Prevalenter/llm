# llm

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

# 加载Llama 7B模型和分词器
model_name = "hfl/chinese-macbert-mini"  # 或者根据具体路径指定模型
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token


# 创建LoRA配置
lora_config = LoraConfig(
    r=4,  # LoRA的秩
    lora_alpha=32,  # 权重缩放
    lora_dropout=0.1,  # Dropout率
    task_type="CAUSAL_LM"  # 任务类型，适用于Causal LM
)

# 使用LoRA对模型进行微调
model = get_peft_model(model, lora_config)

# 加载数据集（你可以选择合适的数据集进行微调）
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# small_train_dataset = dataset["train"]

# 准备数据
# def preprocess_function(examples):
#     return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
def preprocess_function(examples):
    # Use the same tokens for labels as input_ids for causal language modeling
    encodings = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
    encodings["labels"] = encodings["input_ids"].copy()  # Set labels equal to input_ids
    return encodings


train_dataset = dataset["train"].select(range(1)).map(preprocess_function, batched=True)
eval_dataset = dataset["validation"].select(range(1)).map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./lora_llama_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    deepspeed="data/ds_config.json",  # 需要提供 DeepSpeed 配置文件
)


# 使用Trainer API进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始微调
trainer.train()

# 保存微调后的模型
model.save_pretrained("./lora_llama_model")
tokenizer.save_pretrained("./lora_llama_model")
