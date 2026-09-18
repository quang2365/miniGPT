import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer
# Phải lấy model đã được SFT tức BASEMODEL + SFT (LoRA, QLoRA)
MODEL_NAME = "./models/Qwen3-8B"
DATA_FILE = "./clawer/data/data/dataset.jsonl"  # File có 3 cột: prompt, chosen, rejected

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
#formart chuẩn là sử dụng dataset có format 3 cột prompt, chosen, reject
raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset_split = raw_dataset.train_test_split(test_size=0.1)

training_args = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1, 
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,  # LR của DPO thường nhỏ hơn SFT (thường từ 5e-7 đến 5e-6)
    max_length=512,  # Độ dài tối đa của full sequence (prompt + response)
    max_prompt_length=256,  # Độ dài tối đa riêng cho prompt
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    peft_config=peft_config,
    args=training_args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./my_dpo_adapter")
tokenizer.save_pretrained("./my_dpo_adapter")