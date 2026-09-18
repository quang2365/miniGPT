from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments
from peft import prepare_model_for_kbit_training,LoraConfig,get_peft_model
from datasets import load_dataset
import torch 

MODEL_NAME = r"./models/Qwen3-8B"
TRAIN_FILE = r"./clawer/data/sft_qwen_messages_oke.jsonl"
VAL_FILE = r"./clawer/data/sft_qwen_messages_oke_val.jsonl"

MAX_LENGTH =256
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = (tokenizer.eos_token)

quant_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
model = prepare_model_for_kbit_training(model)

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    }
)

model.config.use_cache = False
lora_config = LoraConfig(
    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "v_proj",
    ],

    bias="none",

    task_type="CAUSAL_LM",)

model = get_peft_model(
    model,
    lora_config,
)


model.print_trainable_parameters()


# ============================================================
# DATASET
# ============================================================

dataset = load_dataset(

    "json",

    data_files={
        "train":
            TRAIN_FILE,

        "validation":
            VAL_FILE,
    },
)

print(dataset)
print("Train:", len(dataset["train"]))
print("Validation:", len(dataset["validation"]))

print("\nExample:")
print(dataset["train"][0])


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_sample(sample):

    messages = sample["messages"]

    # Tin nhắn cuối cùng phải là assistant
    assistant_message = messages[-1]

    if assistant_message["role"] != "assistant":
        raise ValueError(
            "Sample không kết thúc bằng assistant message"
        )

    # Phần prompt = tất cả message trước assistant
    prompt_messages = messages[:-1]

    # Full conversation = toàn bộ messages
    full_messages = messages

    # ----------------------------
    # Prompt
    # ----------------------------

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ----------------------------
    # Full conversation
    # ----------------------------

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    full_encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=False,
    )

    prompt_encoded = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = full_encoded["input_ids"]
    attention_mask = full_encoded["attention_mask"]


    labels = input_ids.copy()

    prompt_length = len(
        prompt_encoded["input_ids"]
    )

    mask_length = min(
        prompt_length,
        len(labels)
    )

    labels[:mask_length] = [-100] * mask_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

tokenized_dataset = (
    dataset.map(
        preprocess_sample,

        remove_columns=(
            dataset["train"]
            .column_names
        ),
    )
)


# ============================================================
# COLLATOR
# ============================================================

class SFTCollator:

    def __init__( self, tokenizer, ):
        self.tokenizer = (tokenizer)


    def __call__( self, features, ):

        max_len = max(len(x["input_ids"]) for x in features)


        input_ids = []
        attention_mask = []
        labels = []


        for item in features:

            length = len( item[ "input_ids" ] )


            pad_length = ( max_len - length )


            input_ids.append( item[ "input_ids" ] + [self.tokenizer.pad_token_id] * pad_length )


            attention_mask.append(item["attention_mask"] + [0] * pad_length )


            labels.append(item["labels"] + [-100] * pad_length)


        return {

            "input_ids": torch.tensor( input_ids, dtype=torch.long, ),

            "attention_mask": torch.tensor( attention_mask, dtype=torch.long, ),

            "labels": torch.tensor(labels, dtype=torch.long, ),
        }


collator = SFTCollator(
    tokenizer
)


# ============================================================
# TRAINING CONFIG
# ============================================================

training_args = TrainingArguments(

    output_dir=(
        "./qlora_output"
    ),

    num_train_epochs=3,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=16,

    learning_rate=2e-4,

    logging_steps=10,

    eval_strategy="steps",

    eval_steps=100,

    save_steps=100,

    save_total_limit=2,

    bf16=True,

    report_to="none",

    remove_unused_columns=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=(
        tokenized_dataset[
            "train"
        ]
    ),

    eval_dataset=(
        tokenized_dataset[
            "validation"
        ]
    ),

    data_collator=collator,
)


# ============================================================
# TRAIN
# ============================================================

trainer.train()


# ============================================================
# SAVE ADAPTER
# ============================================================

model.save_pretrained(
    "./my_lora_adapter"
)

tokenizer.save_pretrained(
    "./my_lora_adapter"
)
