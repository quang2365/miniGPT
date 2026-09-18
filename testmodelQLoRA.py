import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = r'./models/Qwen3-8B'
LORA_ADAPTER_PATH = r'./my_lora_adapter'


tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)

quant_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= True
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config = quant_config,
    device_map = {"":0},
    dtype = torch.bfloat16,
    trust_remote_code = True
)

model = PeftModel.from_pretrained(base_model,LORA_ADAPTER_PATH)

model.eval()

message = [{'role':'user','content':'Cô mình ra câu đố: Con gì có cánh mà không biết bay? Bạn ơi, con gì có cánh mà không biết bay vậy?'}]

prompt_text = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
Input = tokenizer(prompt_text,return_tensors='pt').to(model.device)
with torch.no_grad():
    output = model.generate(**Input,max_new_tokens=256,temperature = 1,top_p=0.9,do_sample = True,pad_token_id = tokenizer.pad_token_id,eos_token_id = tokenizer.eos_token_id)

response = tokenizer.decode(output[0][Input.input_ids.shape[1]:],skip_special_token = True)

print(response)