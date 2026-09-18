from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("nhuvo/nllb-600m-en-vimedner-direct-trans-ner-en2vi")
model = AutoModelForSeq2SeqLM.from_pretrained("nhuvo/nllb-600m-en-vimedner-direct-trans-ner-en2vi", device_map="auto")
prefix = "translate English to Vietnamese with inline named entity tags: "
text = "Patients with type 2 diabetes mellitus were enrolled."
inputs = tok(prefix + text, return_tensors="pt")
outputs = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("vie_Latn"),
    max_new_tokens=256,
)
print(tok.batch_decode(outputs, skip_special_tokens=True)[0])