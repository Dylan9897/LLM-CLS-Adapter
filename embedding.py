import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from QwenBase.qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

tokenizer = AutoTokenizer.from_pretrained(
    'QwenBase',
    pad_token='<|extra_0|>',
    eos_token='<|endoftext|>',
    padding_side='left',
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    'QwenBase',
    pad_token_id=tokenizer.pad_token_id,
    device_map="auto",
    trust_remote_code=True
).eval()

# model.generation_config = GenerationConfig.from_pretrained('./', pad_token_id=tokenizer.pad_token_id)

input_demo = "我马上迟到了，怎么做才能不迟到"

input_ids = tokenizer(input_demo, padding='longest')
input_ids['input_ids'] = torch.LongTensor(input_ids['input_ids']).to(model.device)
# input_ids['token_type_ids'] = torch.LongTensor(input_ids['token_type_ids']).to(model.device)
input_ids.pop('token_type_ids')
input_ids['attention_mask'] = torch.LongTensor(input_ids['attention_mask']).to(model.device)
print(input_ids)
output_demo = model(**input_ids)
print(output_demo.logits.shape)
past_key_values = output_demo.past_key_values


