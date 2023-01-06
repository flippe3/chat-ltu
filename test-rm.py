import transformers
from transformers import GPTNeoForSequenceClassification, GPT2Tokenizer
import torch

torch.manual_seed(42)
transformers.set_seed(42)

model = GPTNeoForSequenceClassification.from_pretrained("reward_model_3").cuda()
model.config.pad_token_id = model.config.eos_token_id

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

prompt = "The world is amazing."

encoded_dict = tokenizer.encode_plus(
                    prompt,            
                    add_special_tokens = True,
                    max_length = 512,
                    padding = 'max_length',
                    return_attention_mask = True,
                    truncation=True,
                    return_tensors = 'pt')
                    
input_ids = encoded_dict['input_ids'].cuda()
attention_masks = encoded_dict['attention_mask'].cuda()

with torch.no_grad():
    output = model(input_ids, attention_mask=attention_masks)
    print(output.logits.softmax(dim=-1))
    print(f"Negative: {output.logits.softmax(dim=-1)[0][0]} Positive: {output.logits.softmax(dim=-1)[0][1]}")