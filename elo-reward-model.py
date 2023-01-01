import transformers
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
device = torch.device("cuda")

# Load models and tokenizer 
model_one = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
model_two = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

# Load datasets


input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

output_one = model_one.generate(input_ids, 
                                do_sample=True, 
                                early_stopping=True, 
                                temperature=0.7, 
                                max_length=100, 
                                pad_token_id=tokenizer.eos_token_id, 
                                repetition_penalty=0.5)

output_two = model_two.generate(input_ids, 
                                do_sample=True,   
                                early_stopping=True, 
                                temperature=0.9, 
                                max_length=100, 
                                pad_token_id=tokenizer.eos_token_id)

text_one = tokenizer.batch_decode(output_one)[0]
text_two = tokenizer.batch_decode(output_two)[0]



print()

