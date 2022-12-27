import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
import torch

torch.manual_seed(42)
transformers.set_seed(42)

#model = AutoModel.from_pretrained("/home/fleip/models/gpt-neo-1.3B").half().cuda()
#tokenizer = AutoTokenizer.from_pretrained("/home/fleip/models/gpt-neo-1.3B")
generator = pipeline('text-generation', model='/home/fleip/models/gpt-neo-1.3B')
prompt = "One day in the future the gravity will flip and everything that was once attached to earth will be flipped."
generator(prompt, do_sample=True, min_length=20)


input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=200)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
