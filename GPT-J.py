from transformers import AutoModel, AutoTokenizer, GPTJForCausalLM
import torch
model = GPTJForCausalLM.from_pretrained("/home/fleip/models/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("/home/fleip/models/gpt-j-6B")

prompt = "Tell me a story about a genius girl"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

gen_tokens = model.generate(input_ids, do_sample=True, num_return_sequences=4, max_length=100)
#gen_text = tokenizer.batch_decode(gen_tokens)[0]


for i, output  in enumerate(gen_tokens):
    print(f"{i}: {tokenizer.decode(output, skip_special_tokens=True)}\n**********************************************************")
