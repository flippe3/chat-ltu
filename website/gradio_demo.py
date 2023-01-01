import gradio as gr
from transformers import AutoModel, AutoTokenizer, GPTJForCausalLM
import torch
model = GPTJForCausalLM.from_pretrained("/home/fleip/models/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("/home/fleip/models/gpt-j-6B")
tokenizer.pad_token_id = tokenizer.eos_token_id

def language_model(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    gen_tokens = model.generate(input_ids, do_sample=True, num_return_sequences=4, max_length=150, early_stopping=True)

    outputs = []
    for output in gen_tokens:
        outputs.append(tokenizer.decode(output, skip_special_tokens=True))
    return outputs


demo = gr.Interface(title="Open-Fnet prompting for reward model",
                    fn=language_model, 
                    inputs="text", 
                    outputs=["text", "text", "text", "text"],
                    )

demo.launch(share=True)   