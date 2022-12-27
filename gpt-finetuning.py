# https://github.com/dredwardhyde/gpt-neo-fine-tuning-example

import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained("~/models/gpt-j-6B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

training_args = TrainingArguments(output_dir='./results', num_train_epochs=4.3, logging_steps=100, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=15, per_device_eval_batch_size=15, warmup_steps=100, 
                                  weight_decay=0.01, logging_dir='./logs', fp16=True, low_cpu_mem_usage=True, deepspeed='./ds_config_gpt_j.json')

model = AutoModelForCausalLM.from_pretrained("~/models/gpt-j-6B").cuda()
model.resize_token_embeddings(len(tokenizer))
max_length = max([len(tokenizer.encode(description)) for description in descriptions])
print("Max length: {}".format(max_length))

trainer = Trainer(model=model, 
                  args=training_args, 
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset, 
                  data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                               'attention_mask': torch.stack([f[1] for f in data]),
                                               'labels': torch.stack([f[0] for f in data])})
trainer.train()                                    
generated = tokenizer("<|startoftext|>", return_tensors="pt").input_ids.cuda()
sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                bos_token='<|startoftext|>',
                                eos_token='<|endoftext|>', pad_token='<|pad|>',
                                max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)

for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))