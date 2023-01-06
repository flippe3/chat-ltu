import transformers
from transformers import GPTNeoForSequenceClassification, GPT2Tokenizer, get_linear_schedule_with_warmup, Trainer, TrainingArguments, GPTNeoForCausalLM
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.nn import BCELoss
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import os
import torch
from tqdm import tqdm
tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# Variables
device = torch.device("cuda")
batch_size = 24
epochs = 4
learning_rate = 3e-5

# Load initial model [Step 1] 
initial_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
initial_model.config.pad_token_id = initial_model.config.eos_token_id

# Load reward model [Step 2]
reward_model = GPTNeoForSequenceClassification.from_pretrained("reward_model_3").to(device)
reward_model.config.pad_token_id = reward_model.config.eos_token_id

# Loading tokenizer (same tokenizer for all models since everything is the same model)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

# Load final fine-tuned model [Step 3] (This should ideally be mostly frozen)
config = PPOConfig(
    model_name="EleutherAI/gpt-neo-125M",
    learning_rate=1.41e-5,
)

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 12
}

prompts = []
for i in open("data/train.wp_source", 'r').readlines():
    prompts.append(i)


encoded_dict = tokenizer.batch_encode_plus(
                    prompts,            
                    add_special_tokens = True,
                    max_length = 512,
                    padding = 'max_length',
                    return_attention_mask = True,
                    truncation=True,
                    return_tensors = 'pt')
                    
input_ids = encoded_dict['input_ids']
attention_masks = encoded_dict['attention_mask']

dataset = DataLoader(input_ids, batch_size=12)

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

generation_kwargs = {
    "min_length":10,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch['input_ids']

    response_tensors = []
    for query in query_tensors:
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # texts = [q + r for q,r in zip(batch['query'], batch['response'])]
    with torch.no_grad():
        pipe_outputs = reward_model(batch['response'], **sent_kwargs)
        rewards = pipe_outputs.logits
        print("Rewards:", rewards)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    # Run PPO step 
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)