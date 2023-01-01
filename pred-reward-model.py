import transformers
from transformers import GPTNeoForSequenceClassification, GPT2Tokenizer, get_linear_schedule_with_warmup, Trainer, TrainingArguments
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.nn import BCELoss
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

# Variables
device = torch.device("cuda")
batch_size = 24
epochs = 4
learning_rate = 3e-5

# Load model and tokenizer 
model = GPTNeoForSequenceClassification.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
model.config.pad_token_id = model.config.eos_token_id

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

# Load data
train_dataset = torch.load("train_gpt_2_tokenized_data.pickle")
test_dataset = torch.load("test_gpt_2_tokenized_data.pickle")
# train_dataset = torch.load("train_small.pickle")
# test_dataset = torch.load("test_small.pickle")


train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size=batch_size)

# Training configs
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps=total_steps)

# Training and validation
for epoch in range(epochs):
    print(f"===== Epoch {epoch+1} / {epochs} =====")
    # Training
    model.train()
    avg_epoch_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        model.zero_grad()

        b_ids = batch[0].to(device)
        b_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
        avg_epoch_loss += outputs.loss.item() 

        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    print("Average epoch loss:", avg_epoch_loss/len(train_dataloader))

    model.save_pretrained(f"reward_model_{epoch}")

    # Validation
    model.eval()
    with torch.no_grad():
            true_labels = []
            pred_labels = []
            for step, batch in enumerate(tqdm(test_dataloader)):
                b_ids = batch[0].to(device)
                b_mask = batch[1].to(device)

                outputs = model(input_ids=b_ids, attention_mask=b_mask)

                logits = outputs.logits.detach().cpu().numpy().tolist()
                true_labels.extend(batch[2])
                pred_labels.extend(np.argmax(logits, axis=1))

            print(classification_report(pred_labels, true_labels, digits=6))

            output_file = open("training_scores.txt", 'a')
            output_file.write(classification_report(pred_labels, true_labels, digits=6) + "\n")
            output_file.close()
            print("===== Finished Validation =====") 