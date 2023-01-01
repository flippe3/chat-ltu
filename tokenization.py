import transformers
from transformers import GPTNeoForSequenceClassification, GPT2Tokenizer
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset

# Variables
device = torch.device("cuda")
batch_size = 32
epochs = 4

# Load model and tokenizer 
model = GPTNeoForSequenceClassification.from_pretrained("EleutherAI/gpt-neo-125M").to(device)

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

# Load data
raw_data = load_dataset("Anthropic/hh-rlhf")

train_texts = []
test_texts = []
train_sentiment = []
test_sentiment = []

count = 0 

for i in raw_data['train']:
    train_texts.append(i['chosen'])
    train_sentiment.append(1)
    train_texts.append(i['rejected'])
    train_sentiment.append(0)
    count += 1
    if count == 500:
        break
count = 0 
for i in raw_data['test']:
    test_texts.append(i['chosen'])
    test_sentiment.append(1)
    test_texts.append(i['rejected'])
    test_sentiment.append(0)
    count += 1
    if count == 100:
        break

encoded_dict = tokenizer.batch_encode_plus(
                    train_texts,            
                    add_special_tokens = True,
                    max_length = 512,
                    padding = 'max_length',
                    return_attention_mask = True,
                    truncation=True,
                    return_tensors = 'pt')

train_input_ids = encoded_dict['input_ids']
train_attention_masks = encoded_dict['attention_mask']
train_sentiments = torch.tensor(train_sentiment, dtype=torch.long)


encoded_dict = tokenizer.batch_encode_plus(
                    test_texts,            
                    add_special_tokens = True,
                    max_length = 512,
                    padding = 'max_length',
                    return_attention_mask = True,
                    truncation=True,
                    return_tensors = 'pt')

test_input_ids = encoded_dict['input_ids']
test_attention_masks = encoded_dict['attention_mask']
test_sentiments = torch.tensor(test_sentiment, dtype=torch.long)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_sentiments)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_sentiments)

torch.save(train_dataset, "train_small.pickle")
torch.save(test_dataset, "test_small.pickle")