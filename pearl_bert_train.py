################################################################
# purpose: fine-tuning the pretrain bert model to predict 
# asthma excerbation in mild asthma
################################################################

##import packages
################################################################
import sys
import os
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from accelerate import Accelerator
################################################################

torch.cuda.empty_cache()
accelerator = Accelerator()

# tunning parameter spaces
################################################################
max_length = 512 # max sequence length
batch_size = 16 # batch size
num_epochs = 3 # number of epoch
lr=5e-6  # learning rate
################################################################

# define the pretrain model path to save the model, modify this to your setting accordingly
pretrain_model_path = "study/model/pretrain"

# define asthma excerbation model path after downstream fine-tuning, modify this to your setting accordingly
model_path = "study/model/finetune"

# load study data, please change it to your study dataset
datafiles={"train": "study/data/simulated_sample_bert.csv"}
dataset = load_dataset('csv', data_files = datafiles)
dataset = dataset.remove_columns(["study_id","index_date","doc_id"])

# split the dataset into training (90%) and testing (10%) for internal validation, please modify the ratio as needed
dataset = dataset["train"].train_test_split(test_size=0.1)

# check split dataset dimension
print(dataset.shape)

# tokenizer based on the pretrain bert model
tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)

def tokenize_function(item):
    return tokenizer(item["text"], max_length=max_length,padding ="max_length", truncation=True)

# map text tokens into ids
tokenized_datasets = dataset.map(tokenize_function, batched = True)

# remove the original text column
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# set torch format
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=64)

# loading the pretrain bert model
model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_path, num_labels=2)

# initiating the optimizer
optimizer = AdamW(model.parameters(), lr=lr)

train_dataloader,eval_dataloader,model,optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

#save model
model.save_pretrained(model_path)
