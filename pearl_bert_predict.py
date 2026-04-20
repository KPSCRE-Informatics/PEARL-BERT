##############################################################
# purpose: generate the probability of asthma excerbation
# base on fine-tuning bert model
# input data source: dataset used for prediction
##############################################################

##import packages
##############################################################
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
##############################################################

# empty cache
torch.cuda.empty_cache()

# initiate accelerator
accelerator = Accelerator()

# define the pretrain model path to save the model, please modify this to your setting accordingly
pretrain_model_path = "study/model/pretrain"

# define asthma excerbation model path after downstream fine-tuning, please modify this to your setting accordingly
model_path = "study/model/finetune"

# max sequence length
max_length=512

# load data, please change it to your study specific dataset
datafiles={"test": "study/data/simulated_sample_bert.csv"}
dataset = load_dataset('csv', data_files = datafiles)
dataset = dataset.remove_columns(["study_id","index_date","doc_id"])

# check dataset dimension
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

eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=1024)

# load finetune model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# prepare for gpu running
eval_dataloader, model = accelerator.prepare(eval_dataloader, model)

# file to save the prediction,please change the file name as needed
outputfile=open("study/data/simulated_sample_prediction.csv",'w')
# true label, probability, label_0 for logit of zero, label_1 for logit of 1
outputfile.write('label,prob,label_0,label_1\n')

# generate the predicted probability
metric = evaluate.load("accuracy")

for batch in eval_dataloader:
   # batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    for i in range(logits.shape[0]):
       lg=logits.cpu().numpy()
       prob=np.exp(lg[i,1])/(np.exp(lg[i,0])+np.exp(lg[i,1]))
#      prob=np.exp(logits[i,1])/(np.exp(logits[i,0])+np.exp(logits[i,1]))
       outputfile.write('%1d,%9.7f,%8.6f,%8.6f\n' % (batch["labels"][i],prob,logits[i,0],logits[i,1]))

outputfile.close()
