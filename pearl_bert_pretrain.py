#############################################################
# purpose: BERT Pretrain based on mild ashtma cohort dataset
#############################################################

## import packages
#############################################################
from datasets import *
from transformers import *
from tokenizers import *
from itertools import chain
import os
import json
#############################################################

# special tokens for pretrain
special_tokens = [ "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>" ]

# define the path to save the model, please modify this to your setting
model_path = "study/model/pretrain"

# load the study dataset, change the name to use your study dataset
files = ["study/data/simulated_sample_bert.csv"]
dataset = load_dataset("csv", data_files = files, split="train")

# drop the outcome column
dataset = dataset.remove_columns(["study_id","index_date","labels","doc_id"])

# check the dimension
print(dataset.shape)

# split the dataset into training (90%) and testing (10%), please modify the ratio as needed
d = dataset.train_test_split(test_size=0.1)

# 30,522 vocab is BERT's default vocab size, feel free to modify per your study corpus
vocab_size = 2000

# maximum sequence length, lowering will result to faster training (when increasing batch size), modify it as needed
max_length = 512

# whether to truncate
truncate_longer_samples = True

# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()

# train the tokenizer
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)

# enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

# make the directory if not already there
if not os.path.isdir(model_path):
  os.mkdir(model_path)

# save the tokenizer  
tokenizer.save_model(model_path)

# when the tokenizer is trained and configured, load it as BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(model_path)

# dumping some of the tokenizer config to config file, 
# including special tokens, whether to lower case and the maximum sequence length
with open(os.path.join(model_path, "config.json"), "w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "max_position_embeddings": max_length,
      "model_max_length": max_length,
      "max_len": max_length,
      "vocab_size":vocab_size,
  }
  json.dump(tokenizer_cfg, f)


def encode_with_truncation(examples):
  """Mapping function to tokenize the sentences passed with truncation"""
  return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
  """Mapping function to tokenize the sentences passed without truncation"""
  return tokenizer(examples["text"], return_special_tokens_mask=True)

# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

# tokenizing the train dataset
train_dataset = d["train"].map(encode, batched=True)

# tokenizing the testing dataset
test_dataset = d["test"].map(encode, batched=True)

if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as PyTorch tensors
  train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
  # remove other columns, and remain them as Python lists
  test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
  train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

# initialize the model with the config
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)

# initialize the data collator, randomly masking 20% of the tokens for the Masked Language, modify it as needed
# Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,      # over the directory/files if existed
    num_train_epochs=20,            # number of training epochs, feel free to modify
    per_device_train_batch_size=32, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,                # save every 1000 step
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=5,           # whether you don't have much space so you let only 5 model weights saved in the disk
)


# initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# train the model
trainer.train()

# save the pretrained model
trainer.save_model(model_path)
