# Imports
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AlbertTokenizerFast, AutoTokenizer, AutoConfig, AlbertModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd
from tqdm import tqdm
import multiprocessing
from itertools import chain
from time import time
from datetime import datetime
from platform import python_version
import torch
import os
from accelerate import Accelerator
from huggingface_hub import Repository, get_full_repo_name
from torch.utils.data.dataloader import DataLoader


def create_dataset():
    with open('../../input_data/train.sliding.full.txt') as f:
        train_lines = f.readlines()
    train_lines_df = pd.DataFrame(train_lines)
    train_lines_df = train_lines_df.rename(columns={0: 'text'})
    train_dataset = Dataset.from_pandas(train_lines_df)
    
    with open('../../input_data/eval.sliding.full.txt') as f:
        eval_lines = f.readlines()
    eval_lines_df = pd.DataFrame(eval_lines)
    eval_lines_df = eval_lines_df.rename(columns={0: 'text'})
    eval_dataset = Dataset.from_pandas(eval_lines_df)
    
    return concatenate_datasets([train_dataset, eval_dataset])
    
def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, len(raw_datasets), batch_size)):
        yield raw_datasets[i : i + batch_size]["text"]
        
def group_texts(examples):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    tokenized_inputs = tokenizer(
       examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}
    
def tokenize_dataset(raw_datasets, num_proc):
    tokenized_datasets = raw_datasets.map(group_texts, batched=True, remove_columns=["text"], num_proc=num_proc)
    tokenized_datasets = tokenized_datasets.shuffle(seed=34)
    return tokenized_datasets
     
def initialize_model(tokenizer):
    context_length = 512
    config = AutoConfig.from_pretrained(
        "albert-base-v2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = AlbertModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"HAGALBERT size: {model_size/1000**2:.1f}M parameters")
    model.save_pretrained("HAGALBERT")
    model.push_to_hub("Hielke/HAGALBERT")
    return model

def create_save_tokenizer():
    tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
    hagalbert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=30_000)
    hagalbert_tokenizer.save_pretrained("tokenizer")
    return hagalbert_tokenizer
    
def initialize_data_collator(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)
    return tokenizer, data_collator
    
def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()    

def initialize_trainer(tokenized_datasets, model, data_collator, tokenizer):
    tokenized_datasets_split = tokenized_datasets.train_test_split(test_size=0.2)
    args = TrainingArguments(
        output_dir="HAGALBERT",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        max_steps=125_000,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=0.00176,
        save_steps=5_000,
        fp16=True,
        push_to_hub=True,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets_split["train"],
        eval_dataset=tokenized_datasets_split["test"],
    )
    return tokenized_datasets_split, trainer
    

if __name__ == "__main__":

    # get starting time of script
    start_time = time()
    now = datetime.now()
    print("Python version: {}".format(python_version()))
    print("Amount of visible GPUs: {}".format(torch.cuda.device_count()))
    print("Current date/time: {}".format(now))
    # print("Logging on to Huggingface...")
    # !huggingface-cli login --token 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
    num_proc = multiprocessing.cpu_count()
    print("Amount of CPUs: {}".format(num_proc))
    print("Creating repository...")
    model_name = "HAGALBERT"
    repo_name = get_full_repo_name(model_name)
    print(repo_name)
    # output_dir = "HAGALBERT"
    # repo = Repository(output_dir, clone_from=repo_name)

    print("Creating datasets...")
    raw_datasets = create_dataset()
    print("Creating and saving tokenizer...")
    hagalbert_tokenizer = create_save_tokenizer()
    print("Tokenizing dataset...")
    tokenized_datasets = tokenize_dataset(raw_datasets, num_proc)
    print("Initializing model...")
    model = initialize_model(hagalbert_tokenizer)
    print("Initializing data collator...")
    hagalbert_tokenizer, data_collator = initialize_data_collator(hagalbert_tokenizer)
    print("Initializing trainer...")
    tokenized_datasets_split, trainer = initialize_trainer(tokenized_datasets, model, data_collator, hagalbert_tokenizer)
    print("Training model...")
    trainer.train()
    print("Pushing model to hub...")
    trainer.push_to_hub()
    print("Python test finished (running time: {0:.1f}s)".format(time() - start_time))
