import pandas as pd
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, RobertaTokenizer
import numpy as np
from time import time
from datetime import datetime
from platform import python_version
import torch
import os
from accelerate import Accelerator
from huggingface_hub import Repository, get_full_repo_name
from torch.utils.data.dataloader import DataLoader
import multiprocessing
import evaluate

def tokenize_texts(dataset):
    tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

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
    
    print('Create training and test sets...')
    fhlo = pd.read_csv("../../../input_data/fully_hand_labelled_output.csv")
    fhlo.Roken = fhlo.Roken.astype('category').cat.codes
    fhlo.Alcohol = fhlo.Alcohol.astype('category').cat.codes
    fhlo.Drugs = fhlo.Drugs.astype('category').cat.codes
    fhlo["label"] = fhlo.Alcohol
    fhlo2 = fhlo[["text", "label"]]
    train = fhlo2.sample(frac=0.8,random_state=200)
    test = fhlo2.drop(train.index)
    train.to_csv('train.csv')
    test.to_csv('test.csv')

    file_dict = {
      "train" : "train.csv",
      "test" : "test.csv"
    }
    
    dataset = load_dataset(
      'csv',
      data_files=file_dict,
      delimiter=",",
      column_names=['text', 'label'],
      skiprows=[0, 1583]
    )
    
    print('Create labels...')
    dataset['train'] = dataset['train'].cast_column("label", ClassLabel(num_classes=4, names=["Geen gebruiker", "Huidige gebruiker", "Niets gevonden", "Voormalige gebruiker"]))
    dataset['test'] = dataset['test'].cast_column("label", ClassLabel(num_classes=4, names=["Geen gebruiker", "Huidige gebruiker", "Niets gevonden", "Voormalige gebruiker"]))
    
    print('Tokenizing datasets')
    tokenized_datasets = tokenize_texts(dataset)
    model = AutoModelForSequenceClassification.from_pretrained("Hielke/RobBERT-HAGA", num_labels=4)
    training_args = TrainingArguments(output_dir='finetuned_RobBERT_alcohol', evaluation_strategy="epoch", num_train_epochs = 10, save_strategy="no")
    metric = evaluate.load("accuracy")
    
    print('Training model...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.push_to_hub("Hielke/finetuned_RobBERT_alcohol")