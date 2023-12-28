from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tqdm
import zipfile
import torch
import pandas as pd
from platform import python_version
from time import time
from transformers import pipeline
import pickle
from joblib import parallel_backend, Parallel, delayed
from datasets import Dataset
import os
from datetime import datetime
import random


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("../opus-mt-nl-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("../opus-mt-nl-en")
    return tokenizer, model
    
def translate(text):
    global output_file
    if isinstance(text, str):
        translated = translator(text)[0]['translation_text']
        output_file.write('"' + translated + '"' + "\n")
        return 
    translated  = "[" + "\n".join(['"' + translator(t)[0]['translation_text'] + '"' for t in text]) + "]" + "\n" 
    output_file.write(translated)
    if random.randint(0, 1000) < 1:
        output_file.close()
        output_file = open("translated_output.txt", "a")
    return
    
def translate_dataset(example):
    example["translated"] = translator(example['text'])[0]['translation_text']
    return example

def preprocess_input_texts(lines):
    full_to_translate = []
    for i, line in enumerate(lines):
        separator = "."
        new_line = line.replace("\t", separator)
        if len(new_line) > 512:
            last_index = new_line[:512].rfind(".") + 1
            if last_index < 1:
                last_index = 512
            lines_to_translate = [new_line[:last_index]]
            current_string = new_line[last_index:]
            while len(current_string) > 512:
                last_index = current_string[:512].rfind(".") + 1
                if last_index < 1:
                    last_index = 512
                lines_to_translate.append(current_string[:last_index])
                current_string = current_string[last_index:]
            if len(current_string) > 0:
                lines_to_translate.append(current_string)
            full_to_translate.append(lines_to_translate)
            del(lines_to_translate)
            continue
        full_to_translate.append(new_line)
    return full_to_translate
    
def get_current_to_translate(full_to_translate):
    if os.path.isfile('translated_output.txt'):
        with open('translated_output.txt', 'r') as f:
            translated_lines = f.readlines()
            current_index = len(translated_lines)
            if current_index < len(full_to_translate):
                current_to_translate = full_to_translate[current_index:]
        return current_to_translate
    with open('translated_output.txt', 'w') as fp:
        return full_to_translate

def get_results_df(full_to_translate):
    results = []
    for i, item in enumerate(full_to_translate):
        if isinstance(item, str):
            results.append([i, 0, item])
            continue
        for j, subitem in enumerate(item):
            results.append([i, j, subitem])
    results_df = pd.DataFrame(results)
    results_df = results_df.rename(columns={0: 'id', 1:'order', 2:'text'})
    return results_df
    

if __name__ == "__main__":

    # get starting time of script
    start_time = time()
    now = datetime.now()
    print("Python version: {}".format(python_version()))
    print("Amount of visible GPUs: {}".format(torch.cuda.device_count()))
    print("Current date/time: {}".format(now))

    print('Creating tokenizer, model...')
    tokenizer, model = load_model()
    print('Loading all input text...')
    with open('../train.sliding.full.txt') as f:
        lines = f.readlines()
    #print('Creating pipeline on GPU..')
    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    print('Preprocessing input texts...')
    full_to_translate = preprocess_input_texts(lines)
    print('Obtaining previous progress...')
    current_to_translate = get_current_to_translate(full_to_translate)
    
    #print('Creating dataset...')
    # results_df = get_results_df(full_to_translate)
    # dataset = Dataset.from_pandas(results_df)
    
    print("Translating input texts...")
    #updated_dataset = dataset.map(translate_dataset, batched=True)
    output_file = open("translated_output.txt", "a")
    with parallel_backend('threading', n_jobs=-1):
        Parallel()(delayed(translate)(i) for i in tqdm.tqdm(current_to_translate))
    #updated_dataset.save_to_disk('updated_dataset')

    print("Python test finished (running time: {0:.1f}s)".format(time() - start_time))