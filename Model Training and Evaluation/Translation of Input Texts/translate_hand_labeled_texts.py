import pandas as pd
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification, AdamW, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

def translate_dataset(example): 
    return translator(example['text'])[0]['translation_text']

def get_full_to_translate(lines):
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


if __name__ == "__main__":

    fhlo = pd.read_csv("../../input_data/fully_hand_labelled_output.csv")
    fhlo2 = fhlo["text"]
    fhlo2.to_csv('texts.csv')
    
    lines = list(fhlo['text'])
    full_to_translate = get_full_to_translate(lines)
    results = []
    for i, item in enumerate(full_to_translate):
        if isinstance(item, str):
            results.append([i, 1, item])
            continue
        for j, subitem in enumerate(item):
            results.append([i, j, subitem])
            
    result_df = pd.DataFrame(results)
    result_df = result_df.rename(columns={0: 'doc_id', 1:'segment_id', 2:'text'})
    dataset = Dataset.from_pandas(result_df)
    
    tokenizer = AutoTokenizer.from_pretrained("../../opus-mt-nl-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("../../opus-mt-nl-en")
    translator = pipeline("translation", model=model, tokenizer=tokenizer, device = 0)
    translated_dataset = dataset.map(lambda example: {"translated": translate_dataset(example)})
    translated_dataset_df = translated_dataset.to_pandas()
    translated_dataset_df.to_csv('translated_dataset_df.csv')

    
