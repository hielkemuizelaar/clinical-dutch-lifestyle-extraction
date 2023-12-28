from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification, AdamW, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, concatenate_datasets
import pandas as pd
from huggingface_hub import Repository, get_full_repo_name

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("jwouts/belabBERT_115k")
    model = RobertaForMaskedLM.from_pretrained("jwouts/belabBERT_115k")
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

    raw_datasets = concatenate_datasets([train_dataset, eval_dataset])
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    tokenized_datasets_split = tokenized_datasets.train_test_split(test_size=0.2)
    args = TrainingArguments(
        output_dir="belabBERT-HAGA",
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
        learning_rate=5e-4,
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
    print("Creating repository...")
    model_name = "belabBERT-HAGA"
    repo_name = get_full_repo_name(model_name)
    model.save_pretrained(model_name)
    model.push_to_hub(repo_name)
    trainer.train()

    print("Pushing model to hub...")
    trainer.push_to_hub()
    print("Python test finished (running time: {0:.1f}s)".format(time() - start_time))
