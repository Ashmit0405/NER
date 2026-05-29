from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

from dataset import (
    load_dataset_data,
    preprocess
)

from model import load_model
from utils import compute_metrics

def main():

    dataset = load_dataset_data()

    label_names = (
        dataset['train']
        .features['ner_tags']
        .feature.names
    )

    model, tokenizer = load_model(label_names)

    tokenized_dataset = preprocess(
        dataset,
        tokenizer
    )

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2
    )

    data_collator = (
        DataCollatorForTokenClassification(
            tokenizer
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=lambda p:
            compute_metrics(p, label_names)
    )

    trainer.train()

    trainer.save_model(
        "./checkpoints/final_model"
    )


if __name__ == "__main__":
    main()