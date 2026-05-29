from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

def load_model(label_names):

    tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
    id2label = {i: l for i, l in enumerate(label_names)}
    label2id = {l: i for i, l in enumerate(label_names)}

    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id
    )

    return model, tokenizer