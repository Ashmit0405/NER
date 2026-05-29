from datasets import load_dataset

def load_dataset_data():
    return load_dataset("tomaarsen/conll2003")

def tokenize_align(data, tokenizer):

    tokenized = tokenizer(
        data['tokens'],
        truncation=True,
        is_split_into_words=True
    )
    all_labels = []
    for i, label in enumerate(data['ner_tags']):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized['labels'] = all_labels
    return tokenized

def preprocess(dataset, tokenizer):
    return dataset.map(
        lambda x: tokenize_align(x, tokenizer),
        batched=True
    )