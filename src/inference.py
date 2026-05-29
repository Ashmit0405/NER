from transformers import pipeline

ner = pipeline(
    "ner",
    model="./checkpoints/checkpoint-2634",
    aggregation_strategy="simple"
)


texts = [
    "Elon Musk founded Tesla",
    "Google is in California",
    "Ashmit studies at IIIT Kota"
]


results = ner(texts)

print(results)