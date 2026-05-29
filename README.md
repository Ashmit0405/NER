
# Transformer-Based Named Entity Recognition (NER)
## Overview
This project implements a transformer-based Named Entity Recognition (NER) system using the CoNLL-2003 dataset and the BERT architecture.

The model is fine-tuned using Hugging Face Transformers for token classification tasks. The system is capable of identifying entities such as:
- Persons (PER)
- Organizations (ORG)
- Locations (LOC)
- Miscellaneous entities (MISC)

### Example:
#### Input:
Elon Musk founded Tesla in California

#### Output:
- Elon Musk → PERSON
- Tesla → ORGANIZATION
- California → LOCATION


## Project Structure
```text
root/ 
│ 
├── src/ 
│ ├── train.py 
│ ├── inference.py 
│ ├── dataset.py 
│ ├── model.py 
│ └── utils.py 
│ 
├── configs/ 
│ └── train_config.yaml 
│ 
├── ner_model/ 
│ └── checkpoints/
| 
├── requirements.txt 
├── README.md 
└── .gitignore
```

## Dataset
The project uses the CoNLL-2003 dataset.
Dataset source:
- Reuters news articles
- Annotated for Named Entity Recognition

| Label | Meaning |
|---|---|
| B-PER | Beginning of Person |
| I-PER | Inside Person |
| B-ORG	| Beginning of Organization |
| I-ORG	| Inside Organization |
| B-LOC	| Beginning of Location |
| I-LOC	| Inside Location |
| B-MISC	| Beginning of Miscellaneous |
| I-MISC	| Inside Miscellaneous |
| O	| Outside any entity |

Dataset  Statistics:
|Split | Samples |
|---|---|
| Train | 14,041 |
| Validation | 3,250 |
| Test | 3,453 |

## Model Architecture
The project uses:
- BERT Base Cased
- Token Classification Head
- BIO tagging scheme

Pipeline:

```text
Input Text
   ↓
Tokenizer
   ↓
BERT Transformer
   ↓
Token Classification Layer
   ↓
NER Predictions
```

## File Descriptions
```
src/train.py
```  
Main training script.

Responsibilities:
- Load dataset
- Preprocess data
- Initialize model
- Configure trainer
- Train model
- Save checkpoints

Run:
``` bash
python src/train.py
```
###
```
src/inference.py
```  
Runs inference using the trained model.  

Responsibilities:
- Load trained checkpoint
- Predict entities from text
- Aggregate BIO spans

Run:
``` bash
python src/inference.py
```   
###
```
src/dataset.py  
```
Handles:
- dataset loading
- tokenization
- label alignment
- preprocessing

Important functionality:
- aligns labels with BERT subword tokenization

###
```
src/model.py
```
Responsible for:
- tokenizer initialization
- model initialization
- label mappings

Uses:
- bert-base-cased
###
```
src/utils.py  
```
Contains utility functions such as:
- evaluation metrics
- seqeval computation
- helper functions
###


## Evaluation Metrices
The project uses seqeval metrics:
- Precision
- Recall
- F1 Score
- Accuracy

Metric computation occurs during validation after each epoch.

## Future Improvements

Potential extensions:
- RoBERTa / DeBERTa models
- CRF decoding layer
- Span-based NER
- Multi-task learning
- Entity linking
- Knowledge graph integration
- Quantization for deployment
- FastAPI inference server
- Docker deployment

```
Note: The above code for training a model is only divided into different files to maintain modularity the original precursor of the above code was written in a single notebook
```