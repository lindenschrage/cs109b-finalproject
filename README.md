# README

## Project Overview
This repository contains all the data, models, and helper functions needed for fine-tuning and evaluating BERT and LLaMA models on a sentiment prediction task. The main focus is on fine-tuning these models and training regression models using the embeddings. 

### Repository Structure
- **`data/`**  
  Folder containing the preprocessed dataset in CSV format.

- **`embeddings/`**  
  Folder with LLaMA and BERT embeddings saved as `.pkl` files.

- **`helper_functions/`**  
  Folder with utility functions:
  - **`embedding_data.py`**: Functions for creating BERT and LLaMA embeddings.
  - **`plot_functions.py`**: Functions for plotting loss, MSE, and prediction vs. actual graphs.
  - **`pre_process_data.py`**: Code used to preprocess the original dataset.

- **`plots/`**  
  Folder with all generated plots.

- **`FINETUNE-BERT.py`**  
  Script for fine-tuning the BERT model.

- **`FINETUNE-llama.py`**  
  Script for fine-tuning the LLaMA model.

- **`REGRESSION-bert.py`**  
  Script for training a BERT regression model on embeddings.

- **`REGRESSION-llama.py`**  
  Script for training a LLaMA regression model on embeddings.

- **`depreciated-FINETUNE-llama.py`**  
  First attempt at fine-tuning LLaMA (unsuccessful approach).

- **`requirements.txt`**  
  Required Python packages.

### Usage Instructions

**Installation**  
   Clone the repository and install required packages:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   pip install -r requirements.txt
   
