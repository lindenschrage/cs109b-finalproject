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

1. **Installation**  
   Clone the repository and install required packages:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   pip install -r requirements.txt
   
2. **Fine-tuning Models**
To fine-tune the BERT model, use 'FINETUNE-BERT.py'.
To fine-tune the LLaMA model, use 'FINETUNE-llama.py'.

3. **Training Regression Models**
Run 'REGRESSION-bert.py' to train the BERT regression model using BERT embeddings.
Run 'REGRESSION-llama.py' to train the LLaMA regression model using LLaMA embeddings.

5. **Generating Embeddings**
The embeddings can be created using the 'embedding_data.py' script found in helper_functions/.

6. **Data Preprocessing**
If the dataset needs further preprocessing, refer to 'pre_process_data.py'.

7. **Plotting and Visualization##
Use the plotting functions in 'plot_functions.py' to visualize the results and model performance.

8. **Depreciated Files**
'depreciated-FINETUNE-llama.py' contains the initial attempt to fine-tune the LLaMA model using custom prompts. While ultimately unsuccessful, it remains in the repo for reference.
