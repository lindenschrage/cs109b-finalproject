from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from dotenv import load_dotenv, dotenv_values 

load_dotenv() 

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")



def get_bert_embeddings(df, path):

    bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = AutoModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True).to('cuda')  
    inputs = bert_tokenizer(list(df['Tweet']), add_special_tokens=True, truncation=True, padding=True, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = bert_model(**inputs)
        hidden_states = outputs.hidden_states
        last = hidden_states[-1][:, 0, :]  # taking  CLS token embeddings from the last layer
        sec = hidden_states[-2][:, 0, :]    # 2nd to last layer
        thr = hidden_states[-3][:, 0, :]    
        frth = hidden_states[-4][:, 0, :] 
        embeddings = last + sec + thr + frth 

    df['Tweet-tokens'] = embeddings.cpu().numpy().tolist()

    top_layers = list(df['Tweet-tokens'])

    with open(path, 'wb') as f:
        pickle.dump(top_layers, f)

def get_llama_embeddings(df, path):

    base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=ACCESS_TOKEN, output_hidden_states=True).to('cuda')

    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=ACCESS_TOKEN, return_tensors = 'tf')
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    top_layers = []

    tweet_text = list(df['Tweet'])
    with torch.no_grad():
        for tweet in tweet_text:
            tokens = tokenizer(tweet, return_tensors='pt', padding=True).to('cuda')
            output = base_model(**tokens)
            sentence_embeddings = output.hidden_states[-1].mean(dim=1)
            top_layers.append(sentence_embeddings.cpu().detach().numpy())

    with open(path, 'wb') as f:
        pickle.dump(top_layers, f)