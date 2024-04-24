
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
import torch
from datasets import Dataset


model_path = '/n/home09/lschrage/projects/cs109b/finetuned_model'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

url = '/n/home09/lschrage/projects/cs109b/cs109b-finalproject/dataframe.csv'
df = pd.read_csv(url)
df.head()

y = df['TweetAvgAnnotation']
X = df

X_train_full, X_test, y_true_train_full, y_true_test = train_test_split(X, y, test_size=0.2, random_state=109, stratify=X['Sentiment'])

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_true_train_full, test_size=0.5, random_state=109, stratify=X_train_full['Sentiment'])

def generate_train_prompt(tweet):
  return f"""
          Analyze the sentiment of the tweet enclosed in square brackets,
          determine if it is positive, neutral, or negative, and return the answer as a float value rounded to two decimal places
          between -3 (corresponding to a  negative sentiment) and 3 (corresponding to a positive sentiment).

          [{tweet["Tweet"]}] = {tweet["TweetAvgAnnotation"]}
          """.strip()
def generate_test_prompt(tweet):
  return f"""
          Analyze the sentiment of the tweet enclosed in square brackets,
          determine if it is positive, neutral, or negative, and return the answer as a float value rounded to two decimal places
          between -3 (corresponding to a  negative sentiment) and 3 (corresponding to a positive sentiment).

          [{tweet["Tweet"]}] =
          """.strip()

X_train_full['Prompt'] = X_train_full.apply(generate_train_prompt, axis=1)
X_train['Prompt'] = X_train.apply(generate_train_prompt, axis=1)
X_test['Prompt'] = X_test.apply(generate_test_prompt, axis=1)
X_val['Prompt'] = X_val.apply(generate_test_prompt, axis=1)

df_val = pd.DataFrame({
    "text": X_val['Prompt'],
    "labels": y_val
})

val_dataset = Dataset.from_pandas(df_val)

def evaluate_model(model, tokenizer, dataset):
    model.eval()
    predictions, true_labels = [], []

    for _, row in dataset.iterrows():
        inputs = tokenizer(row['text'], return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = logits.argmax(-1).item()
        predictions.append(predicted_label)
        true_labels.append(row['labels'])

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['Negative', 'Neutral', 'Positive'])
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

# Use the function
evaluate_model(model, tokenizer, val_dataset.to_pandas())
