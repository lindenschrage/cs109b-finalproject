
import matplotlib.pyplot as plt

def plot_loss(history, path):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(path)

def plot_mse(history, path):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(path)

def plot_predictions_vs_actual(model, X_test, y_test, path):
    y_pred = model.predict(X_test)  
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5) 
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--') 
    plt.title('Actual vs Predicted Sentiment Scores')
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.grid(True)
    plt.show()
    plt.savefig(path)