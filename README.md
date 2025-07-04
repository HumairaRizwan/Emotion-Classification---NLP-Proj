# Emotion-Classification---NLP-Proj

This project involves fine-tuning a pretrained DistilBERT model on the Hugging Face [emotion dataset](https://huggingface.co/datasets/emotion) to classify English text into one of six emotion categories: **sadness, joy, love, anger, fear, and surprise**.

## 📊 Dataset

The dataset contains ~20,000 short English texts labeled with one of the six emotion categories.

- Source: Hugging Face Datasets
- Classes: `['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']`

## 🧠 Model

- Base model: [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased)
- Framework: Hugging Face Transformers with PyTorch
- Training handled using the `Trainer` API

## 🛠️ Techniques Used

- Tokenization using `AutoTokenizer`
- Sequence classification using `AutoModelForSequenceClassification`
- Preprocessing: truncation, padding, label mapping
- Evaluation: accuracy and F1-score
- Visualizations for class distribution and tweet length

## 🏁 Results

| Metric       | Score     |
|--------------|-----------|
| Test Accuracy| **92.7%** |
| Test F1-Score| **92.7%** |

> The model demonstrates strong generalization performance on unseen data.

## 📦 Libraries Used

- Python
- Hugging Face Transformers & Datasets
- PyTorch
- Scikit-learn
- Matplotlib

## 🔍 Sample Inference

```python
text = "I feel amazing and joyful today!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()
print(classes[pred])  # 'joy'
