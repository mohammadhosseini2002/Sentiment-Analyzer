import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# دانلود مدل پیش‌آموزش‌داده‌شده BERT برای تشخیص احساسات
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)