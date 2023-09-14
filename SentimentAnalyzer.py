import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# دانلود مدل پیش‌آموزش‌داده‌شده BERT برای تشخیص احساسات
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# تابعی برای تشخیص احساسات از متن
def detect_sentiment(text):
    # توکن‌بندی متن
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")

    # اجرای مدل بر روی متن
    with torch.no_grad():
        outputs = model(tokens)

    # استخراج احساس از نتایج مدل
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # ترجمه‌ی احساس به عبارت معنایی
    if prediction == 0:
        sentiment = "Negative"
    elif prediction == 2:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    return sentiment

# نمونه استفاده از تابع
text = "این یک متن مثبت است."
result = detect_sentiment(text)