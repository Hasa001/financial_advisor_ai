from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_local_llm(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


def generate_advice(prediction, confidence, news_docs):
    tokenizer, model = load_local_llm()

    news_text = "\n".join(
        [f"- {doc['content']}" for doc in news_docs]
    )

    prompt = f"""
You are a financial advisor AI combining ML + RAG.

Prediction:
    Direction = {prediction}
    Confidence = {confidence:.2f}

Recent News:
{news_text}

Now explain what is most likely to happen in the market and give
a safe, simple investment recommendation.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=300)

    return tokenizer.decode(output[0], skip_special_tokens=True)
