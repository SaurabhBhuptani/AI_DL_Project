import json
import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
TOKENIZER_DIR = os.path.join(ROOT_DIR, "tokenizer")
THRESH_PATH = os.path.join(ROOT_DIR, "threshold.json")
LABEL_MAP_PATH = os.path.join(ROOT_DIR, "label_map.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

thr_payload = load_json(THRESH_PATH, {"threshold": 0.5})
threshold = float(thr_payload.get("threshold", 0.5))
label_map = load_json(LABEL_MAP_PATH, {"0": "Different subdomain", "1": "Same subdomain"})

NO_ID = 150
YES_ID = 4273

def predict_pair(abstract, conclusion, max_len=512):
    prompt = (
        "classify whether the abstract and conclusion are in the same subdomain: "
        f"abstract: {abstract} conclusion: {conclusion}"
    )
    enc = tokenizer(
        prompt,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    decoder_input_ids = torch.full(
        (enc["input_ids"].size(0), 1),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=DEVICE
    )
    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input_ids
        )
        step0 = out.logits[:, 0, :]
        two_logits = step0[:, [NO_ID, YES_ID]]
        prob_same = torch.softmax(two_logits, dim=-1)[0, 1].item()
    pred = int(prob_same >= threshold)
    return {
        "label": pred,
        "label_name": label_map.get(str(pred), str(pred)),
        "probability_same_subdomain": prob_same,
        "threshold": threshold
    }

if __name__ == "__main__":
    print(predict_pair("Sample abstract.", "Sample conclusion."))
