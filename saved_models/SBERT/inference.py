import json
import os
import re
import unicodedata
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
THRESH_PATH = os.path.join(ROOT_DIR, "threshold.json")
LABEL_MAP_PATH = os.path.join(ROOT_DIR, "label_map.json")
BEST_CKPT = os.path.join(ROOT_DIR, "best_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_text(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\[\d[\d,;\s\-]*\]", " ", text)
    text = re.sub(r"\(\d[\d,;\s\-]*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text(text, max_words=300):
    text = normalize_text(text)
    if not text:
        return ""
    words = text.split()
    return " ".join(words[:max_words])

class SBERTClassifier(nn.Module):
    def __init__(self, emb_dim=768, dropout=0.25):
        super().__init__()
        feat_dim = emb_dim * 4 + 1
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, a, c):
        a = nn.functional.normalize(a, p=2, dim=-1)
        c = nn.functional.normalize(c, p=2, dim=-1)
        diff = torch.abs(a - c)
        prod = a * c
        cos = torch.sum(a * c, dim=-1, keepdim=True)
        x = torch.cat([a, c, diff, prod, cos], dim=-1)
        return self.net(x)

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default

THRESHOLD = float(load_json(THRESH_PATH, {"threshold": 0.5})["threshold"])
LABEL_MAP = load_json(LABEL_MAP_PATH, {"0": "Different subdomain", "1": "Same subdomain"})

# Local backbone saved in the package, no internet dependency
backbone = SentenceTransformer(MODEL_DIR, device=str(DEVICE))
backbone.max_seq_length = 256
backbone.eval()

ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
emb_dim = int(ckpt.get("emb_dim", 768))
dropout = float(ckpt.get("dropout", 0.25))

classifier = SBERTClassifier(emb_dim=emb_dim, dropout=dropout).to(DEVICE)
classifier.load_state_dict(ckpt["state_dict"])
classifier.eval()

def encode_text(text):
    text = clean_text(text)
    emb = backbone.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return torch.tensor(emb, dtype=torch.float32, device=DEVICE)

def predict_pair(abstract, conclusion):
    a = encode_text(abstract)
    c = encode_text(conclusion)
    with torch.no_grad():
        logits = classifier(a, c)
        prob_1 = torch.softmax(logits, dim=-1)[0, 1].item()
        pred = int(prob_1 >= THRESHOLD)
    return {
        "label": pred,
        "label_name": LABEL_MAP.get(str(pred), str(pred)),
        "probability_same_subdomain": prob_1,
        "threshold": THRESHOLD
    }

if __name__ == "__main__":
    print(predict_pair("Sample abstract.", "Sample conclusion."))
