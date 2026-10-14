import torch
import torch.nn as nn

# _________ Neural Network ________________________

def train_nn(num_features, train_dl, val_dl):

    model = nn.Sequential(
        nn.Linear(num_features, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(200):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = total = 0
            for xb, yb in val_dl:
                preds = (model(xb).squeeze(1) > 0).float()
                correct += (preds == yb).sum().item()
                total += len(yb)
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}  val acc {correct / total:.3f}")

    return model


def test_nn(model, X_test, y_test):

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test).squeeze(1))
        preds = (probs > 0.5).float()

    # Evaluate
    tp = ((preds == 1) & (y_test == 1)).sum().item()
    tn = ((preds == 0) & (y_test == 0)).sum().item()
    fp = ((preds == 1) & (y_test == 0)).sum().item()
    fn = ((preds == 0) & (y_test == 1)).sum().item()

    print(f"accuracy  {(tp + tn) / len(y_test):.3f}")
    print(f"precision {tp / max(tp + fp, 1):.3f}")
    print(f"recall    {tp / max(tp + fn, 1):.3f}")
    print(f"confusion: TP={tp} FP={fp} TN={tn} FN={fn}")


# _________ XGBoost ________________________

from utils import transform_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

def train_xgb(data, mean, std):

    # Same pair features as the NN; split the DataSamples' feature matrix, not the DataSamples themselves
    X, y = transform_data(data, mean, std)
    X_train, X_val, y_train, y_val = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)

    # Gradient-boosted trees (sklearn's equivalent of XGBoost, no extra system libs needed)
    model = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.01,
        max_depth=10,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Validation (same source as training) and test (the synthetic set from above)

    print(f"val acc  {accuracy_score(y_val, model.predict(X_val)):.3f}")

    return model

def test_xgb(model, X_test, y_test):
    print(f"test acc {accuracy_score(y_test.numpy(), model.predict(X_test.numpy())):.3f}")


# _________ Speaker encoder ________________________

import random

import torch.nn.functional as F
from torch.utils.data import DataLoader

from encoder import SpeakerDataset, SpeakerEncoder, SmallSpeakerEncoder, load_wave, log_mel


def train_encoder(files_by_speaker, EncoderClass, epochs=30, val_frac=0.1, batch_size=64, num_workers=4):
    """Train the encoder as a speaker classifier; its embedding layer is what we use afterwards"""

    # Hold out some clips of every speaker to check classification on unseen clips
    train_files, val_files = {}, {}
    for speaker, files in files_by_speaker.items():
        files = random.sample(files, len(files))
        n_val = max(1, int(val_frac * len(files)))
        train_files[speaker], val_files[speaker] = files[n_val:], files[:n_val]

    train_ds, val_ds = SpeakerDataset(train_files), SpeakerDataset(val_files)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    model = EncoderClass(n_speakers=len(train_ds.speakers))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = total = 0
            for xb, yb in val_dl:
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += len(yb)
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}  val speaker acc {correct / total:.3f}")

    return model


def embed_file(model, path):
    model.eval()
    with torch.no_grad():
        return model.embedding(log_mel(load_wave(path)).unsqueeze(0))[0]  # whole clip, no crop


def test_encoder(model, samples, threshold=0.5):
    """Same/different speaker on DataSample pairs via cosine similarity of the embeddings"""

    cache = {}
    def emb(path):
        if path not in cache:
            cache[path] = embed_file(model, path)
        return cache[path]

    sims = torch.tensor([F.cosine_similarity(emb(s.sample_1), emb(s.sample_2), dim=0).item() for s in samples])
    y_test = torch.tensor([float(s.same) for s in samples])
    preds = (sims > threshold).float()

    # Evaluate
    tp = ((preds == 1) & (y_test == 1)).sum().item()
    tn = ((preds == 0) & (y_test == 0)).sum().item()
    fp = ((preds == 1) & (y_test == 0)).sum().item()
    fn = ((preds == 0) & (y_test == 1)).sum().item()

    print(f"threshold {threshold}")
    print(f"accuracy  {(tp + tn) / len(y_test):.3f}")
    print(f"precision {tp / max(tp + fp, 1):.3f}")
    print(f"recall    {tp / max(tp + fn, 1):.3f}")
    print(f"confusion: TP={tp} FP={fp} TN={tn} FN={fn}")

    # Best possible threshold on this set, to see how far off the chosen one is
    best = max(sims.tolist(), key=lambda t: ((sims > t).float() == y_test).float().mean().item())
    print(f"best threshold {best:.3f} -> accuracy {((sims > best).float() == y_test).float().mean().item():.3f}")

    return sims
