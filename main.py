import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import wandb
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
os.environ['WANDB_SILENT'] = "true"
import pickle
import random

if __name__ == "__main__":
    wandb.init(project="seqvae")

from tqdm import tqdm

TRAIN=True

EPOCHS = 25
BATCH_SIZE = 64
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data_pol"
EMBED_DIM = 2
NUM_WORKERS = 2
LSTM_LAYERS = 4

LOSS_CLASS = 1
LOSS_KLD = 0.001
LOSS_SIMILAR = 0.1
LOSS_DISSIMILAR = 0.1



class SeqDataset(Dataset):
    def __init__(self, X, X_one_hot, y, contrasts):
        self.X = X
        self.X_one_hot = X_one_hot
        self.y = y
        self.contrasts = contrasts

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        similar_idx = random.choice(self.contrasts[idx]["similar"])
        dissimilar_idx = random.choice(self.contrasts[idx]["dissimilar"])
        similar_x = self.X[similar_idx]
        dissimilar_x = self.X[dissimilar_idx]
        similar_one_hot = self.X_one_hot[similar_idx]
        dissimilar_one_hot = self.X_one_hot[dissimilar_idx]
        return self.X[idx], self.X_one_hot[idx], self.y[idx], similar_x, dissimilar_x, similar_one_hot, dissimilar_one_hot


class SeqVAE(nn.Module):
    def __init__(
        self,
        sequence_length,
        num_subtypes,
        nucl_alphabet_size=5,
        nucl_embedding_dim=EMBED_DIM,
        latent_dim=2,
    
    ):
        super().__init__()

        self.embed = nn.Embedding(nucl_alphabet_size, nucl_embedding_dim)

        self.encoder = nn.LSTM(
            input_size=nucl_embedding_dim,
            hidden_size=latent_dim,
            num_layers=LSTM_LAYERS,
            batch_first=True,
        )

        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)

        # Decoder will map from a single point in latent space to a sequence
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, sequence_length),
            nn.ReLU(),
            unsqueeze_last(),
            nn.LSTM(
                input_size=1,
                hidden_size=nucl_embedding_dim,
                num_layers=LSTM_LAYERS,
                batch_first=True,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, num_subtypes),
            nn.ReLU(),
            nn.Linear(num_subtypes, num_subtypes),
            nn.ReLU(),
        )

        self.embedding_to_nucl = nn.Linear(nucl_embedding_dim, nucl_alphabet_size)

    def reparameterize(self, mu, logvar):
        # We sample from the gaussian using the reparameterization trick
        # https://stats.stackexchange.com/a/16338
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.encoder(x)
        x:torch.Tensor = x[:, -1, :]  # only take the last hidden state
        x_enc = x.clone()
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        x, _ = self.decoder(z)
        x = self.embedding_to_nucl(x)
        return x, mu, logvar, x_enc

    def generate(self, z):
        x,_ = self.decoder(z)
        x = self.embedding_to_nucl(x)
        x = F.softmax(x, dim=-1)
        return x

    def encode(self, x):
        x = self.embed(x)
        x, _ = self.encoder(x)
        x = x[:, -1, :]  # only take the last hidden state
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return z


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())






def main():
    data = get_data()
    model = SeqVAE(sequence_length=data["X"].shape[1])
    if TRAIN:
        model = train(model, data)
        save(model)
    else:
        model.load_state_dict(torch.load("saved_models/model.pt"))
        model.to(DEVICE)
    # plot_latent_space(model, data)
    print("avg_acc", generate(model, data))


def train(model, data):
    model = model.to(DEVICE)
    train_loader, test_loader = _get_data_loaders(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.5)
    recon_loss = nn.CrossEntropyLoss()
    kld_loss = KLDivergenceLoss()
    embed_distance = nn.MSELoss() # TODO: try cosine distance

    for epoch in tqdm(range(EPOCHS)):
        avg_loss = 0
        for i, (x, x_one_hot, label, similar_x, dissimilar_x, similar_one_hot, dissimilar_one_hot) in enumerate(train_loader):
            # if epoch % 5 == 0:
            #     plot_latent_space(model, data, epoch=epoch)
            
            x = x.to(DEVICE)
            x_one_hot = x_one_hot.to(DEVICE)
            similar_x = similar_x.to(DEVICE)
            dissimilar_x = dissimilar_x.to(DEVICE)

            model.train()

            optimizer.zero_grad()

            x_pred, mu, logvar, x_enc = model(x)
            print(x_pred.shape, x.shape)
            loss = recon_loss(x_pred, x)
            loss += kld_loss(mu, logvar) * LOSS_KLD
            loss += 

            sim_enc = model.encode(similar_x)
            dissim_enc = model.encode(dissimilar_x)
            loss += embed_distance(sim_enc, x_enc) * LOSS_SIMILAR
            loss -= embed_distance(dissim_enc, x_enc) * LOSS_DISSIMILAR

            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        wandb.log({"epoch loss": avg_loss / len(train_loader)})
        scheduler.step(avg_loss / len(train_loader))
        print(f"{avg_loss / len(train_loader)=}")


    return model



def plot_latent_space(model, data, epoch='final'):
    model.eval()
    latent_space = []
    labels = []
    if TRAIN:
        for seq, label in tqdm(zip(data["X"], data["y"]), total=len(data["X"]), desc="Plotting Latent Space"):
            seq = seq.unsqueeze(0)
            seq = seq.to(DEVICE)
            z = model.encode(seq)
            latent_space.append(z.cpu().detach().numpy())
            labels.append(label)
    else:
        latent_space = pickle.load(open("latent_space.pkl", "rb"))
        labels = pickle.load(open("labels.pkl", "rb"))
    latent_space = np.array(torch.tensor(latent_space).squeeze(1))
    labels = np.array(labels)
    print(latent_space.shape)
    print(labels.shape)
    with open("latent_space.pkl", "wb") as f:
        pickle.dump(latent_space, f)
    with open("labels.pkl", "wb") as f:
        pickle.dump(labels, f)
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels)
    

    plt.savefig(f"latent_space_{epoch}.png")
    plt.clf()



def generate(model, data):
    # for each sequence, generate 10 new similar sequences
    model.eval()
    avg_acc = 0
    for seq in tqdm(data["X"]):
        seq = seq.unsqueeze(0)
        seq = seq.to(DEVICE)
        z = model.encode(seq)
        for i in range(5):
            new_seq = model.generate(z).squeeze(1)
            new_seq = torch.argmax(new_seq, dim=-1)
            acc = _get_accuracy_integer(new_seq, seq.float())
            avg_acc += acc
            torch.set_printoptions(profile="full")
            print("new_seq", new_seq)
            print("seq", seq)
            print("acc", acc)
            sys.exit()

    return avg_acc / (len(data["X"]) * 5)



def get_data():
    X = torch.load(f"{DATA_DIR}/X.pt")
    X_one_hot = torch.load(f"{DATA_DIR}/X_one_hot.pt")
    y = torch.load(f"{DATA_DIR}/y.pt")
    with open(f"{DATA_DIR}/contrasts.pkl", "rb") as f:
        contrasts = pickle.load(f)
    with open(f"{DATA_DIR}/map_label_to_subtype.pkl", "rb") as f:
        map_label_to_subtype = pickle.load(f)
    with open(f"{DATA_DIR}/map_row_to_seqid.pkl", "rb") as f:
        map_row_to_seqid = pickle.load(f)
    result = {
        "X": X,
        "X_one_hot": X_one_hot,
        "y": y,
        "contrasts": contrasts,
        "map_label_to_subtype": map_label_to_subtype,
        "map_row_to_seqid": map_row_to_seqid,
    }
    return result


def _get_data_loaders(data):
    dataset = SeqDataset(data["X"], data["X_one_hot"], data["y"], data["contrasts"])
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader


def save(model):
    torch.save(model.state_dict(), "saved_models/model.pt")



def _get_accuracy(x_pred, x):
    with torch.no_grad():
        corr = 0
        tota = 0
        for seq in range(x.shape[0]):
            for i in range(x.shape[1]):
                if torch.argmax(x_pred[seq][i]) == int(x[seq][i]):
                    corr += 1
                tota += 1
    return corr / tota


def _get_accuracy_integer(x_pred, x):
    """
    x_pred is the predicted sequence
    x is the true sequence
    Compute their hamming distance.
    return 1- normalized hamming distance
    """
    corr = 0
    inco = 0
    tota = 0
    for seq in range(x.shape[0]):
        for i in range(x.shape[1]):
            if int(x[seq][i]) == int(x_pred[seq][i]):
                corr += 1
            else:
                inco += 1
            tota += 1
    assert corr + inco == tota
    return corr / tota

class printSize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x
    
class unsqueeze_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.unsqueeze(-1)





if __name__ == "__main__":
    main()
