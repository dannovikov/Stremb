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

#for formatting stack trace
import traceback

if __name__ == "__main__":
    wandb.init(project="seqvae")

from tqdm import tqdm

TRAIN=True

EPOCHS = 25
BATCH_SIZE = 8
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data_pol"
EMBED_DIM = 2
NUM_WORKERS = 0
LSTM_LAYERS = 6

#Loss weights
LOSS_RECON = 1
LOSS_CLASS = 1
LOSS_CONTRAST = 0.5



class SeqDataset(Dataset):
    def __init__(self, X, y, contrasts):
        self.X = X.float()
        self.y = y
        self.contrasts = contrasts

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)
        similar_idx = random.choice(self.contrasts[idx]["similar"])
        similar_x = self.X[similar_idx].unsqueeze(0)
        dissimilar_idx = random.choice(self.contrasts[idx]["dissimilar"])
        dissimilar_x = self.X[dissimilar_idx].unsqueeze(0)
        return x, self.y[idx], similar_x, dissimilar_x, self.y[similar_idx], self.y[dissimilar_idx]

def get_data():
    X = torch.load(f"{DATA_DIR}/X.pt")
    y = torch.load(f"{DATA_DIR}/y.pt")
    with open(f"{DATA_DIR}/contrasts.pkl", "rb") as f:
        contrasts = pickle.load(f)
    with open(f"{DATA_DIR}/map_label_to_subtype.pkl", "rb") as f:
        map_label_to_subtype = pickle.load(f)
    with open(f"{DATA_DIR}/map_row_to_seqid.pkl", "rb") as f:
        map_row_to_seqid = pickle.load(f)
    result = {
        "X": X,
        "y": y,
        "contrasts": contrasts,
        "map_label_to_subtype": map_label_to_subtype,
        "map_row_to_seqid": map_row_to_seqid,
    }
    return result

class SeqAEClassifier(nn.Module):
    def __init__(self, sequence_length, sequence_embedding_dim, num_classes, num_LSTM_layers):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_embedding_dim = sequence_embedding_dim
        self.num_classes = num_classes
        self.num_LSTM_layers = num_LSTM_layers


        # the sequence embedding will embed entire sequences into a vector
        self.sequence_embedding = nn.LSTM(input_size=sequence_length, hidden_size=sequence_embedding_dim, num_layers=num_LSTM_layers, batch_first=True)

        # the classifier will take the sequence embedding and classify it into a class
        self.classifier = nn.Sequential(
            nn.Linear(sequence_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.decoder = nn.LSTM(input_size=sequence_embedding_dim, hidden_size=sequence_length, num_layers=num_LSTM_layers, batch_first=True)


    def forward(self, x):
        x, _ = self.sequence_embedding(x)
        x = x[:, -1, :] # take the last hidden state
        x = F.normalize(x, p=2, dim=1)
        classification = self.classifier(x)
        reconstruction, _ = self.decoder(x.unsqueeze(1)) # add a dimension for the sequence length
        reconstruction = reconstruction.squeeze(1) # remove the dimension for the sequence length
        return x, reconstruction, classification
    

class simpleNet(nn.Module):
    #just linear
    def __init__(self, sequence_length, sequence_embedding_dim, num_classes):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_embedding_dim = sequence_embedding_dim
        self.num_classes = num_classes
        #sequences are integer encoded like A = 1, C = 2, G = 3, T = 4, else = 0
        self.sequence_embedding = nn.Sequential(
            nn.Linear(sequence_length, sequence_length//2),
            nn.ReLU(),
            nn.Linear(sequence_length//2, sequence_embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(sequence_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.decoder = nn.Sequential(
            nn.Linear(sequence_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, sequence_length)
        )

    def forward(self, x):
        embedding = self.sequence_embedding(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        classification = self.classifier(embedding)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction, classification


def plot_latent_space(model, dataloader, epoch):
    model.eval()
    latent_space = []
    labels = []
    with torch.no_grad():
        for x, y, _, _, _, _ in tqdm(dataloader, desc="Plotting latent space"):
            x = x.to(DEVICE)
            emb, rec, clas = model(x)
            latent_space.append(emb.squeeze(1).cpu().numpy())
            labels.append(y.cpu().numpy())
    latent_space = np.concatenate(latent_space)
    print(latent_space.shape)
    labels = np.concatenate(labels)
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels)
    plt.colorbar()
    plt.savefig(f"./latent_space_{epoch}.png")
    plt.close()

if __name__ == '__main__':
    data = get_data()
    X = data["X"]
    y = data["y"]
    print(X.shape)
    print(y.shape)
    contrasts = data["contrasts"]
    map_row_to_seqid = data["map_row_to_seqid"]
    map_label_to_subtype = data["map_label_to_subtype"]

    # don't split into train and test because the contrast indices considered the whole dataset. Future: split during preproc
    dataset = SeqDataset(X, y, contrasts)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # create the model
    # model = SeqAEClassifier(sequence_length=X.shape[1], sequence_embedding_dim=EMBED_DIM, num_classes=len(map_label_to_subtype), num_LSTM_layers=LSTM_LAYERS)
    model = simpleNet(sequence_length=X.shape[1], sequence_embedding_dim=EMBED_DIM, num_classes=len(map_label_to_subtype))
    model.to(DEVICE)

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # create the loss functions
    reconstruction_loss = nn.MSELoss()
    classification_loss = nn.CrossEntropyLoss()
    contrastive_loss = nn.CosineEmbeddingLoss()

    # create the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # train the model

    for epoch in range(EPOCHS):
        plot_latent_space(model, dataloader, epoch)
        for batch_idx, (x, y, similar_x, dissimilar_x, similar_y, dissimilar_y) in enumerate(dataloader):
            model.train()
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            similar_x = similar_x.to(DEVICE)
            dissimilar_x = dissimilar_x.to(DEVICE)
            similar_y = similar_y.to(DEVICE)
            dissimilar_y = dissimilar_y.to(DEVICE)

            optimizer.zero_grad()
            embedding, reconstruction, classification = model(x)
            similar_embedding, _, similar_classification = model(similar_x)
            dissimilar_embedding, _, dissimilar_classification = model(dissimilar_x)

            rec_loss = reconstruction_loss(reconstruction, x.squeeze(1))
            class_loss = classification_loss(classification, y)

            similar_x_class_loss = classification_loss(similar_classification, similar_y)
            dissimilar_x_class_loss = classification_loss(dissimilar_classification, dissimilar_y)

            # contrast_loss = contrastive_loss(embedding, similar_embedding, torch.ones(embedding.shape[0]).to(DEVICE)) # we want the embeddings to be similar
            # contrast_loss += contrastive_loss(embedding, dissimilar_embedding, -1 * torch.ones(embedding.shape[0]).to(DEVICE)) # we want the embeddings to be dissimilar
            
            # loss = rec_loss + class_loss + similar_x_class_loss + dissimilar_x_class_loss + contrast_loss
            loss = rec_loss * LOSS_RECON + class_loss * LOSS_CLASS + similar_x_class_loss * LOSS_CLASS + dissimilar_x_class_loss * LOSS_CLASS# + contrast_loss * LOSS_CONTRAST
            
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item()} = {rec_loss.item():.2f} * {LOSS_RECON} + {class_loss.item():.2f} * {LOSS_CLASS} + {similar_x_class_loss.item():.2f} * {LOSS_CLASS} + {dissimilar_x_class_loss.item():.2f} * {LOSS_CLASS} + {contrast_loss.item():.2f} * {LOSS_CONTRAST}")




