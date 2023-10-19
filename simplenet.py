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
import traceback

if __name__ == "__main__":
    wandb.init(project="seqvae")

from tqdm import tqdm

TRAIN=True

EPOCHS = 100
BATCH_SIZE = 16
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data_pol"
EMBED_DIM = 2
NUM_WORKERS = 0

LOSS_RECON = 0.5
LOSS_CLASS = 2.5
LOSS_CONTRAST_SIM = 1.0
LOSS_CONTRAST_DISSIM = 1.2



class SeqDataset(Dataset):
    def __init__(self, X, y, contrasts):
        self.X = X.float()
        self.y = y
        self.contrasts = contrasts

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        similar_idx = random.choice(self.contrasts[idx]["similar"])
        similar_x = self.X[similar_idx]
        dissimilar_idx = random.choice(self.contrasts[idx]["dissimilar"])
        dissimilar_x = self.X[dissimilar_idx]
        return x, self.y[idx], similar_x, dissimilar_x, self.y[similar_idx], self.y[dissimilar_idx], idx

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


class simpleNet(nn.Module):
    def __init__(self, sequence_length, sequence_embedding_dim, num_classes):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_embedding_dim = sequence_embedding_dim
        self.num_classes = num_classes

        self.sequence_embedding = nn.Sequential(
            nn.Linear(sequence_length, sequence_length),
            nn.ReLU(),
            nn.Linear(sequence_length, sequence_length//2),
            nn.ReLU(),
            nn.Linear(sequence_length//2, sequence_length//4),
            nn.ReLU(),
            nn.Linear(sequence_length//4, sequence_length//8),
            nn.ReLU(),
            nn.Linear(sequence_length//8, sequence_length//16),
            nn.ReLU(),
            nn.Linear(sequence_length//16, sequence_length//32),
            nn.ReLU(),
            nn.Linear(sequence_length//32, sequence_length//64),
            nn.ReLU(),
            nn.Linear(sequence_length//64, sequence_length//128),
            nn.ReLU(),
            nn.Linear(sequence_length//128, sequence_embedding_dim),
            nn.BatchNorm1d(sequence_embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(sequence_embedding_dim, sequence_embedding_dim*2),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*2, sequence_embedding_dim*4),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*4, sequence_embedding_dim*8),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*8, sequence_embedding_dim*16),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*16, sequence_embedding_dim*32),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*32, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes),
        )
        self.decoder = nn.Sequential(
            nn.Linear(sequence_embedding_dim, sequence_embedding_dim*2),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*2, sequence_embedding_dim*4),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*4, sequence_embedding_dim*8),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*8, sequence_embedding_dim*16),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*16, sequence_embedding_dim*32),
            nn.ReLU(),
            nn.Linear(sequence_embedding_dim*32, sequence_length//4),
            nn.ReLU(),
            nn.Linear(sequence_length//4, sequence_length//2),
            nn.ReLU(),
            nn.Linear(sequence_length//2, sequence_length),
        )

    def forward(self, x):
        embedding = self.sequence_embedding(x)
        classification = self.classifier(embedding)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction, classification

    def classify(self, z):
        return self.classifier(z)

def plot_latent_space(model, dataloader, epoch, with_labels = False, map_label_to_subtype=None, map_row_to_seqid=None):
    model.eval()
    latent_space = []
    seq_ids = []
    labels = []
    with torch.no_grad():
        for i, (x, y, _, _, _, _, idx) in tqdm(enumerate(dataloader), desc="Plotting latent space"):
            x = x.to(DEVICE)
            emb, rec, clas = model(x)
            latent_space.append(emb.squeeze(1).cpu().numpy())
            labels.append(y.cpu().numpy())
            if with_labels:
                seq_ids.extend([map_row_to_seqid[i] for i in idx.cpu().numpy()])
    latent_space = np.concatenate(latent_space)
    # seq_ids = np.concatenate(seq_ids)
    print(latent_space.shape)
    labels = np.concatenate(labels)

    plt.figure(figsize=(10, 10))
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels)
    plt.colorbar()
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)

    if with_labels:
        # for i, label in enumerate(labels):
        #     annotation = f"{map_label_to_subtype[label]}+{map_row_to_seqid[i]}"
        #     plt.annotate(annotation, (latent_space[i, 0], latent_space[i, 1]))

        for i, seq_id in enumerate(seq_ids):
            annotation = f"{seq_id}"
            plt.annotate(annotation, (latent_space[i, 0], latent_space[i, 1]))


        plt.savefig(f"./latent_space_{epoch}_labeled.png")
    else:
        plt.savefig(f"./latent_space_{epoch}.png")
    plt.close()

def classify_latent_space(model, dataloader, epoch):
    # create a grid of points in the latent space and classify them

    grid = np.mgrid[-2:2:0.005, -2:2:0.005].reshape(2, -1).T
    grid = torch.from_numpy(grid).float().to(DEVICE)

    model.eval()
    # now classify the grid
    
    with torch.no_grad():
        classification = model.classify(grid)
        _, predicted = torch.max(classification.data, 1)

    grid = grid.cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(grid[:, 0], grid[:, 1], c=predicted.cpu().numpy()) 
    plt.colorbar()

    plt.savefig(f"./latent_space_{epoch}_classified.png")






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
    model.load_state_dict(torch.load("./model.pt"))
    model.to(DEVICE)

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # create the loss functions
    reconstruction_loss = nn.MSELoss()
    classification_loss = nn.CrossEntropyLoss()
    contrastive_loss = nn.CosineEmbeddingLoss()

    # create the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1500, verbose=True)

    classify_latent_space(model, dataloader, 550)
    sys.exit()

    # train the model

    for epoch in range(450, 550):
        plot_latent_space(model, dataloader, epoch)
        # plot_latent_space(model, dataloader, epoch, with_labels=True, map_label_to_subtype=map_label_to_subtype, map_row_to_seqid=map_row_to_seqid)
        for batch_idx, (x, y, similar_x, dissimilar_x, similar_y, dissimilar_y, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}"):
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

            rec_loss = reconstruction_loss(reconstruction, x)
            class_loss = classification_loss(classification, y)

            similar_x_class_loss = classification_loss(similar_classification, similar_y)
            dissimilar_x_class_loss = classification_loss(dissimilar_classification, dissimilar_y)

            contrast_loss_sim = contrastive_loss(embedding, similar_embedding, torch.ones(embedding.shape[0]).to(DEVICE)) # we want the embeddings to be similar
            contrast_loss_dissim = contrastive_loss(embedding, dissimilar_embedding, -1 * torch.ones(embedding.shape[0]).to(DEVICE)) # we want the embeddings to be dissimilar
            
            # loss = rec_loss + class_loss + similar_x_class_loss + dissimilar_x_class_loss + contrast_loss
            loss = rec_loss * LOSS_RECON + class_loss * LOSS_CLASS + similar_x_class_loss * LOSS_CLASS + dissimilar_x_class_loss * LOSS_CLASS + contrast_loss_sim * LOSS_CONTRAST_SIM + contrast_loss_dissim * LOSS_CONTRAST_DISSIM
            # loss = class_loss * LOSS_CLASS # + similar_x_class_loss * LOSS_CLASS + dissimilar_x_class_loss * LOSS_CLASS + contrast_loss_sim * LOSS_CONTRAST_SIM + contrast_loss_dissim * LOSS_CONTRAST_DISSIM
            
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)
            # print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item()} = {rec_loss.item():.2f} * {LOSS_RECON} + {class_loss.item():.2f} * {LOSS_CLASS} + {similar_x_class_loss.item():.2f} * {LOSS_CLASS} + {dissimilar_x_class_loss.item():.2f} * {LOSS_CLASS} + {contrast_loss.item():.2f} * {LOSS_CONTRAST}")
        torch.save(model.state_dict(), f"./model_{epoch}.pt")

    # save the model
    torch.save(model.state_dict(), "./model.pt")

    # plot the latent space
    plot_latent_space(model, dataloader, epoch+1)

    # plot the latent space with labels
    plot_latent_space(model, dataloader, epoch+1, with_labels=True, map_label_to_subtype=map_label_to_subtype, map_row_to_seqid=map_row_to_seqid)

    # evaluate the model on classification accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _, _, _, _, _ in tqdm(dataloader, desc="Evaluating model"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            _, _, classification = model(x)
            _, predicted = torch.max(classification.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Accuracy of the network on the {total} sequences: {100 * correct / total}%")

    from sklearn.metrics import balanced_accuracy_score

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y, _, _, _, _, _ in tqdm(dataloader, desc="Evaluating model"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            _, _, classification = model(x)
            _, predicted = torch.max(classification.data, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(f"Balanced accuracy of the network on the {total} sequences: {balanced_accuracy_score(y_true, y_pred)}")

    # plot the latent space with classification
    classify_latent_space(model, dataloader, epoch+1)



