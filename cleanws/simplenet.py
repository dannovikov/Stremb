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
    wandb.init(project="seqvae2")

from tqdm import tqdm

TRAIN=False

DATA_DIR = r"E:\projects\hiv_deeplearning\stremb\cleanws\preproc"
TRAIN_IMG_DIR = r"E:\projects\hiv_deeplearning\stremb\cleanws\images"
TRAIN_MODEL_DIR = r"E:\projects\hiv_deeplearning\stremb\cleanws\models"
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DATA_DIR = "./data_pol"

EMBED_DIM = 2
NUM_WORKERS = 4

LOSS_RECON = 0.5
LOSS_CLASS = 2.5
LOSS_CONTRAST_SIM = 0.7#1.0
LOSS_CONTRAST_DISSIM = 1.2



class SeqDataset(Dataset):
    def __init__(self, X, y, contrasts):
        self.X = X.float()
        self.y = y.long()
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
    #train
    X_train = torch.load(f"{DATA_DIR}/train_seqs_tensor.pt")
    y_train = torch.load(f"{DATA_DIR}/train_labels_tensor.pt")
    with open(f"{DATA_DIR}/train_map_seqid_to_row.pkl", "rb") as f:
        train_map_seqid_to_row = pickle.load(f)
    with open(f"{DATA_DIR}/train_map_row_to_seqid.pkl", "rb") as f:
        train_map_row_to_seqid = pickle.load(f)
    with open(f"{DATA_DIR}/train_map_subtype_to_seqids.pkl", "rb") as f:
        train_map_subtype_to_seqids = pickle.load(f)
    with open(f"{DATA_DIR}/train_map_seqid_to_subtype.pkl", "rb") as f:
        train_map_seqid_to_subtype = pickle.load(f)
    with open(f"{DATA_DIR}/train_contrasts.pkl", "rb") as f:
        train_contrasts = pickle.load(f)
    #test
    X_test = torch.load(f"{DATA_DIR}/test_seqs_tensor.pt")
    y_test = torch.load(f"{DATA_DIR}/test_labels_tensor.pt")
    with open(f"{DATA_DIR}/test_map_seqid_to_row.pkl", "rb") as f:
        test_map_seqid_to_row = pickle.load(f)
    with open(f"{DATA_DIR}/test_map_row_to_seqid.pkl", "rb") as f:
        test_map_row_to_seqid = pickle.load(f)
    with open(f"{DATA_DIR}/test_map_subtype_to_seqids.pkl", "rb") as f:
        test_map_subtype_to_seqids = pickle.load(f)
    with open(f"{DATA_DIR}/test_map_seqid_to_subtype.pkl", "rb") as f:
        test_map_seqid_to_subtype = pickle.load(f)
    with open(f"{DATA_DIR}/test_contrasts.pkl", "rb") as f:
        test_contrasts = pickle.load(f)
    #common
    with open(f"{DATA_DIR}/map_label_to_subtype.pkl", "rb") as f:
        map_label_to_subtype = pickle.load(f)
    with open(f"{DATA_DIR}/map_subtype_to_label.pkl", "rb") as f:
        map_subtype_to_label = pickle.load(f)

    result = {
        "train": {
            "X": X_train,
            "y": y_train,
            "map_seqid_to_row": train_map_seqid_to_row,
            "map_row_to_seqid": train_map_row_to_seqid,
            "map_subtype_to_seqids": train_map_subtype_to_seqids,
            "map_seqid_to_subtype": train_map_seqid_to_subtype,
            "contrasts": train_contrasts,
        },
        "test": {
            "X": X_test,
            "y": y_test,
            "map_seqid_to_row": test_map_seqid_to_row,
            "map_row_to_seqid": test_map_row_to_seqid,
            "map_subtype_to_seqids": test_map_subtype_to_seqids,
            "map_seqid_to_subtype": test_map_seqid_to_subtype,
            "contrasts": test_contrasts,
        },
        "map": {
            "label_to_subtype": map_label_to_subtype,
            "subtype_to_label": map_subtype_to_label,
        }
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

def plot_latent_space(model, dataloader, epoch, with_labels = False, map_label_to_subtype=None, map_row_to_seqid=None, map_subtype_to_seqids=None):
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
    print(latent_space.shape)
    labels = np.concatenate(labels)

    plt.figure(figsize=(10, 10))
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=labels)
    plt.colorbar()

    if with_labels:
        subtype_label_positions = {}
        for subtype in map_subtype_to_seqids.keys():
            subtype_label_positions[subtype] = np.mean(latent_space[np.isin(seq_ids, map_subtype_to_seqids[subtype])], axis=0)
        for subtype, pos in subtype_label_positions.items():
            plt.annotate(subtype, pos, fontsize=8)

        plt.savefig(f"{TRAIN_IMG_DIR}/ls_labeled/latent_space_{epoch}_labeled.png")
    else:
        plt.savefig(f"{TRAIN_IMG_DIR}./images/ls/latent_space_{epoch}.png")
    plt.close()

def classify_latent_space(model, dataloader, epoch):
    # create a grid of points in the latent space and classify them

    grid = np.mgrid[-3:3:0.005, -3:3:0.005].reshape(2, -1).T
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

    plt.savefig(f"{TRAIN_IMG_DIR}/ls_classified/latent_space_{epoch}_classified.png")


def main():
    # Initialize data structures
    data = get_data()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]
    train_contrasts = data["train"]["contrasts"]
    test_contrasts = data["test"]["contrasts"]
    map_row_to_seqid = data["train"]["map_row_to_seqid"]
    map_label_to_subtype = data["map"]["label_to_subtype"]

    # Initialize dataloaders
    train_dataset = SeqDataset(X_train, y_train, train_contrasts)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    test_dataset = SeqDataset(X_test, y_test, test_contrasts)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # create the model
    model = simpleNet(sequence_length=X_train.shape[1], sequence_embedding_dim=EMBED_DIM, num_classes=len(map_label_to_subtype))
    # model.load_state_dict(torch.load('saved96_model_26.pt'))
    model.to(DEVICE)

    # Set up criteria, optimizer, and scheduler
    reconstruction_loss = nn.MSELoss()
    classification_loss = nn.CrossEntropyLoss()
    contrastive_loss = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    # Train the model
    for epoch in range(EPOCHS):
        plot_latent_space(model, train_dataloader, epoch, with_labels=True, map_label_to_subtype=map_label_to_subtype, map_row_to_seqid=map_row_to_seqid, map_subtype_to_seqids=data["train"]["map_subtype_to_seqids"])
        for i, (x, y, similar_x, dissimilar_x, similar_y, dissimilar_y, rowidx) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch: {epoch}'):
            model.train()
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            similar_x = similar_x.to(DEVICE)
            dissimilar_x = dissimilar_x.to(DEVICE)
            similar_y = similar_y.to(DEVICE)
            dissimilar_y = dissimilar_y.to(DEVICE)

            optimizer.zero_grad()

            embedding, reconstruction, classification = model(x)
            rec_loss = reconstruction_loss(reconstruction, x)
            class_loss = classification_loss(classification, y)


            similar_embedding, similar_reconstruction, similar_classification = model(similar_x)
            similar_x_class_loss = classification_loss(similar_classification, similar_y)
            dissimilar_embedding, dissimilar_reconstruction, dissimilar_classification = model(dissimilar_x)
            dissimilar_x_class_loss = classification_loss(dissimilar_classification, dissimilar_y)

            contrast_loss_sim = contrastive_loss(embedding, similar_embedding, torch.ones(embedding.shape[0]).to(DEVICE)) 
            contrast_loss_dissim = contrastive_loss(embedding, dissimilar_embedding, -1 * torch.ones(embedding.shape[0]).to(DEVICE))

            loss = rec_loss * LOSS_RECON + \
                class_loss * LOSS_CLASS + \
                similar_x_class_loss * LOSS_CLASS + \
                dissimilar_x_class_loss * LOSS_CLASS + \
                contrast_loss_sim * LOSS_CONTRAST_SIM + \
                contrast_loss_dissim * LOSS_CONTRAST_DISSIM
            
            loss.backward()
            wandb.log({"train_loss": loss.item()})
            optimizer.step()

        model.eval()
        with torch.no_grad():
            avg_loss = 0
            for x, y, similar_x, dissimilar_x, similar_y, dissimilar_y, rowidx in test_dataloader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                similar_x = similar_x.to(DEVICE)
                dissimilar_x = dissimilar_x.to(DEVICE)
                similar_y = similar_y.to(DEVICE)
                dissimilar_y = dissimilar_y.to(DEVICE)

                embedding, reconstruction, classification = model(x)
                rec_loss = reconstruction_loss(reconstruction, x)
                class_loss = classification_loss(classification, y)


                similar_embedding, similar_reconstruction, similar_classification = model(similar_x)
                similar_x_class_loss = classification_loss(similar_classification, similar_y)
                dissimilar_embedding, dissimilar_reconstruction, dissimilar_classification = model(dissimilar_x)
                dissimilar_x_class_loss = classification_loss(dissimilar_classification, dissimilar_y)

                contrast_loss_sim = contrastive_loss(embedding, similar_embedding, torch.ones(embedding.shape[0]).to(DEVICE)) 
                contrast_loss_dissim = contrastive_loss(embedding, dissimilar_embedding, -1 * torch.ones(embedding.shape[0]).to(DEVICE))

                loss = rec_loss * LOSS_RECON + \
                    class_loss * LOSS_CLASS + \
                    similar_x_class_loss * LOSS_CLASS + \
                    dissimilar_x_class_loss * LOSS_CLASS + \
                    contrast_loss_sim * LOSS_CONTRAST_SIM + \
                    contrast_loss_dissim * LOSS_CONTRAST_DISSIM
                avg_loss += loss.item()

            avg_loss /= len(test_dataloader)
            wandb.log({"test_loss": avg_loss})
            scheduler.step(avg_loss)

        torch.save(model.state_dict(), f"{TRAIN_MODEL_DIR}/model_{epoch}.pt")
        
        # evaluate the model on classification accuracy of the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _, _, _, _, _ in tqdm(test_dataloader, desc="Evaluating model"):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                _, _, classification = model(x)
                _, predicted = torch.max(classification.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()


        print(f"Accuracy of the network on the {total} sequences: {100 * correct / total}%")


def eval_model(model_path):
    # Initialize data structures
    data = get_data()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]
    train_contrasts = data["train"]["contrasts"]
    test_contrasts = data["test"]["contrasts"]
    map_row_to_seqid = data["train"]["map_row_to_seqid"]
    map_label_to_subtype = data["map"]["label_to_subtype"]

    # Initialize dataloaders
    train_dataset = SeqDataset(X_train, y_train, train_contrasts)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    test_dataset = SeqDataset(X_test, y_test, test_contrasts)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # create the model
    model = simpleNet(sequence_length=X_train.shape[1], sequence_embedding_dim=EMBED_DIM, num_classes=len(map_label_to_subtype))
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)

    plot_latent_space(model, train_dataloader, 16, with_labels=True, map_label_to_subtype=map_label_to_subtype, map_row_to_seqid=map_row_to_seqid, map_subtype_to_seqids=data["train"]["map_subtype_to_seqids"])

    # evaluate the model on classification accuracy of the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _, _, _, _, _ in tqdm(test_dataloader, desc="Evaluating model"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            _, _, classification = model(x)
            _, predicted = torch.max(classification.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Accuracy of the network on the {total} sequences: {100 * correct / total}%")
    classify_latent_space(model, train_dataloader, 16)



if __name__ == "__main__":
    # main()
    eval_model(r'E:\projects\hiv_deeplearning\stremb\cleanws\models\model_47.pt')





