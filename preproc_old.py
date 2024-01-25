"""
This file preprocesses a fasta file for use in pytorch.
It's meant to be run on the train and the test sets seperately, i.e., it doesn't split them internally. 

Input:

"""


import numpy as np
import pickle
import torch
import pandas as pd
from tqdm import tqdm

import multiprocessing as mp
from multiprocessing import Value
from ctypes import c_int
from subprocess import check_output

CORES = 10

#contrast set hyperparameters
SIMILAR_SIZE = 5
DISSIMILAR_SIZE = 20

SEQUENCE_LENGTH = 2909

fasta_file = "data_pol/new_noncurated/sequences_2line.fasta"
metadata_file = "data_pol/new_noncurated/metadata.csv"
output_dir = "data_pol/new_noncurated/preproc"
dists_csv = f"data_pol/new_noncurated/distances.csv"


def read_fasta(fasta_file):
    seqs = {}
    max_len = 0

    with open(fasta_file, "r") as f:
        last_id = ""
        for i, line in enumerate(f):
            if i % 2 == 0:
                last_id = line.strip()[1:]
                seqs[last_id] = ""
            else:
                seqs[last_id] = line.strip()
                if len(line.strip()) > max_len:
                    max_len = len(line.strip())
    return seqs,max_len


def create_contrast_set(i, dists, map_row_to_seqid, map_seqid_to_row, labels):
    """
    i: row index of the sequence in the sequence matrix
    dists: pandas dataframe distance matrix, with sequence ids as indices and columns
    map_row_to_seqid: dict mapping row indices to sequence ids
    map_seqid_to_row: dict mapping sequence ids to row indices
    labels: dict mapping sequence ids to subtypes
    """
    contrast = {'similar': [], 'dissimilar': []}
    s1 = map_row_to_seqid[i] # get the sequence id of row i
    s1_dists = dists.loc[s1] # get the distances of s1 to all other sequences
    s1_dists = s1_dists.sort_values() 

    #Finding similar sequences
    j = 0 #iterator over sorted distances to s1.

    while len(contrast['similar']) < SIMILAR_SIZE and j < len(s1_dists):
        s2 = s1_dists.index[j]
        if s1 == s2:
            j += 1
            continue
        if s1_dists[j] == 0:
            j += 1
            continue
        if labels[s1] == labels[s2]:
            contrast['similar'].append(map_seqid_to_row[s2])

        j += 1

    if len(contrast['similar']) == 0:
        contrast['similar'].append(map_seqid_to_row[s1_dists.index[1]])

    #Finding dissimilar sequences
    j = len(s1_dists) - 1

    while len(contrast['dissimilar']) < 5 and j > 0:
        s2 = s1_dists.index[j]
        if s1 == s2:
            j -= 1
            continue
        if s1_dists[j] == 0:
            j -= 1
            continue
        if labels[s1] != labels[s2]:
            contrast['dissimilar'].append(map_seqid_to_row[s2])

        j -= 1

    j = 0
    while len(contrast['dissimilar']) < DISSIMILAR_SIZE and j < len(s1_dists):
        s2 = s1_dists.index[j]
        if s1 == s2:
            j += 1
            continue
        if labels[s1] != labels[s2]:
            assert s1_dists[j] != 0, f"Distance between {s1} and {s2} is 0, but they are different subtypes {labels[s1]} and {labels[s2]}"
            contrast['dissimilar'].append(map_seqid_to_row[s2])
        j += 1

    # print(f"Done: {i}, {contrast}")
    return contrast


if __name__ == "__main__":

    seqs, max_len = read_fasta(fasta_file)

    # Create integer encoding tensor
    N = len(seqs.keys())
    # L = max_len
    X = torch.zeros((N, L), dtype=torch.long)
    map_row_to_seqid = {} #maps rows in the matrix to sequence ids
    for row, (id,seq) in tqdm(enumerate(seqs.items()), desc="Integer Encoding", total=len(seqs.keys())):
        map_row_to_seqid[row] = id
        # for col, nucl in enumerate(seq):
        #     nucl = nucl.upper()
        #     if nucl not in ["A", "C", "G", "T"]:
        #         X[row][col] = 0
        #     elif nucl == "A":
        #         X[row][col] = 1
        #     elif nucl == "C":
        #         X[row][col] = 2
        #     elif nucl == "G":
        #         X[row][col] = 3
        #     elif nucl == "T":
        #         X[row][col] = 4
    torch.save(X, f"{output_dir}/X.pt")
    X = torch.load(f"{output_dir}/X.pt")

    # maps back from sequence ids to rows in the matrix
    map_seqid_to_row = {v: k for k, v in map_row_to_seqid.items()}

    # create the labels tensor.
    # match the seq id using "Accession" column, and get the subtype from the "Rhee Subtype" column
    metadata = pd.read_csv(metadata_file,  encoding = "ISO-8859-1")
    labels = {}
    # for seq_id in seqs.keys():
    #     try:
    #         labels[seq_id] = metadata[metadata["Accession"] == seq_id]["Rhee Subtype"].values[0]
    #     except IndexError as e:
    #         print(f"Could not find subtype for {seq_id}")
    #         raise e
    for i, row in metadata.iterrows():
        labels[row["Accession"]] = row["Subtype"]
        


    # create pytorch integer tensor of the labels
    y = torch.zeros(N, dtype=torch.long)
    subtypes = list(set(labels.values()))
    for i, key in enumerate(seqs.keys()):
        y[i] = subtypes.index(labels[key])

    # create a dict mapping label ids as ints to subtype names as strings
    map_label_to_subtype = {i: subtypes[i] for i in range(len(subtypes))}
    print(f"Number of sequences: {N}")


    # plot subtype distribution
    import seaborn as sns
    import matplotlib.pyplot as plt
    #set wide figure size
    plt.figure(figsize=(10,5))
    ax = sns.countplot(x=[map_label_to_subtype[i.item()] for i in y], order=pd.value_counts([map_label_to_subtype[i.item()] for i in y]).index)
    #rotate x labels, set font size, and annotate bars with counts rotated 90 degrees, sitting a constant distance above the bar
    for item in ax.get_xticklabels():
        item.set_rotation(90)
        item.set_fontsize(8)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01), rotation=40, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/10k_ea_subtype_distribution.png")


    seq_ids = list(seqs.keys())


    # create distance matrix from tn93 output csv
    # dists = pickle.load(open("data_pol/dists.pkl", "rb"))
    # dists = np.zeros((N, N))
    # with open(dists_csv, "r") as f:
    #     for i, line in tqdm(enumerate(f), desc="Reading distance matrix", total=18632461):
    #         if i == 0: continue
    #         s,t,d = line.strip().split(",")
    #         dists[seq_ids.index(s)][seq_ids.index(t)] = float(d)
    #         dists[seq_ids.index(t)][seq_ids.index(s)] = float(d)
    # dists = pd.DataFrame(dists, index=seq_ids, columns=seq_ids)
    # pickle.dump(dists, open("data_pol/generated_dists.pkl", "wb"))

    dists = pickle.load(open("data_pol/generated_dists.pkl", "rb"))
    
    contrasts = {}
    tasks = []
    for i in range(N): # N is the number of sequences
        contrasts[i] = {"similar": [], "dissimilar": []}
        tasks.append((i, dists, map_row_to_seqid, map_seqid_to_row, labels))

    with mp.Pool(CORES) as pool:
        results = pool.starmap(create_contrast_set, tasks) 
        for i, result in enumerate(results):
            contrasts[i] = result
    # contrasts = pickle.load(open("data_pol/contrasts.pkl", "rb"))


    print(X, X.shape)
    print(y, y.shape)


    torch.save(X, f"{output_dir}/X.pt")
    torch.save(y, f"{output_dir}/y.pt")

    with open(f"{output_dir}/map_label_to_subtype.pkl", "wb") as f:
        pickle.dump(map_label_to_subtype, f)

    with open(f"{output_dir}/map_row_to_seqid.pkl", "wb") as f:
        pickle.dump(map_row_to_seqid, f)

    with open(f"{output_dir}/contrasts.pkl", "wb") as f:
        pickle.dump(contrasts, f)
