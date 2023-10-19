# from Bio import SeqIO
import numpy as np
import pickle
import torch
import pandas as pd
from tqdm import tqdm

import multiprocessing as mp
from multiprocessing import Value
from ctypes import c_int





import sys

CORES = 16

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

if __name__ == "__main__":
    fasta_file = "data_pol/sequences.fasta"
    output_dir = "data_pol"

    seqs, max_len = read_fasta(fasta_file)

    # Create integer encoding tensor
    N = len(seqs.keys())
    print(f"Number of sequences: {N}")
    L = max_len
    X = torch.zeros((N, L), dtype=torch.long)

    # X = torch.load(f"{output_dir}/X.pt")

    map_row_to_seqid = {} #maps rows in the matrix to sequence ids
    for row, (id,seq) in tqdm(enumerate(seqs.items()), desc="Integer Encoding", total=len(seqs.keys())):
        map_row_to_seqid[row] = id
        for col, nucl in enumerate(seq):
            nucl = nucl.upper()
            if nucl not in ["A", "C", "G", "T"]:
                X[row][col] = 0
            elif nucl == "A":
                X[row][col] = 1
            elif nucl == "C":
                X[row][col] = 2
            elif nucl == "G":
                X[row][col] = 3
            elif nucl == "T":
                X[row][col] = 4

    # maps back from sequence ids to rows in the matrix
    map_seqid_to_row = {v: k for k, v in map_row_to_seqid.items()}

    # create the labels tensor.
    # assumes that the sequence ids are of the form "subtype.id"
    labels = {}
    for id in seqs.keys():
        subtype = id.split(".")[0]
        labels[id] = subtype

    # create pytorch integer tensor of the labels
    y = torch.zeros(N, dtype=torch.long)
    subtypes = list(set(labels.values()))
    for i, key in enumerate(seqs.keys()):
        y[i] = subtypes.index(labels[key])

    # create a dict mapping label ids as ints to subtype names as strings
    map_label_to_subtype = {i: subtypes[i] for i in range(len(subtypes))}

#Create a one-hot encoding of the integer encoding
# X_one_hot = torch.nn.functional.one_hot(X, num_classes=5).float()



# Create contrast sets for each sequence 
# giving the indices of the top 10 most similar sequences, and the top 10 most dissimilar sequences.

# contrasts = {}
# dists = pd.read_csv('data_pol/dists.csv') #columns are Source, Target, Distance where Source and Target are sequence ***names***
# for i in tqdm(range(N), desc="Creating Contrast Sets"): # N is the number of sequences
#     if i < 2340: continue
#     print(f"Creating contrast set for sequence {i}")
#     contrasts[i] = {"similar": [], "dissimilar": []} 

#     s1 = map_row_to_seqid[i] # get the name of sequence i
#     s1_dists = dists[dists["Source"] == s1] # get all rows where the source is sequence i
#     s1_dists = s1_dists.sort_values(by="Distance") # sort the rows in ascending order by distance
#     print("got sorted distances", len(s1_dists))
#     #Finding similar sequences
#     j = 0 #iterator over sorted distances to s1. 
#     while len(contrasts[i]["similar"]) < 10 and j < len(s1_dists):
#         print(f"{j}")
#         s2 = s1_dists.iloc[j]["Target"] 
#         if s1 == s2:
#             j += 1
#             continue
#         if s1_dists.iloc[j]["Distance"] == 0:
#             j += 1
#             continue
#         if labels[s1] == labels[s2]:
#             contrasts[i]["similar"].append(map_seqid_to_row[s2])
#         j += 1

#     #Finding dissimilar sequences
#     j = len(s1_dists) - 1
#     while len(contrasts[i]["dissimilar"]) < 10 and j > 0:
#         print(f"{j}")
#         s2 = s1_dists.iloc[j]["Target"]
#         if s1 == s2:
#             continue
#         if s1_dists.iloc[j]["Distance"] == 0:
#             continue
#         if labels[s1] != labels[s2]:
#             contrasts[i]["dissimilar"].append(map_seqid_to_row[s2])
        
#         j -= 1



# contrasts = {}
# dists = pd.read_csv('data_pol/dists.csv') 

#dists gives the lower triangle of the distance matrix
#so not all sequences apepar in the source or target columns
#but we want a symmetric matrix so we need to add the upper triangle
#we can do this by swapping the source and target columns and then appending

# dists_swapped = dists.rename(columns={"Source": "Target", "Target": "Source"})
# dists = dists.append(dists_swapped, ignore_index=True)

#We're gonna manually read the file to create this distance matrix




# #same as above but parallel architecture 
# def create_contrast_set(i, dists, map_row_to_seqid, map_seqid_to_row, labels):
#     contrast = {"similar": [], "dissimilar": []}
#     s1 = map_row_to_seqid[i] # get the name of sequence i
#     use_column = "Source"
#     s1_dists = dists[dists["Source"] == s1] # get all rows where the source is sequence i
#     print(f"got {len(s1_dists)} distances")
#     s1_dists = s1_dists.sort_values(by="Distance") # sort the rows in ascending order by distance

#     #Finding similar sequences
#     j = 0 #iterator over sorted distances to s1.

#     while len(contrast["similar"]) < 10 and j < len(s1_dists):
#         s2 = s1_dists.iloc[j]["Target"]
#         if s1 == s2:
#             j += 1
#             continue
#         if s1_dists.iloc[j]["Distance"] == 0:
#             j += 1
#             continue
#         if labels[s1] == labels[s2]:
#             contrast["similar"].append(map_seqid_to_row[s2])

#         j += 1

#     #Finding dissimilar sequences
#     j = len(s1_dists) - 1

#     while len(contrast["dissimilar"]) < 10 and j > 0:
#         s2 = s1_dists.iloc[j]["Target"]
#         if s1 == s2:
#             j -= 1
#             continue
#         if s1_dists.iloc[j]["Distance"] == 0:
#             j -= 1
#             continue
#         if labels[s1] != labels[s2]:
#             contrast["dissimilar"].append(map_seqid_to_row[s2])

#         j -= 1

#     with done_counter.get_lock():
#         done_counter.value += 1
#         print(f"Done: {done_counter.value}/{N}, {s1}, {s2}\r", end="")
#     return contrast



def create_contrast_set(i, dists, map_row_to_seqid, map_seqid_to_row, labels):
    contrast = {'similar': [], 'dissimilar': []}
    s1 = map_row_to_seqid[i] # get the name of sequence i
    # now lets access dists as a matrix
    s1_dists = dists.loc[s1]
    s1_dists = s1_dists.sort_values() # sort the rows in ascending order by distance

    #Finding similar sequences
    j = 0 #iterator over sorted distances to s1.

    while len(contrast['similar']) < 10 and j < len(s1_dists):
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
        contrast['similar'].append(map_seqid_to_row[s1_dists.index[0]])

    #Finding dissimilar sequences
    j = len(s1_dists) - 1

    while len(contrast['dissimilar']) < 10 and j > 0:
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

    print(f"Done: {i}, {contrast}")
    return contrast



if __name__ == '__main__':

    seq_ids = list(seqs.keys())

    #i want a symmetric matrix who's indices and columns are the sequence ids

    # dists = pd.DataFrame(index=seq_ids, columns=seq_ids)
    # total = int(N * (N-1) / 2)

    # with open("data_pol/dists.csv", "r") as f:
    #     for i, line in tqdm(enumerate(f), desc="Reading distance matrix", total=total):
    #         if i == 0: continue
    #         s,t,d = line.strip().split(",")
    #         dists.loc[s,t] = d
    #         dists.loc[t,s] = d


    # pickle.dump(dists, open("data_pol/dists.pkl", "wb"))
    dists = pickle.load(open("data_pol/dists.pkl", "rb"))

    contrasts = {}
    pool = mp.Pool(CORES)
    tasks = []


    for i in range(N): # N is the number of sequences
        contrasts[i] = {"similar": [], "dissimilar": []}
        tasks.append((i, dists, map_row_to_seqid, map_seqid_to_row, labels))


    results = pool.starmap(create_contrast_set, tasks) 
    for i, result in enumerate(results):
        contrasts[i] = result
    # contrasts = pickle.load(open("data_pol/contrasts.pkl", "rb"))




    # print(contrasts)
    # print(X, X.shape)
    # # print(X_one_hot)
    # print(y)



    # torch.save(X, f"{output_dir}/X.pt")
    # torch.save(y, f"{output_dir}/y.pt")
    # torch.save(X_one_hot, f"{output_dir}/X_one_hot.pt")

    with open(f"{output_dir}/map_label_to_subtype.pkl", "wb") as f:
        pickle.dump(map_label_to_subtype, f)

    with open(f"{output_dir}/map_row_to_seqid.pkl", "wb") as f:
        pickle.dump(map_row_to_seqid, f)

    with open(f"{output_dir}/contrasts.pkl", "wb") as f:
        pickle.dump(contrasts, f)
