"""
Proprocesses molecular data for use in pytorch contrastive learning model.

INPUT: 
    a. Fasta file with all sequences. Sequence name is of format >{subtype}.{sequence_id}
    b. Subtype distance matrix (csv file)

0. Perform train test split
1. Create integer encoding tensor 
2. Create the following dictionaries for both train and test sets:
    a. map_seqid_to_row: maps sequence id to row number in integer encoding tensor
    b. map_row_to_seqid: maps row number in integer encoding tensor to sequence id
    c. map_subtype_to_seqids: maps subtype to list of sequence ids
    d. map_seqid_to_subtype: maps sequence id to subtype
    e. map_seqid_to_sequence: maps sequence id to sequence
3. Create contrast set for each sequence
    1. Find similar sequences (same subtype)
    2. Find dissimilar sequences (different subtype)
        include both the closest and farther subtypes which are not from the same subtype
    Set size = 5 similar + 5 close dissimilar + 5 far dissimilar = 15 sequences per contrast set

"""


import torch
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from tqdm import tqdm
import pickle
import random


# input parameters
fasta_file = ""
distance_matrix_file = ""
output_dir = ""

# Global variables
SEQ_LEN = 0
RAND_SEED = 42
random.seed(RAND_SEED)


def main():
    seqs_dictionary = read_fasta(fasta_file)
    subtype_distance_matrix = read_distance_matrix(distance_matrix_file)
    train_seqs, test_seqs = train_test_split(seqs_dictionary)

    train_seqs_tensor = create_integer_encoding(train_seqs)
    test_seqs_tensor = create_integer_encoding(test_seqs)

    train_labels_tensor = create_labels_tensor(train_seqs)
    test_labels_tensor = create_labels_tensor(test_seqs)

    (
        train_map_seqid_to_row,
        train_map_row_to_seqid,
        train_map_subtype_to_seqids,
        train_map_seqid_to_subtype,
        test_map_seqid_to_row,
        test_map_row_to_seqid,
        test_map_subtype_to_seqids,
        test_map_seqid_to_subtype,
        map_label_to_subtype,
        map_subtype_to_label,
    ) = create_maps(train_seqs, test_seqs)

    train_contrasts = create_contrast_sets(train_seqs, subtype_distance_matrix, train_map_subtype_to_seqids)
    test_contrasts = create_contrast_sets(test_seqs, subtype_distance_matrix, test_map_subtype_to_seqids)

    save(
        (train_seqs_tensor, "train_seqs_tensor"),
        (test_seqs_tensor, "test_seqs_tensor"),
        (train_labels_tensor, "train_labels_tensor"),
        (test_labels_tensor, "test_labels_tensor"),
        (train_map_seqid_to_row, "train_map_seqid_to_row"),
        (train_map_row_to_seqid, "train_map_row_to_seqid"),
        (train_map_subtype_to_seqids, "train_map_subtype_to_seqids"),
        (train_map_seqid_to_subtype, "train_map_seqid_to_subtype"),
        (test_map_seqid_to_row, "test_map_seqid_to_row"),
        (test_map_row_to_seqid, "test_map_row_to_seqid"),
        (test_map_subtype_to_seqids, "test_map_subtype_to_seqids"),
        (test_map_seqid_to_subtype, "test_map_seqid_to_subtype"),
        (train_contrasts, "train_contrasts"),
        (test_contrasts, "test_contrasts"),
    )


def read_fasta(fasta_file):
    global SEQ_LEN

    seqs = {}
    max_len = 0

    with open(fasta_file, "r") as f:
        last_id = ""
        for i, line in enumerate(f):
            if i % 2 == 0:
                last_id = line.strip()[1:]  # removing ">"
                seqs[last_id] = ""
            else:
                seqs[last_id] = line.strip()
                if len(line.strip()) > max_len:
                    max_len = len(line.strip())

    SEQ_LEN = max_len
    return seqs


def read_distance_matrix(distance_matrix_file):
    distance_matrix = {}
    with open(distance_matrix_file, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            s, t, d = line.strip().split(",")
            if s not in distance_matrix:
                distance_matrix[s] = {}
            distance_matrix[s][t] = float(d)
    return distance_matrix


def train_test_split(seqs_dict):
    seqs = list(seqs_dict.keys())
    train_seqs, test_seqs = sklearn_train_test_split(seqs, test_size=0.2, random_state=RAND_SEED)
    train_seqs_dict = {seq: seqs_dict[seq] for seq in train_seqs}
    test_seqs_dict = {seq: seqs_dict[seq] for seq in test_seqs}
    return train_seqs_dict, test_seqs_dict


def create_integer_encoding(seqs_dict):
    X = torch.zeros(len(seqs_dict), SEQ_LEN, dtype=torch.float)
    for row, (id, seq) in tqdm(enumerate(seqs_dict.items()), desc="Integer Encoding", total=len(seqs_dict.keys())):
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
    return X


def create_labels_tensor(seqs_dict, map_subtype_to_label):
    # create a vector of length N giving the integer label for each sequence
    labels = torch.zeros(len(seqs_dict))
    for row, (id, seq) in tqdm(enumerate(seqs_dict.items()), desc="Creating Labels Tensor", total=len(seqs_dict.keys())):
        subtype = id.split(".")[0]
        labels[row] = map_subtype_to_label[subtype]
    return labels


def create_contrast_sets(seqs_dict, subtype_distance_matrix, map_subtype_to_seqids, map_row_to_seqid, map_seqid_to_row):
    contrasts = {}
    for seq_id, sequence in seqs_dict.items():
        row_num = map_seqid_to_row[seq_id]
        subtype = seq_id.split(".")[0]

        contrast = {"similar": [], "dissimilar": []}
        subtype_distances = subtype_distance_matrix[subtype]
        similar_sequences = create_similar_set(map_subtype_to_seqids, seq_id, subtype)
        dissimilar_sequences = create_dissimilar_set(subtype_distance_matrix, map_subtype_to_seqids, subtype, subtype_distances)

        contrast["similar"] = [map_seqid_to_row[seq] for seq in similar_sequences]
        contrast["dissimilar"] = [map_seqid_to_row[seq] for seq in dissimilar_sequences]

        contrasts[row_num] = contrast

    return contrasts


def create_similar_set(map_subtype_to_seqids, seq_id, subtype):
    # Similar examples: 5 random sequences from the same subtype
    subtype_sequences = map_subtype_to_seqids[subtype]
    subtype_sequences.remove(seq_id)
    similar_sequences = random.sample(subtype_sequences, 5)
    return similar_sequences


def create_dissimilar_set(subtype_distance_matrix, map_subtype_to_seqids, subtype, subtype_distances):
    # Dissimilar examples: 5 random other subtypes taking 1 sequence from each
    dissimilar_sequences = []
    other_subtypes = list(subtype_distance_matrix.keys())
    other_subtypes.remove(subtype)
    for other_subtype in random.sample(other_subtypes, 5):
        other_subtype_sequences = map_subtype_to_seqids[other_subtype]
        dissimilar_sequences.append(random.choice(other_subtype_sequences))

    # Adding to Dissimilar: 5 examples from the 5 closest subtypes
    closest_subtypes = sorted(subtype_distances, key=subtype_distances.get)[:5]
    for other_subtype in closest_subtypes:
        other_subtype_sequences = map_subtype_to_seqids[other_subtype]
        dissimilar_sequences.append(random.choice(other_subtype_sequences))
    return dissimilar_sequences


# function to save an arbitrary length list of parameters by checking their type, if torch tensor use torch save else use pickle
def save(*args):
    for arg, name in args:
        if type(arg) == torch.Tensor:
            torch.save(arg, f"{output_dir}/{name}.pt")
        else:
            with open(f"{output_dir}/{name}.pkl", "wb") as f:
                pickle.dump(arg, f)


def create_maps(train_seqs, test_seqs):
    """
    creating these maps:
        1. train_map_seqid_to_row,
        2. train_map_row_to_seqid,
        3. train_map_subtype_to_seqids,
        4. train_map_seqid_to_subtype,
        5. test_map_seqid_to_row,
        6. test_map_row_to_seqid,
        7. test_map_subtype_to_seqids,
        8. test_map_seqid_to_subtype,
        9. map_label_to_subtype,
        10. map_subtype_to_label,
    """

    train_map_seqid_to_row = {}
    train_map_row_to_seqid = {}
    train_map_subtype_to_seqids = {}
    train_map_seqid_to_subtype = {}
    train_subtypes = []
    for row, (id, seq) in tqdm(enumerate(train_seqs.items()), desc="Creating Train Maps", total=len(train_seqs.keys())):
        train_map_seqid_to_row[id] = row
        train_map_row_to_seqid[row] = id
        subtype = id.split(".")[0]
        train_map_seqid_to_subtype[id] = subtype
        if subtype not in train_map_subtype_to_seqids:
            train_map_subtype_to_seqids[subtype] = []
        train_map_subtype_to_seqids[subtype].append(id)
        train_subtypes.append(subtype)

    test_map_seqid_to_row = {}
    test_map_row_to_seqid = {}
    test_map_subtype_to_seqids = {}
    test_map_seqid_to_subtype = {}
    test_subtypes = []
    for row, (id, seq) in tqdm(enumerate(test_seqs.items()), desc="Creating Test Maps", total=len(test_seqs.keys())):
        test_map_seqid_to_row[id] = row
        test_map_row_to_seqid[row] = id
        subtype = id.split(".")[0]
        test_map_seqid_to_subtype[id] = subtype
        if subtype not in test_map_subtype_to_seqids:
            test_map_subtype_to_seqids[subtype] = []
        test_map_subtype_to_seqids[subtype].append(id)
        test_subtypes.append(subtype)

    map_label_to_subtype = {}
    map_subtype_to_label = {}

    for i, subtype in enumerate(set(train_subtypes + test_subtypes)):
        map_label_to_subtype[i] = subtype
        map_subtype_to_label[subtype] = i

    return (
        train_map_seqid_to_row,
        train_map_row_to_seqid,
        train_map_subtype_to_seqids,
        train_map_seqid_to_subtype,
        test_map_seqid_to_row,
        test_map_row_to_seqid,
        test_map_subtype_to_seqids,
        test_map_seqid_to_subtype,
        map_label_to_subtype,
        map_subtype_to_label,
    )
