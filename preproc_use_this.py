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
    Set size = 5 similar + 5 close dissimilar + 5 random dissimilar = 15 sequences per contrast set

"""


import torch
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import pandas as pd
from tqdm import tqdm
import pickle
import random
import subprocess

TEST = False #skips contrast sets creation if True

# input parameters
# fasta_file = r"E:\projects\hiv_deeplearning\stremb\data_pol\new_noncurated\sequences_2line.fasta" 
fasta_file = r"E:\projects\hiv_deeplearning\stremb\data_pol\new_noncurated\generated\final_output.fasta" 

sequence_distance_matrix_file = r"E:\projects\hiv_deeplearning\stremb\data_pol\new_noncurated\generated\gen_distances.csv" 
metadata_file = r"E:\projects\hiv_deeplearning\stremb\data_pol\new_noncurated\metadata.csv"
output_dir = r"E:\projects\hiv_deeplearning\stremb\data_pol\new_noncurated\preproc"



# Global variables
SEQ_LEN = 2950
RAND_SEED = 42
random.seed(RAND_SEED)
meta = pd.read_csv(metadata_file, encoding="ISO-8859-1")
# we only use meta to map sequence ids to subtypes, so let's just read that map as a dict, where the key is the sequence id and the value is the subtype
get_subtype_dict = {seqid: subtype for seqid, subtype in zip(meta["Accession"].values, meta["Subtype"].values)}

def get_subtype(seqid):
    if "spawned" in seqid:
        return get_subtype_dict[seqid.split("_")[0]]
    return get_subtype_dict[seqid]


def main():
    print("Reading fasta file...")
    seqs_dictionary = read_fasta(fasta_file)
    print("Reading distance matrix file...")
    # subtype_distance_matrix = get_subtype_distance_matrix_from_sequence_matrix(sequence_distance_matrix_file)
    subtype_distance_matrix = get_subtype_distance_matrix_from_sequences(seqs_dictionary)
    print("Splitting into train and test sets...")
    train_seqs, test_seqs = train_test_split(seqs_dictionary)

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

    train_seqs_tensor = create_integer_encoding(train_seqs)
    save((train_seqs_tensor, "train_seqs_tensor"))
    test_seqs_tensor = create_integer_encoding(test_seqs)
    save((test_seqs_tensor, "test_seqs_tensor"))

    # train_seqs_tensor = load("train_seqs_tensor.pt")
    # test_seqs_tensor = load("test_seqs_tensor.pt")

    train_labels_tensor = create_labels_tensor(train_seqs, map_subtype_to_label)
    test_labels_tensor = create_labels_tensor(test_seqs, map_subtype_to_label)

    if not TEST:
        train_contrasts = create_contrast_sets(train_seqs, subtype_distance_matrix, train_map_subtype_to_seqids, train_map_row_to_seqid, train_map_seqid_to_row)
        test_contrasts = create_contrast_sets(test_seqs, subtype_distance_matrix, test_map_subtype_to_seqids, test_map_row_to_seqid, test_map_seqid_to_row)
    else:
        train_contrasts = {}
        test_contrasts = {}

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
        (map_label_to_subtype, "map_label_to_subtype"),
        (map_subtype_to_label, "map_subtype_to_label"),
    )


def read_fasta(fasta_file):
    # global SEQ_LEN

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

    # SEQ_LEN = max_len
    return seqs


def get_consensus(seqs):
    # seqs is a list of sequences
    # returns the consensus sequence
    consensus = ""
    for i in range(SEQ_LEN):
        counts = {}
        for seq in seqs:
            if len(seq) <= i:
                continue
            if seq[i] == "-":
                continue
            if seq[i] not in counts:
                counts[seq[i]] = 0
            counts[seq[i]] += 1
        if counts == {}:
            consensus += "-"
        else:
            consensus += max(counts, key=counts.get)
    return consensus
            

def get_subtype_distance_matrix_from_sequences(seqs_dict):
    # for each subtype, get the consensus sequence
    #then write all consensus sequences to a fasta file
    # then run java -jar seqruler.jar -i <fasta file> -o <output file> -a resolve -f 0.05 
    # then read the output file and create a distance matrix from it

    subtype_consensus_seqs = {}
    for seqid, seq in seqs_dict.items():
        subtype = get_subtype(seqid)
        if subtype not in subtype_consensus_seqs:
            subtype_consensus_seqs[subtype] = []
        subtype_consensus_seqs[subtype].append(seq)


    for subtype, seqs in tqdm(subtype_consensus_seqs.items()):
        subtype_consensus_seqs[subtype] = get_consensus(seqs)


    with open("consensus_seqs.fasta", "w") as f:
        for subtype, seq in subtype_consensus_seqs.items():
            f.write(f">{subtype}\n{seq}\n")

    subprocess.run(["java", "-jar", "seqruler.jar", "-i", "consensus_seqs.fasta", "-o", "consensus_distances.csv", "-a", "resolve", "-f", "0.05"])

    subtype_distances = {}
    with open("consensus_distances.csv", "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            subtype1, subtype2, dist = line.split(',')
            if subtype1 not in subtype_distances:
                subtype_distances[subtype1] = {}
            if subtype2 not in subtype_distances:
                subtype_distances[subtype2] = {}
            subtype_distances[subtype1][subtype2] = float(dist)
            subtype_distances[subtype2][subtype1] = float(dist)

    return subtype_distances
    

def get_subtype_distance_matrix_from_sequence_matrix(distance_matrix_file):
    subtype_counts = {}
    subtype_distances = {}

    def parse(line):
        source, target, dist = line.split(',')
        source_type = get_subtype(source)
        target_type = get_subtype(target)
        return source_type, target_type, float(dist)

    with open(distance_matrix_file, "r") as f:
        for i, line in enumerate(tqdm(f, total = 18632461)):
            if i == 0:
                continue
            subtype1, subtype2, dist = parse(line)

            if subtype1 not in subtype_counts:
                subtype_counts[subtype1] = 0

            if subtype2 not in subtype_counts:
                subtype_counts[subtype2] = 0

            if subtype1 not in subtype_distances:
                subtype_distances[subtype1] = {}

            if subtype2 not in subtype_distances:
                subtype_distances[subtype2] = {}

            if subtype2 not in subtype_distances[subtype1]:
                subtype_distances[subtype1][subtype2] = 0

            if subtype1 not in subtype_distances[subtype2]:
                subtype_distances[subtype2][subtype1] = 0

            subtype_counts[subtype1] += 1
            subtype_counts[subtype2] += 1
            subtype_distances[subtype1][subtype2] += dist
            subtype_distances[subtype2][subtype1] += dist

    for subtype1 in subtype_counts:
        for subtype2 in subtype_counts:
            if subtype1 == subtype2:
                subtype_distances[subtype1][subtype2] = 0
            else:
                subtype_distances[subtype1][subtype2] /= ((subtype_counts[subtype1] + subtype_counts[subtype2]) * 2)

    return subtype_distances


def train_test_split(seqs_dict):
    seqs = list(seqs_dict.keys())
    random.shuffle(seqs)
    # train_seqs, test_seqs = sklearn_train_test_split(seqs, test_size=0.2, random_state=RAND_SEED)
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
        subtype = get_subtype(id)
        labels[row] = map_subtype_to_label[subtype]
    return labels


def create_contrast_sets(seqs_dict, subtype_distance_matrix, map_subtype_to_seqids, map_row_to_seqid, map_seqid_to_row):
    contrasts = {}
    for seq_id, sequence in tqdm(seqs_dict.items(), desc="Creating Contrast Sets", total=len(seqs_dict.keys())):
        row_num = map_seqid_to_row[seq_id]
        subtype = get_subtype(seq_id)

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
    subtype_sequences = map_subtype_to_seqids[subtype].copy()
    subtype_sequences.remove(seq_id)
    try:
        similar_sequences = random.sample(subtype_sequences, 5)
    except ValueError as e:
        print(f"Subtype {subtype} has only {len(subtype_sequences)} sequences")
        similar_sequences = subtype_sequences
        similar_sequences.append(seq_id)
        while len(similar_sequences) < 5:
            similar_sequences.append(random.choice(subtype_sequences))
    return similar_sequences


def create_dissimilar_set(subtype_distance_matrix, map_subtype_to_seqids, subtype, subtype_distances):
    # Dissimilar examples: 5 random other subtypes taking 1 sequence from each
    dissimilar_sequences = []
    other_subtypes = list(subtype_distance_matrix.keys())
    other_subtypes.remove(subtype)
    random.shuffle(other_subtypes)
    for other_subtype in other_subtypes:

        try: other_subtype_sequences = map_subtype_to_seqids[other_subtype]
        except KeyError: continue

        dissimilar_sequences.append(random.choice(other_subtype_sequences))
        if len(dissimilar_sequences) >= 5:
            break

    # Adding to Dissimilar: 5 examples from the 5 closest subtypes
    closest_subtypes = sorted(subtype_distances, key=subtype_distances.get)
    for other_subtype in closest_subtypes:

        try: other_subtype_sequences = map_subtype_to_seqids[other_subtype]
        except KeyError: continue  

        dissimilar_sequences.append(random.choice(other_subtype_sequences))
        if len(dissimilar_sequences) >= 10: 
            break
    return dissimilar_sequences


def save(*args):
    # function to save an arbitrary length list of parameters by checking their type, if torch tensor use torch save else use pickle
    for arg, name in args:
        if type(arg) == torch.Tensor:
            torch.save(arg, f"{output_dir}/{name}.pt")
        else:
            with open(f"{output_dir}/{name}.pkl", "wb") as f:
                pickle.dump(arg, f)

def load(name):
    if name.endswith(".pt"):
        return torch.load(f"{output_dir}/{name}")
    else:
        with open(f"{output_dir}/{name}", "rb") as f:
            return pickle.load(f)

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
        # subtype = id.split(".")[0]
        subtype = get_subtype(id)
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
        subtype = get_subtype(id)
        test_map_seqid_to_subtype[id] = subtype
        if subtype not in test_map_subtype_to_seqids:
            test_map_subtype_to_seqids[subtype] = []
        test_map_subtype_to_seqids[subtype].append(id)
        test_subtypes.append(subtype)

    map_label_to_subtype = {}
    map_subtype_to_label = {}

    # for i, subtype in enumerate(set(train_subtypes + test_subtypes)):
    #     map_label_to_subtype[i] = subtype
    #     map_subtype_to_label[subtype] = i

    #i want to use the metadata file to ensure all subtypes are included in the label map
    for i, subtype in enumerate(meta["Subtype"].unique()):
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

if __name__ == "__main__":
    main()