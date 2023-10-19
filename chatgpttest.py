"""

this file will iterate over a 700GB distance file to obtain the distance between subtypes

distance file format:

source, target, distance
0, 1, 0.1
0, 2, 0.2
1, 2, 0.3

There's another file mapping integers to sequence ids as :

0, subtype1.seq_id1
1, subtype1.seq_id2
2, subtype2.seq_id3

where seq_id 1, 2, and 3 are the actual sequence names, just pointing out that they also have a subtype label in the name which you can obtain from split(".")[0]

The task is to find the average distance between subtypes, and the average distance between sequences within a subtype.

algorithm:
subtype_counts = {} #maps subtype to number of sequences in that subtype
subtype_distances[subtype1][subtype2] = float average distance between subtype1 and subtype2
for each row in the dist file:
    subtype1, subtype2, dist = parse(row)
    subtype_counts[subtype1] += 1
    subtype_counts[subtype2] += 1
    subtype_distances[subtype1][subtype2] += dist
    
for each subtype1 in subtype_counts:
    for each other subtype2 in subtype_counts:
        subtype_distances[subtype1][subtype2] /= subtype_counts[subtype1] + subtype_counts[subtype2]
"""
from tqdm import tqdm
from pread import parallel_read

distances_file = r"G:\3k_ea_training_dists.csv"
map_file = r"G:\3k_ea_training_dists.csv.map"
outfile = r"G:\subtype_dists.csv"

map_int_to_seqid = {} #maps integers to sequence ids
with open(map_file, 'r') as f:
    for line in f:
        i, seqid = line.split(',')
        map_int_to_seqid[int(i)] = seqid

def parse(line):
    """
    parses a line from the distance file
    returns the source, target, and distance
    """
    source, target, dist = line.split(',')
    source_type = map_int_to_seqid[int(source)].split('.')[0]
    target_type = map_int_to_seqid[int(target)].split('.')[0]
    return source_type, target_type, float(dist)

# read the distance file
subtype_counts = {} #maps subtype to number of sequences in that subtype
subtype_distances = {} #maps subtype1 to subtype2 to distance between them

with open(distances_file, 'r') as f:
    # for i,line in tqdm(enumerate(f), total=32394414917):
    for i,line in tqdm(enumerate(parallel_read(distances_file)), total=32394414917):
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
        subtype_counts[subtype1] += 1
        subtype_counts[subtype2] += 1
        subtype_distances[subtype1][subtype2] += dist

# normalize the distances
for subtype1 in tqdm(subtype_counts):
    for subtype2 in subtype_counts:
        subtype_distances[subtype1][subtype2] /= subtype_counts[subtype1] + subtype_counts[subtype2]

# write the distances to file
with open(outfile, 'w') as f:
    for subtype1 in subtype_counts:
        for subtype2 in subtype_counts:
            f.write(subtype1 + ',' + subtype2 + ',' + subtype_distances[subtype1][subtype2] + '\n')



