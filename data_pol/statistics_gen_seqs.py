

"""infile format:

//subtype
>ref
seq
>spawned1
seq
>spawned2
seq
//anothersubtype
>ref
seq
>spawned1
seq
>spawned2
seq
.... and so on
"""

"""
 about the generated sequences in each subtype

Count nucleotide distributions, percentage of ambiguous nucleotides, 
and average tn93 distance to reference sequence of that subtype, 
and average tn93 distance to refence sequence of other subtypes. 
 """

from tn93 import tn93
from Bio import SeqIO
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

tn93 = tn93.TN93().tn93_distance

subtypes = defaultdict(list)
nucl_counts = defaultdict(lambda: defaultdict(int)) 
ambig_percents = defaultdict(float)
tn93_dists = defaultdict(list)
avg_tn93_dists_to_subtype = defaultdict(float)
avg_tn93_dists_to_other_subtype = defaultdict(lambda: defaultdict(float))

infile = r"E:\projects\hiv_deeplearning\stremb\data_pol\generated\final_output.fasta"

#Read the generated seqs per subtype
data = {}
with open(infile, 'r') as f:
    name = ''
    for i, line in enumerate(f):
        if line.startswith('//'):
            subtype = line.split('//')[1].strip()
            data[subtype] = []
        elif line.startswith('>'):
            name = line.split('>')[1].strip()
        else:
            data[subtype].append((name, line.strip()))

#Calculate nucleotide counts, ambiguous nucleotide percentages, and tn93 distances

for subtype, seqs in tqdm(data.items(), desc="Getting stats"): #seqs is a list of tuples (name, seq)
    for name, seq in tqdm(seqs, leave=False, desc='seqs'): #name is the name of the sequence, seq is the sequence
        ambig_count = 0
        for nucl in seq:
            nucl_counts[subtype][nucl] += 1
            if nucl not in 'ATCG':
                ambig_count += 1
        ambig_percents[subtype] = ambig_count / len(seq)
        # reference is first (index 0), grabs sequence from the tuple (index 1)
        reference = SeqIO.SeqRecord(seqs[0][1], id=seqs[0][0])
        seq = SeqIO.SeqRecord(seq, id=name)
        tn93_dist = tn93(reference, seq, 'RESOLVE')
        tn93_dist = float(tn93_dist.split(',')[-1])
        tn93_dists[subtype].append(tn93_dist)

#Calculate average tn93 distance to reference sequence of that subtype,

for subtype, dists in tn93_dists.items():
    avg_tn93_dists_to_subtype[subtype] = sum(dists) / len(dists)

#Calculate average tn93 distance to reference sequence of other subtypes.

for subtype, seqs in tqdm(data.items(), desc="Getting distances"):
    for other_subtype, other_seqs in tqdm(data.items(), leave=False, desc='other seqs'):
        if subtype != other_subtype:
            reference = SeqIO.SeqRecord(seqs[0][1], id=seqs[0][0])
            other_reference = SeqIO.SeqRecord(other_seqs[0][1], id=other_seqs[0][0])
            tn93_dist = tn93(reference, other_reference, 'RESOLVE')
            tn93_dist = float(tn93_dist.split(',')[-1])
            avg_tn93_dists_to_other_subtype[subtype][other_subtype] = tn93_dist

#Put all this information into a table, where the rows are subtypes,
# the first few columns are their stats, and then remaining are the distance to each other subtypes references.

#Create a dataframe with the subtype as the index
df = pd.DataFrame(index=subtypes.keys())

#Add columns for nucleotide counts
for subtype, counts in nucl_counts.items():
    for nucl, count in counts.items():
        df.loc[subtype, nucl] = count

#Add column for ambiguous nucleotide percentages
for subtype, percent in ambig_percents.items():
    df.loc[subtype, 'ambig%'] = percent

#Add column for average tn93 distance to reference sequence of that subtype
for subtype, avg_dist in avg_tn93_dists_to_subtype.items():
    df.loc[subtype, 'divergence'] = avg_dist

#Add columns for average tn93 distance to reference sequence of other subtypes.
for subtype, avg_dists in avg_tn93_dists_to_other_subtype.items():
    for other_subtype, avg_dist in avg_dists.items():
        df.loc[subtype, other_subtype] = avg_dist

#Save the dataframe to a csv file
df.to_csv(r"E:\projects\hiv_deeplearning\stremb\data_pol\10k_generated_seqs_output_stats.csv")

#plot subtype distribution bar plot in seaborn counts for each type
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
counts = {}
for subtype, seqs in data.items():
    counts[subtype] = len(seqs)
ax = sns.barplot(x=df.index, y=counts.values())
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()




            





