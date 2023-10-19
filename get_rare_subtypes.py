import numpy as np
infile = 'stremb/data_pol/sequences.fasta'
outfile = 'stremb/data_pol/sequences_rare_subtypes.fasta'

seqs = []
subtypes = []

with open(infile, 'r') as f:
    last_name = ''
    for i, line in enumerate(f):
        if line.startswith('>'):
            last_name = line.strip()[1:]
            subtype = last_name.split('.')[0]
            subtypes.append(subtype)
        else:
            seqs.append((last_name, line.strip()))

subtypes = np.array(subtypes)
unique_subtypes, counts = np.unique(subtypes, return_counts=True)

rare_subtypes = unique_subtypes[counts < 10]
rare_subtypes_idx = np.isin(subtypes, rare_subtypes)

# Write sequences to file, choosing 1 sequence from each rare subtype

with open(outfile, 'w') as f:
    visited_subtypes = {}
    for i, subtype in enumerate(subtypes):
        if subtype in rare_subtypes and subtype not in visited_subtypes:
            visited_subtypes[subtype] = True
            f.write('>' + seqs[i][0] + '\n')
            f.write(seqs[i][1] + '\n')


        