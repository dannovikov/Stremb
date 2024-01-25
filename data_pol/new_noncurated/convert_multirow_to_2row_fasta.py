#I have a fasta file with a header and then many lines for a sequences before next header. 
# i want to convert this to a fasta file with 2 lines per sequence. the first for the id and 
# the second for the sequence.

import Bio
from Bio import SeqIO
import sys

fasta_file = "sequences.fas"
out_file = "sequences_2line.fasta"

with open(out_file, "w") as out_handle:
    for record in SeqIO.parse(fasta_file, "fasta"):
        SeqIO.write(record, out_handle, "fasta-2line")

