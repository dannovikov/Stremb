"""
in format:
//subtype
>ref
seq
>spawned1
seq
...
//anothersubtype
...

out format:
>ref
seq
>spawned1
seq
...

essentially just remove the lines that start with "//"

"""

infile = r"E:\projects\hiv_deeplearning\stremb\cleanws\generated\final_output.fasta"
outfile = r"E:\projects\hiv_deeplearning\stremb\cleanws\generated\final_output_no_subtypes.fasta"

with open(infile, 'r') as f:
    with open(outfile, 'w') as g:
        for line in f:
            if line.startswith('//'):
                continue
            else:
                g.write(line)

                