Given 2 files:
1. fasta
2. metadata csv with header 'Accession' and 'Subtype'



Preprocessing pipeline

1. create fasta file with format >subtype.sequence

2. obtain sequence distance matrix
	seqruler
		input: fasta
		output: sequence distance matrix

3. obtain subtype distance matrix 
	obtain_subtype_distances.py
		input: sequence distance matrix
		output: subtype distance matrix

4. give fasta to seqSpawnR to generate new sequences
	gen_seqs.R (seqspawnR)
		input: fasta
		output: augmented fasta

5. convert output format from SeqSpawnr to fasta 
	subtype_fmt_to_fasta.py
		Give seqSpawnR augmented fasta
		Returns final augmented fasta

6. run preproc_new.py on augmented fasta
	Creates contrast sets for each sequence
		similar:
			5 sequences from same subtype
		dissimilar:
			5 sequences from closest different subtype
			5 sequences from 5 randomly chosen different subtypes
	
	and makes the following data structures for training:
	
	preproc/
		train_seqs_tensor
		train_labels_tensor
		train_contrasts

		train_map_row_to_seqid
		train_map_seqid_to_row
		train_map_seqid_to_subtype
		train_map_subtype_to_seqids
	
		test_seqs_tensor
		test_labels_tensor
		test_contrasts

		test_map_row_to_seqid
		test_map_seqid_to_row
		test_map_seqid_to_subtype
		test_map_subtype_to_seqids
		
		map_label_to_subtype
		map_subtype_to_label
	
	
7. Give the preproc directory as input to train.py