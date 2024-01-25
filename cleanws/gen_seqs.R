.libPaths("C:\\Users\\Dan\\Documents\\R_Packages")
library(devtools)
devtools::install_local("E:\\projects\\seqspawnR_with_loadingbars\\SeqSpawnR", force=TRUE)
 


library(SeqSpawnR)
library(seqinr)

fasta_path= "E:\\projects\\hiv_deeplearning\\stremb\\data_pol\\new_noncurated\\sequences_2line.fasta"
output_dir= "E:\\projects\\hiv_deeplearning\\stremb\\data_pol\\new_noncurated\\generated\\"
metadata_path= "E:\\projects\\hiv_deeplearning\\stremb\\data_pol\\new_noncurated\\metadata.csv"

N <- 2000 #Number of sequences each subtype should have after generation

# Read sequences from a FASTA file
fasta_sequences <- read.fasta(fasta_path, whole.header=TRUE, forceDNAtolower = FALSE)

# 1. Count unique subtypes and sequences

# read the metadata file and match "Accession" column and get "Subtype" column
meta <- read.csv(metadata_path, header = TRUE, sep = ",", stringsAsFactors = FALSE, encoding = "ISO-8859-1")
# count the occurence of each subtype in the fasta file by matching the sequence names with the "Accession" column in the metadata file
subtype_counter <- table(meta[match(names(fasta_sequences), meta$Accession), "Subtype"])
print("Subtypes and their counts:")
print(subtype_counter)

# 2. Calculate spawn quotas
spawn_quotas <- ifelse((N+1) < subtype_counter, 0, ceiling((N+1 - subtype_counter) / subtype_counter))
print("Spawn quotas:")
print(spawn_quotas)

# 3. & 4. Modified loop for spawn quotas and file handling


for (name in names(fasta_sequences)) {
  # subtype <- unlist(strsplit(name, "\\."))[1]
  subtype <- meta[match(name, meta$Accession), "Subtype"]

  output_file <- paste0(output_dir, subtype, "_generated.fasta")
  
  if (!file.exists(output_file)) {
    file.create(output_file)
  }
  
  ref_sequence <- toupper(paste(fasta_sequences[[name]], collapse = ""))
  
  write(paste(">", name, sep = ""), output_file, append = TRUE)
  write(ref_sequence, output_file, append = TRUE)
  
  quota <- spawn_quotas[subtype]
  if (quota == 0) {
    next
  }
  print(quota)
  
  spawned <- SeqSpawnR::spawn_sequences(quota, seed = ref_sequence, snps = 10)
  
  for (i in 1:length(spawned)) {
    new_name <- paste(name, "_spawned_", i, sep = "")
    spawned_sequence <- toupper(paste(spawned[[i]], collapse = ""))
    
    write(paste(">", new_name, sep = ""), output_file, append = TRUE)
    write(spawned_sequence, output_file, append = TRUE)
  }
}

# 5. Concatenate all output files
output_files <- list.files(path = output_dir, pattern = "*_generated.fasta", full.names = TRUE)
final_output <- "E:\\projects\\hiv_deeplearning\\stremb\\data_pol\\new_noncurated\\generated\\final_output.fasta"

for (file in output_files) {
  lines <- readLines(file)
  write(lines, final_output, append = TRUE, ncolumns = 1)
}
