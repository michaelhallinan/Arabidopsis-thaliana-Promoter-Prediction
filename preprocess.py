#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from Bio import SeqIO

def load_fasta(file_path):
    """Loads sequences and labels from a FASTA file."""
    sequences, labels = [], []
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq).upper()
        label = 1 if "Positive" in record.id else 0
        sequences.append(seq)
        labels.append(label)
    return sequences, labels

def one_hot_encode(sequences, sequence_length=600):
    """Converts DNA sequences into one-hot encoded matrix."""
    nuc_dict = {'A': [1,0,0,0], 'T': [0,1,0,0], 'C': [0,0,1,0], 'G': [0,0,0,1]} 
    encoded = np.zeros((len(sequences), sequence_length, 4))
    for i, seq in enumerate(sequences):
        for j, nuc in enumerate(seq[:sequence_length]):
            encoded[i, j] = nuc_dict.get(nuc, [0,0,0,0])
    return encoded

def calculate_gc_content(sequence):
    """Calculates GC content as a feature."""
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if sequence else 0

def add_gc_content_to_data(sequences, one_hot_encoded, sequence_length=600):
    """Adds GC content as an extra feature."""
    gc_contents = np.array([calculate_gc_content(seq) for seq in sequences]).reshape(-1, 1)
    gc_repeated = np.repeat(gc_contents, sequence_length, axis=1)[..., np.newaxis]
    return np.concatenate([one_hot_encoded, gc_repeated], axis=-1)

