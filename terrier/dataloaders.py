import sys
import re
from pathlib import Path
from Bio import SeqIO
from corgi.transforms import PadBatchX
from corgi.dataloaders import SeqIODataloader
from corgi.tensor import dna_seq_to_tensor


class MaskedDataloader(SeqIODataloader):
    def __init__(self, files, device, batch_size:int=1, min_length:int=128, max_length:int=5_000, max_seqs:int=None, format:str=""):
        self.files = list(files)
        self.device = device
        self.format = format
        self.chunk_details = []
        self.max_length = max_length
        self.batch_size = batch_size
        self.min_length = min_length
        self.pad = PadBatchX()
        self.repeat_details = None
        self.count = 0
        self.max_seqs = max_seqs
        seqs = 0

        self.matcher = re.compile(r"[acgtn]+")
        for file in self.files:
            for record in self.parse(file):

                if self.max_seqs and seqs >= self.max_seqs:
                    break

                
                repeats = self.matcher.findall(str(record.seq))
                for repeat in repeats:
                    if len(repeat) < min_length:
                        continue
                        
                    self.count += 1
                seqs += 1

    def __iter__(self):
        self.repeat_details = []
        batch = []
        seqs = 0

        for file in self.files:
            for record in self.parse(file):
                if self.max_seqs and seqs >= self.max_seqs:
                    break

                seqs += 1
                repeats = self.matcher.finditer(str(record.seq))
                for repeat in repeats:
                    start = repeat.start()
                    end = repeat.end()
                    if end - start < self.min_length:
                        continue

                    self.repeat_details.append( (str(file), record.id, start+1, end) )
                    batch.append(dna_seq_to_tensor(repeat.group(0).upper()))

                    if len(batch) >= self.batch_size:
                        batch = self.pad(batch)
                        yield batch
                        batch = []

        if batch:
            batch = self.pad(batch)
            yield batch
