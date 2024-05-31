import sys
import re
from pathlib import Path
from Bio import SeqIO
from corgi.transforms import Transform, slice_tensor
import torch
from corgi.dataloaders import SeqIODataloader
from corgi.tensor import dna_seq_to_tensor


class PadBatch(Transform):
    def __init__(self,min_length:int=0, max_length:int=None, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def encodes(self, batch):
        length = self.min_length
        for item in batch:
            length = max(item[0].shape[0], length)

        if self.max_length:
            length = min(self.max_length, length)

        def pad(tensor):
            # return slice_tensor(tensor[0], length).unsqueeze(dim=0)
            return (slice_tensor(tensor[0], length),) + tensor[1:]

        return list(map(pad, batch))

        # return torch.cat(list(map(pad, batch)))


class MaskedDataloader(SeqIODataloader):
    def __init__(self, files, device, batch_size:int=1, min_length:int=128, max_length:int=5_000, max_seqs:int = None, max_repeats:int=None, format:str=""):
        self.files = list(files)
        self.device = device
        self.format = format
        self.chunk_details = []
        self.max_length = max_length
        self.batch_size = batch_size
        self.min_length = min_length
        self.pad = PadBatch(min_length=min_length)
        self.repeat_details = None
        self.count = 0
        self.max_repeats = max_repeats
        self.max_seqs = max_seqs
        seqs = 0

        self.matcher = re.compile(r"[acgtn]+")
        for file in self.files:
            for record in self.parse(file):
                if self.max_seqs and seqs >= self.max_seqs:
                    break

                
                repeats = self.matcher.findall(str(record.seq))
                for repeat in repeats:                        
                    self.count += 1
                    if self.max_repeats and self.count >= self.max_repeats:
                        return
                seqs += 1

    def __iter__(self):
        self.repeat_details = []
        batch = []
        seqs = 0
        repeat_index = 0

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
                    
                        repeat_index += 1
                        if self.max_repeats and repeat_index >= self.max_repeats:
                            return


        if batch:
            batch = self.pad(batch)
            yield batch


