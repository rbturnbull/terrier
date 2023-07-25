from pathlib import Path
from corgi.tensor import dna_seq_to_tensor


class RepBase():
    def __init__(self, base_dir:Path, **kwargs):
        super().__init__(**kwargs)
        self.base_dir = Path(base_dir)
        self.index_fasta()

    def index_fasta(self):
        self.refs = {}

        import pyfastx
        for ref in self.base_dir.glob("*.ref"):
            self.refs[ref.stem] = pyfastx.Fasta(str(ref), uppercase=True)

    def __getitem__(self, value):
        value = value.strip()
        slash_position = value.find("/")
        if slash_position <= 0:
            raise ValueError(f"key {value} not understood")
        
        ref_name = value[:slash_position]
        accession = value[slash_position+1:]
        assert ref_name in self.refs
        ref = self.refs[ref_name]
        if accession not in ref:
            raise ValueError(f"accession {accession} not in ref {ref_name}")
        
        return ref[accession]
    
    def to_csv(self, filename):
        with open(filename, "w") as f:
            print("ref", "id", "key", "classification", "species", sep=",", file=f)
            for ref, fasta in self.refs.items():
                for record in fasta:
                    components = record.description.split("\t")
                    if len(components) != 3:
                        print(f"incorrect number of tabs in description: {record.description}")
                        continue
                    id = components[0]
                    classification = components[1]
                    species = components[2]
                    key = f"{ref}/{id}"
                    print(ref, id, key, classification, species, sep=",", file=f)
                    print(ref, id, key, classification, species, sep=",")
    
    def __getstate__(self):
        # don't include fasta indexes in pickle
        state = self.__dict__.copy()
        del state['refs']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.index_fasta()


class RepBaseGetter():
    def __init__(self, repbase):
        self.repbase = repbase

    def __call__(self, key:str):
        assert self.repbase
        record = self.repbase[key]
        return dna_seq_to_tensor(record.seq)


