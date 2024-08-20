import typer
from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
import h5py
from anytree import PreOrderIter
from corgi.seqtree import SeqTree
from collections import Counter
app = typer.Typer()

count = 0

@app.command()
def create_dfam_seqtree(output:Path, dfam:Path, label_smoothing:float=0.0, gamma:float=0.0, partitions:int=5):
    # Create Tree From Text File
    classification_tree = SoftmaxNode(name="root", label_smoothing=label_smoothing, gamma=gamma, repeat_masker_name="")
    classification_nodes = {}

    seqtree = SeqTree(classification_tree)

    file = h5py.File(dfam, 'r')

    subsubtypes = set()

    def visitor_func(name, node):
        global count
        if isinstance(node, h5py.Dataset):
            accession = node.attrs['accession']
            partition = count % partitions

            repeat_type = node.attrs['repeat_type']
            repeat_subtype = node.attrs['repeat_subtype'] if 'repeat_subtype' in node.attrs else ""

            if "-" in repeat_subtype:
                subsubtypes.add(repeat_subtype)

            repeat_name = f"{repeat_type}/{repeat_subtype}" if repeat_subtype else repeat_type
            if repeat_type == "Unknown":
                return

            if repeat_name not in classification_nodes:
                # add to tree
                if repeat_type not in classification_nodes:
                    classification_nodes[repeat_type] = SoftmaxNode(repeat_type, parent=classification_tree, label_smoothing=label_smoothing, gamma=gamma, repeat_masker_name=repeat_type)
                repeat_type_node = classification_nodes[repeat_type]
                if repeat_subtype:
                    classification_nodes[repeat_name] = SoftmaxNode(repeat_subtype, parent=repeat_type_node, label_smoothing=label_smoothing, gamma=gamma, repeat_masker_name=repeat_name)
            
            node = classification_nodes[repeat_name]
            seqtree.add(accession, node, partition)
            count += 1

    file['Families/DF'].visititems(visitor_func)
    classification_tree.render(print=True)
    # for node in PreOrderIter(classification_tree):
    #     node.render(print=True)
    #     counter = Counter([child.name.split("-")[0] for child in node.children])
    #     children_to_split = [name for name, occurrences in counter.items() if occurrences >= 2]

    #     print('children', node.children)
    #     for child in node.children:
    #         print('child.repeat_masker_name', child.repeat_masker_name)
    #         assert child.parent == node
    #         stub = child.name.split("-")[0]
    #         remainder = child.name[len(stub):]
    #         if stub in children_to_split and remainder:
    #             print('stub', stub)
    #             new_name = child.repeat_masker_name[:-len(remainder)]
    #             print('new_name', new_name)
    #             if not new_name in classification_nodes:
    #                 classification_nodes[new_name] = SoftmaxNode(stub, parent=node, label_smoothing=label_smoothing, gamma=gamma, repeat_masker_name=new_name)
    #             assert child.parent == node
                
    #             new = classification_nodes[new_name]
    #             assert new.parent == node
    #             assert new.name == stub
    #             assert new.repeat_masker_name == new_name
                
    #             assert child.parent == node 

    #             print('new.children_dict', new.children_dict)
    #             assert child.parent == node
    #             child.parent = new

    #             del new.children_dict[child.name]
    #             child.name = remainder[1:]
    #             new.children_dict[child.name] = child

    #             assert child.parent == new
    #             print('new.children_dict after', new.children_dict)



    classification_tree.render(print=True)
    breakpoint()
    seqtree.save(output)


def list_accessions(path: Path):
    famdb = FamDB(path)

    for accession in famdb.get_family_accessions():
        family = famdb.get_family_by_accession(accession)
        if not hasattr(family, "classification") or not family.classification:
            continue
        if family.length < 64:
            continue
    
        print(accession)


if __name__ == "__main__":
    app()
