=============
Preprocessing
=============

Use this tutorial to learn how to generate the files needed to train Terrier from scratch. 
You can use this to replicate the results found in the paper (still to be released).

Terrier takes two input files, a SeqTree and a SeqBank. 
The SeqBank holds the sequence data for each accession in the dataset. 
The SeqTree has the information about the cross-validation partition for each accession and which node in the taxonomic tree that the accession corresponds to.
First we will download the Repbase library and then we will generate the SeqBank and SeqTree files from it.

Repbase
-------

Repbase is available with a subscription from the Genetic Information Research Institute (GIRI) and is released under an academic user agreement which is available on the GIRI site.
Terrier was trained using the 29.07 release of Repbase (2024-07-24) release.

Download Repbase database in FASTA format from https://www.girinst.org/server/RepBase/index.php
Untar the file and you will have a directory with a few dozen FASTA files with a `.ref` extension. 
We will refer to this directory as `REPBASE_DIR`. 
We will ignore the files in the `archive` directory.

SeqBank
-------

To create the SeqBank file, we use the seqbank CLI utility which is included as a dependency of Terrier.

.. code-block:: bash

    seqbank add $REPBASE_DIR/RepBase29.07.sb $REPBASE_DIR/*.ref --format fasta

This will create a SeqBank file called ``RepBase29.07.sb`` inside the ``$REPBASE_DIR``.

SeqTree
-------

To create the SeqTree file we use the `create-repeatmasker-seqtree` command from the `terrier-tools` CLI utility.

.. code-block:: bash

    terrier-tools create-repeatmasker-seqtree $REPBASE_DIR/RepBase29.07.st $REPBASE_DIR

This will create a SeqTree file at ``$REPBASE_DIR/RepBase29.07.st`` with five cross-validation partitions and the taxonomic tree from the Repbase library using the RepeatMasker schema.

To create a different number of partitions, run the command with the `--partitions` flag. For more options see the help:

.. code-block:: bash

    terrier-tools create-repeatmasker-seqtree --help

Now you are ready to train Terrier using the SeqBank and SeqTree files you have created.