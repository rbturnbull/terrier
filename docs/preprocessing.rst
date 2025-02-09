=============
Preprocessing
=============

Use this tutorial to learn how to generate the files needed to train Terrier from scratch. 
You can use this to replicate the results found in the paper (still to be released).

Repbase
-------

Terrier is trained using Repbase, a database of repetitive elements. 
Repbase is available with a subscription from the Genetic Information Research Institute (GIRI) and is released under an academic user agreement which is available on the GIRI site.
Terrier was trained using the 29.07 release of Repbase (2024-07-24) release.

Download Repbase database in FASTA format from https://www.girinst.org/server/RepBase/index.php
Untar the file and you will have a directory with a few dozen FASTA files with a ``.ref`` extension. 
We will refer to this directory as ``REPBASE_DIR``. 
We will ignore the files in the ``archive`` directory.

Preprocess
----------

Terrier takes two input files, a SeqTree and a SeqBank. 
The SeqBank holds the sequence data for each accession in the dataset. 
The SeqTree has the information about the cross-validation partition for each accession and which node in the taxonomic tree that the accession corresponds to.

These two files can be generated from the Repbase database using the ``terrier-tools`` CLI utility:

.. code-block:: bash

    terrier-tools preprocess --repbase $REPBASE_DIR --seqbank $REPBASE_DIR/Repbase-seqbank.sb --seqtree $REPBASE_DIR/Repbase-seqtree.st

This will create a SeqBank file called ``Repbase-seqbank.sb`` and a SeqTree files called ``Repbase-seqtree.st`` and place them the ``$REPBASE_DIR``. 

The SeqTree file will have five cross-validation partitions and a taxonomic tree using the RepeatMasker schema.

To create a different number of partitions, run the command with the ``--partitions`` flag. For more options see the help:

.. code-block:: bash

    terrier-tools preprocess --help

Now you are ready to train Terrier using the SeqBank and SeqTree files you have created.


Optional: Display the SeqTree
------------------------------

You can list the number of accessions for each node in the SeqTree file with this command:

.. code-block:: bash

    seqtree render $REPBASE_DIR/Repbase-seqtree.st --print --count

That will output a tree with the number of accessions like this:

.. code-block:: text

    root
    ├── SINE (107)
    │   ├── 7SL (192)
    │   ├── tRNA (2312)
    │   ├── 5S (35)
    │   └── U (17)
    ├── LTR (1169)
    │   ├── ERV (9127)
    │   ├── Gypsy (28119)
    │   ├── DIRS (1332)
    │   ├── Copia (10619)
    │   ├── Pao (5320)
    │   └── Caulimovirus (207)
    ├── LINE (446)
    │   ├── L1 (4833)
    │   ├── R2 (336)
    │   ├── CR1 (1237)
    │   ├── I (1093)
    │   ├── R1 (410)
    │   ├── Rex-Babar (141)
    │   ├── Dong-R4 (58)
    │   ├── L2 (889)
    │   ├── RTE (1065)
    │   ├── Proto2 (69)
    │   ├── CRE (99)
    │   ├── Dualen (13)
    │   ├── Proto1 (10)
    │   └── Tad1 (550)
    ├── DNA (2404)
    │   ├── hAT (6083)
    │   ├── PiggyBac (627)
    │   ├── TcMar (4884)
    │   ├── Kolobok (857)
    │   ├── MULE (2531)
    │   ├── CMC (1667)
    │   ├── Merlin (151)
    │   ├── Maverick (237)
    │   ├── P (320)
    │   ├── Harbinger (2381)
    │   ├── Dada (170)
    │   ├── Crypton (294)
    │   ├── Ginger (91)
    │   ├── Academ (526)
    │   ├── Zator (102)
    │   ├── IS3EU (79)
    │   ├── Zisupton (44)
    │   ├── Sola (385)
    │   └── Novosib (9)
    ├── Satellite (741)
    ├── RC
    │   └── Helitron (1999)
    ├── Structural_RNA (86)
    ├── PLE (1045)
    └── Other (11)

You can also display the SeqTree in a Sunburst chart by running:

.. code-block:: bash

    seqtree sunburst $REPBASE_DIR/Repbase-seqtree.st --show --output $REPBASE_DIR/Repbase-seqtree.html

This will create an HTML file with the Sunburst chart of the SeqTree like this:

.. raw:: html
    :file: ./images/Repbase-seqtree.html

You can open the HTML file in a browser to view the chart.

You can also output the SeqTree with a .png, .svg, or .pdf extension by changing the extension of the output file.