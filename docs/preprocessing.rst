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

    terrier-tools preprocess --input $REPBASE_DIR --seqbank $REPBASE_DIR/Repbase-seqbank.sb --seqtree $REPBASE_DIR/Repbase-seqtree.st

This will create a SeqBank file called ``Repbase-seqbank.sb`` and a SeqTree files called ``Repbase-seqtree.st`` and place them the ``$REPBASE_DIR``. 

The SeqTree file will have five cross-validation partitions and a taxonomic tree using the RepeatMasker schema.

Some accessions appear in more than one file with different classifications.
Each accession is assigned once: a general label (e.g. ``DNA``) is dropped in favour of a more specific one from the same lineage (e.g. ``DNA/hAT``).
If labels from different lineages conflict, the most specific one is used, then the most common, then the first alphabetically, so the result does not depend on the order of the input files.
The summary reports how many labels were superseded or dropped.

A few sequences have a label which is too generic or wrong in Repbase, so classifications for individual accessions can be set in ``terrier/data/repbase-accession-rules.toml``.
A rule applies to every occurrence of that accession and takes priority over the label mapping.
Each classification must be one that the label mapping can produce, and the summary reports how many rules were applied.

SeqBanks use the ``lmdb`` storage engine by default. Use ``--engine rocksdict`` or ``--engine speedict`` to choose another one
(``speedict`` needs ``seqbank[speedict]`` and Python below 3.13).
An existing SeqBank must already use the requested engine, otherwise it is an error.

Partitions are assigned deterministically using a SHA-256 hash of the seed and the
uppercase sequence. The default seed is ``0``; use ``--seed`` to choose another
seed. With the same seed and partition count, reordering files or adding sequences
does not change existing assignments, and identical sequences share a partition.
Partition sizes are approximately balanced. Changing the seed or partition count
changes assignments. This replaces the previous round-robin assignment, so
regenerating a SeqTree can change partitions compared with earlier versions.

To create a different number of partitions, run the command with the ``--partitions`` flag. For more options see the help:

.. code-block:: bash

    terrier-tools preprocess --help

Now you are ready to train Terrier using the SeqBank and SeqTree files you have created.


Optional: Display the SeqTree
------------------------------

You can list the number of sequences for each node in the SeqTree file with this command:

.. code-block:: bash

    seqtree render $REPBASE_DIR/Repbase-seqtree.st --print --count

That will output a tree with the number of sequences like this:

.. code-block:: text

    root
    ├── DNA (2494)
    │   ├── TcMar (5313)
    │   ├── MULE (2793)
    │   ├── hAT (7796)
    │   ├── Merlin (188)
    │   ├── Kolobok (1015)
    │   ├── Maverick (264)
    │   ├── PiggyBac (716)
    │   ├── CMC (2121)
    │   ├── Academ (708)
    │   ├── P (357)
    │   ├── Crypton (326)
    │   ├── Zisupton (127)
    │   ├── Harbinger (2898)
    │   ├── Dada (171)
    │   ├── IS3EU (173)
    │   ├── Novosib (9)
    │   ├── Ginger (118)
    │   ├── Sola (428)
    │   └── Zator (106)
    ├── Satellite (631)
    ├── RC
    │   └── Helitron (2226)
    ├── LTR (1204)
    │   ├── Gypsy (41309)
    │   ├── DIRS (2095)
    │   ├── Pao (9880)
    │   ├── Copia (12549)
    │   ├── ERV (9638)
    │   └── Caulimovirus (217)
    ├── LINE (460)
    │   ├── R2 (356)
    │   ├── CR1 (1458)
    │   ├── L1 (5514)
    │   ├── L2 (1142)
    │   ├── Jockey-I (1176)
    │   ├── Dualen (13)
    │   ├── RTE (1793)
    │   ├── Rex-Babar (222)
    │   ├── Dong-R4 (316)
    │   ├── CRE (133)
    │   ├── R1 (426)
    │   ├── Proto2 (91)
    │   ├── Proto1 (10)
    │   └── Tad1 (699)
    ├── Structural_RNA (157)
    ├── SINE (113)
    │   ├── 7SL (244)
    │   ├── tRNA (2664)
    │   ├── 5S (50)
    │   └── U (20)
    ├── PLE (1713)
    ├── Simple_repeat (139)
    └── Other (11)

You can also display the SeqTree in a Sunburst chart by running:

.. code-block:: bash

    seqtree sunburst $REPBASE_DIR/Repbase-seqtree.st --show --output $REPBASE_DIR/Repbase-seqtree.html

This will create an HTML file with the Sunburst chart of the SeqTree like this:

.. raw:: html
    :file: ./images/Repbase-seqtree.html

You can open the HTML file in a browser to view the chart.

You can also output the SeqTree with a .png, .svg, or .pdf extension by changing the extension of the output file.

Custom Datasets
----------------

You can create a custom repeat library in FASTA format, with the classification of each sequence like this:

.. code-block:: text

    >SeqID#DNA/Academ
    ACTGACTGACTG...

Or with the classification separated with a tab character like this:

.. code-block:: text

    >SeqID	LTR/Caulimovirus
    ACTGACTGACTG...

Then preprocess like this:

.. code-block:: bash

        terrier-tools preprocess --input custom.fasta --seqbank custom-seqbank.sb --seqtree custom-seqtree.st

.. warning ::

    Terrier will only be able to classify sequences into classes found here: https://github.com/rbturnbull/terrier/blob/main/terrier/data/repbase-to-repeatmasker.toml

You can include Repbase with your custom dataset like this:

.. code-block:: bash

        terrier-tools preprocess --input $REPBASE_DIR --input custom.fasta --seqbank combined-seqbank.sb --seqtree combined-seqtree.st



Preprocessing summaries
-----------------------

Preprocessing prints Rich tables with dataset totals, every input classification,
and retained classification counts and percentages. The input classification table
and CSV contain three columns: ``provided`` (unique input label), ``repeat_masker``
(resulting classification, or ``<UNSUPPORTED>``), and ``count`` (input sequences
with that label). Labels already in the supported schema retain their names.
Unsupported and unknown records are excluded from the SeqTree; failed additions
are counted separately in the dataset summary. The SeqBank contains the input
sequences.

Counts in the retained table are direct assignments, not totals including child
classes. Percentages use all successfully retained sequences as the denominator.
The input classification table includes rejected records, so its counts sum to the
number of input sequences.

Save any or all tables using these optional arguments:

.. code-block:: bash

    terrier-tools preprocess --input custom.fasta \
        --seqbank custom-seqbank.sb --seqtree custom-seqtree.st \
        --summary-csv summary.csv \
        --classifications-csv classifications.csv \
        --retained-csv retained.csv

CSV files contain the same columns and rows as the corresponding displayed tables.
Parent directories are created automatically; existing CSV files are overwritten.
