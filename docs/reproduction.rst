================
Reproduction
================

This document describes how to reproduce the results of the paper. The test data comes from: 

   - Bickmann, Lucas, Matias Rodriguez, Xiaoyi Jiang, and Wojciech Makalowski. 
     "TEclass2: Classification of transposable elements using Transformers." 
     *bioRxiv* (2023). `doi:10.1101/2023.10.13.562246 <https://doi.org/10.1101/2023.10.13.562246>`_. 
     Accessed at: `https://www.biorxiv.org/content/early/2023/10/16/2023.10.13.562246 <https://www.biorxiv.org/content/early/2023/10/16/2023.10.13.562246>`_


Fruit Fly Genome
================

Bickmann et al. (2023) provide Transposable Elements (TE) models of a fruit-fly genome. Download it with the following command:

.. code-block:: bash

   wget https://raw.githubusercontent.com/IOB-Muenster/TEclass2/refs/heads/main/tests/Drosophila_melanogaster.fasta

Run inference using Terrier like this:

.. code-block:: bash

   terrier --file Drosophila_melanogaster.fasta \
        --output-csv drosophila-terrier.final.TEs.csv \
        --min-length 0 \
        --threshold 0

.. note::

   We set the theshold to zero so that we can see all the predictions, the threshold can be adjusted in the evaluation steps below.
   We also set the minimum length to zero to include all the sequences in the evaluation. 
   By default, Terrier will only evaluate sequences with a minimum length of 128.

Now evaluate the results with the following command:

.. code-block:: bash

   terrier-tools evaluate --csv drosophila-terrier.final.TEs.csv  \
        --threshold 0.7 \
        --map /I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC 
        
That will produce the following output:

   Total: 667
   Total with ground truth: 661
   Number classified: 424/661 (64.15%)
   Correct predictions: 386/424 (91.04%)
   
.. note::

   The dataset includes six sequences with the 'Unknown' label. That is why there is a difference between the 'Total' and 'Total with ground truth' values.

To get the accuracy results just for the 'Order' level and ignoring the 'Superfamily' level, use the following command:

.. code-block:: bash

   terrier-tools evaluate --csv drosophila-terrier.final.TEs.csv  \
        --threshold 0.7 \
        --no-superfamily \
        --map /I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC

That will produce the following output:

.. code-block:: text

   Total: 667
   Total with ground truth: 661
   Number classified: 539/661 (81.54%)
   Correct predictions: 474/539 (87.94%)

To generate a confusion matrix, use the following command:

.. code-block:: bash

   terrier-tools confusion-matrix --csv drosophila-terrier.final.TEs.csv  \
        --output drosophila-terrier-confusion-matrix-threshold-0.7.html \
        --threshold 0.7 \
        --map /I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC

That will generate an HTML file like this:

.. raw:: html
    :file: ./images/drosophila-terrier-confusion-matrix-threshold-0.7.html

You can also output the confusion matrix with a .csv, .png, .svg, or .pdf extension by changing the extension of the output file.    

These results reflect the default Terrier threshold of 0.7. You can output the confusion matrix for different thresholds by changing the threshold value in the command above.

To see the effect of the threshold on the results, you can run the following command:

.. code-block:: bash

   terrier-tools threshold-plot --csv drosophila-terrier.final.TEs.csv  \
        --output drosophila-terrier-threshold-plot.html \
        --map /I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC

That will generate an HTML file like this:

.. raw:: html
    :file: ./images/drosophila-terrier-threshold-plot.html


Rice Genome
================

Bickmann et al. (2023) also provide Transposable Elements (TE) models of a rice genome. Download it with the following command:

.. code-block:: bash

   wget https://raw.githubusercontent.com/IOB-Muenster/TEclass2/refs/heads/main/tests/Oryza_sativa.fasta

Run inference using Terrier like this:

.. code-block:: bash

   terrier --file Oryza_sativa.fasta --output-csv oryza-terrier.final.TEs.csv --threshold 0

.. note::

   We set the theshold to zero so that we can see all the predictions, the threshold can be adjusted in the evaluation steps below.
   We can use the default minimum length because the sequences are long enough.

Now evaluate the results with the following command:

.. code-block:: bash

   terrier-tools evaluate --csv oryza-terrier.final.TEs.csv  \
        --threshold 0.7 \
        --map /I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC

That will produce the following output:

.. code-block:: text

    Total: 75
    Total with ground truth: 75
    Number classified: 68/75 (90.67%)
    Correct predictions: 67/68 (98.53%)

To get the accuracy results just for the 'Order' level and ignoring the 'Superfamily' level, use the following command:

.. code-block:: bash

   terrier-tools evaluate --csv oryza-terrier.final.TEs.csv  \
        --threshold 0.7 \
        --no-superfamily \
        --map /I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC

That will produce the following output:

.. code-block:: text

    Total: 75
    Total with ground truth: 75
    Number classified: 71/75 (94.67%)
    Correct predictions: 67/71 (94.37%)

To generate a confusion matrix, use the following command:

.. code-block:: bash

   terrier-tools confusion-matrix --csv oryza-terrier.final.TEs.csv  \
        --output oryza-terrier-confusion-matrix-threshold-0.7.html \
        --threshold 0.7 \
        --map /I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC

That will generate an HTML file like this:

.. raw:: html
    :file: ./images/oryza-terrier-confusion-matrix-threshold-0.7.html

You can also output the confusion matrix with a .csv, .png, .svg, or .pdf extension by changing the extension of the output file.    

These results reflect the default Terrier threshold of 0.7. You can output the confusion matrix for different thresholds by changing the threshold value in the command above.

To see the effect of the threshold on the results, you can run the following command:

.. code-block:: bash

   terrier-tools threshold-plot --csv oryza-terrier.final.TEs.csv  \
        --output oryza-terrier-threshold-plot.html \
        --map /I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC

That will generate an HTML file like this:

.. raw:: html
    :file: ./images/oryza-terrier-threshold-plot.html



Human Genome
================


.. code-block:: bash

   wget https://raw.githubusercontent.com/rbturnbull/terrier/refs/heads/main/comparison-test-data/Homo_sapiens.fasta

Run inference using Terrier like this:

.. code-block:: bash

   terrier --file Homo_sapiens.fasta --output-csv Terrier-human.csv --threshold 0


Now evaluate the results with the following command:

.. code-block:: bash

   terrier-tools evaluate --csv Terrier-human.csv  \
        --threshold 0.7 \
        --map "/Pao=/Bel-Pao,TIR=DNA,DNA/CMC-.*=DNA/CACTA,DNA/CMC=DNA/CACTA,TcMar-.*=Tc1,Tc1-.*=Tc1,hAT-.*=hAT,LTR/ERV.*=LTR/ERV,L1-.*=L1,PIF-Harbinger=Harbinger,Crypton-.*=Crypton,RTE-.*=RTE,Retroposon/L1=LINE/L1,Satellite/.*=Satellite,^tRNA=SINE/tRNA,SINE/tRNA-.*=SINE/tRNA,TcMar=Tc1,SINE/5S-.*=SINE/5S,SINE/Alu=SINE/7SL,SINE/B2=SINE/tRNA,SINE/B4=SINE/tRNA,SINE/MIR=SINE/tRNA,SINE/ID=SINE/tRNA,/I-Jockey=/I,/Jockey.*=/I,/MULE-.*=/MULE,LINE/R1-.*=LINE/R1"


To generate a confusion matrix, use the following command:

.. code-block:: bash

   terrier-tools confusion-matrix --csv Terrier-human.csv  \
        --output Terrier-human-confusion-matrix-threshold-0.7.html \
        --threshold 0.7 \
        --map "/Pao=/Bel-Pao,TIR=DNA,DNA/CMC-.*=DNA/CACTA,DNA/CMC=DNA/CACTA,TcMar-.*=Tc1,Tc1-.*=Tc1,hAT-.*=hAT,LTR/ERV.*=LTR/ERV,L1-.*=L1,PIF-Harbinger=Harbinger,Crypton-.*=Crypton,RTE-.*=RTE,Retroposon/L1=LINE/L1,Satellite/.*=Satellite,^tRNA=SINE/tRNA,SINE/tRNA-.*=SINE/tRNA,TcMar=Tc1,SINE/5S-.*=SINE/5S,SINE/Alu=SINE/7SL,SINE/B2=SINE/tRNA,SINE/B4=SINE/tRNA,SINE/MIR=SINE/tRNA,SINE/ID=SINE/tRNA,/I-Jockey=/I,/Jockey.*=/I,/MULE-.*=/MULE,LINE/R1-.*=LINE/R1"

That will generate an HTML file like this:

.. raw:: html
    :file: ../comparison-test-data/Terrier/Terrier-human-confusion-matrices/Terrier-0.7-human-superfamily-confusion-matrix.html



Mouse Genome
================


.. code-block:: bash

   wget https://raw.githubusercontent.com/rbturnbull/terrier/refs/heads/main/comparison-test-data/Mus_musculus.fasta

Run inference using Terrier like this:

.. code-block:: bash

   terrier --file Mus_musculus.fasta --output-csv Terrier-mouse.csv --threshold 0


Now evaluate the results with the following command:

.. code-block:: bash

   terrier-tools evaluate --csv Terrier-mouse.csv  \
        --threshold 0.7 \
        --map "/Pao=/Bel-Pao,TIR=DNA,DNA/CMC-.*=DNA/CACTA,DNA/CMC=DNA/CACTA,TcMar-.*=Tc1,Tc1-.*=Tc1,hAT-.*=hAT,LTR/ERV.*=LTR/ERV,L1-.*=L1,PIF-Harbinger=Harbinger,Crypton-.*=Crypton,RTE-.*=RTE,Retroposon/L1=LINE/L1,Satellite/.*=Satellite,^tRNA=SINE/tRNA,SINE/tRNA-.*=SINE/tRNA,TcMar=Tc1,SINE/5S-.*=SINE/5S,SINE/Alu=SINE/7SL,SINE/B2=SINE/tRNA,SINE/B4=SINE/tRNA,SINE/MIR=SINE/tRNA,SINE/ID=SINE/tRNA,/I-Jockey=/I,/Jockey.*=/I,/MULE-.*=/MULE,LINE/R1-.*=LINE/R1"


To generate a confusion matrix, use the following command:

.. code-block:: bash

   terrier-tools confusion-matrix --csv Terrier-mouse.csv  \
        --output Terrier-mouse-confusion-matrix-threshold-0.7.html \
        --threshold 0.7 \
        --map "/Pao=/Bel-Pao,TIR=DNA,DNA/CMC-.*=DNA/CACTA,DNA/CMC=DNA/CACTA,TcMar-.*=Tc1,Tc1-.*=Tc1,hAT-.*=hAT,LTR/ERV.*=LTR/ERV,L1-.*=L1,PIF-Harbinger=Harbinger,Crypton-.*=Crypton,RTE-.*=RTE,Retroposon/L1=LINE/L1,Satellite/.*=Satellite,^tRNA=SINE/tRNA,SINE/tRNA-.*=SINE/tRNA,TcMar=Tc1,SINE/5S-.*=SINE/5S,SINE/Alu=SINE/7SL,SINE/B2=SINE/tRNA,SINE/B4=SINE/tRNA,SINE/MIR=SINE/tRNA,SINE/ID=SINE/tRNA,/I-Jockey=/I,/Jockey.*=/I,/MULE-.*=/MULE,LINE/R1-.*=LINE/R1"

That will generate an HTML file like this:

.. raw:: html
    :file: ../comparison-test-data/Terrier/Terrier-mouse-confusion-matrices/Terrier-0.7-mouse-superfamily-confusion-matrix.html
