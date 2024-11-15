================
Reproduction
================

This document describes how to reproduce the results of the paper. The test data comes from: 

   - Bickmann, Lucas, Matias Rodriguez, Xiaoyi Jiang, and Wojciech Makalowski. 
     "TEclass2: Classification of transposable elements using Transformers." 
     *bioRxiv* (2023). `doi:10.1101/2023.10.13.562246 <https://doi.org/10.1101/2023.10.13.562246>`_. 
     Accessed at: `https://www.biorxiv.org/content/early/2023/10/16/2023.10.13.562246 <https://www.biorxiv.org/content/early/2023/10/16/2023.10.13.562246>`_


Fruit-fly genome
================

Bickmann et al. (2023) provide Transposable Elements (TE) models of a fruit-fly genome. Download it with the following command:

.. code-block:: bash

   wget https://raw.githubusercontent.com/IOB-Muenster/TEclass2/refs/heads/main/tests/drosophila.final.TEs.fa

Run inference using Terrier like this:

.. code-block:: bash

   terrier --file drosophila.final.TEs.fa --output drosophila-terrier.final.TEs.csv --threshold 0

.. note::

   We set the theshold to zero so that we can see all the predictions, the threshold can be adjusted in the evaluation steps below.

Now evaluate the results with the following command:

.. code-block:: bash

   terrier-tools evaluate --csv drosophila-terrier.final.TEs.csv  \
        --threshold 0.7 \
        --map I-Jockey=I,Jockey-I=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC \
        --ignore Unknown

That will produce the following output:

.. code-block:: text

    Total: 619
    Total with ground truth: 613
    Number classified: 412/613 (67.21%)
    Correct predictions: 375/412 (91.02%)

To get the accuracy results just for the 'Order' level and ignoring the 'Superfamily' level, use the following command:

.. code-block:: bash

   terrier-tools evaluate --csv drosophila-terrier.final.TEs.csv  \
        --threshold 0.7 \
        --no-superfamily \
        --map I-Jockey=I,Jockey-I=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC \
        --ignore Unknown

That will produce the following output:

.. code-block:: text

    Total: 619
    Total with ground truth: 613
    Number classified: 412/613 (67.21%)
    Correct predictions: 391/412 (94.90%)

To generate a confusion matrix, use the following command:

.. code-block:: bash

   terrier-tools confusion-matrix --csv drosophila-terrier.final.TEs.csv  \
        --output drosophila-terrier-confusion-matrix-threshold-0.7.html \
        --threshold 0.7 \
        --map I-Jockey=I,Jockey-I=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC \
        --ignore Unknown

That will generate an HTML file like this:

.. raw:: html
    :file: ./images/drosophila-terrier-confusion-matrix-threshold-0.7.html

You can also output the confusion matrix with a .csv, .png, .svg, or .pdf extension by changing the extension of the output file.    

These results reflect the default Terrier threshold of 0.7. You can output the confusion matrix for different thresholds by changing the threshold value in the command above.

To see the effect of the threshold on the results, you can run the following command:

.. code-block:: bash

   terrier-tools threshold-plot --csv drosophila-terrier.final.TEs.csv  \
        --output drosophila-terrier-threshold-plot.html \
        --map I-Jockey=I,Jockey-I=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC \
        --ignore Unknown

That will generate an HTML file like this:

.. raw:: html
    :file: ./images/drosophila-terrier-threshold-plot.html


Rice genome
================

Bickmann et al. (2023) also provide Transposable Elements (TE) models of a rice genome. Download it with the following command:

.. code-block:: bash

   wget https://raw.githubusercontent.com/IOB-Muenster/TEclass2/refs/heads/main/tests/oryza.final.TEs.fa

Run inference using Terrier like this:

.. code-block:: bash

   terrier --file oryza.final.TEs.fa --output oryza-terrier.final.TEs.csv --threshold 0

.. note::

   We set the theshold to zero so that we can see all the predictions, the threshold can be adjusted in the evaluation steps below.

Now evaluate the results with the following command:

.. code-block:: bash

   terrier-tools evaluate --csv oryza-terrier.final.TEs.csv  \
        --threshold 0.7 \
        --map I-Jockey=I,Jockey-I=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC \
        --ignore Unknown

That will produce the following output:

.. code-block:: text

    Total: 75
    Total with ground truth: 75
    Number classified: 68/75 (90.67%)
    Correct predictions: 67/68 (98.53%)

To generate a confusion matrix, use the following command:

.. code-block:: bash

   terrier-tools confusion-matrix --csv oryza-terrier.final.TEs.csv  \
        --output oryza-terrier-confusion-matrix-threshold-0.7.html \
        --threshold 0.7 \
        --map I-Jockey=I,Jockey-I=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC \
        --ignore Unknown

That will generate an HTML file like this:

.. raw:: html
    :file: ./images/oryza-terrier-confusion-matrix-threshold-0.7.html

You can also output the confusion matrix with a .csv, .png, .svg, or .pdf extension by changing the extension of the output file.    

These results reflect the default Terrier threshold of 0.7. You can output the confusion matrix for different thresholds by changing the threshold value in the command above.

To see the effect of the threshold on the results, you can run the following command:

.. code-block:: bash

   terrier-tools threshold-plot --csv oryza-terrier.final.TEs.csv  \
        --output oryza-terrier-threshold-plot.html \
        --map I-Jockey=I,Jockey-I=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC \
        --ignore Unknown

That will generate an HTML file like this:

.. raw:: html
    :file: ./images/oryza-terrier-threshold-plot.html
