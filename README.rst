.. image:: https://raw.githubusercontent.com/rbturnbull/terrier/main/docs/images/terrier-banner.png

.. start-badges

|pypi badge| |colab badge| |testing badge| |docs badge| |black badge| |torchapp badge| |doi badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/bio-terrier?color=blue
   :alt: PyPI - Version
   :target: https://pypi.org/project/bio-terrier/

.. |testing badge| image:: https://github.com/rbturnbull/terrier/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/terrier/actions

.. |docs badge| image:: https://github.com/rbturnbull/terrier/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/terrier
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/5e0c3115955fde132a8b7c131da68b86/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/terrier/coverage/

.. |torchapp badge| image:: https://img.shields.io/badge/torch-app-B1230A.svg
    :target: https://rbturnbull.github.io/torchapp/

.. |colab badge| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/rbturnbull/terrier/blob/main/terrier_colab.ipynb

.. |doi badge| image:: https://img.shields.io/badge/DOI-10.1093%2Fbib%2Fbbaf442-blue
   :target: https://doi.org/10.1093/bib/bbaf442
    
.. end-badges

.. start-quickstart

Transposable Element Repeat Result classifIER

Terrier is a Neural Network model to classify transposable element sequences.

It is based on ‘corgi’ which was trained to do hierarchical taxonomic classification of DNA sequences.

This model was trained using the Repbase library of repetitive DNA elements and trained to do hierarchical classification according to the RepeatMasker schema.

An online version of Terrier (using CPUs only) is available at `https://portal.cpg.unimelb.edu.au/tools/terrier <https://portal.cpg.unimelb.edu.au/tools/terrier>`_.

Installation
==================================

Install using pip:

.. code-block:: bash

    pip install bio-terrier

.. warning ::

    Do not try just ``pip install terrier`` because that is a different package.

Or install the latest version from GitHub:

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/terrier.git


Google Colab Version
==================================

Follow this link to launch a Google Colab notebook where you can run the model on your own data: |colab badge2|

.. |colab badge2| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/rbturnbull/terrier/blob/main/terrier_colab.ipynb

Usage
==================================

To run inference on a FASTA file, run this command:

.. code-block:: bash

    terrier --file INPUT.fa --output-fasta OUTPUT.fa

That will add the classification to after the sequence ID in the `OUTPUT.fa` FASTA file.

If you want to save the probabilities for all classes run this:

.. code-block:: bash

    terrier --file INPUT.fa --output-csv OUTPUT.csv

The columns will be the probability of each classification and the rows correspond to each sequence in ``INPUT.fa``.

If you want to output a visualization of the prediction probabilities:

.. code-block:: bash

    terrier --file INPUT.fa --image-dir OUTPUT-IMAGES/

The outputs for the above can be combined together. For more options run 

.. code-block:: bash

    terrier --help

To see the options to train the model, run:

.. code-block:: bash

    terrier-tools --help

Programmatic Usage
==================================

You can also use the model programmatically:

.. code-block:: python

    from terrier import Terrier

    terrier = Terrier()
    terrier(file="INPUT.fa", output_fasta="OUTPUT.fa")


Potential Use Case
==================================

A potential workflow is to use `RepeatModeler <https://github.com/Dfam-consortium/RepeatModeler>`_ first to generate a repeat library.
Then you can use Terrier to attempt to classify the remaining unknown repeats. 
If you only want highly confident classifications from Terrier, you can set the threshold to 0.9 or higher.
If you wish to have more coverage, then you can set the threshold lower (or keep it at the default value of 0.7). 
The modified repeat library can then be used with `RepeatMasker <http://www.repeatmasker.org/>`_ to mask the repeats in your genome assembly.

.. end-quickstart


Credits
==================================

.. start-credits

Terrier was developed by:

- `Robert Turnbull <https://robturnbull.com>`_
- `Neil D. Young <https://findanexpert.unimelb.edu.au/profile/249669-neil-young>`_
- `Edoardo Tescari <https://findanexpert.unimelb.edu.au/profile/428364-edoardo-tescari>`_
- `Lee F. Skerratt <https://findanexpert.unimelb.edu.au/profile/451921-lee-skerratt>`_
- `Tiffany A. Kosch <https://findanexpert.unimelb.edu.au/profile/775927-tiffany-kosch>`_

If you use this software, please cite the following preprint:

    Robert Turnbull, Neil D. Young, Edoardo Tescari, Lee F. Skerratt, and Tiffany A. Kosch. (2025). 'Terrier: A Deep Learning Repeat Classifier'. `arXiv:2503.09312 <https://arxiv.org/abs/2503.09312>`_.

`Wytamma Wirth <https://wytamma.com/>`_ set up Terrier as a tool at the `Centre for Pathogen Genomics Portal <https://portal.cpg.unimelb.edu.au/>`_ at the University of Melbourne.

This command will generate a bibliography for the Terrier project.

.. code-block:: bash

    terrier --bibliography

Here it is in BibTeX format:

.. code-block:: bibtex

    @article{terier,
        author = {Turnbull, Robert and Young, Neil D and Tescari, Edoardo and Skerratt, Lee F and Kosch, Tiffany A},
        title = {Terrier: a deep learning repeat classifier},
        journal = {Briefings in Bioinformatics},
        volume = {26},
        number = {4},
        pages = {bbaf442},
        year = {2025},
        month = {08},
        abstract = {Repetitive DNA sequences underpin genome architecture and evolutionary processes, yet they remain challenging to classify accurately. Terrier is a deep learning model designed to overcome these challenges by classifying repetitive DNA sequences using a publicly available, curated repeat sequence library trained under the RepeatMasker schema. Poor representation of taxa within repeat databases often limits the classification accuracy and reproducibility of current repeat annotation methods, limiting our understanding of repeat evolution and function. Terrier overcomes these challenges by leveraging deep learning for improved accuracy. Trained on Repbase, which includes over 100,000 repeat families—four times more than Dfam—Terrier maps 97.1\% of Repbase sequences to RepeatMasker categories, offering the most comprehensive classification system available. When benchmarked against DeepTE, TERL, and TEclass2 in model organisms (rice, fruit flies, humans, and mice), Terrier achieved superior accuracy while classifying a broader range of sequences. Further validation in non-model amphibian, flatworm, and Northern krill genomes highlights its effectiveness in improving classification in non-model species, facilitating research on repeat-driven evolution, genomic instability, and phenotypic variation.},
        issn = {1477-4054},
        doi = {10.1093/bib/bbaf442},
        url = {https://doi.org/10.1093/bib/bbaf442},
        eprint = {https://academic.oup.com/bib/article-pdf/26/4/bbaf442/64143069/bbaf442.pdf},
    }

Run the following command to get the latest BibTeX entry:

.. code-block:: bash

    terrier --bibtex


This will be updated with the final publication details when available.



Created using torchapp (https://github.com/rbturnbull/torchapp).

.. end-credits

