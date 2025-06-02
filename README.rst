.. image:: https://raw.githubusercontent.com/rbturnbull/terrier/main/docs/images/terrier-banner.png

.. start-badges

|pypi badge| |colab badge| |testing badge| |docs badge| |black badge| |torchapp badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/terrier?color=blue
   :alt: PyPI - Version
   :target: https://pypi.org/project/terrier/

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

    
.. end-badges

.. start-quickstart

Transposable Element Repeat Result classifIER

Terrier is a Neural Network model to classify transposable element sequences.

It is based on ‘corgi’ which was trained to do hierarchical taxonomic classification of DNA sequences.

This model was trained using the Repbase library of repetitive DNA elements and trained to do hierarchical classification according to the RepeatMasker schema.

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

This command will generate a bibliography for the Terrier project.

.. code-block:: bash

    terrier --bibliography

Here it is in BibTeX format:

.. code-block:: bibtex

    @article{terrier,
        title = {{Terrier: A Deep Learning Repeat Classifier}},
        author = {Turnbull, Robert and Young, Neil D. and Tescari, Edoardo and Skerratt, Lee F. and Kosch, Tiffany A.},
        year = {2025},
        journal = {arXiv},
        url = {https://arxiv.org/abs/2503.09312},
        doi = {10.48550/arXiv.2503.09312}
    }

Run the following command to get the latest BibTeX entry:

.. code-block:: bash

    terrier --bibtex


This will be updated with the final publication details when available.



Created using torchapp (https://github.com/rbturnbull/torchapp).

.. end-credits

