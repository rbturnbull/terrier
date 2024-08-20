=============
Training
=============


After performing the instructions on the :ref:`preprocessing:Preprocessing` page, you will have a SeqBank and SeqTree file that you can use to train Terrier.

To train Terrier, you will need to use the `terrier-tools` CLI utility.

To use the same hyperparameters as in the main release of Terrier, you can run the following command:

.. code-block:: bash

    terrier-tools train \
        --seqtree $REPBASE_DIR/RepBase29.07.st \
        --seqbank $REPBASE_DIR/RepBase29.07.sb \
        --learning-rate 0.001 \
        --macc 10000000000 \
        --epochs 130 \
        --output-dir outputs/$RUN_NAME 

You can see the command-line options by running:

.. code-block:: bash

    terrier-tools train --help