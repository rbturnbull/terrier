=============
Training
=============


After performing the instructions on the :ref:`preprocessing:Preprocessing` page, you will have a SeqBank and SeqTree file that you can use to train Terrier.

To train Terrier, you will need to use the `terrier-tools` CLI utility.

To use the same hyperparameters as in the main release of Terrier, you can run the following command:

.. code-block:: bash

    SEQBANK=$REPBASE_DIR/Repbase-seqbank.sb
    SEQTREE=$REPBASE_DIR/Repbase-seqtree.st
    terrier-tools train \
        --seqtree $SEQTREE \
        --seqbank $SEQBANK

You can see other command-line options by running:

.. code-block:: bash

    terrier-tools train --help

To reproduce the training process of the main release of Terrier, you can use the following command:

.. code-block:: bash

    terrier-tools train         \
        --seqtree $SEQTREE         \
        --seqbank $SEQBANK         \
        --max-learning-rate 0.001         \
        --macc 20000000000         \
        --cnn-layers 4 \
        --dropout 0.2479560973202271 \
        --embedding-dim 18 \
        --factor 1.959254226973812 \
        --kernel-size 7 \
        --penultimate-dims 1953 \
        --phi 1.0196823166741456 \
        --max-epochs 100 \
        --test-partition -2 \
        --validation-partition -1
