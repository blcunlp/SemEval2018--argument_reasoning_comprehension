# argument_reasoning_code

This repository provides the implement of the model described in the paper *BLCU_NLP at SemEval-2018 Task 12: An Ensemble Model for Argument Reasoning Based on Hierarchical Attention*. 

The experiment is based on TensorFlow platform.

## modules
### hierarchical model
The model_code directory contains the train/dev/test data and codes of hierarchical attention model. 
To get the predict answer, run
```
    $ cd model_code
    $ python model_tf.py
    
```    
The answer will be saved as test_answer in the current directory.

### ensemble model
The ensemble directory contains scripts for majority voting. Put the different dev answer files into dev_answers directory, and run 
    $ perl voting_dev.pl 
The accuracy after voted will be print on the screen, and the new answer file will be saved as dev_voted_answer 
    
Put the different test answer files into dev_answers directory, and run 
    $ perl voting_test.pl 
The accuracy after voted will be print on the screen, and the new answer file will be saved as test_voted_answer
