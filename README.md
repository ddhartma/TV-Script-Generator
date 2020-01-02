[//]: # (Image References)
[image1]: Train_16_Plot.png "Training loss as a function of batches (every 500 batches)"

# TV Script Generation
In this project, I generate my own Seinfeld TV scripts using RNNs. I am using part of the Seinfeld dataset of scripts from 9 seasons. The Neural Network which I built generates a new ,"fake" TV script, based on patterns it recognizes in this training data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- To run this script you will need to use a Terminal (Mac OS) or Command Line Interface (Git Bash on Windows).
- If you are unfamiliar with Command Line check the free [Shell Workshop](https://www.udacity.com/course/shell-workshop--ud206) lesson at Udacity.


### Installing

- Clone this project to your local machine
- Install jupyter notebook via

```
pip3 install --upgrade pip
```
Then install the Jupyter Notebook using:
```
pip3 install jupyter
```
(Use pip if using legacy Python 2.)

For further help installing Jupyter Notebook check [Jupyter-ReadTheDocs](https://jupyter.readthedocs.io/en/latest/install.html)


## Running the tests

The following files were used for this project:

- dlnd_tv_script_generation.ipynb
- helper.py
- problem_unittests.py
- workspace_utils.py
- data/Seinfeld_Scripts.txt

The data for RNN training process is provided in the text file  *data/Seinfeld_Scripts.txt*

In order to run the Jupyter Notebook *dlnd_tv_script_generation.ipynb* use the Terminal/CLI and write

```
jupyter notebook dlnd_tv_script_generation.ipynb
```

### Dataset Stats
- Roughly the number of unique words: 46367
- Number of lines: 109233
- Average number of words in each line: 5.544240293684143

### Hyperparameters

Set and train the neural network with the following parameters:

- Set **sequence_length** --- to the length of a sequence
- Set **batch_size** --- to the batch size
- Set **dropout (LSTM)** --- dropout probability for LSTM cell
- Set **dropout (Layer)** --- dropout probability for Dropout-Layer
- Set **num_epochs** --- to the number of epochs to train for
- Set **learning_rate** --- to the learning rate for an Adam optimizer
- Set **vocab_size** --- to the number of uniqe tokens in our vocabulary
- Set **output_size** --- to the desired size of the output
- Set **embedding_dim** --- to the embedding dimension; smaller than the vocab_size
- Set **hidden_dim** --- to the hidden dimension of your RNN
- Set **n_layers** --- to the number of layers/cells in your RNN
- Set **show_every_n_batches** --- to the number of batches at which the neural network should print progress

### Parameter: num_epochs

Parameters          | 5th Train-Run | 6th Train-Run |
----------          | ------------- | ------------- |
sequence_length     |   5           |   5           |  
batch_size          |   128         |   128         |   
dropout (LSTM)      |   0.5         |   0.5         |  
dropout (Layer)     |   no dropout  |   no dropout  |   
num_epochs          |   `20`        |   `5`         |   
learning_rate       |   0.001       |   0.001       |
vocab_size          |   vocab_size  |   vocab_size  |   
output_size         |   vocab_size  |   vocab_size  |   
embedding_dim       |   200         |   200         |  
hidden_dim          |   256         |   256         |  
n_layers            |   2           |   2           |  
show_every_n_batches|   500         |   500         |   
**Loss**            |   **3.26**    |   **3.65**     

### Parameter: learning_rate

Parameters          | 2nd Train-Run | 3rd Train-Run |
----------          | ------------- | ------------- |
sequence_length     |   5           |   5           |  
batch_size          |   128         |   128         |   
dropout (LSTM)      |   0.5         |   0.5         |   
dropout (Layer)     |   0.5         |   0.5         |   
num_epochs          |   5           |   5           |  
learning_rate       |   `0.001`     |   `0.01`      |   
vocab_size          |   vocab_size  | vocab_size    |   
output_size         |   vocab_size  |   vocab_size  |  
embedding_dim       |   400         |   400         |  
hidden_dim          |   512         |   512         |   
n_layers            |   2           |   2           |   
show_every_n_batches|   500         |   500         |   
**Loss**            |   **3.96**    |   **4.89**    |   

### Parameter: batch_size

Parameters          | 7th Train-Run | 6th Train-Run | 8th Train-Run | 9th Train-Run |
----------          | ------------- | ------------- | ------------- | ------------- |
sequence_length     |   5           |   5           |   5           |   5      
batch_size          |   `64`        |   `128`       |   `256`       |   `512`
dropout (LSTM)      |   0.5         |   0.5         |   0.5         |   0.5     
dropout (Layer)     |   no dropout  |   no dropout  |   no dropout  |   no dropout  
num_epochs          |   5           |   5           |   5           |   5     
learning_rate       |   0.001       |   0.001       |   0.001       |   0.001
vocab_size          |   vocab_size  |   vocab_size  |   vocab_size  |   vocab_size
output_size         |   vocab_size  |   vocab_size  |   vocab_size  |   vocab_size
embedding_dim       |   200         |   200         |   200         |   200
hidden_dim          |   256         |   256         |   256         |   256
n_layers            |   2           |   2           |   2           |   2
show_every_n_batches|   500         |   500         |   500         |   500
**Loss**            |   **3.81**    |   **3.65**    |   **3.63**    |   **3.74**

### Parameter: embedding_dim

Parameters          | 8th Train-Run | 10th Train-Run | 11th Train-Run |
----------          | ------------- | -------------  | -------------  |
sequence_length     |   5           |   5            |  5             |
batch_size          |   256         |   256          |  256           |
dropout (LSTM)      |   0.5         |   0.5          |  0.5
dropout (Layer)     |   no dropout  |   no dropout   |  no dropout
num_epochs          |   5           |   5            |  5
learning_rate       |   0.001       |   0.001        |  0.001
vocab_size          |   vocab_size  |   vocab_size   |  vocab_size
output_size         |   vocab_size  |   vocab_size   |  vocab_size
embedding_dim       |   `200`       |   `400`        |  `600`         |
hidden_dim          |   256         |   256          |  256           |
n_layers            |   2           |   2            |  2
show_every_n_batches|   500         |   500          |  500
**Loss**            |   **3.63**    |   **3.58**     |  **3.60**

### Parameter: hidden_dim

Parameters          | 13th Train-Run| 8th Train-Run  | 12th Train-Run|
----------          | ------------- | -------------  | -------------  |
sequence_length     |   5           |   5            |  5             |
batch_size          |   256         |   256          |  256           |
dropout (LSTM)      |   0.5         |   0.5          |  0.5
dropout (Layer)     |   no dropout  |   no dropout   |  no dropout
num_epochs          |   5           |   5            |  5
learning_rate       |   0.001       |   0.001        |  0.001
vocab_size          |   vocab_size  |   vocab_size   |  vocab_size
output_size         |   vocab_size  |   vocab_size   |  vocab_size
embedding_dim       |   200         |   200          |  200           |
hidden_dim          |   `128`       |   `256`        |  `512`
n_layers            |   2           |   2            |  2
show_every_n_batches|   500         |   500          |  500
**Loss**            |   **3.80**    |   **3.63**     |  **3.34**

### Parameter: sequence_length

Parameters          | 8th Train-Run | 14th Train-Run | 15th Train-Run|
----------          | ------------- | -------------  | -------------  |
sequence_length     |   `5`         |   `10`         |  `15`          |
batch_size          |   256         |   256          |  256           |
dropout (LSTM)      |   0.5         |   0.5          |  0.5
dropout (Layer)     |   no dropout  |   no dropout   |  no dropout
num_epochs          |   5           |   5            |  5
learning_rate       |   0.001       |   0.001        |  0.001
vocab_size          |   vocab_size  |   vocab_size   |  vocab_size
output_size         |   vocab_size  |   vocab_size   |  vocab_size
embedding_dim       |   200         |   200          |  200           |
hidden_dim          |   256         |   256          |  256
n_layers            |   2           |   2            |  2
show_every_n_batches|   500         |   500          |  500
**Loss**            |   **3.63**    |   **3.61**       |  **3.59**

### Influence of an additional Dropout layer

Parameters          | 4th Train-Run | 5th Train-Run |
----------          | ------------- | ------------- |
sequence_length     |   10          |   10          |   
batch_size          |   128         |   128         |   
dropout (LSTM)      |   0.5         |   0.5         |  
dropout (Layer)     |   `0.3`       |   `no dropout`|   
num_epochs          |   20          |   20          |  
learning_rate       |   0.001       |   0.001       |   
vocab_size          |   vocab_size  |   vocab_size  |  
output_size         |   vocab_size  |   vocab_size  |   
embedding_dim       |   200         |   200         |   
hidden_dim          |   256         |   256         |   
n_layers            |   2           |   2           |  
show_every_n_batches|   500         |   500         |   
**Loss**            |   **3.65**    |   **3.26**    |   

The influence of an additional Dropout Layer before the Fully Connected Layer can be studied between the 4th and 5th Train-Run. In the 4th Train-Run an additional Dropout-Layer with a Dropout Probability of 0.3 was used. For the 5th Train-Run the same parameter set as for the 4th Train-Run was used, however this additional Dropout layer was left out. As it can be seen from the Loss-vs-Batches-Plot this additional Dropout layer does not lead to an improvement in performance, as the loss increases.


### Tested best-of hyperparameter setting

Parameters          | 16th Train-Run |
----------          | ------------- |
sequence_length     |   15          |   
batch_size          |   256         |
dropout (LSTM)      |   0.5         |   
dropout (Layer)     |   no dropout  |
num_epochs          |   20          |  
learning_rate       |   0.001       |   
vocab_size          |   vocab_size  |   
output_size         |   vocab_size  |   
embedding_dim       |   400         |   
hidden_dim          |   512         |  
n_layers            |   2           |   
show_every_n_batches|   500         |  
**Loss**            |   **2.63**     |   

### Training loss (tested best-of hyperparameter setting) as a function of batches

![image1]

## Acknowledgments

* README was inspired by https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
