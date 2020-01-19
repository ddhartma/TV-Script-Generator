[//]: # (Image References)
[image1]: Train_16_Plot.png "Training loss as a function of batches (every 500 batches)"

# TV Script Generation
In this project, I generate my own Seinfeld TV scripts using Recurrent Neural Networks (RNNs). I am using part of the Seinfeld dataset of scripts from 9 seasons. The Neural Network which I built generates a new ,"fake" TV script, based on patterns it recognizes in this training data.  

In more detail, I am using word embedding (Lookup Tables) and Recurrent Neural Networks based on Long-Term-Short-Term (LSTM) memory cells.
The code is written in PyTorch. If you are interested in the specific coding steps open the Jupyter Notebook *dlnd_tv_script_generation.ipynb*. A description to *open/execute* the file is provided in the Getting Started section.

## Typical TV script data (original script)
This is a typical script abstract from the dataset which I used for training the neural network.

**george:** are you through?  
**jerry:** you do of course try on, when you buy?  
**george:** yes, it was purple, i liked it, i dont actually recall considering the buttons.   
**jerry:** oh, you dont recall?   
**george:** (on an imaginary microphone) uh, no, not at this time.   
**jerry:** well, senator, id just like to know, what you knew and when you knew it.   
**claire:** mr. seinfeld. mr. costanza.   
**george:** are, are you sure this is decaf? wheres the orange indicator?   
**claire:** its missing, i have to do it in my head decaf left, regular right, decaf left, regular right...its very challenging work.   
**jerry:** can you relax, its a cup of coffee. claire is a professional waitress.   
**claire:** trust me george. no one has any interest in seeing you on caffeine.   
**george:** how come youre not doing the second show tomorrow?   
**jerry:** well, theres this uh, woman might be coming in.   
**george:** wait a second, wait a second, what coming in, what woman is coming in?   
**jerry:** i told you about laura, the girl i met in michigan?   
**george:** no, you didnt!   

## Typical output "fake" TV script data
Here I provide the TV "fake" script result after the training was finished. This script output is based on the best-of hyperparameter setting (see 'Tested best-of hyperparameter setting'):

**george:**(to jerry) hey, how 'bout giving the computer?(to kramer) i don't know, i'm not buying a soak with her. shes questioning the flesh- daily suspension.  
**kramer:** yeah? well, i gotta go, i gotta get the cloth out, and then i got the one i can embrace.  
**jerry:** oh, i don't want you to do this.  
**jerry:** no, i'm not.  
**george:** well, what did she say?  
**jerry:** i don't know.  
**kramer:** i can't. i'm going to do something, i'm gonna call nbc and talk to him.  
**george:**(smiling) oh...  
**george:** you know, i really should go upstairs, but i can't tell you what you do.  
**george:** i don't want to.  
**jerry:** no.  
**kramer:**(looking around) well, i'm sorry.  
**jerry:** oh, i'm sorry, i just wanted to see what mary todd wore for you, but...  
**kramer:**(interrupting) alright, alright, alright.  
**jerry:** bye shmoopy.  
**george:** i am not calling her.(to jerry) i can't believe it. you know, i was wondering if i had a little tiff here, i would have picked it up at puddys  
**jerry:** hey jerry, you know i think i was interested in availing myself......  
**elaine:** you know, if i wasn't, you know, i don't like the sound of this. you know, i just cashed my nana's birthday checks.  
**jerry:** i don't know what i can do, and i know, i was just curious.  
**jerry:** you know, i'm thinking about this, but i can't afford to do anything.  
**jerry:** well, you don't have a proper workstation.  
**george:** oh, yeah.  
**elaine:**(quietly) i can't believe you called me going to do it  

You can see that there are multiple characters that say (somewhat) complete sentences. As the focus is on the algorithm and the principle modeling architecture the generated output does not need to be be perfect! It takes quite a while to get good results, and often, you'll have to use a smaller vocabulary (and discard uncommon words), or get more data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing (via pip)

- To run this script you will need to use a Terminal (Mac OS) or Command Line Interface (e.g. Git Bash on Windows [git](https://git-scm.com/)), if you don't have it already.
- If you are unfamiliar with Command Line coding check the free [Shell Workshop](https://www.udacity.com/course/shell-workshop--ud206) lesson at Udacity.

Next, clone this repository by opening a terminal and typing the following commands:

```
$ cd $HOME  # or any other development directory you prefer
$ git clone https://github.com/ddhartma/TV-Script-Generator.git
$ cd project-tv-script-generation_cloned_2
```
A Python installation is needed. Python 3 is already preinstalled on many systems nowadays. You can check which version you have by typing the following command (you may need to replace python3 with python):

```
$ python3 --version  # for Python 3
```
A Python 3.5 version or above should be fine. If you don't have Python 3, you can just download it from [python.org](https://www.python.org/downloads/).

You need to install several scientific Python libraries that are necessary for this project, in particular NumPy, Matplotlib, Pandas, Jupyter Notebook, Torch and Torchvision. For this, you can either use Python's integrated packaging system, pip, or you may prefer to use your system's own packaging system (if available, e.g. on Linux, or on MacOSX when using MacPorts or Homebrew). The advantage of using pip is that it is easy to create multiple isolated Python environments with different libraries and different library versions (e.g. one environment for each project). The advantage of using your system's packaging system is that there is less risk of having conflicts between your Python libraries and your system's other packages. Since I have many projects with different library requirements, I prefer to use pip with isolated environments.

These are the commands you need to type in a terminal if you want to use pip to install the required libraries. Note: in all the following commands, if you chose to use Python 2 rather than Python 3, you must replace pip3 with pip, and python3 with python.

First you need to make sure you have the latest version of pip installed:

```
$ python3 -m pip install --user --upgrade pip
```
The ```--user``` option will install the latest version of pip only for the current user. If you prefer to install it system wide (i.e. for all users), you must have administrator rights (e.g. use sudo python3 instead of python3 on Linux), and you should remove the ```--user``` option. The same is true of the command below that uses the ```--user``` option.

Next, you can optionally create an isolated environment. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially very different libraries, and different versions:

```
$ python3 -m pip install --user --upgrade virtualenv
$ python3 -m virtualenv -p `which python3` env
```

This creates a new directory called env in the current directory, containing an isolated Python environment based on Python 3. If you installed multiple versions of Python 3 on your system, you can replace ```which python3``` with the path to the Python executable you prefer to use.

Now you must activate this environment. You will need to run this command every time you want to use this environment.

```
$ source ./env/bin/activate
```

On Windows, the command is slightly different:

```
$ .\env\Scripts\activate
```

Next, use pip to install the required python packages. If you are not using virtualenv, you should add the --user option (alternatively you could install the libraries system-wide, but this will probably require administrator rights, e.g. using sudo pip3 instead of pip3 on Linux).

```
$ python3 -m pip install --upgrade -r requirements.txt
```

Great! You're all set, you just need to start Jupyter now.

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

For further help regarding Jupyter Notebook check [Jupyter-ReadTheDocs](https://jupyter.readthedocs.io/en/latest/index.html)

### Dataset Stats
This is a short summary of the characteristics of the Seinfeld_Scripts.txt file:
- The number of unique words: 46367
- Number of lines: 109233
- Average number of words in each line: 5.544240293684143

### Hyperparameters
The Neural network training success depends strongly on the right choice of hyperparameters. If the network isn't getting the desired results, tweak these parameters and/or the layers in the RNN class.
Hence, set and train the neural network with the following parameters:

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

In the following section the parameter tuning investigated in separated training runs is depicted. Furthermore the resulting **Loss** for each training run (parameter variation) is shown as a measure of training success.

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
The following table shows the best-of setting of the parameter tuning.

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
In the following picture the training loss for the best-off hyperparameter setting is shown as function of cumulated batches. As from the tabel above the batch_size was set to 256 and the number of epochs to 20. The training loss is remarkably low with 2.63 after more than 60000 batches of training.

![image1]

### Reasons for the chosen best-of hyperparameter setting:
- A sequence_length of 15 was chosen accordingly to the number of words in typical dialog sentences.  
- A batch_size of 256 was chosen based on a trade-off of
    - speed up training (getting better for higher batch_sizes),
    - no out-of-memory errors (getting worse for higher batch_sizes)
    - good training loss  
- Leave out additional dropout layers: An additional dropout layer right before the fully connected layer increased the training loss significantly form (3.26 to 3.65).
- An increase of the num_epochs value reduces the training error. For num_epochs=20 a training loss goal of <3.5 was achieved.
- The embedding_dim (see table Parameter: embedding_dim) in the tested regime between 200 and 600 had no strong influence on the training loss. Hence an embedding_dim of 400 was arbitrarily chosen.
- Increasing the hidden_dim from 128 to 512 showed a significant influence on the training loss (see table Parameter: hidden_dim). The training loss decreased from 3.80 down to 3.34. Hence for the final test a hidden_dim of 512 was chosen.
- Two LSTM layer were stacked as typical values for LSTM stacking are 2 or 3.

## Acknowledgments

* README was inspired by https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
