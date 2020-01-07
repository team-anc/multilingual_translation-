# Multilingual Translation
NMT with Fasttext Embedding using TensorFlow (Encoder Attention Decoder Architecture)

We’ve implemented a Seq-to-Seq Model (Encoder-Decoder-Attention Mechanism) on tensorflow backend for text to text translation of languages like Sinhalese, Persian, Chinese (simplified) and Indonesian. We’ve used pretrained Fasttext models to generate word embeddings.


## Requirements
* jieba
(to segment Chinese data)

pip install jieba
* hazm
(to segment Persian data)

pip install hazm
* TextCleaner
(to clean Chinese data)

pip install TextCleaner


## Model Architecture
* Encoder: Single Layer GRU
* Attention: *bahdanau* Style
* Decoder: Single Layer GRU
* Optimiser: Adam
* Loss Function: Sparse Categorical Entropy



## Overview of the Model
The language of the input sentences detected using pretrained fasttext model. Input sentences are sent to the encoder word by word after which a hidden vector is obtained which becomes input to the decoder. The attention mechanism ensures that each encoder hidden state is assigned a unique weight in every timestep. The hidden state obtained from the encoder is processed by the decoder and later passed into a softmax layer which determines the final probability vector.
The word having the maximum probability becomes the next word in the sentence.


## Datasets
Chinese Dataset was obtained from https://www.manythings.org/anki/
The other language datasets were obtained by using the English phrases (present in the Chinese datasets) and putting them through Google Translate to obtain outputs of Sinhalese, Persian, Indonesian and Chinese.

### Load Datasets 
Ensure that the datasets are uploaded on your google drive and code is opened on Google Colabs. After mounting your drive, copy the id of the dataset file name and replace it with the id existing in the code. In case you change the file names, remember to replace that too. The same is applicable to pretrained Fasttext Models.


## Preprocessing Language Corpuses
Punctuations, numbers were removed from the language datasets.
In order to denote the beginning and end of each sentence, <start> and <end> were appended to each sentence of all the datasets.  
Vocabulary was created and sentences were tokenized using the Keras tokenizer

### Segmentation of Chinese and Persian Dataset
Modules like jieba and hazm were used to segment Chinese and Persian data respectively as white spaces aren’t present in these languages. 


## Training the Model
Models were trained on Google Colabs. Our models were trained on an average of 20,000 sentence pairs. The model was trained using batch training. To produce good results, each model should be trained on minimum of 10 epochs. After every second epoch the model was saved as a checkpoint. While training, teacher forcing is used.

## Inference
Using the *evaluate* function, translated sentences are obtained.
