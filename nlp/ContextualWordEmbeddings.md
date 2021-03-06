# Contextual Word Embeddings

Word Embeddings like Word2Vec,Glove etc. provide a fixed meaning to words. This is a major drawback of word embeddings as the meaning of words are based on context, thus wasn't much helpful for Language Modelling

For example, the word *plane* has different meaning in each of the following sentences-
1. The plane took off at exactly nine o’clock.
2. The plane surface is a must for any cricket pitch.
3. Plane geometry is fun to study.

Hence, there's a need for contextualized word embeddings.

At the core of the recent advances in NLP lies the paradigm shift: going from initializing just the first layer of our model to pretraining the entire model with hierarchical representations. If learning word vectors is like only learning edges, these approaches are like learning the full hierarchy of features, from edges to shapes to high-level semantic concepts.

## Solution
Running a RNN over the word embeddings provides a hidden vector for each word. These hidden representations essentially give ***"context-specific representation of words"***. Essentially when we are running any language model, we generate context dependent representations. 

## ELMo
Unlike Glove or Word2Vec, ELMo looks at the entire sentence before assigning each word an embedding.
The main idea of the Embeddings from Language Models (ELMo) can be divided into two main tasks, first we train an LSTM-based language model on some corpus, and then we use the hidden states of the LSTM for each token to generate a vector representation of each word.

### Features
1. ELMo uses character based embeddings, which allows network to form robust representations for *out-of-vocabulary* word unseen during the training.
2. It generates word vectors *during run-time*.
3. It gives embedding of anything you put in — characters, words, sentences, paragraphs, but it is built for sentence embeddings in mind.

### Observations
The two BiLSTM NLM layers have different uses/meanings
* Lower layer is useful for lower-level syntax such as pos tagging, syntactic dependencies, NER
* Higher layer is better for higher-level semantics such as sentiment, question answering etc.

## ULMfit 
* Howard & Ruder(2018)
* Universal Language Model Fine-tuning for Text Classication
* Transfer the learning of a big language model for text classification task

3 stages involved
1. Train a LM on big general corpus (using BiLSTM)
2. Fine-tune this LM on target domain data/corpus
3. Classifier fine-tuning. Convert the model from LM to text classifier.

If we train the LM on large amount of unsupervised data then we will be able to do better on supervised task even with lesser data.
![ULMfit](https://www.bualabs.com/wp-content/uploads/2019/08/ulmfit_imdb.png)

## BERT
Bidirectional Encoder Representations from Transformers

### Advantages
The output of BERT training is the trained model and vectors - not just vectors. The trained model is used
* as is for generating word or sentence vectors (which are not only of the same quality as the older models like word2vec/Glove, but combine some of their useful characteristics - e.g. OOV/misspell handling)
* or analogous to reusing CNN models trained on images for downstream tasks like object detection, reusing the trained language model with the addition of a layer or two, for downstream supervised tasks like NER. This reduces the need for large amounts of labeled data for the specific supervised task.
* trained BERT model can be used for a wide variety of tasks with just very little additional architectural add-on and fine-tuned for specific tasks such as NER, Q&A etc. This eliminates the need for task specific architectures that we needed before.

## How Contextual are Contextualized Word Representations?
1. In all layers of BERT, ELMo, and GPT-2, the representations of all words are anisotropic: they occupy a narrow cone in the embedding space instead of being distributed throughout.
2. The models contextualize words very differently from one another.
3. If a word’s contextualized representations were not at all contextual, we’d expect 100% of their variance to be explained by a static embedding. Instead, on average - less than 5% of the variance can be explained by a static embedding.


## General Difference
Word2Vec and Glove vectors can be used directly for downstream tasks. All we need is the vectors for the words. There is no need for the model itself that was used to train these vectors.

ELMo & BERT are context dependent, hence we need the model that was used during training even for inference. We can just use the context independent vectors for a word if we choose too, but that would defeat the purpose of these models.

* Glove & Word2Vec - Word based
* ELMo - Character based
* BERT - Subword based 


