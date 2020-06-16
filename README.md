# Data Science Stuff
Data Science stuff that I find interesting are here.

## Machine Learning
* [BatchNorm](https://github.com/adimyth/datascience_stuff/blob/master/machine-learning/BatchNorm.ipynb)
* [AutoEncoders](https://github.com/adimyth/datascience_stuff/blob/master/machine-learning/AutoEncoders.ipynb) - Compares PCA vs AutoEncoders. Working example for Dense & Convolutional AutoEncoder
* [Variational AutoEncoders](https://github.com/adimyth/datascience_stuff/blob/master/machine-learning/VariationalAutoEncoders.ipynb) - Compares AutoEncoders vs VAEs for generative modelling & their latent spaces. Implements CVAE from tensorflow tutorial & visualizes the latent spaces
* [WhiteDistance](https://github.com/adimyth/datascience_stuff/blob/master/machine-learning/WhiteDistance.ipynb) - Metric to calculate similarity between texts

## Model Interpretation
* [LIME](https://github.com/adimyth/datascience_stuff/blob/master/model_interpretation/lime_multiclass.py)
* [Occlusion](https://github.com/adimyth/datascience_stuff/blob/master/model_interpretation/occlusion.ipynb)
* [CAM](https://github.com/adimyth/datascience_stuff/blob/master/model_interpretation/cam.ipynb)
* [Grad CAM](https://github.com/adimyth/datascience_stuff/blob/master/model_interpretation/grad_cam.ipynb)

## Neural Arithmetic Logic Unit (NALU)

## NLP
* [TextGeneration](https://github.com/adimyth/datascience_stuff/blob/master/nlp/TextGeneration.ipynb) - Copy of Karpathy's work really
* [Seq2Seq](https://github.com/adimyth/datascience_stuff/blob/master/nlp/Seq2Seq.ipynb) - Notebook for a word based Encoder-Decoder Seq2Seq network for English to Spanish translation
* [Attention](https://github.com/adimyth/datascience_stuff/blob/master/nlp/LanguageModelling.ipynb) - An extension of previous Seq2Seq work which uses `attention` mechanism for english to spanish translation
* [Transformers](https://github.com/adimyth/datascience_stuff/blob/master/nlp/Transformers.ipynb) - Implements individual components of a transformer from scratch

## Optimization
[Differential Evolution](https://github.com/adimyth/datascience_stuff/blob/master/optimization/DifferentialEvolution.ipynb)

Will add more methods as I study them

## Cool Stuffs
* [stitch_images](https://github.com/adimyth/datascience_stuff/blob/master/cool_stuffs/stitch_images.py) - Script to create a single image stitched horizontally or vertically from a list of images
* [csv_stats.sh](https://github.com/adimyth/datascience_stuff/blob/master/cool_stuffs/csv_stats.sh) - Simple shell script to give summary of all *csv* files
* [code_snipets](https://github.com/adimyth/datascience_stuff/blob/master/cool_stuffs/code_snippets.py) - Utility Script

## Translate App
A simple sanic app which translates english sentence to spanish. It uses an attention based Seq2Seq model. Try and run this app on your local machine - 
### How to run
1. Install tensorflow 2.0 and sanic
2. Clone the git repo
3. Navigate to *translate_app* directory
4. Run server on *localhost:5000/translate*. 
```python
python server.py
```
5. Evaluate by making curl request at the above url with `input_sentence` as a parameter. Pass in the desired English sentence as value with it. 
```bash
curl -X POST -H "Content-Type: application/json"  -d '{"input_sentence":"What did you decide?"}' "localhost:5000/translate"
```
6. Alternatively, you could run the script `make_requests.py` and change the sentences in `sentences` list.

More interesting stuff to follow

