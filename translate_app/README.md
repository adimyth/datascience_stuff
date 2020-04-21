# Translate App

A simple sanic app which translates english sentence to spanish. It uses the attention based Seq2Seq model trained. For more details refer to [Attention.ipynb](https://github.com/adimyth/interesting_stuff/blob/master/nlp/Attention.ipynb) or [Seq2Seq.ipynb](https://github.com/adimyth/interesting_stuff/blob/master/nlp/Seq2Seq.ipynb) in `nlp` directory.

## How to Run
1. Install tensorflow 2.0 and sanic
2. Clone the git repo
3. Navigate to translate_app directory
4. Run server on *localhost:5000/translate*. 
```python
python server.py
```
5. Evaluate by making curl request at the above url with `input_sentence` as a parameter. Pass in the desired English sentence as value with it. 
```bash
curl -X POST -H "Content-Type: application/json"  -d '{"input_sentence":"What did you decide?"}' "localhost:5000/translate"
```
6. Alternatively, you could run the script `make_requests.py` and change the sentences in `sentences` list.

