from app import decode_sequence
import config
from sanic import Sanic
from sanic import response
from tensorflow.keras.models import load_model

app = Sanic(__name__)
url = config.url
print("Loading trained models....")
model = load_model(config.model_path)
encoder = load_model(config.encoder_path)
decoder = load_model(config.decoder_path)


@app.route("/translate", methods=["POST"])
async def translate_seq(request):
    english_seq = request["input_sentence"]
    print(f"Input English Sentence: {english_seq}")
    spanish_seq = decode_sequence(model, encoder, decoder, english_seq)
    print(f"Predicted Spanish Sentence: {spanish_seq}")
    act_spa_seq = get_spanish_translation(english_seq)
    print(f"Actual Spanish Sentence: {act_spa_seq}")
    spa_to_eng = get_english_translation(spanish_seq)
    print(f"Predicted Spanish to English Sentence: {spa_to_eng}")
    return response.json({"english_seq": english_seq, 
        "spanish_seq": spanish_seq, 
        "act_spa_seq": act_spa_seq, 
        "spa_to_eng": spa_to_eng})


def get_english_translation(seq):
    response = requests.post(url, data={"q": seq, "langpair": "es|en"})
    translated_text = response.json()["responseData"]["translatedText"]
    return translated_text


def get_spanish_translation(seq):
    response = requests.post(url, data={"q": seq, "langpair": "en|es"})
    translated_text = response.json()["responseData"]["translatedText"]
    return translated_text


if __name__ == "__main__":
    app.run(host="localhost", port=5000)

