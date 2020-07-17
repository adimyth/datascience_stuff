import requests

url = "http://127.0.0.1:5000/translate"
with open("text", "r") as file:
    sentences = file.read().splitlines()

if __name__ == "__main__":
    for seq in sentences:
        print("="*60)
        data = {}
        data["input_sentence"] = seq
        data["Content-Type"] = "application/json"
        response = requests.post(url, json=data)
        response_json = response.json()
        print(f"Input English Sentence: {response_json['english_seq']}")
        print(
            f"Predicted Spanish Sentence: {response_json['spanish_seq'][:-4]}")
        print(f"Actual Spanish Sentence: {response_json['act_spa_seq']}")
        print(
            f"Predicted Spanish to English Sentence: {response_json['spa_to_eng']}")
        print("\n\n")
