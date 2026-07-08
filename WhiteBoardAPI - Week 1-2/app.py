from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

FILE_NAME = "board.json"

if not os.path.exists(FILE_NAME):
    with open(FILE_NAME, "w") as file:
        json.dump({"lines": []}, file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/save", methods=["POST"])
def save_board():
    data = request.get_json()

    with open(FILE_NAME, "w") as file:
        json.dump(data, file)

    return jsonify({"message": "Board saved successfully!"})


@app.route("/load", methods=["GET"])
def load_board():
    with open(FILE_NAME, "r") as file:
        data = json.load(file)

    return jsonify(data)


@app.route("/clear", methods=["DELETE"])
def clear_board():
    with open(FILE_NAME, "w") as file:
        json.dump({"lines": []}, file)

    return jsonify({"message": "Board cleared!"})


if __name__ == "__main__":
    app.run(debug=True)