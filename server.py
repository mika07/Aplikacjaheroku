from classification_nazwa_emocji2 import Classificator
from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
app = Flask(__name__)


em4_feature_path = os.path.join('resources', 'feature_matrix_4_4_emotions.csv')
em4_classes_path = os.path.join('resources', '4_feature_4_emotions.json')
em5_feature_path = os.path.join('resources', 'feature_matrix_3.csv')
em5_classes_path = os.path.join('resources', '3_feature.json')
em6_feature_path = os.path.join('resources', 'test10.csv')
em6_classes_path = os.path.join('resources', '10_feature.json')


@app.route('/emotions_4', methods=['POST'])
def emotions_4():
    image = decode_image(request.files['image'])
    Classificator.CreateClassificator(em4_feature_path, em4_classes_path)
    response = analyze_image(image, 4)

    return jsonify(response)


@app.route('/emotions_5', methods=['POST'])
def emotions_5():
    image = decode_image(request.files['image'])
    Classificator.CreateClassificator(em5_feature_path, em5_classes_path)
    response = analyze_image(image, 5)

    return jsonify(response)


@app.route('/emotions_6', methods=['POST'])
def emotions_6():
    image = decode_image(request.files['image'])
    Classificator.CreateClassificator(em6_feature_path, em6_classes_path)
    response = analyze_image(image, 6)

    return jsonify(response)


def decode_image(image_storage):
    """
    Converts FileStorage to numpy's ndarray
    :param image_storage: <FileStorage> Flask FileStorage with image data
    :return: <ndarray> representing the image
    """
    byte_data = image_storage.read()
    npimg = np.fromstring(byte_data, np.uint8)

    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)


def analyze_image(image, emotion):
    """
    Analyzes image to find emotion.
    :param image: <ndarray> containing image data
    :param emotion: <int> number of emotions to check against
    :return: <dict> request response
    """
    emotion_4 = False
    emotion_5 = False
    emotion_6 = False

    if emotion == 4:
        emotion_4 = True
    elif emotion == 5:
        emotion_5 = True
    elif emotion == 6:
        emotion_6 = True

    # try:
    result, probability, probSum, allClasses, allProbs = Classificator.ClassifyImages(
        "classificator.pkl",
        image,
        emotion_4,
        emotion_5,
        emotion_6
    )
    # # except:
    # response = {
    #     'error': 'Face couldn\'t be analyzed!'
    # }
    # return response

    response = {
        'emotion': result,
        'probability': float(probSum)
    }

    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)