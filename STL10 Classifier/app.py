from flask import Flask, jsonify, request
from torch_utils import transform_image
from torch_utils import get_prediction


app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        img = transform_image(img_bytes)
        pred = get_prediction(img)
        return jsonify({'classification': int(pred[0])})
    
if __name__ == '__main__':
    app.run(port=5000, debug=True)