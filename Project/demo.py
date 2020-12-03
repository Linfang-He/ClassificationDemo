# coding:utf-8
 
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from train_test import demo
from datetime import timedelta
 
# file format allowed
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)

# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "Please check file format，only png,PNG,jpg,JPG, and bmp are allowed"})
 
        data_type = request.form.get("name")
 
        basepath = os.path.dirname(__file__)  # current file path
 
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)
 
        # Transform image format and name by Opencv
        img = cv2.imread(upload_path)
        path = os.path.join(basepath, 'static/images', 'test.jpg')
        cv2.imwrite(path, img)
        
        output = demo(data_type, path)

        if data_type == 'digit':
            l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif data_type == 'character':
            l = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]
        else:
            l = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", \
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

        cat = l[output[0]]
        return render_template('upload_ok.html',userinput=cat,val1=time.time())
 
    return render_template('upload.html')
 
 
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)