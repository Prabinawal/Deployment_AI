from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        data = pd.read_csv(filepath)
        return render_template('analysis.html', tables=[data.head().to_html()], titles=['Data Preview'])

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    data = pd.read_csv(file)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    model = LinearRegression()
    model.fit(X, y)
    score = model.score(X, y)
    return render_template('result.html', score=score)

if __name__ == '__main__':
    app.run(debug=True)
