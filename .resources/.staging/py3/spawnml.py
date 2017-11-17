from flask import Flask, request, json

import utility

app = Flask(__name__)
words = []
predict = []
classes = []
documents = []
ignore_words = ['?']

utility.loadModel()


@app.route('/')
def hello_world():
    return 'Welcome to SpawNML'


@app.route('/train')
def train():
    utility.train()
    return "training successfull"


@app.route('/accuracy')
def accuracy():
    sum = 0
    f = open("C:/Users/Amar/Downloads/training_data.csv", 'rU')
    for line in f:
        cells = line.split(",")
        predict.append(cells[1])

    f.close()

    sum = 0
    for i in range(len(predict)):
        value = float(utility.classifyPredict(predict[i]))
        print("query " + predict[i] + "\n" + "score " + str(value))
        sum += value

    print(sum / len(predict))
    js = {
        "model_accurary": str(sum / len(predict))
    }

    return json.dumps(js)


@app.route('/classify', methods=["GET"])
def classify():
    sentence = request.args.get('query')
    return_list = utility.classify(sentence)
    return return_list


if __name__ == '__main__':
    app.run(threaded=True)