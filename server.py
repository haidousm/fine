import cv2
import numpy as np
import pyparsing

import matplotlib.pyplot as plt

from utils.Model import Model
from datetime import timedelta
from flask import Flask, make_response, request, current_app
from functools import update_wrapper


def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))

    # use str instead of basestring if using Python 3.x
    if not isinstance(origin, pyparsing.basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """

        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator


model = None
app = Flask(__name__)


def load_model():
    global model
    model = Model.load("trained_models/digits_mnist_rot.model")


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
@crossdomain(origin='*')
def get_prediction():
    if request.method == 'POST':
        image_data = request.get_json(force=True)
        image_data = np.array(image_data).astype(np.float32) / 255
        image_data = cv2.resize(image_data, dsize=(28, 28))
        image_data = image_data.reshape((1, -1))
        confidences = model.predict(image_data)
        prediction = model.output_layer_activation.predictions(confidences)
        res = ""
        res += str(prediction[0])
        _confidences = confidences.tolist()[0]
        for i in range(len(_confidences)):
            res += "," + str(_confidences[i])

    return res


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0')
