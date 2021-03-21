#! usr/bin/env python
# *-- coding : utf-8 --*

import re
from nltk.tokenize import TweetTokenizer
from flask import Flask, request, jsonify
from model import predict_meme_class
import os
import image_processing

app = Flask(__name__)




@app.route('/post/', methods=['POST', "GET"])
def post_something():
    meme_path = request.args.get('meme')
    result_dict = {}
    if (not request.data):
        return "No data was sent !"
        # getting the image from client
    image_name = image_processing.get_image(request)

    # check if the file less than 1 MB
    if os.stat(image_name).st_size > 1000000:
        # reduce the size of the received image and delete the old one
        new_image_name = image_processing.get_image_less_than_1MB(image_name)
        os.remove(image_name)
    else:
        new_image_name = image_name
    predicted_label = predict_meme_class(new_image_name)
    return jsonify(predicted_label)


@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"


def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass

def create_app():
    app = Flask(__name__)

    # Initialize our ngrok settings into Flask
    app.config.from_mapping(
        BASE_URL="http://localhost:6007",
        USE_NGROK=os.environ.get("USE_NGROK", "False") == "True" and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
    )

    if app.config.get("ENV") == "development" and app.config["USE_NGROK"]:
        # pyngrok will only be installed, and should only ever be initialized, in a dev environment
        from pyngrok import ngrok

        # Get the dev server port (defaults to 5000 for Flask, can be overridden with `--port`
        # when starting the server
        port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 6007

        # Open a ngrok tunnel to the dev server
        public_url = ngrok.connect(port).public_url
        print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

        # Update any base URLs or webhooks to use the public ngrok URL
        app.config["BASE_URL"] = public_url
        init_webhooks(public_url)

    # ... Initialize Blueprints and the rest of our app

    return app

from pyngrok import ngrok

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    #app = Flask(__name__)
    port = 6007
    public_url = ngrok.connect(port).public_url
    print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))
    app.run(threaded=True, port=6007)
