#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import json
import time

from flask import request
from flask_restful import Resource
from app import app, api

from image_proc import img_to_b64, score_word, b64_remove_header, b64_preprocess
from model import load_word_predictor

DEBUG = False

@api.route("/api/word")
class Prediction(Resource):
    def post(self):
        """API that expects an b64 image as input, analyze it
        and returns the word it represents
        Language has to be provided as different alphabets are handled differently
        Expected word can be provided too

        :return: json
            a json containing the read word (string) and if expected word was provided,
            a b64 images flagging discrepancies between read and expected (if any)
        """
        start = time.time()

        # Read parameters
        data = request.data.decode("utf-8")
        params = json.loads(data)
        img_in = params['img']
        word_in = params['word']
        lang_in = params['lang']
        nb_output = params['nb_output']

        # Ensure image has no header
        img_in = b64_remove_header(img_in)

        # Convert image to process-able format
        img = b64_preprocess(img_in)

        # Convert image to word
        response = dict()
        if (lang_in=='en'):
            words_out = word_predictor_en(img,nb_output)
        elif (lang_in=='ru'):
             words_out = word_predictor_ru(img,nb_output)
        word_out = ''.join(list(words_out[:,0]))

        response["word"] = word_out

        if DEBUG:
            print("Found word: %s" % word_out)
            if  nb_output > 1:
                for i in range(1, nb_output):
                    print("Alternatively word could be %s" % ''.join(list(words_out[:,i])))

        # Compare read word with expected word
        if len(word_in) != 0:
            img, correct = score_word(word_in, words_out, img)
            response["correct"] = correct
            # Convert image back to base64 to be sent to the requestor
            response["img"] = img_to_b64(img)

        print("Time spent handling the request: %f" % (time.time() - start))
        return json.dumps(response)


if __name__ == "__main__":

    # Read arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=DEBUG, type=bool, help="Image shape")
    args = parser.parse_args()

    DEBUG = args.debug

    # Load model and start API
    print('Loading models')
    word_predictor_en = load_word_predictor("weights/comnist_keras_en.hdf5", 26, lang_in='en')
    word_predictor_ru = load_word_predictor("weights/comnist_keras_ru.hdf5", 34, lang_in='ru')
    print('Starting the API')
    app.run(host="0.0.0.0", port=5002)
