from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam

import numpy as np
import pickle
import os
import glob
from sister.tokenizers import SimpleTokenizer
from sister.word_embedders import FasttextEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import easyocr
import cv2
import sys
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

word_embedder = FasttextEmbedding("en")
tokenizer = SimpleTokenizer()
vgg_model = VGG16()
vgg_model = Model(inputs = vgg_model.inputs, outputs = vgg_model.layers[-2].output)

def extract_text_features(word_embedder, tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence)
    text_features = word_embedder.get_word_vectors(tokens)
    text_features = np.mean(text_features, axis=0)
    return text_features

def extract_image_features(img_path, img_model):
    img = load_img(img_path, target_size=(224,224))
    img = np.array(img)
    reshaped_img = img.reshape(1,224,224,3)
    imgx = preprocess_input(reshaped_img)
    img_features = img_model.predict(imgx, use_multiprocessing=True)
    img_features = img_features.reshape((4096,))
    return img_features

def extract_features(word_embedder, tokenizer, sentence, img_path, img_model):
    text_features = extract_text_features(word_embedder, tokenizer, sentence)
    image_features = extract_image_features(img_path, img_model)
    return np.concatenate((text_features, image_features), axis=0)

def train_model():
    clf = Sequential()
    clf.add(Dense(1000, kernel_initializer="uniform", input_shape=(4396,)))
    clf.add(Dropout(0.5))
    clf.add(Dense(500))
    clf.add(Dropout(0.5))
    clf.add(Dense(100))
    clf.add(Dropout(0.5))
    clf.add(Dense(50))
    clf.add(Dropout(0.5))
    clf.add(Dense(2, activation="softmax"))


    clf.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    train_id, X_train, Y_train = pickle.load(open("train_dev_nonbenignimg.pkl", "rb"))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    Y_train = to_categorical(Y_train)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.7, random_state=42)
    
    file_path = f"hate_meme_detection_model.hdf5"
    if not os.path.exists(file_path):
        clf.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=32)
        clf.save_weights(file_path)
    else:
        clf.load_weights(file_path)
    return clf



from predict_object import obj_co


#OBJECT_COORDINATES = [[(344,144),(530,403)]]

def apply_sampling(sentence, meme_path):
    
    obj_coord = obj_co(meme_path)
    image = cv2.imread(meme_path)
    
    for i, each_object in enumerate(obj_coord):
        start_point = each_object[0]
        end_point = each_object[1]


        # Black color in BGR
        color = (0, 0, 0)

        # Line thickness of -1 px
        # Thickness of -1 will fill the entire shape
        thickness = -1
        image_sample = cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.imwrite(os.path.join(os.path.dirname(meme_path),'masked_sample_{}.png'.format(str(i))), image_sample)
    text_samples = []
    for j, token in enumerate(tknzr.tokenize(sentence)):
        text_samples.append(' '.join(tknzr.tokenize(sentence)[:j]+tknzr.tokenize(sentence)[j+1:]))

    return text_samples, tknzr.tokenize(sentence), obj_coord
    

def interpretable_region_scores(clf, meme_path, sentence):
    x_text = extract_features(word_embedder, tokenizer, sentence, meme_path, vgg_model)
    x_text = np.reshape(x_text,(1,4396))

    predict_label = clf.predict(x_text)

    return predict_label[0][1]

def predict_meme_class( meme_path):
    clf = train_model()

    prediction_result = {}
    reader = easyocr.Reader(['en']) # need to run only once to load model into memory
    sentence = reader.readtext(meme_path)

    result = []

    for text in sentence:
        result.append(text[1])

    sentence = ' '.join(result)

    # TODO:update meme_path with inpainting
    x_text = extract_features(word_embedder, tokenizer, sentence, meme_path, vgg_model)
    x_text = np.reshape(x_text,(1,4396))

    predict_label = clf.predict(x_text)
    prediction = np.argmax(predict_label)
    print(predict_label)
    if prediction == 0:
        prediction_result['meme_label'] = 'non-hate'
        prediction_result['classifier_score'] = str(predict_label[0][0])
        prediction_result['interpretable_text_scores'] = []
        prediction_result['interpretable_object_scores'] = []
        print("Non-hateful")
    elif prediction == 1:
        prediction_result['meme_label'] = 'hate'
        prediction_result['classifier_score'] = str(predict_label[0][1])
        prediction_result['interpretable_text_scores'] = []
        prediction_result['interpretable_object_scores'] = []
        text_samples, token_text, coord = apply_sampling(sentence, meme_path)
        #total_sample = len(text_samples) + len(glob.glob(os.path.dirname(meme_path),"*masked_sample*")) 
        for k,txt in enumerate(text_samples):
            prediction_result['interpretable_text_scores'].append({'token':str(token_text[k]), 'hate_score':str(1-interpretable_region_scores(clf, meme_path, txt))})
            print(token_text[k])
            print(interpretable_region_scores(clf, meme_path, txt))
        for l, samp_img in enumerate(glob.glob(os.path.join(os.path.dirname(meme_path),"*masked_sample*"))):
            print(coord)
            prediction_result['interpretable_object_scores'].append({'image_coordinate':str(coord[l]), 'hate_score':str(1-interpretable_region_scores(clf, samp_img, sentence))})
            print(coord[l])
            print(interpretable_region_scores(clf, samp_img, sentence))
        print("Hateful")

    fileList = glob.glob(os.path.join(os.path.dirname(meme_path),"*masked_sample*"))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    return prediction_result

def test_model(clf):
    test_id, X_test, Y_test = pickle.load(open("test_benignimg.pkl", "rb"))
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    Y_pred = clf.predict(X_test)
    Y_pred = [np.argmax(pred) for pred in Y_pred]
    print("----- Classification report -----")
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))
    print("Accuracy score:", accuracy_score(Y_test, Y_pred))
    print("F1 score:", f1_score(Y_test, Y_pred))
    print("Precision score:", precision_score(Y_test, Y_pred))
    print("Recall score:", recall_score(Y_test, Y_pred))

def main():
    image_path = sys.argv[1]
    clf = train_model()
    predict_meme_class(image_path)
   # test_model(clf)
    return

if __name__ == '__main__':
    main()
