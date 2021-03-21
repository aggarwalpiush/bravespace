import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
import cv2
import PIL as pil
import io
import os
import random
import sqlite3
import emoji
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import socket
import ssl
import requests
import ast

sns.set_style('darkgrid')

option1 = 'API demo'
option2 = 'Annotation'
option3 = 'Moderator Dashboard'
selected_option = st.sidebar.selectbox(
    'Choose a view',
    (option1, option2, option3)
)

img_path = './img'

## Databases
random_con = sqlite3.connect('random.db')
random_cur = random_con.cursor()
racist_con = sqlite3.connect('racist.db')
racist_cur = racist_con.cursor()
sexist_con = sqlite3.connect('sexist.db')
sexist_cur = sexist_con.cursor()
religion_con = sqlite3.connect('religion.db')
religion_cur = religion_con.cursor()
disabled_con = sqlite3.connect('disabled.db')
disabled_cur = disabled_con.cursor()

# ----- API demo -----
if selected_option == option1:
    st.title("BraveSpace")

    st.markdown("BraveSpace is an API that uses deep learning methods to identify hateful images, making it easier for social media moderators to watch over their platforms. Our aim is to lessen the emotional or psychological burden carried by the moderators after long periods of reviewing content, since hateful text or images could potentially trigger past traumas or even produce new ones.")

    st.subheader("**Test it yourself here**👇")
    file_obj = st.file_uploader("Upload a meme", type=["jpg", "png", "jpeg"])
    if file_obj is not None:
        img = cv2.imdecode(np.fromstring(file_obj.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imwrite('./img/temp.png', img)

        # Send image to the model
        img_pil = pil.Image.open('./img/temp.png')
        output = io.BytesIO()
        img_pil.save(output, format='png')
        hex_data = output.getvalue()
        
        url = 'http://dfae02b1c4bb.ngrok.io/post/'
        r = requests.post(url, data=hex_data)

        print(r.text)
        pred = r.json()

        object_scores = pred['interpretable_object_scores']
        for obj in object_scores:
            obj_coor = obj['image_coordinate']
            obj_coor = list(ast.literal_eval(obj_coor)) # [(12,13),(16,17)]
            x1,y1 = obj_coor[0]
            x2,y2 = obj_coor[1]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        col1, col2 = st.beta_columns(2)

        with col1:
            # Display image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, use_column_width='always')
        with col2:
            # Display predictions
            if pred['meme_label'] == 'hate':
                st.write("{}% likely to be hateful.".format(float(pred['classifier_score'])*100))
            elif pred['meme_label'] == 'non-hate':
                st.write("{}% likely to be non-hateful.".format(float(pred['classifier_score'])*100))
            st.write("Model output:")
            st.json(pred)

# ----- Annotation -----
elif selected_option == option2: 
    st.markdown("<h1 style='text-align: center;'>Hello! Please label some of the hateful memes below.</h1>", unsafe_allow_html=True)
    random_cur.execute("SELECT * FROM random")
    results = random_cur.fetchall()

    if len(results) == 0:
        st.markdown("<h3 style='text-align: center;'>Congratulations, you have nothing to label!​​​​​​​​​​​​​​​​​​​​​ &#x1F60a;</h3>", unsafe_allow_html=True)

    for result in results:
        img_id, img_name, img_score = result
        img = cv2.imread(os.path.join(img_path,img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        
        col1, col2 = st.beta_columns(2)
        with col1:
            st.image(img, width=int(width*0.5))
            st.write("Hatefulness score: {}".format(img_score))
        with col2:
            st.markdown("<h2>What kind of hateful meme is this?</h2>", unsafe_allow_html=True)
            racist = st.button("Racist", key="{}.a".format(img_id))
            sexist = st.button("Sexist", key="{}.b".format(img_id))
            religion = st.button("Attacks to a religion", key="{}.c".format(img_id))
            disabled = st.button("Attacks to disabled people", key="{}.d".format(img_id))
            if racist or sexist or religion or disabled:
                t = (img_id,)
                random_cur.execute("DELETE FROM random WHERE id = ?", t)
                random_con.commit()

# ----- Moderator Dashboard ------
elif selected_option == option3:
    
    # Moderator statistics
    st.markdown("<h1 style='text-align: center;'>How are you doing this week?</h1>", unsafe_allow_html=True)
    hateful_memes = [["racist", 101], ["sexist", 24], ["religion", 79], ["disabled", 30]]
    hate_df = pd.DataFrame(data=hateful_memes, columns=["category", "amount"])
    fig = Figure()
    ax = fig.subplots()
    sns.barplot(x=hate_df['category'],
                y=hate_df['amount'], color='goldenrod', ax=ax)
    ax.set_xlabel('Category')
    ax.set_ylabel('Amount reviewed')
    st.pyplot(fig)

    # Content review
    st.markdown("<h1 style='text-align: center;'>What kind of memes would you like to review today?</h1>", unsafe_allow_html=True)
    genre = st.selectbox("", ('---', 'Racist', 'Sexist', 'Attacks to a religion', 'Attacks to disabled people'))

    if genre == 'Racist':
        racist_cur.execute("SELECT * FROM racist")
        results = racist_cur.fetchall()

        if len(results) == 0:
            st.markdown("<h3 style='text-align: center;'>Congratulations, you have reviewed everything!​​​​​​​​​​​​​​​​​​​​​ &#x1F60a;</h3>", unsafe_allow_html=True)

        for result in results:
            img_id, img_name, img_score = result
            img = cv2.imread(os.path.join(img_path,img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img.shape
            
            col1, col2 = st.beta_columns(2)
            with col1:
                st.image(img, width=int(width*0.5))
                st.write("Hatefulness score: {}".format(img_score))
            with col2:
                st.markdown("<h2>Is this meme hateful?</h2>", unsafe_allow_html=True)
                hate = st.button("Hateful", key="{}.1".format(img_id))
                nonhate = st.button("Non-hateful", key="{}.0".format(img_id))
                if hate or nonhate:
                    t = (img_id,)
                    racist_cur.execute("DELETE FROM racist WHERE id = ?", t)
                    racist_con.commit()

    elif genre == 'Sexist':
        sexist_cur.execute("SELECT * FROM sexist")
        results = sexist_cur.fetchall()

        if len(results) == 0:
            st.markdown("<h3 style='text-align: center;'>Congratulations, you have reviewed everything!​​​​​​​​​​​​​​​​​​​​​ &#x1F60a;</h3>", unsafe_allow_html=True)

        for result in results:
            img_id, img_name, img_score = result
            img = cv2.imread(os.path.join(img_path,img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img.shape
            
            col1, col2 = st.beta_columns(2)
            with col1:
                st.image(img, width=int(width*0.3))
                st.write("Hatefulness score: {}".format(img_score))
            with col2:
                st.markdown("<h3>Is this meme hateful?</h3>", unsafe_allow_html=True)
                hate = st.button("Hateful", key="{}.1".format(img_id))
                nonhate = st.button("Non-hateful", key="{}.0".format(img_id))
                if hate or nonhate:
                    t = (img_id,)
                    sexist_cur.execute("DELETE FROM sexist WHERE id = ?", t)
                    sexist_con.commit()

    elif genre == 'Attacks to a religion':
        religion_cur.execute("SELECT * FROM religion")
        results = religion_cur.fetchall()

        if len(results) == 0:
            st.markdown("<h3 style='text-align: center;'>Congratulations, you have reviewed everything!​​​​​​​​​​​​​​​​​​​​​ &#x1F60a;</h3>", unsafe_allow_html=True)

        for result in results:
            img_id, img_name, img_score = result
            img = cv2.imread(os.path.join(img_path,img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img.shape
            
            col1, col2 = st.beta_columns(2)
            with col1:
                st.image(img, width=int(width*0.3))
                st.write("Hatefulness score: {}".format(img_score))
            with col2:
                st.markdown("<h2>Is this meme hateful?</h2>", unsafe_allow_html=True)
                hate = st.button("Hateful", key="{}.1".format(img_id))
                nonhate = st.button("Non-hateful", key="{}.0".format(img_id))
                if hate or nonhate:
                    t = (img_id,)
                    religion_cur.execute("DELETE FROM religion WHERE id = ?", t)
                    religion_con.commit()

    elif genre == 'Attacks to disabled people':
        disabled_cur.execute("SELECT * FROM disabled")
        results = disabled_cur.fetchall()

        if len(results) == 0:
            st.markdown("<h3 style='text-align: center;'>Congratulations, you have reviewed everything!​​​​​​​​​​​​​​​​​​​​​ &#x1F60a;</h3>", unsafe_allow_html=True)

        for result in results:
            img_id, img_name, img_score = result
            img = cv2.imread(os.path.join(img_path,img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img.shape
            
            col1, col2 = st.beta_columns(2)
            with col1:
                st.image(img, width=int(width*0.5))
                st.write("Hatefulness score: {}".format(img_score))
            with col2:
                st.markdown("<h2>Is this meme hateful?</h2>", unsafe_allow_html=True)
                hate = st.button("Hateful", key="{}.1".format(img_id))
                nonhate = st.button("Non-hateful", key="{}.0".format(img_id))
                if hate or nonhate:
                    t = (img_id,)
                    disabled_cur.execute("DELETE FROM disabled WHERE id = ?", t)
                    disabled_con.commit()
