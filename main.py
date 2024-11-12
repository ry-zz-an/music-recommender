import streamlit as st
import streamlit.components.v1 as comp
from PIL import Image
import numpy as np
from keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import cv2
import time
import os
import warnings
import random

st.set_page_config(
    layout = 'wide',
)

warnings.filterwarnings("ignore", category=UserWarning, message="Exception ignored")

model = load_model('model.keras')

client_id = '92400d1dc54b4c598388cfce96849dfb'
client_secret = '3598e16423ca471ba40fee18bcccb461'

spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, 
                                                                client_secret=client_secret))

emotion_playlist_map = {
    'Happy': '0jrlHA5UmxRxJjoykf7qRY',
    'Sad': '2GevOeTWtEEX4EVFEdX5zE',
    'Angry': '2ptLyv2A6vjgXi2Wo3dRlg',
    'Neutral': '4oblctaZrOuwS4YCeRG6wO',
    'Fear': '0PLteLOoJNWvHh7OdxugTJ',
    'Surprise': '4pbZUPxfAby7ar8vH6Z8yQ',
}

if 'history' not in st.session_state:
    st.session_state['history']=[]

def capture_image():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the webcam (0 is usually the default camera)
    cam = cv2.VideoCapture(0)

    # Create a window to display the webcam feed
    cv2.namedWindow("Face Detection Webcam Screenshot App")
    img_counter = 0
    mode = 'manual'  # Start in manual mode
    last_capture_time = 0  # Variable to store the last time an image was saved
    screenshot_taken = False  # Flag to track if a screenshot was already taken in auto mode
    mode_switch_time = 0  # Time when mode switched to auto

    while True:    
        ret, frame = cam.read()
        
        # Check if the frame was successfully captured
        if not ret:
            st.error("Failed to grab frame.")
            break
        
        # Convert the frame to grayscale (Haar Cascade works with grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the resulting frame
        cv2.imshow("Face Detection Webcam Screenshot App", frame)
        
        # Wait for a key event
        k = cv2.waitKey(1) & 0xFF  # Use & 0xFF to handle 64-bit systems correctly
        
        if k == 32: 
            if mode == 'manual' and len(faces) > 0:
                img_name = "opencv_face_{}.jpg".format(img_counter)
                image_roi = 'ROI_{}.jpg'.format(img_counter)
                for i, (x,y,w,h) in enumerate(faces):
                    roi = frame[y:y+h, x:x+w]
                    cv2.imwrite(img_name, frame)
                    print(f"{img_name} saved!")
                    cv2.imwrite(image_roi, roi) 
                    img_counter += 1
                break
    cam.release()
    cv2.destroyAllWindows()
    return img_name, image_roi

def detect_emotion(image):
    img = Image.open(image).convert('L')    
    img = img.resize((112,112))
    img = np.array(img)
    img = img/255.0
    img = img.reshape(1,112,112,1)

    labels = ["Angry","Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    predictions = model.predict(img)
    label = labels[predictions.argmax()]
    return label


def suggestion(emotion, song_list):
    playlist_id = emotion_playlist_map[emotion]
    playlist = spotify.playlist_tracks(playlist_id)
    tracks=[track for track in playlist['items'] if not track['track']['explicit']]

    if tracks:
         while len(song_list) <= 5:
            selected_track = random.choice(tracks)
            if selected_track['track']['id'] not in song_list:
                track_id = selected_track['track']['id']
                track_name = selected_track['track']['name']
                song_list.append(track_id)
                return track_name, f"https://open.spotify.com/embed/track/{track_id}"
            else:
                continue
        

    else:
        return "Bad playlist; no such songs found"
    


#UI

st.markdown("<h1 style='text-align: center;'>Playmotion: A Song Recommender Based on Your Current Mood! </h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Click the following button to show us how you feel, maybe a song could cheer you up! </h1>", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .singleline {
        white-space: nowrap;
        overflow: hidden;   
        text-overflow: ellipsis;
    }
    </style>
    """,
    unsafe_allow_html=True
)
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

if col3.button("Capture!"):
    image, roi = capture_image()
    poc1,poc2,poc3,poc4=st.columns(4)
    if image:
        img = Image.open(image)
        with poc2:    
            st.image(img, caption='Captured Image', width=600, use_container_width=False, channels='RGB')

        emotion = detect_emotion(roi)
        st.write("Here's a few songs that fit your mood!")
        suggested = []
        coll1, coll2, coll3, coll4 = st.columns(4)
        song, embed = suggestion(emotion, suggested)
        if embed:
            with coll1:
                st.markdown(f'<div class="single-line">Song: {song} </div>', unsafe_allow_html=True)
                comp.iframe(embed, width=300, height=400)
        else:
            st.write('Sorry, we do not currently have songs that fit your mood.')
        
        st.session_state['history'].append({'Emotion':emotion, 'Song':song})

        song, embed = suggestion(emotion, suggested)
        if embed:
            with coll2:
                st.markdown(f'<div class="single-line">Song: {song} </div>', unsafe_allow_html=True)
                comp.iframe(embed, width=300, height=400)
        else:
            st.write('Sorry, we do not currently have songs that fit your mood.')
        
        st.session_state['history'].append({'Emotion':emotion, 'Song':song})

        song, embed = suggestion(emotion, suggested)
        if embed:
            with coll3:
                st.markdown(f'<div class="single-line">Song: {song} </div>', unsafe_allow_html=True)
                comp.iframe(embed, width=300, height=400)
        else:
            st.write('Sorry, we do not currently have songs that fit your mood.')
        
        st.session_state['history'].append({'Emotion':emotion, 'Song':song})

        song, embed = suggestion(emotion, suggested)
        if embed:
            with coll4:
                st.markdown(f'<div class="single-line">Song: {song} </div>', unsafe_allow_html=True)
                comp.iframe(embed, width=300, height=400)
        else:
            st.write('Sorry, we do not currently have songs that fit your mood.')
        
        st.session_state['history'].append({'Emotion':emotion, 'Song':song})

if col5.button('View History'):
    if st.session_state['history']:
        st.markdown("<h3 style='text-align:center;'> Suggestion History:</h3>", unsafe_allow_html=True)
        for entry in st.session_state['history']:
            st.markdown(f"<p style='text-align:center;'>Emotion: {entry['Emotion']} | Suggested Song: {entry['Song']} </p>", unsafe_allow_html=True)

    else:
        st.markdown(f"<p style='text-align:center;'> No history found. </p>", unsafe_allow_html=True)
    