from sre_constants import SUCCESS
from flask import Flask, render_template, request, jsonify,Response,url_for,redirect,session,flash
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

###################### Database ko lagi import gareko 

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError,DataRequired,Email
from flask_bcrypt import Bcrypt
from flask_mysqldb import MySQL
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

app = Flask(__name__)
#########################################Roshani code
img_shape = 48
batch_size = 64
test_data_path =  r'data/test'
# test_data_path = r'C:\Sentisymphonics_final\data\test'
test_preprocessor = ImageDataGenerator(
    rescale=1.0/255.0
)

test_data = test_preprocessor.flow_from_directory(
    test_data_path,
    class_mode="categorical",
    target_size=(img_shape,img_shape),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size,
)
######################################################




user_emotions = {}
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']=''
app.config['MYSQL_DB']='mydatabase'
app.secret_key='helloworld'

mysql=MySQL(app)
bcrypt = Bcrypt(app)

#########################################################


class RegisterForm(FlaskForm):
    name = StringField("Name",validators=[DataRequired()])
    email = StringField("Email",validators=[DataRequired(),Email()])
    password = PasswordField("Password",validators=[DataRequired()])
    repassword = PasswordField("RePassword",validators=[DataRequired()])
    submit = SubmitField('Register')

   

class LoginForm(FlaskForm):
    email = StringField("Email",validators=[DataRequired(), Email()])
    password = PasswordField("Password",validators=[DataRequired()])
    submit = SubmitField("Login")






#########################################################


# Load the trained emotion recognition model (CNN or ResNet50V2)
emotion_model = load_model('models/CNN_Model.h5')  # Update with your model file
music_clf = joblib.load('models/music_classifier.pkl')  # or 'models/...'
le = joblib.load('models/label_encoder.pkl')


# Load the face detection classifier (Haar Cascades or other)
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')  # Update with your classifier

# Define a list of emotion labels for reference
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Calm', 'Energetic']

# Load a sample music dataset for recommendations (e.g., CSV with name, artist, mood, and popularity)
music_data = pd.read_csv('data/data_moods.csv')  # Update with your dataset


cap = cv2.VideoCapture(0)  # Open a connection to the camera
def generate_frames():
    while True:
        SUCCESS,frame=cap.read()
        if not SUCCESS:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type:image/jpeg\r\n\r\n'+frame+b'\r\n')    


emotion_to_audio_features = {
    'Happy':     [0.8, 0.2, 0.9, 0.1, 0.2, 0.9, -5.0, 0.05, 120],
    'Sad':       [0.3, 0.8, 0.3, 0.0, 0.1, 0.2, -15.0, 0.03, 70],
    'Angry':     [0.7, 0.1, 0.95, 0.2, 0.3, 0.4, -4.0, 0.07, 140],
    'Disgust':   [0.2, 0.7, 0.3, 0.1, 0.1, 0.1, -20.0, 0.02, 65],
    'Fear':      [0.4, 0.6, 0.4, 0.1, 0.2, 0.3, -12.0, 0.03, 85],
    'Calm':      [0.4, 0.6, 0.4, 0.1, 0.3, 0.3, -12.0, 0.02, 90],
    'Energetic': [0.9, 0.1, 1.0, 0.0, 0.6, 0.85, -3.0, 0.06, 150]
}

def recommend_music(emotion):
    if emotion not in emotion_to_audio_features:
        return []

    input_features = [emotion_to_audio_features[emotion]]
    predicted_mood_encoded = music_clf.predict(input_features)[0]
    predicted_mood = le.inverse_transform([predicted_mood_encoded])[0]

    top_songs = music_data[music_data['mood'] == predicted_mood].sort_values(by='popularity', ascending=False).head(3)
    return top_songs.to_dict(orient='records')


@app.route('/')
def home():   
    return render_template('AIScreen.html')
    
@app.route('/evaluate_accuracy', methods=['GET'])
def evaluate_accuracy():
    
   
    accuracy = emotion_model.evaluate(test_data)
    test_accuracy = accuracy[1]
    
    # Return the evaluation results as JSON
    return jsonify({'test_accuracy': test_accuracy*100})


############################################################################ for login and logout







    


@app.route('/emotion',methods=['GET','POST'])
def emot():
    try:
        data = request.get_json()
        emotionArray = data.get('arrayVariable', [])
        print(emotionArray)
        if 'user_id' in session:
            user_id = session['user_id']
            print(f"User ID from session: {user_id}")

            # Calculate the most common emotion
            user_emotions.setdefault(user_id, []).extend(emotionArray)

            

            return jsonify({'message': 'Emotion data received successfully'})
        else:
            return jsonify({'error': 'User not logged in'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/videos')
def videos():
    return render_template('videos.html')

@app.route('/process_emotion', methods=['POST'])
def process_emotion():
    try:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            return jsonify({"error": "Failed to capture frame from the camera."})

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use the face detection classifier to detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        emotion_results = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            # Convert grayscale to a 3-channel image (RGB)
            face_img = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)
            face_img = cv2.resize(face_img, (48, 48))
            face_img = np.reshape(face_img, [1, 48, 48, 3])  # Prepare the face image for prediction
            face_img = face_img / 255.0  # Normalize pixel values

            emotion_prediction = emotion_model.predict(face_img)
            predicted_emotion = emotion_labels[np.argmax(emotion_prediction)]

            # Get the music recommendations for the predicted emotion
            music_recommendations = recommend_music(predicted_emotion)

            # Convert int32 values to regular integers
            x, y, w, h = int(x), int(y), int(w), int(h)

            emotion_results.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "emotion": predicted_emotion,
                "music_recommendations": music_recommendations
            })

        # Return the results as JSON to the web page
        return jsonify(emotion_results)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    

        # Run the app
    app.run(debug=True)

# Release the camera and close the window when the app is stopped
cap.release()
cv2.destroyAllWindows()