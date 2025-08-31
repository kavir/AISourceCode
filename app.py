from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

########################################
# Testing Accuracy Dataset
img_shape = 48
batch_size = 64
test_data_path = 'data/test'

test_preprocessor = ImageDataGenerator(rescale=1.0/255.0)
test_data = test_preprocessor.flow_from_directory(
    test_data_path,
    class_mode="categorical",
    target_size=(img_shape, img_shape),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size,
)
########################################

# In-memory reflection storage (no DB, single user)
user_emotions = []
user_feedback = []

# Load models
emotion_model = load_model('models/CNN_Model.h5')
music_clf = joblib.load('models/music_classifier.pkl')
le = joblib.load('models/label_encoder.pkl')

# Face detection
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Calm', 'Energetic']

# Music dataset
music_data = pd.read_csv('data/data_moods.csv')

# Camera
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type:image/jpeg\r\n\r\n'+frame+b'\r\n')    

# Emotion to audio features
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

    # Get top songs for predicted mood
    top_songs = music_data[music_data['mood'] == predicted_mood].sort_values(by='popularity', ascending=False)

    # Exclude songs previously disliked for this emotion
    disliked_songs = [fb.get('song') for fb in user_feedback
                      if fb['feedback'] == 'dislike' and fb['emotion'] == emotion]
    if disliked_songs:
        top_songs = top_songs[~top_songs['song'].isin(disliked_songs)]

    # Optionally, prioritize liked songs
    liked_songs = [fb.get('song') for fb in user_feedback
                   if fb['feedback'] == 'like' and fb['emotion'] == emotion]
    if liked_songs:
        top_songs_liked = top_songs[top_songs['song'].isin(liked_songs)]
        top_songs_other = top_songs[~top_songs['song'].isin(liked_songs)]
        top_songs = pd.concat([top_songs_liked, top_songs_other])

    return top_songs.head(3).to_dict(orient='records')

# def recommend_music(emotion):
#     if emotion not in emotion_to_audio_features:
#         return []

#     input_features = [emotion_to_audio_features[emotion]]
#     predicted_mood_encoded = music_clf.predict(input_features)[0]
#     predicted_mood = le.inverse_transform([predicted_mood_encoded])[0]

#     top_songs = music_data[music_data['mood'] == predicted_mood].sort_values(by='popularity', ascending=False).head(3)
#     return top_songs.to_dict(orient='records')

@app.route('/')
def home():   
    return render_template('AIScreen.html')

@app.route('/evaluate_accuracy', methods=['GET'])
def evaluate_accuracy():
    accuracy = emotion_model.evaluate(test_data)
    test_accuracy = accuracy[1]
    return jsonify({'test_accuracy': test_accuracy*100})

@app.route('/emotion', methods=['POST'])
def emot():
    try:
        data = request.get_json()
        emotionArray = data.get('arrayVariable', [])
        user_emotions.extend(emotionArray)   # store emotions in memory
        return jsonify({'message': 'Emotion data stored'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        feedback_entry = {
            "emotion": data.get('emotion'),
            "feedback": data.get('feedback'),  # like/dislike
            "song": data.get('song')           # add this
        }
        user_feedback.append(feedback_entry)
        return jsonify({'message': 'Feedback stored', 'all_feedback': user_feedback})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reflect', methods=['GET'])
def reflect():
    # Reflection summary: what emotions were detected & feedback trends
    from collections import Counter
    emotion_summary = Counter(user_emotions)
    feedback_summary = Counter([fb['feedback'] for fb in user_feedback])
    return jsonify({
        "emotion_summary": dict(emotion_summary),
        "feedback_summary": dict(feedback_summary)
    })

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/process_emotion', methods=['POST'])
def process_emotion():
    try:
        ret, frame = cap.read()
        if not ret:
            return jsonify({"error": "Failed to capture frame."})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        emotion_results = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_img = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)
            face_img = cv2.resize(face_img, (48, 48))
            face_img = np.reshape(face_img, [1, 48, 48, 3]) / 255.0

            emotion_prediction = emotion_model.predict(face_img)
            predicted_emotion = emotion_labels[np.argmax(emotion_prediction)]

            # Save emotion for reflection
            user_emotions.append(predicted_emotion)

            music_recommendations = recommend_music(predicted_emotion)

            emotion_results.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "emotion": predicted_emotion,
                "music_recommendations": music_recommendations
            })

        return jsonify(emotion_results)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

cap.release()
cv2.destroyAllWindows()


