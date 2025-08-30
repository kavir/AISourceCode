let lastDetectedEmotion = null;
let lastMusicRecommendations = [];
let myArray=[];
// Function to request camera permissions
function requestCameraPermissions() {
    return navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            // Permission granted, load the index.html
            console.log("Camera Stream:", stream);
            loadIndexHtml();
        })
        .catch((error) => {
            // Permission denied or error occurred
            console.error("Error accessing camera:", error);
            alert("Camera permission is required to proceed.");
        });
}

// Function to load index.html
function loadIndexHtml() {
    // Assuming you have an 'index.html' file in the same directory as your script
    window.location.href = 'index.html';
}

// Call the function to request camera permissions
requestCameraPermissions();

function analyzeEmotion() {
    // Send a POST request to the server to analyze the emotion
    fetch("/process_emotion", {
        method: "POST",
    })
    .then(response => response.json())
    .then(data => {
        // Clear the previous results
        const musicRecommendationsDiv = document.getElementById("music-recommendations");
        const detectedEmotionDiv = document.getElementById("detected-emotion");

        musicRecommendationsDiv.innerHTML = ""; // Clear music recommendations
        detectedEmotionDiv.innerHTML = ""; // Clear detected emotion

        // Process and display the detected emotions and up to 10 music recommendations
        data.forEach(result => {
            if (result.emotion) {
                const emotionParagraph = document.createElement("p");
                emotionParagraph.innerText = `Emotion: ${result.emotion}`;
                detectedEmotionDiv.appendChild(emotionParagraph);
                //////////////////////////////////////////////sending a myArray to flask for storing emotion
                fetch("/emotion", {
                    method: "POST",
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ arrayVariable: myArray }),
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Response from Flask:', data);
                })
                .catch(error => {
                    console.error('Error:', error);
                });
                /////////////////////////////////////////////

                // Check if the emotion has changed
                if (result.emotion !== lastDetectedEmotion) {
                    // Clear the previous results
                     // Clear music recommendations
                    
                
                    myArray.push(result.emotion);
                    // Update lastDetectedEmotion
                    lastDetectedEmotion = result.emotion;
                    // alert(lastDetectedEmotion);
                    lastMusicRecommendations=result.music_recommmendations;
                    // Update music recommendations only if they are available
                    if (result.music_recommendations.length > 0) {
                        lastMusicRecommendations = result.music_recommendations;
                
                        // Create an unordered list to display up to 10 music recommendations
                        const musicList = document.createElement("ul");
                        const numRecommendations = Math.min(10, lastMusicRecommendations.length);
                        for (let i = 0; i < numRecommendations; i++) {
                            const recommendation = lastMusicRecommendations[i];
                            const musicItem = document.createElement("li");
                            musicItem.innerText = `Music Track: ${recommendation.name} by ${recommendation.artist} (Album: ${recommendation.album}, Mood: ${recommendation.mood})`;
                            musicList.appendChild(musicItem);
                        }
                        musicRecommendationsDiv.appendChild(musicList);
                    }
                    var preloader = document.getElementById("preloader");
                    preloader.style.display = "none";
                }
                
            }
        });
    })
    .catch(error => {
        console.error("Error: " + error);
    });
}

function recommendMusicBasedOnEmotion() {
    // Check the last detected emotion
    if (lastDetectedEmotion === "Happy") {
        selectHappy();
    } else if (lastDetectedEmotion === "Sad") {
        selectSad();
    } else if (lastDetectedEmotion === "Energetic") {
        selectEnergetic();
    } else if (lastDetectedEmotion === "Calm") {
        selectCalm();
    }else if (lastDetectedEmotion === "Angry") {
        selectCalm();
    }else if (lastDetectedEmotion === "Fear") {
        selectCalm();
    } else {
        console.log("Unknown emotion. Cannot recommend music.");
    }
}



// Initial call to analyze emotion
analyzeEmotion();
// Automatically trigger the emotion analysis every 3 seconds
setInterval(analyzeEmotion, 3000);

