function toggleMenu() {
    document.getElementById('sidebar').classList.toggle('active');
}

let isPlaying = false;
let currentAudioFile = null;

function togglePlayPause() {
    const button = document.getElementById('play-pause');
    const audioPlayer = document.getElementById('audio-player');
    
    if (!currentAudioFile) {
        console.log("No audio file available to play");
        alert("No audio available. Please capture a gesture first.");
        return;
    }
    
    if (!isPlaying) {
        // Play audio
        audioPlayer.src = `/static/audio/${currentAudioFile}`;
        audioPlayer.play()
            .then(() => {
                console.log("Audio playing");
                button.textContent = "Pause";
                isPlaying = true;
            })
            .catch(error => {
                console.error("Error playing audio:", error);
                alert("Error playing audio");
            });
    } else {
        // Pause audio
        audioPlayer.pause();
        button.textContent = "Play";
        isPlaying = false;
        console.log("Audio paused");
    }
}

// Handle audio end event
document.getElementById('audio-player').addEventListener('ended', function() {
    document.getElementById('play-pause').textContent = "Play";
    isPlaying = false;
});

function clearText() {
    document.getElementById('text-output').value = "Text: Waiting for gestures...";
    const audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = '';
    audioPlayer.pause();
    currentAudioFile = null;
    isPlaying = false;
    document.getElementById('play-pause').textContent = "Play";
}

// Function to update text output with current prediction
function updateCurrentPrediction() {
    fetch('/api/current_prediction')
        .then(response => response.json())
        .then(data => {
            const textOutput = document.getElementById('text-output');
            if (data.prediction && data.confidence > 0.3) {
                textOutput.value = `Text: ${data.prediction.toUpperCase()} (${(data.confidence * 100).toFixed(1)}%)`;
            } else {
                textOutput.value = "Text: Waiting for gestures...";
            }
        })
        .catch(error => {
            console.error('Error fetching prediction:', error);
        });
}

// Function to capture current webcam prediction and generate audio
function captureWebcamPrediction() {
    const button = document.getElementById('capture-btn');
    const originalText = button.textContent;
    
    // Show loading state
    button.textContent = "Capturing...";
    button.disabled = true;
    
    // Send request to capture prediction
    fetch('/conversion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'action=predict'
    })
    .then(response => response.text())
    .then(html => {
        // Parse the response to extract text and audio
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        
        // Check for error message
        const errorElement = doc.querySelector('.error-message');
        if (errorElement) {
            alert(errorElement.textContent);
            return;
        }
        
        // Extract text result
        const textElement = doc.querySelector('#text-result');
        if (textElement && textElement.textContent.trim()) {
            document.getElementById('text-output').value = `Text: ${textElement.textContent}`;
            
            // Extract audio file name
            const audioElement = doc.querySelector('#audio-result');
            if (audioElement && audioElement.textContent.trim()) {
                currentAudioFile = audioElement.textContent;
                console.log("Audio file available:", currentAudioFile);
            }
        }
    })
    .catch(error => {
        console.error('Error capturing prediction:', error);
        alert('Error capturing prediction');
    })
    .finally(() => {
        // Reset button state
        button.textContent = originalText;
        button.disabled = false;
    });
}

// Start real-time prediction updates when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Update predictions every 500ms
    setInterval(updateCurrentPrediction, 500);
    
    // Add event listener for capture button if it exists
    const captureBtn = document.getElementById('capture-btn');
    if (captureBtn) {
        captureBtn.addEventListener('click', captureWebcamPrediction);
    }
});

// Handle video upload form submission
document.addEventListener('DOMContentLoaded', function() {
    const videoForm = document.getElementById('video-upload-form');
    if (videoForm) {
        videoForm.addEventListener('submit', function(e) {
            const submitBtn = videoForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            
            // Show loading state
            submitBtn.textContent = "Processing...";
            submitBtn.disabled = true;
            
            // Re-enable button after form submission
            setTimeout(() => {
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            }, 3000);
        });
    }
});
