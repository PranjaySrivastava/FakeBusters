const { jsPDF } = window.jspdf;

// Backend API URL
const API_URL = 'https://salil-ind-fake-buster.hf.space/api';

// Store results for PDF generation
let imageResult = null;
let videoResult = null;
let aiResult = null;

function previewImage(event, previewId) {
    const file = event.target.files[0];
    const preview = document.getElementById(previewId);
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
        preview.src = reader.result;
        preview.style.display = "block";
    };
    reader.readAsDataURL(file);
}

function previewVideo(event) {
    const file = event.target.files[0];
    const video = document.getElementById("videoPreview");
    if (!file) return;

    video.src = URL.createObjectURL(file);
    video.style.display = "block";
}

async function analyzeImage() {
    const fileInput = document.getElementById("imageUpload");
    const spinner = document.getElementById("imageSpinner");
    const result = document.getElementById("imageResult");
    const confidenceText = document.getElementById("imageConfidenceText");

    if (!fileInput.files[0]) {
        alert("Please upload an image first!");
        return;
    }

    spinner.style.display = "block";
    result.innerHTML = "Analyzing...";
    confidenceText.innerHTML = "0%";

    try {
        const file = fileInput.files[0];
        const base64 = await fileToBase64(file);

        const response = await fetch(`${API_URL}/analyze/image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64 })
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const data = await response.json();
        imageResult = data;

        spinner.style.display = "none";
        confidenceText.innerHTML = data.confidence + "%";
        result.innerHTML = data.prediction;
        result.style.color = data.is_fake ? "#ff4d4d" : "#00ff99";

    } catch (error) {
        spinner.style.display = "none";
        result.innerHTML = "Error: " + error.message;
        result.style.color = "#ff4d4d";
        console.error('Analysis error:', error);
    }
}

async function analyzeAI() {
    const fileInput = document.getElementById("aiImageUpload");
    const spinner = document.getElementById("aiSpinner");
    const result = document.getElementById("aiResult");
    const confidenceText = document.getElementById("aiConfidenceText");

    if (!fileInput.files[0]) {
        alert("Please upload an image first!");
        return;
    }

    spinner.style.display = "block";
    result.innerHTML = "Analyzing...";
    confidenceText.innerHTML = "0%";

    try {
        const file = fileInput.files[0];
        const base64 = await fileToBase64(file);

        const response = await fetch(`${API_URL}/analyze/ai`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: base64 })
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const data = await response.json();
        aiResult = data;

        spinner.style.display = "none";
        confidenceText.innerHTML = data.confidence + "%";
        result.innerHTML = data.prediction;
        result.style.color = data.is_ai ? "#ff4d4d" : "#00ff99";

    } catch (error) {
        spinner.style.display = "none";
        result.innerHTML = "Error: " + error.message;
        result.style.color = "#ff4d4d";
        console.error('Analysis error:', error);
    }
}

async function analyzeVideo() {
    const fileInput = document.getElementById("videoUpload");
    const spinner = document.getElementById("videoSpinner");
    const result = document.getElementById("videoResult");
    const confidenceText = document.getElementById("videoConfidenceText");
    const timeline = document.getElementById("videoTimeline");

    if (!fileInput.files[0]) {
        alert("Please upload a video first!");
        return;
    }

    spinner.style.display = "block";
    timeline.style.display = "none";
    result.innerHTML = "Analyzing video... This may take a few moments...";
    confidenceText.innerHTML = "0%";

    try {
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('video', file);

        const response = await fetch(`${API_URL}/analyze/video`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const data = await response.json();
        videoResult = data;

        spinner.style.display = "none";
        confidenceText.innerHTML = data.confidence + "%";

        if (data.segments && data.segments.length > 0) {
            let segmentText = data.prediction + "<br><br>";
            data.segments.forEach((seg, i) => {
                segmentText += `${seg.start_time} → ${seg.end_time} (${seg.confidence}%)<br>`;
            });
            result.innerHTML = segmentText;
            result.style.color = "#ff4d4d";
        } else {
            result.innerHTML = data.prediction;
            result.style.color = data.is_fake ? "#ff4d4d" : "#00ff99";
        }

        updateTimeline(data.timeline);
        timeline.style.display = "block";

    } catch (error) {
        spinner.style.display = "none";
        result.innerHTML = "Error: " + error.message;
        result.style.color = "#ff4d4d";
        console.error('Analysis error:', error);
    }
}

function updateTimeline(timelineData) {
    const timelineBar = document.querySelector('.timeline-bar');
    
    timelineBar.innerHTML = '';
    
    if (!timelineData || timelineData.length === 0) {
        return;
    }
    
    timelineData.forEach((segment, index) => {
        const segDiv = document.createElement('div');
        segDiv.className = 'fake-segment';
        segDiv.style.left = segment.position + '%';
        segDiv.style.width = '2%';
        segDiv.title = `Confidence: ${segment.confidence}%`;
        timelineBar.appendChild(segDiv);
    });
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
        reader.readAsDataURL(file);
    });
}

function downloadImageReport() {
    if (!imageResult) {
        alert("Please analyze an image first!");
        return;
    }
    generatePDF("Image Deepfake Detection", imageResult.confidence + "%", imageResult.prediction);
}

function downloadVideoReport() {
    if (!videoResult) {
        alert("Please analyze a video first!");
        return;
    }
    
    const doc = new jsPDF();
    
    doc.setFontSize(22);
    doc.text("FakeBusters Forensic Report", 20, 20);
    
    doc.setFontSize(14);
    doc.text("Module: Video Deepfake Detection", 20, 40);
    doc.text("Result: " + videoResult.prediction, 20, 55);
    doc.text("Confidence: " + videoResult.confidence + "%", 20, 70);
    doc.text("Frames Analyzed: " + videoResult.frames_analyzed, 20, 85);
    
    if (videoResult.segments && videoResult.segments.length > 0) {
        doc.text("Manipulated Segments:", 20, 100);
        let y = 115;
        videoResult.segments.forEach((seg, i) => {
            doc.setFontSize(12);
            doc.text(`${i+1}. ${seg.start_time} → ${seg.end_time} (${seg.confidence}%)`, 25, y);
            y += 15;
        });
    }
    
    doc.setFontSize(10);
    doc.text("Generated on: " + new Date().toLocaleString(), 20, 280);
    
    doc.save("FakeBusters_Video_Report.pdf");
}

function downloadAIReport() {
    if (!aiResult) {
        alert("Please analyze an image first!");
        return;
    }
    generatePDF("AI Generated Image Detection", aiResult.confidence + "%", aiResult.prediction);
}

function generatePDF(title, confidence, resultText) {
    const doc = new jsPDF();

    doc.setFontSize(22);
    doc.text("FakeBusters Forensic Report", 20, 20);

    doc.setFontSize(14);
    doc.text("Module: " + title, 20, 40);
    doc.text("Result: " + resultText, 20, 55);
    doc.text("Confidence: " + confidence, 20, 70);
    doc.text("Generated on: " + new Date().toLocaleString(), 20, 90);

    doc.save("FakeBusters_Report.pdf");
}

function resetImage() {
    document.getElementById("imageUpload").value = "";
    document.getElementById("imagePreview").style.display = "none";
    document.getElementById("imageResult").innerHTML = "";
    document.getElementById("imageConfidenceText").innerHTML = "0%";
    imageResult = null;
}

function resetVideo() {
    document.getElementById("videoUpload").value = "";
    document.getElementById("videoPreview").src = "";
    document.getElementById("videoPreview").style.display = "none";
    document.getElementById("videoResult").innerHTML = "";
    document.getElementById("videoConfidenceText").innerHTML = "0%";
    document.getElementById("videoTimeline").style.display = "none";
    videoResult = null;
}

function resetAI() {
    document.getElementById("aiImageUpload").value = "";
    document.getElementById("aiPreview").style.display = "none";
    document.getElementById("aiResult").innerHTML = "";
    document.getElementById("aiConfidenceText").innerHTML = "0%";
    aiResult = null;
}

window.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'ok') {
            console.log('✓ Backend connected');
            console.log('Available models:', data.models);
        }
    } catch (error) {
        console.error('⚠ Backend not connected. Please start the Flask server.');
        console.error('Run: python app.py');
    }
});