document.addEventListener('DOMContentLoaded', () => {
    // API Base URL: defaults to empty string for same-origin requests (e.g. local or all-in-one Render),
    // or can be overridden via window.API_BASE_URL when frontend is hosted separately on Vercel.
    const API_BASE_URL = (window.API_BASE_URL || '').replace(/\/+$/, '');

    // ==================== IMAGE MODE ELEMENTS ====================
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnReset = document.getElementById('btn-reset');
    const btnAnalyze = document.getElementById('btn-analyze');
    const uploadSection = document.getElementById('upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    const messageContainer = document.getElementById('message-container');
    const systemStatus = document.getElementById('system-status');
    const systemStatusMsg = document.getElementById('system-status-msg');
    
    // Image Results Elements
    const predictionBanner = document.getElementById('prediction-banner');
    const predictionLabel = document.getElementById('prediction-label');
    const probRealText = document.getElementById('prob-real-text');
    const probRealBar = document.getElementById('prob-real-bar');
    const probFakeText = document.getElementById('prob-fake-text');
    const probFakeBar = document.getElementById('prob-fake-bar');
    const metaModel = document.getElementById('meta-model');
    const metaFaces = document.getElementById('meta-faces');
    const metaThreshold = document.getElementById('meta-threshold');
    const warningsContainer = document.getElementById('warnings-container');
    
    // Image Visualization Elements
    const imgFace = document.getElementById('img-face');
    const imgHeatmap = document.getElementById('img-heatmap');
    const imgOverlay = document.getElementById('img-overlay');
    
    let currentFile = null;

    // ==================== VIDEO MODE ELEMENTS ====================
    const videoDropZone = document.getElementById('video-drop-zone');
    const videoFileInput = document.getElementById('video-file-input');
    const videoPreviewContainer = document.getElementById('video-preview-container');
    const videoPreview = document.getElementById('video-preview');
    const btnVideoReset = document.getElementById('btn-video-reset');
    const btnVideoAnalyze = document.getElementById('btn-video-analyze');
    const videoUploadSection = document.getElementById('video-upload-section');
    const videoLoadingSection = document.getElementById('video-loading-section');
    const videoResultsSection = document.getElementById('video-results-section');

    let currentVideoFile = null;

    // ==================== HEALTH CHECK ====================
    checkHealth();

    // ==================== MODE SWITCHING ====================
    window.switchMode = function(mode) {
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
        
        document.querySelectorAll('.mode-content').forEach(mc => mc.classList.remove('active'));
        document.getElementById(`${mode}-mode`).classList.add('active');
        
        hideMessage();
    };

    // ==================== IMAGE MODE LOGIC ====================
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleImageFile(files[0]);
    }, false);
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleImageFile(e.target.files[0]);
    }, false);
    
    btnReset.addEventListener('click', resetImageUI, false);
    btnAnalyze.addEventListener('click', analyzeImage, false);

    function handleImageFile(file) {
        hideMessage();
        const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            showError("Please upload a valid image file (JPG, PNG, WEBP).");
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            showError("File size exceeds the 10MB limit.");
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    function resetImageUI() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        resultsSection.classList.add('hidden');
        hideMessage();
    }

    async function analyzeImage() {
        if (!currentFile) return;
        hideMessage();
        uploadSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch(`${API_BASE_URL}/api/predict-and-explain`, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.detail || result.error || "An error occurred during analysis.");
            }
            displayImageResults(result);
        } catch (error) {
            console.error('Analysis error:', error);
            showError(error.message);
            loadingSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
        }
    }

    function displayImageResults(data) {
        loadingSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        resultsSection.classList.remove('hidden');

        predictionLabel.textContent = data.prediction;
        predictionBanner.className = 'prediction-banner ' + 
            (data.prediction === 'REAL' ? 'prediction-real' : 'prediction-fake');

        const realPct = (data.real_probability * 100).toFixed(1) + '%';
        const fakePct = (data.deepfake_probability * 100).toFixed(1) + '%';
        
        probRealText.textContent = realPct;
        probFakeText.textContent = fakePct;
        
        setTimeout(() => {
            probRealBar.style.width = realPct;
            probFakeBar.style.width = fakePct;
        }, 100);

        metaModel.textContent = data.model_name;
        metaFaces.textContent = data.face_count;
        metaThreshold.textContent = data.threshold;

        if (data.warnings && data.warnings.length > 0) {
            warningsContainer.innerHTML = data.warnings.map(w => `<p>${w}</p>`).join('');
            warningsContainer.classList.remove('hidden');
        } else {
            warningsContainer.classList.add('hidden');
            warningsContainer.innerHTML = '';
        }

        if (data.processed_face_image) {
            imgFace.src = 'data:image/png;base64,' + data.processed_face_image;
        } else {
            imgFace.src = '';
            imgFace.alt = 'No face extracted';
        }

        if (data.explanation_available && data.gradcam_heatmap && data.gradcam_overlay) {
            imgHeatmap.src = 'data:image/png;base64,' + data.gradcam_heatmap;
            imgOverlay.src = 'data:image/png;base64,' + data.gradcam_overlay;
        } else {
            imgHeatmap.src = '';
            imgHeatmap.alt = 'Explanation not available';
            imgOverlay.src = '';
            imgOverlay.alt = 'Explanation not available';
        }
        
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // ==================== VIDEO MODE LOGIC ====================
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        videoDropZone.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        videoDropZone.addEventListener(eventName, () => videoDropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        videoDropZone.addEventListener(eventName, () => videoDropZone.classList.remove('dragover'), false);
    });

    videoDropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleVideoFile(files[0]);
    }, false);

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleVideoFile(e.target.files[0]);
    }, false);

    btnVideoReset.addEventListener('click', resetVideoUI, false);
    btnVideoAnalyze.addEventListener('click', analyzeVideo, false);

    function handleVideoFile(file) {
        hideMessage();
        const validTypes = ['video/mp4', 'video/avi', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/webm'];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
            showError("Please upload a valid video file (MP4, AVI, MOV, MKV, WEBM).");
            return;
        }
        if (file.size > 100 * 1024 * 1024) {
            showError("File size exceeds the 100MB limit.");
            return;
        }
        currentVideoFile = file;
        
        const url = URL.createObjectURL(file);
        videoPreview.src = url;
        videoDropZone.classList.add('hidden');
        videoPreviewContainer.classList.remove('hidden');
    }

    function resetVideoUI() {
        currentVideoFile = null;
        videoFileInput.value = '';
        if (videoPreview.src) {
            URL.revokeObjectURL(videoPreview.src);
        }
        videoPreview.src = '';
        videoDropZone.classList.remove('hidden');
        videoPreviewContainer.classList.add('hidden');
        videoResultsSection.classList.add('hidden');
        hideMessage();
    }

    async function analyzeVideo() {
        if (!currentVideoFile) return;
        hideMessage();
        videoUploadSection.classList.add('hidden');
        videoResultsSection.classList.add('hidden');
        videoLoadingSection.classList.remove('hidden');

        // Animate progress bar indeterminately
        const progressFill = document.getElementById('video-progress-fill');
        progressFill.style.width = '0%';
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 3;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
        }, 500);

        const formData = new FormData();
        formData.append('file', currentVideoFile);

        try {
            const response = await fetch(`${API_BASE_URL}/api/predict-video`, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();

            clearInterval(progressInterval);
            progressFill.style.width = '100%';

            if (!response.ok) {
                throw new Error(result.detail || result.error || "An error occurred during video analysis.");
            }

            setTimeout(() => displayVideoResults(result), 300);

        } catch (error) {
            clearInterval(progressInterval);
            console.error('Video analysis error:', error);
            showError(error.message);
            videoLoadingSection.classList.add('hidden');
            videoUploadSection.classList.remove('hidden');
        }
    }

    function displayVideoResults(data) {
        videoLoadingSection.classList.add('hidden');
        videoUploadSection.classList.remove('hidden');
        videoResultsSection.classList.remove('hidden');

        // Overall prediction
        const videoPredictionLabel = document.getElementById('video-prediction-label');
        const videoPredictionBanner = document.getElementById('video-prediction-banner');
        videoPredictionLabel.textContent = data.prediction;
        
        if (data.prediction === 'REAL') {
            videoPredictionBanner.className = 'prediction-banner prediction-real';
        } else if (data.prediction === 'DEEPFAKE') {
            videoPredictionBanner.className = 'prediction-banner prediction-fake';
        } else {
            videoPredictionBanner.className = 'prediction-banner prediction-unknown';
        }

        // Probabilities
        const realPct = (data.real_probability * 100).toFixed(1) + '%';
        const fakePct = (data.deepfake_probability * 100).toFixed(1) + '%';
        
        document.getElementById('video-prob-real-text').textContent = realPct;
        document.getElementById('video-prob-fake-text').textContent = fakePct;
        
        setTimeout(() => {
            document.getElementById('video-prob-real-bar').style.width = realPct;
            document.getElementById('video-prob-fake-bar').style.width = fakePct;
        }, 100);

        // Meta info
        document.getElementById('video-meta-model').textContent = data.model_name;
        document.getElementById('video-meta-threshold').textContent = data.threshold;
        document.getElementById('video-meta-duration').textContent = data.video_duration_seconds.toFixed(1) + 's';
        document.getElementById('video-meta-fps').textContent = data.video_fps.toFixed(1);
        document.getElementById('video-meta-resolution').textContent = 
            data.video_resolution.length === 2 ? `${data.video_resolution[0]}×${data.video_resolution[1]}` : '-';

        // Stats
        document.getElementById('video-stat-total-frames').textContent = data.total_frames_extracted;
        document.getElementById('video-stat-faces-found').textContent = data.frames_with_faces;
        document.getElementById('video-stat-fake-frames').textContent = data.fake_frame_count || 0;
        document.getElementById('video-stat-fake-ratio').textContent = 
            ((data.fake_frame_ratio || 0) * 100).toFixed(1) + '%';

        // Warnings
        const videoWarnings = document.getElementById('video-warnings-container');
        if (data.warnings && data.warnings.length > 0) {
            videoWarnings.innerHTML = data.warnings.map(w => `<p>${w}</p>`).join('');
            videoWarnings.classList.remove('hidden');
        } else {
            videoWarnings.classList.add('hidden');
            videoWarnings.innerHTML = '';
        }

        // Frame-by-frame table
        const tbody = document.getElementById('frame-results-tbody');
        tbody.innerHTML = '';

        if (data.frame_results && data.frame_results.length > 0) {
            data.frame_results.forEach(frame => {
                const tr = document.createElement('tr');
                
                const fakeProb = frame.deepfake_probability;
                const hasFace = frame.face_detected;
                
                let resultBadge = '';
                let confidenceBar = '';
                
                if (hasFace && fakeProb !== null) {
                    const pct = (fakeProb * 100).toFixed(1);
                    if (frame.prediction === 'DEEPFAKE') {
                        resultBadge = '<span class="badge badge-fake">FAKE</span>';
                    } else {
                        resultBadge = '<span class="badge badge-real">REAL</span>';
                    }
                    confidenceBar = `
                        <div class="mini-prob-bar">
                            <div class="mini-prob-fill ${fakeProb >= 0.5 ? 'fill-fake' : 'fill-real'}" 
                                 style="width: ${pct}%"></div>
                        </div>`;
                } else {
                    resultBadge = '<span class="badge badge-none">No Face</span>';
                    confidenceBar = '-';
                }
                
                tr.innerHTML = `
                    <td>#${frame.frame_index + 1}</td>
                    <td>${frame.timestamp_seconds.toFixed(1)}s</td>
                    <td>${hasFace ? '✅ ' + frame.face_count : '❌'}</td>
                    <td>${hasFace && fakeProb !== null ? (fakeProb * 100).toFixed(1) + '%' : '-'}</td>
                    <td>${resultBadge}</td>
                    <td>${confidenceBar}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        videoResultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // ==================== COMMON UTILITIES ====================
    function showError(msg) {
        messageContainer.textContent = msg;
        messageContainer.className = 'message message-error';
        messageContainer.classList.remove('hidden');
    }

    function hideMessage() {
        messageContainer.classList.add('hidden');
        messageContainer.textContent = '';
    }

    async function checkHealth() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/health`);
            const data = await response.json();
            
            if (!data.model_loaded) {
                systemStatusMsg.textContent = data.message || "Model is not loaded. Predictions will fail.";
                systemStatus.classList.remove('hidden');
            } else {
                systemStatus.classList.add('hidden');
            }
        } catch (error) {
            console.error("Health check failed:", error);
            systemStatusMsg.textContent = "Connecting to backend... (If using Render Free Tier, the service may take ~50s to wake up).";
            systemStatus.classList.remove('hidden');
        }
    }

    // ==================== PWA INSTALL PROMPT ====================
    let deferredPrompt = null;
    const btnInstall = document.getElementById('btn-install-pwa');

    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        if (btnInstall) {
            btnInstall.classList.remove('hidden');
        }
    });

    if (btnInstall) {
        btnInstall.addEventListener('click', async () => {
            if (!deferredPrompt) return;
            deferredPrompt.prompt();
            const choiceResult = await deferredPrompt.userChoice;
            console.log('User install outcome:', choiceResult.outcome);
            deferredPrompt = null;
            btnInstall.classList.add('hidden');
        });
    }

    window.addEventListener('appinstalled', () => {
        console.log('Deepfake AI App installed as PWA!');
        if (btnInstall) {
            btnInstall.classList.add('hidden');
        }
    });
});
