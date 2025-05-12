document.addEventListener('DOMContentLoaded', function() {
    // DOM Element References
    const uploadBox = document.getElementById('upload-box');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const uploadSection = document.querySelector('.upload-section');
    const resultsSection = document.getElementById('results-section');
    const imageContainer = document.getElementById('image-container');
    const predictionContainer = document.getElementById('prediction-container');
    const meterFill = document.getElementById('meter-fill');
    const newScanBtn = document.getElementById('new-scan-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const projectionContainer = document.getElementById('projection-type');
    const manualAnalysisBtn = document.getElementById('manual-analysis-btn');
    
    // Global variable to store the current file
    let currentFile = null;
    
    // Event Listeners
    uploadBox.addEventListener('click', () => fileInput.click());
    uploadBox.addEventListener('dragover', handleDragOver);
    uploadBox.addEventListener('dragleave', handleDragLeave);
    uploadBox.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    uploadBtn.addEventListener('click', analyzeImage);
    newScanBtn.addEventListener('click', resetInterface);
    manualAnalysisBtn.addEventListener('click', function() {
        if (!currentFile) {
            alert('Please upload an image first');
            return;
        }
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        const formData = new FormData();
        formData.append('file', currentFile);
        
        fetch('/api/manual-analysis', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            // Open manual analysis in new window
            window.open('/manual-analysis', '_blank');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to start manual analysis: ' + error.message);
        })
        .finally(() => {
            loadingOverlay.style.display = 'none';
        });
    });
    
    // Handle drag over event
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.add('highlight');
    }
    
    // Handle drag leave event
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('highlight');
    }
    
    // Handle drop event
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('highlight');
        
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            processFile(files[0]);
        }
    }
    
    // Handle file selection from input
    function handleFileSelect(e) {
        const files = e.target.files;
        if (files.length) {
            processFile(files[0]);
        }
    }
    
    // Process the selected file
    function processFile(file) {
        if (!file) return;
        
        currentFile = file;
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        // Update upload box UI
        uploadBox.classList.add('has-file');
        const fileFormatSpan = uploadBox.querySelector('.file-format');
        fileFormatSpan.textContent = `Selected file: ${file.name}`;
        
        // Enable upload button
        uploadBtn.disabled = false;
        
        // Create form data for preview request
        const formData = new FormData();
        formData.append('file', file);
        
        // Generate preview
        fetch('/api/preview', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) throw new Error('Preview generation failed');
            return response.json();
        })
        .then(data => {
            previewContainer.classList.add('active');
            previewImage.innerHTML = `<img src="${data.image}" alt="DICOM Preview">`;
        })
        .catch(error => {
            console.error('Error:', error);
            previewContainer.classList.add('active');
            previewImage.innerHTML = `
                <div class="error-preview">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Error generating preview</p>
                </div>
            `;
        })
        .finally(() => {
            loadingOverlay.style.display = 'none';
        });
    }
    
    // Show loading overlay
    function showLoading() {
        loadingOverlay.style.display = 'flex';
    }
    
    // Hide loading overlay
    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }
    
    // Show error message
    function showError(title, details, similarityScore = null) {
        const errorContainer = document.getElementById('error-container');
        const errorTitle = document.getElementById('error-title');
        const errorDetails = document.getElementById('error-details');
        const similarityScoreDiv = document.getElementById('similarity-score');
        const scoreFill = document.getElementById('score-fill');
        const scoreText = document.getElementById('score-text');

        errorTitle.textContent = title;
        errorDetails.textContent = details;
        errorContainer.classList.remove('hidden');

        if (similarityScore !== null) {
            similarityScoreDiv.classList.remove('hidden');
            const scorePercent = Math.round(similarityScore * 100);
            scoreFill.style.width = `${scorePercent}%`;
            scoreText.textContent = `Similarity Score: ${scorePercent}%`;
        } else {
            similarityScoreDiv.classList.add('hidden');
        }
    }
    
    // Hide error message
    function hideError() {
        const errorContainer = document.getElementById('error-container');
        errorContainer.classList.add('hidden');
    }
    
    // Analyze the current image
    async function analyzeImage() {
        try {
            const fileInput = document.getElementById('file-input');
            const file = fileInput.files[0];
            
            if (!file) {
                showError('No File Selected', 'Please select a DICOM file to analyze.');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            showLoading();

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                hideLoading();
                if (data.code === 'INVALID_IMAGE') {
                    // Extract similarity score from the error message
                    const similarityMatch = data.details.match(/similarity: ([\d.]+)/);
                    const similarityScore = similarityMatch ? parseFloat(similarityMatch[1]) : null;
                    showError('Invalid Image Type', data.details, similarityScore);
                } else {
                    showError('Error', data.details || 'An error occurred during analysis.');
                }
                return;
            }

            hideLoading();
            hideError();
            displayResults(data);
        } catch (error) {
            hideLoading();
            showError('Error', 'An unexpected error occurred. Please try again.');
            console.error('Error:', error);
        }
    }
    
    // Display the analysis results
    function displayResults(data) {
        // Hide upload section
        uploadSection.style.display = 'none';
        // Show results section
        resultsSection.classList.remove('hidden');
        
        // Display the analyzed image
        imageContainer.innerHTML = `<img src="${data.image}" alt="Analyzed CT Scan">`;
        
        // Format the probability as a percentage
        const probabilityPercent = (data.probability * 100).toFixed(1);
        
        // Set the prediction text and styling
        const isPredictionPositive = data.prediction.includes('Nodule Detected');
        predictionContainer.innerHTML = `
            <h3 style="font-size: 1.5rem; margin-bottom: 15px;">${data.prediction}</h3>
            <p style="font-size: 2.5rem; font-weight: bold;">${probabilityPercent}%</p>
            <p style="margin-top: 10px;">confidence of nodule presence</p>
            ${isPredictionPositive ? `
                <div class="bbox-info" style="margin-top: 15px; font-size: 0.9rem;">
                    <p><i class="fas fa-crosshairs"></i> Nodule Location Detected</p>
                </div>
            ` : ''}
        `;
        
        // Update prediction container class
        predictionContainer.className = isPredictionPositive ? 'prediction-positive' : 'prediction-negative';
        
        // Update confidence meter
        meterFill.style.width = `${probabilityPercent}%`;
        meterFill.style.backgroundColor = isPredictionPositive ? 'var(--success-color)' : 'var(--primary-color)';
        
        // Update projection information
        projectionContainer.textContent = data.projection || 'Unknown';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Reset the interface for a new scan
    function resetInterface() {
        // Reset file input
        fileInput.value = '';
        currentFile = null;
        
        // Reset upload box
        uploadBox.classList.remove('has-file');
        const fileFormatSpan = uploadBox.querySelector('.file-format');
        fileFormatSpan.textContent = 'Supported format: .dcm';
        
        // Reset preview
        previewContainer.classList.remove('active');
        previewImage.innerHTML = '<p class="placeholder-text">Preview will appear here</p>';
        
        // Disable upload button
        uploadBtn.disabled = true;
        
        // Show upload section again
        uploadSection.style.display = 'block';
        
        // Hide results section
        resultsSection.classList.add('hidden');
        
        // Reset prediction container
        predictionContainer.innerHTML = `
            <div class="loading">
                <p>Waiting for analysis...</p>
            </div>
        `;
        predictionContainer.className = '';
        
        // Reset confidence meter
        meterFill.style.width = '0%';
        meterFill.style.backgroundColor = 'var(--primary-color)';
        
        // Reset projection
        projectionContainer.textContent = 'Waiting for analysis...';
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    
    // Initialize the interface
    function init() {
        resetInterface();
    }
    
    // Start the app
    init();
});