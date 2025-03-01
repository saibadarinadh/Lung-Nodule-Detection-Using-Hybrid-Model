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
        // Check if file is a DICOM file
        if (!file.name.toLowerCase().endsWith('.dcm')) {
            alert('Please select a DICOM (.dcm) file');
            return;
        }
        
        currentFile = file;
        uploadBox.classList.add('has-file');
        uploadBtn.disabled = false;
        
        // Display file name in the upload box
        const fileFormatSpan = uploadBox.querySelector('.file-format');
        fileFormatSpan.textContent = `Selected file: ${file.name}`;
        
        // Show loading overlay while generating preview
        loadingOverlay.style.display = 'flex';
        
        // Create form data for preview request
        const formData = new FormData();
        formData.append('file', file);
        
        // Send request to get a preview of the DICOM image
        fetch('/api/preview', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Display the preview image
            previewContainer.classList.add('active');
            previewImage.innerHTML = `<img src="${data.image}" alt="DICOM Preview">`;
        })
        .catch(error => {
            console.error('Error generating preview:', error);
            // Fallback to icon view if preview fails
            previewContainer.classList.add('active');
            previewImage.innerHTML = `
                <div style="text-align: center;">
                    <i class="fas fa-file-medical" style="font-size: 3rem; color: var(--primary-color);"></i>
                    <p style="margin-top: 10px;">${file.name}</p>
                    <p style="margin-top: 5px; font-size: 0.9rem; color: var(--text-light);">DICOM file ready for analysis</p>
                </div>
            `;
        })
        .finally(() => {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
        });
    }
    
    // Analyze the current image
    function analyzeImage() {
        if (!currentFile) return;
        
        // Show loading overlay
        loadingOverlay.style.display = 'flex';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', currentFile);
        
        // Send request to the server
        fetch('/api/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide upload section
            uploadSection.style.display = 'none';
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during analysis. Please try again.');
        })
        .finally(() => {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
        });
    }
    
    // Display the analysis results
    function displayResults(data) {
        // Show results section
        resultsSection.classList.remove('hidden');
        
        // Display the image
        imageContainer.innerHTML = `<img src="${data.image}" alt="CT Scan">`;
        
        // Format the probability as a percentage
        const probabilityPercent = (data.probability * 100).toFixed(1);
        
        // Set the prediction text and styling
        const isPredictionPositive = data.prediction.includes('Nodule Detected');
        predictionContainer.innerHTML = `
            <h3 style="font-size: 1.5rem; margin-bottom: 15px;">${data.prediction}</h3>
            <p style="font-size: 2.5rem; font-weight: bold;">${probabilityPercent}%</p>
            <p style="margin-top: 10px;">confidence of nodule presence</p>
        `;
        
        // Add the appropriate class for styling
        predictionContainer.className = isPredictionPositive 
            ? 'prediction-positive' 
            : 'prediction-negative';
        
        // Update the confidence meter
        meterFill.style.width = `${probabilityPercent}%`;
        
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