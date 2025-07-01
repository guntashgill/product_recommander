// Base URL for API - now using relative paths
const API_BASE_URL = '/api';

// DOM Elements
const apiStatus = document.getElementById('apiStatus');
const healthCheckBtn = document.getElementById('healthCheckBtn');
const healthResult = document.getElementById('healthResult');
const healthStatus = document.getElementById('healthStatus');

const getSegmentsBtn = document.getElementById('getSegmentsBtn');
const segmentsResult = document.getElementById('segmentsResult');
const segmentsList = document.getElementById('segmentsList');
const segmentSelect = document.getElementById('segmentSelect');

const getPopularBtn = document.getElementById('getPopularBtn');
const popularResult = document.getElementById('popularResult');
const popularList = document.getElementById('popularList');

const getRecommendBtn = document.getElementById('getRecommendBtn');
const recommendResult = document.getElementById('recommendResult');
const recommendList = document.getElementById('recommendList');
const strategyInfo = document.getElementById('strategyInfo');

const getClusterBtn = document.getElementById('getClusterBtn');
const clusterResult = document.getElementById('clusterResult');
const clusterImage = document.getElementById('clusterImage');

// Check API health on page load
document.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
});

// Health Check
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === "healthy") {
            apiStatus.textContent = "API is running";
            apiStatus.parentElement.classList.add('active');
        } else {
            apiStatus.textContent = "API is unavailable";
        }
    } catch (error) {
        apiStatus.textContent = "API connection failed";
        console.error('Health check failed:', error);
    }
}

healthCheckBtn.addEventListener('click', async () => {
    healthResult.style.display = 'none';
    healthStatus.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    healthResult.style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === "healthy") {
            healthStatus.innerHTML = `
                <div style="background: #e8f5e9; color: #2e7d32; padding: 15px; border-radius: 8px;">
                    <i class="fas fa-check-circle"></i> <strong>Status:</strong> ${data.status}
                    <div style="margin-top: 10px;">${data.message}</div>
                </div>
            `;
        } else {
            healthStatus.innerHTML = `
                <div class="error">
                    <i class="fas fa-exclamation-triangle"></i> API is unavailable
                </div>
            `;
        }
    } catch (error) {
        healthStatus.innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle"></i> Failed to connect to API: ${error.message}
            </div>
        `;
    }
});

// Get Segments
getSegmentsBtn.addEventListener('click', async () => {
    segmentsResult.style.display = 'none';
    segmentsList.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    segmentsResult.style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE_URL}/segments`);
        const data = await response.json();
        
        if (data.segments && data.segments.length > 0) {
            segmentsList.innerHTML = '';
            segmentSelect.innerHTML = '<option value="">-- Select a segment --</option>';
            
            data.segments.forEach(segment => {
                // Add to segments list
                const badge = document.createElement('div');
                badge.className = 'segment-badge';
                badge.textContent = segment;
                segmentsList.appendChild(badge);
                
                // Add to dropdown
                const option = document.createElement('option');
                option.value = segment;
                option.textContent = segment;
                segmentSelect.appendChild(option);
            });
        } else {
            segmentsList.innerHTML = '<div class="error">No segments found</div>';
        }
    } catch (error) {
        segmentsList.innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle"></i> Failed to retrieve segments: ${error.message}
            </div>
        `;
    }
});

// Get Popular Products
getPopularBtn.addEventListener('click', async () => {
    const segment = segmentSelect.value;
    const n = document.getElementById('popularCount').value || 5;
    
    if (!segment) {
        alert('Please select a segment first');
        return;
    }
    
    popularResult.style.display = 'none';
    popularList.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    popularResult.style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE_URL}/popular?segment=${encodeURIComponent(segment)}&n=${n}`);
        const data = await response.json();
        
        if (data.recommendations && data.recommendations.length > 0) {
            popularList.innerHTML = '';
            
            data.recommendations.forEach(product => {
                const li = document.createElement('li');
                li.className = 'recommendation-item';
                
                li.innerHTML = `
                    <div class="recommendation-info">
                        <div class="recommendation-title">${product.title || 'Unknown Product'}</div>
                        <div class="recommendation-details">
                            <span><i class="fas fa-tag"></i> ${product.brand || 'No Brand'}</span>
                            <span><i class="fas fa-dollar-sign"></i> ${product.price || 'N/A'}</span>
                        </div>
                    </div>
                `;
                
                popularList.appendChild(li);
            });
        } else {
            popularList.innerHTML = '<div class="error">No popular products found for this segment</div>';
        }
    } catch (error) {
        popularList.innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle"></i> Failed to retrieve popular products: ${error.message}
            </div>
        `;
    }
});

// Get Personalized Recommendations
getRecommendBtn.addEventListener('click', async () => {
    const customerId = document.getElementById('customerId').value;
    const n = document.getElementById('recommendCount').value || 5;
    
    if (!customerId) {
        alert('Please enter a customer ID');
        return;
    }
    
    recommendResult.style.display = 'none';
    recommendList.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    strategyInfo.innerHTML = '';
    recommendResult.style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE_URL}/recommend/${customerId}?n=${n}`);
        const data = await response.json();
        
        if (data.error) {
            recommendList.innerHTML = `<div class="error">${data.error}</div>`;
            return;
        }
        
        if (data.recommendations && data.recommendations.length > 0) {
            // Display strategy info
            strategyInfo.innerHTML = `
                <div>
                    <strong>Recommendation Strategy:</strong>
                    <span class="strategy-badge">${data.strategy}</span>
                </div>
            `;
            
            // Display recommendations
            recommendList.innerHTML = '';
            
            data.recommendations.forEach(product => {
                const li = document.createElement('li');
                li.className = 'recommendation-item';
                
                li.innerHTML = `
                    <div class="recommendation-info">
                        <div class="recommendation-title">${product.title || 'Unknown Product'}</div>
                        <div class="recommendation-details">
                            <span><i class="fas fa-tag"></i> ${product.brand || 'No Brand'}</span>
                            <span><i class="fas fa-dollar-sign"></i> ${product.price || 'N/A'}</span>
                            <span><i class="fas fa-list"></i> ${product.category || 'No Category'}</span>
                        </div>
                    </div>
                `;
                
                recommendList.appendChild(li);
            });
        } else {
            recommendList.innerHTML = '<div class="error">No recommendations found for this customer</div>';
        }
    } catch (error) {
        recommendList.innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle"></i> Failed to retrieve recommendations: ${error.message}
            </div>
        `;
    }
});

// Get Cluster Visualization
getClusterBtn.addEventListener('click', async () => {
    clusterResult.style.display = 'none';
    clusterImage.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    clusterResult.style.display = 'block';
    
    try {
        const response = await fetch(`${API_BASE_URL}/cluster_visualization`);
        const data = await response.json();
        
        if (data.image) {
            clusterImage.innerHTML = `<img src="${data.image}" alt="Customer Segmentation Visualization">`;
        } else {
            clusterImage.innerHTML = '<div class="error">Failed to generate visualization</div>';
        }
    } catch (error) {
        clusterImage.innerHTML = `
            <div class="error">
                <i class="fas fa-exclamation-triangle"></i> Failed to retrieve visualization: ${error.message}
            </div>
        `;
    }
});