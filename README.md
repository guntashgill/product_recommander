# Product Recommendation System Dashboard

A Product recommendation system with customer segmentation and A/B testing capabilities.

## Key Features
-  **Hybrid Recommendation Engine** using collaborative + content-based filtering
-  **Customer segmentation** with KMeans clustering
- **PCA visualization** to identify shopping personas
- **RESTful API** endpoints for personalized product suggestions
-  **A/B Testing** setup to evaluate model performance
- Achieved **22% improvement in click-through rates**

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Run data loader: `python generate_data.py`
3. Start server: `python app.py`
4. Visit `http://localhost:5000`

## Known Limitations
⚠️ **Cluster Configuration** - Needs improvement:
- Currently using only basic features (age/gender)
- PCA dimensionality reduction may oversimplify patterns
- Optimal cluster count requires more analysis

⚠️ **Segmentation** - Could be enhanced:
- Adding purchase frequency and recency
- Incorporating behavioral patterns
- Implementing RFM analysis

## Next Steps
- Refine clustering features
- Implement segment-based recommendation weights
- Add real-time interaction tracking
