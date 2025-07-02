# Product Recommendation System Dashboard

A Flask-based recommendation system with customer segmentation and A/B testing capabilities.

## Key Features
- Three recommendation strategies (content-based, collaborative, hybrid)
- Customer segmentation by demographics and interests
- A/B testing framework
- Interactive web dashboard
- Customer cluster visualization

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