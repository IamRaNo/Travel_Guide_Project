import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

preprocessor = joblib.load('preprocessor.pkl')
data = joblib.load('travel_data.pkl')

st.title("🧭 Travel Recommendation System")

# Step 1: Take user input from UI
zone = st.selectbox("Select Zone", sorted(data['zone'].unique()))
type_ = st.selectbox("Type of Place", sorted(data['type'].unique()))
established = st.selectbox("Established", sorted(data['established'].unique()))
time_to_visit = st.selectbox("Preferred Visit Duration", sorted(data['time to visit'].unique()))
entrance_fee = st.selectbox("Entrance Fee Preference", sorted(data['entrance fee'].unique()))
airport = st.selectbox("Airport Nearby?", sorted(data['airport nearby'].unique()))
dslr = st.selectbox("DSLR Allowed?", sorted(data['dslr allowed'].unique()))
best_time = st.selectbox("Best Time to Visit", sorted(data['best time'].unique()))
footfall = st.selectbox("Footfall Level", sorted(data['footfall'].unique()))

# Step 2: When the user clicks the button
if st.button("🎯 Get Recommendations"):
    # Create user input DataFrame (match columns exactly)
    user_input = pd.DataFrame([{
        'zone': zone,
        'type': type_,
        'established': established,
        'time to visit': time_to_visit,
        'entrance fee': entrance_fee,
        'airport nearby': airport,
        'dslr allowed': dslr,
        'best time': best_time,
        'foot fall': footfall  # ⚠️ Must match original column name
    }])

    # Filter dataset based on zone
    filtered_data = data[data['zone'] == zone]

    if filtered_data.empty:
        st.warning("❌ No recommendations found for the selected zone.")
    else:
        # Drop target column
        X_filtered = filtered_data.drop(columns=['name'])

        # Transform filtered data and user input
        X_filtered_processed = preprocessor.transform(X_filtered)
        user_processed = preprocessor.transform(user_input)

        # Fit and get neighbors
        knn = NearestNeighbors(
            n_neighbors=min(3, len(filtered_data)),
            metric='cosine',
            algorithm='brute',
            n_jobs=-1
        )
        knn.fit(X_filtered_processed)
        distances, indices = knn.kneighbors(user_processed)

        # Get recommendations
        recommended_places = filtered_data.iloc[indices[0]]['name'].values

        st.success("✅ Recommended Places:")
        for place in recommended_places:
            st.markdown(f"- **{place}**")
