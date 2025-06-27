import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

# Load files
preprocessor = joblib.load('preprocessor.pkl')
data = joblib.load('data.pkl')

# App Title
st.set_page_config(page_title="Travel Recommendation System", page_icon="🧭")
st.title("🧭 Travel Recommendation System")
st.markdown("#### Discover hand-picked destinations tailored to your travel preferences! 🌍")

# Sidebar for inputs
st.sidebar.header("🧳 Your Preferences")

zone = st.sidebar.selectbox("📍 Select Zone", sorted(data['zone'].unique()))
type_ = st.sidebar.selectbox("🏞️ Type of Place", sorted(data['type'].unique()))
established = st.sidebar.selectbox("🏛️ Established", sorted(data['established'].unique()))
time_to_visit = st.sidebar.selectbox("🕒 Preferred Visit Duration", sorted(data['time to visit'].unique()))
entrance_fee = st.sidebar.selectbox("💵 Entrance Fee Preference", sorted(data['entrance fee'].unique()))
airport = st.sidebar.selectbox("✈️ Airport Nearby?", sorted(data['airport nearby'].unique()))
dslr = st.sidebar.selectbox("📸 DSLR Allowed?", sorted(data['dslr allowed'].unique()))
best_time = st.sidebar.selectbox("📅 Best Time to Visit", sorted(data['best time'].unique()))
foot_fall = st.sidebar.selectbox("👣 Footfall Level", sorted(data['foot fall'].unique()))

st.markdown("---")

# Main recommendation section
if st.button("🎯 Get My Recommendations"):
    with st.spinner("Finding the best places for you... 🔍"):
        user_input = pd.DataFrame([{
            'zone': zone,
            'type': type_,
            'established': established,
            'time to visit': time_to_visit,
            'entrance fee': entrance_fee,
            'airport nearby': airport,
            'dslr allowed': dslr,
            'best time': best_time,
            'foot fall': foot_fall
        }])

        # Filter and process
        filtered_data = data[data['zone'] == zone]

        if filtered_data.empty:
            st.warning("❌ Sorry, no recommendations found for the selected zone.")
        else:
            X_filtered = filtered_data.drop(columns=['name'])
            X_filtered_processed = preprocessor.transform(X_filtered)
            user_processed = preprocessor.transform(user_input)

            knn = NearestNeighbors(
                n_neighbors=min(3, len(filtered_data)),
                metric='cosine',
                algorithm='brute',
                n_jobs=-1
            )
            knn.fit(X_filtered_processed)
            distances, indices = knn.kneighbors(user_processed)

            recommended_places = filtered_data.iloc[indices[0]]['name'].values

            st.success(f"🎉 Top Recommendations for **{zone}** zone:")
            for i, place in enumerate(recommended_places, start=1):
                st.markdown(f"**{i}. {place}**")

            st.balloons()
st.markdown("---")
st.caption("Built with 💙 by Rano using Streamlit and scikit-learn")
