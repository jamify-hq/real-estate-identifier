import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time

# Page configuration
st.set_page_config(
    page_title="EdgeNebula Site Finder",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 1rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .search-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Add near the top of the file, after imports
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False
    st.session_state.search_complete = False
    st.session_state.search_results = None
    st.session_state.search_coords = None

# Cache the data loading
@st.cache_data
def load_data():
    try:
        usecols = [
            'SubstationAlias',
            'SubstationVoltage',
            'demand_headroom',
            'Postcode',
            'Latitude',
            'Longitude',
            'BuildingArea',
            'BuildingAddress'
        ]
        # Remove nrows limit to load full dataset
        df = pd.read_csv("data/ukpn-secondary-sites-3.csv", usecols=usecols)
        
        # Add progress indicator
        st.info(f"📊 Loaded {len(df):,} sites from the database")
        
        # Convert coordinates to numeric immediately
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        # Pre-calculate power estimates for better performance
        df['estimated_power'] = df['demand_headroom'].apply(estimate_power)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def geocode_location(location_query):
    """Geocode a location string to coordinates"""
    try:
        geolocator = Nominatim(user_agent="edgenebula_app")
        location = geolocator.geocode(location_query)
        if location:
            return location.latitude, location.longitude
        return None
    except Exception:
        return None

def estimate_power(utilization):
    """Convert utilization band to estimated power"""
    if not isinstance(utilization, str):
        return 0
    
    band_map = {
        "0-20%": 1800,   # 90% of 2MW
        "20-40%": 1400,  # 70% of 2MW
        "40-60%": 1000,  # 50% of 2MW
        "60-80%": 600,   # 30% of 2MW
        "80-100%": 200   # 10% of 2MW
    }
    return band_map.get(utilization, 0)

def calculate_distances(search_coords, df):
    """Calculate distances from search location to all substations"""
    # Only calculate for rows with valid coordinates
    mask = df['Latitude'].notna() & df['Longitude'].notna()
    distances = pd.Series(index=df.index, dtype=float)
    distances.loc[~mask] = float('inf')
    
    # Vectorized distance calculation for valid coordinates
    valid_coords = df[mask]
    if not valid_coords.empty:
        distances.loc[mask] = valid_coords.apply(
            lambda row: geodesic(
                search_coords, 
                (row['Latitude'], row['Longitude'])
            ).kilometers,
            axis=1
        )
    return distances

# Main app layout
st.title("🏢 EdgeNebula Site Finder")

# Search container
with st.container():
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2,1,1])
    
    with col1:
        location = st.text_input("Enter City or Postcode", placeholder="e.g., London or SW1A 1AA")
    
    with col2:
        min_power = st.slider(
            "Min Power Required (kW)",
            min_value=0,
            max_value=2000,
            value=500,
            step=100
        )
    
    with col3:
        col3_container = st.container()
        if st.session_state.search_performed:
            if col3_container.button("🔄 New Search"):
                st.session_state.search_performed = False
                st.session_state.search_complete = False
                st.rerun()
        else:
            if col3_container.button("🔍 Find Sites"):
                st.session_state.search_performed = True
                st.session_state.location = location
                st.session_state.min_power = min_power
                st.session_state.search_complete = False
    
    st.markdown('</div>', unsafe_allow_html=True)

# Load data
df = load_data()

if df is not None and st.session_state.search_performed:
    if not st.session_state.search_complete:
        with st.spinner('Searching for viable sites...'):
            # Use stored values from session state
            location = st.session_state.location
            min_power = st.session_state.min_power
            
            # Geocode search location
            search_coords = geocode_location(location)
            
            if not search_coords:
                st.error("Could not find the specified location. Please try again.")
            else:
                # Store search coordinates
                st.session_state.search_coords = search_coords
                
                # Calculate distances
                df['distance'] = calculate_distances(search_coords, df)
                
                # Filter results
                filtered_df = df[
                    (df['distance'] <= 10) &  # Within 10km
                    (df['estimated_power'] >= min_power) &
                    (df['Latitude'].notna()) &
                    (df['Longitude'].notna())
                ].copy()
                
                # Store results in session state
                st.session_state.search_results = filtered_df
                st.session_state.search_complete = True
    
    # Use stored results to display
    filtered_df = st.session_state.search_results
    search_coords = st.session_state.search_coords
    
    if filtered_df is None or len(filtered_df) == 0:
        st.warning("No suitable sites found in this area. Try adjusting your criteria.")
    else:
        # Add summary metrics
        total_power = filtered_df['estimated_power'].sum()
        avg_distance = filtered_df['distance'].mean()
        
        # Create metrics row
        metric1, metric2, metric3, metric4 = st.columns(4)
        with metric1:
            st.metric("Sites Found", f"{len(filtered_df):,}")
        with metric2:
            st.metric("Total Available Power", f"{total_power:,.0f} kW")
        with metric3:
            st.metric("Avg Distance", f"{avg_distance:.1f} km")
        with metric4:
            st.metric("High Capacity Sites (>1MW)", 
                    len(filtered_df[filtered_df['estimated_power'] >= 1000]))
        
        # Create two columns for map and table
        col1, col2 = st.columns([2,1])
        
        with col1:
            st.subheader("📍 Available Sites")
            # Create map centered on search location
            m = folium.Map(
                location=search_coords,
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            # Add search location marker
            folium.Marker(
                location=search_coords,
                popup="Search Location",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add markers for each substation (limit to closest 100 for performance)
            for idx, row in filtered_df.nsmallest(100, 'distance').iterrows():
                color = 'green' if row['estimated_power'] >= 1000 else 'orange'
                
                popup_html = f"""
                    <h4>{row['SubstationAlias']}</h4>
                    <b>Available Power:</b> {row['estimated_power']:,.0f} kW<br>
                    <b>Distance:</b> {row['distance']:.1f} km<br>
                    <b>Address:</b> {row['BuildingAddress']}<br>
                    <b>Area:</b> {row.get('BuildingArea', 'N/A')} m²
                """
                
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=8,
                    popup=folium.Popup(popup_html, max_width=300),
                    color=color,
                    fill=True,
                    fill_color=color
                ).add_to(m)
            
            # Display map
            st_folium(m, height=600)
            
            if len(filtered_df) > 100:
                st.info("ℹ️ Showing closest 100 sites on map for performance")
        
        with col2:
            st.subheader("📋 Site Details")
            
            # Add download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                "edgenebula_sites.csv",
                "text/csv",
                key='download-csv'
            )
            
            # Display table
            st.dataframe(
                filtered_df[[
                    'SubstationAlias',
                    'estimated_power',
                    'distance',
                    'BuildingArea',
                    'Postcode'
                ]].sort_values('distance'),
                hide_index=True,
                column_config={
                    'SubstationAlias': 'Site',
                    'estimated_power': 'Power (kW)',
                    'distance': 'Distance (km)',
                    'BuildingArea': 'Area (m²)',
                    'Postcode': 'Postcode'
                }
            )
