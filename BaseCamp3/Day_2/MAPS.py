import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import json
import polyline
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart Route Planner",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MapMyIndiaAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://atlas.mapmyindia.com"
        self.session = requests.Session()
        self.debug_mode = False
        # Set required headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def debug_print(self, message):
        """Print debug information if debug mode is enabled"""
        if self.debug_mode:
            st.write(f"🐛 Debug: {message}")
    
    def geocode(self, address):
        """Geocode an address to get coordinates using Atlas API"""
        # Use the correct Atlas API endpoint for geocoding
        url = f"{self.base_url}/api/places/geocode"
        
        params = {
            'address': address,
            'api_key': self.api_key
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            
            # Debug information
            self.debug_print(f"Geocoding Status Code: {response.status_code}")
            self.debug_print(f"Geocoding URL: {response.url}")
            
            if response.status_code == 200:
                data = response.json()
                self.debug_print(f"Geocoding Response: {data}")
                
                # Check different response formats
                if 'copResults' in data and data['copResults']:
                    result = data['copResults'][0]
                    return {
                        'lat': float(result['latitude']),
                        'lng': float(result['longitude']),
                        'formatted_address': result.get('formatted_address', address)
                    }
                elif 'results' in data and data['results']:
                    result = data['results'][0]
                    return {
                        'lat': float(result.get('lat', result.get('latitude', 0))),
                        'lng': float(result.get('lng', result.get('longitude', 0))),
                        'formatted_address': result.get('formatted_address', address)
                    }
            else:
                # Try alternative geocoding method
                return self.geocode_alternative(address)
                
        except requests.exceptions.RequestException as e:
            if self.debug_mode:
                st.error(f"Network error: {str(e)}")
            return self.geocode_alternative(address)
        except Exception as e:
            if self.debug_mode:
                st.error(f"Geocoding error: {str(e)}")
            return self.geocode_alternative(address)
        
        return None
    
    def geocode_alternative(self, address):
        """Alternative geocoding using different endpoint"""
        try:
            # Try with the legacy API format
            url = f"https://apis.mapmyindia.com/advancedmaps/v1/{self.api_key}/geo_code"
            params = {
                'addr': address
            }
            
            response = self.session.get(url, params=params, timeout=15)
            self.debug_print(f"Alternative Geocoding Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.debug_print(f"Alternative Geocoding Response: {data}")
                
                if 'results' in data and data['results']:
                    result = data['results'][0]
                    return {
                        'lat': float(result.get('lat', 0)),
                        'lng': float(result.get('lng', 0)),
                        'formatted_address': result.get('formatted_address', address)
                    }
            
            # If both fail, try a simple search API
            return self.search_place(address)
            
        except Exception as e:
            if self.debug_mode:
                st.error(f"Alternative geocoding failed: {str(e)}")
            return None
    
    def search_place(self, address):
        """Search for place using text search"""
        try:
            url = f"https://atlas.mapmyindia.com/api/places/search/json"
            params = {
                'query': address,
                'api_key': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=15)
            self.debug_print(f"Place Search Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.debug_print(f"Place Search Response: {data}")
                
                if 'suggestedLocations' in data and data['suggestedLocations']:
                    result = data['suggestedLocations'][0]
                    return {
                        'lat': float(result.get('latitude', 0)),
                        'lng': float(result.get('longitude', 0)),
                        'formatted_address': result.get('placeName', address)
                    }
                    
        except Exception as e:
            if self.debug_mode:
                st.error(f"Place search failed: {str(e)}")
        
        return None
    
    def get_route_with_traffic(self, start_coords, end_coords, profile="driving"):
        """Get route with traffic consideration"""
        try:
            # Try the new Atlas API format first
            url = f"https://apis.mapmyindia.com/advancedmaps/v1/{self.api_key}/route_adv/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
            
            params = {
                'geometries': 'polyline',
                'overview': 'full',
                'steps': 'true',
                'alternatives': 'true',
                'rtype': '1',  # Real-time traffic
                'with_traffic_incidents': 'true'
            }
            
            response = self.session.get(url, params=params, timeout=20)
            self.debug_print(f"Route Status Code: {response.status_code}")
            self.debug_print(f"Route URL: {response.url}")
            
            if response.status_code == 200:
                data = response.json()
                self.debug_print(f"Route Response keys: {list(data.keys()) if data else 'No data'}")
                
                if data.get('routes') and len(data['routes']) > 0:
                    return self.process_route_data(data)
                else:
                    # Try alternative route calculation
                    return self.get_route_alternative(start_coords, end_coords)
            else:
                self.debug_print(f"Route API returned status code: {response.status_code}")
                self.debug_print(f"Response: {response.text}")
                return self.get_route_fallback(start_coords, end_coords)
                
        except Exception as e:
            if self.debug_mode:
                st.error(f"Route calculation error: {str(e)}")
            return self.get_route_fallback(start_coords, end_coords)
    
    def get_route_alternative(self, start_coords, end_coords):
        """Alternative route calculation method"""
        try:
            # Simple route without advanced features
            url = f"https://apis.mapmyindia.com/advancedmaps/v1/{self.api_key}/route/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
            
            response = self.session.get(url, timeout=15)
            self.debug_print(f"Alternative Route Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('routes'):
                    return self.process_route_data(data)
                    
        except Exception as e:
            if self.debug_mode:
                st.error(f"Alternative route failed: {str(e)}")
        
        return None
    
    def get_route_fallback(self, start_coords, end_coords):
        """Fallback route calculation with basic information"""
        try:
            # Create a basic route with estimated values
            import math
            
            # Calculate straight-line distance (Haversine formula)
            lat1, lon1 = math.radians(start_coords[0]), math.radians(start_coords[1])
            lat2, lon2 = math.radians(end_coords[0]), math.radians(end_coords[1])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance_km = 6371 * c  # Earth's radius in km
            
            # Estimate travel time (assuming average speed of 30 km/h in city)
            estimated_time_min = (distance_km / 30) * 60
            
            fallback_route = [{
                'route_index': 0,
                'coordinates': [start_coords, end_coords],
                'distance_km': round(distance_km, 2),
                'duration_min': round(estimated_time_min, 1),
                'duration_traffic_min': round(estimated_time_min * 1.2, 1),  # Add 20% for traffic
                'traffic_delay_min': round(estimated_time_min * 0.2, 1),
                'geometry': '',
                'steps': []
            }]
            
            st.warning("Using estimated route data due to API limitations")
            return fallback_route
            
        except Exception as e:
            st.error(f"Fallback route calculation failed: {str(e)}")
            return None
    
    def process_route_data(self, data):
        """Process route data from API response"""
        routes = []
        
        for i, route in enumerate(data['routes']):
            try:
                # Decode polyline geometry
                geometry = route.get('geometry', '')
                coordinates = polyline.decode(geometry) if geometry else []
                
                # Extract route information
                duration = route.get('duration', 0)
                distance = route.get('distance', 0)
                duration_traffic = route.get('duration_in_traffic', duration)
                
                # Calculate traffic delay
                traffic_delay = max(0, duration_traffic - duration)
                
                route_info = {
                    'route_index': i,
                    'coordinates': coordinates,
                    'distance_km': round(distance / 1000, 2),
                    'duration_min': round(duration / 60, 1),
                    'duration_traffic_min': round(duration_traffic / 60, 1),
                    'traffic_delay_min': round(traffic_delay / 60, 1),
                    'geometry': geometry,
                    'steps': route.get('legs', [{}])[0].get('steps', []) if route.get('legs') else []
                }
                
                routes.append(route_info)
            except Exception as e:
                st.warning(f"Error processing route {i}: {str(e)}")
                continue
        
        # Sort by duration with traffic (shortest first)
        routes.sort(key=lambda x: x['duration_traffic_min'])
        return routes

def create_route_map(start_coords, end_coords, routes):
    """Create a folium map with routes"""
    # Calculate center point
    center_lat = (start_coords[0] + end_coords[0]) / 2
    center_lng = (start_coords[1] + end_coords[1]) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add start and end markers
    folium.Marker(
        start_coords,
        popup="Start Point",
        tooltip="Starting Location",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        end_coords,
        popup="Destination",
        tooltip="Destination",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # Add routes
    colors = ['blue', 'orange', 'purple', 'darkgreen', 'pink']
    
    for i, route in enumerate(routes[:3]):  # Show max 3 routes
        if route['coordinates']:
            color = colors[i % len(colors)]
            weight = 6 if i == 0 else 4  # Highlight best route
            
            folium.PolyLine(
                locations=route['coordinates'],
                color=color,
                weight=weight,
                opacity=0.8,
                popup=f"Route {i+1}: {route['distance_km']}km, {route['duration_traffic_min']}min"
            ).add_to(m)
    
    return m

def main():
    st.title("🗺️ Smart Route Planner with Traffic Intelligence")
    st.markdown("### Find the fastest route considering real-time traffic conditions")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", help="Show API response details")
    
    # Get API key from environment variable
    api_key = os.getenv('MAPMYINDIA_API_KEY')
    
    if not api_key:
        st.error("❌ MapMyIndia API key not found!")
        st.markdown("""
        **Setup Instructions:**
        1. Create a `.env` file in your project directory
        2. Add your API key: `MAPMYINDIA_API_KEY=your_actual_api_key_here`
        3. Get your API key from [MapMyIndia Developer Portal](https://apis.mapmyindia.com/)
        4. Restart the application
        """)
        st.stop()
    
    # Initialize API client
    mmi_api = MapMyIndiaAPI(api_key)
    mmi_api.debug_mode = debug_mode
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Starting Point")
        start_location = st.text_input(
            "Enter starting address",
            placeholder="e.g., Connaught Place, New Delhi"
        )
    
    with col2:
        st.subheader("🎯 Destination")
        end_location = st.text_input(
            "Enter destination address",
            placeholder="e.g., India Gate, New Delhi"
        )
    
    # Route calculation button
    if st.button("🚗 Find Best Route", type="primary", use_container_width=True):
        if not start_location or not end_location:
            st.error("Please enter both starting point and destination")
            return
        
        with st.spinner("🔍 Geocoding addresses..."):
            start_coords_data = mmi_api.geocode(start_location)
            end_coords_data = mmi_api.geocode(end_location)
        
        if not start_coords_data or not end_coords_data:
            st.error("Could not find one or both locations. Please check your addresses.")
            return
        
        start_coords = (start_coords_data['lat'], start_coords_data['lng'])
        end_coords = (end_coords_data['lat'], end_coords_data['lng'])
        
        # Display geocoded addresses
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✓ Start: {start_coords_data['formatted_address']}")
        with col2:
            st.success(f"✓ Destination: {end_coords_data['formatted_address']}")
        
        with st.spinner("🛣️ Calculating optimal routes with traffic data..."):
            routes = mmi_api.get_route_with_traffic(start_coords, end_coords)
        
        if not routes:
            st.error("Could not calculate routes. Please try again.")
            return
        
        # Display route information
        st.subheader("📊 Route Options")
        
        for i, route in enumerate(routes[:3]):
            with st.expander(f"Route {i+1} {'(Recommended)' if i == 0 else ''}", expanded=(i == 0)):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Distance", f"{route['distance_km']} km")
                
                with col2:
                    st.metric("Normal Time", f"{route['duration_min']} min")
                
                with col3:
                    st.metric(
                        "With Traffic", 
                        f"{route['duration_traffic_min']} min",
                        delta=f"+{route['traffic_delay_min']} min" if route['traffic_delay_min'] > 0 else "No delay"
                    )
                
                with col4:
                    if route['traffic_delay_min'] > 0:
                        st.error(f"⚠️ {route['traffic_delay_min']} min delay")
                    else:
                        st.success("✅ Clear roads")
                
                # Show turn-by-turn directions for the best route
                if i == 0 and route['steps']:
                    with st.expander("📋 Turn-by-turn Directions"):
                        for j, step in enumerate(route['steps'][:10]):  # Show first 10 steps
                            instruction = step.get('maneuver', {}).get('instruction', 'Continue')
                            distance = step.get('distance', 0)
                            st.write(f"{j+1}. {instruction} ({distance}m)")
        
        # Create and display map
        st.subheader("🗺️ Route Visualization")
        route_map = create_route_map(start_coords, end_coords, routes)
        st_folium(route_map, width=None, height=500)
        
        # Traffic insights
        best_route = routes[0]
        st.subheader("🚦 Traffic Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            if best_route['traffic_delay_min'] > 5:
                st.warning(f"Heavy traffic detected! Extra {best_route['traffic_delay_min']} minutes expected.")
            elif best_route['traffic_delay_min'] > 0:
                st.info(f"Light traffic. Additional {best_route['traffic_delay_min']} minutes expected.")
            else:
                st.success("Clear roads ahead! No traffic delays expected.")
        
        with col2:
            current_time = datetime.now().strftime("%I:%M %p")
            st.info(f"🕐 Route calculated at {current_time}")
            st.caption("Traffic conditions are updated in real-time")

if __name__ == "__main__":
    main()