from flask import Flask, render_template, request, jsonify
import networkx as nx
from geopy.distance import geodesic
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
import requests
import openai
from flask_cors import CORS
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key= os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

# Predefined list of Indian airports with (Latitude, Longitude, Label)
airports = {
    "DEL": (28.5562, 77.1000, "Indira Gandhi International Airport"),
    "BOM": (19.0896, 72.8656, "Chhatrapati Shivaji Maharaj International Airport"),
    "BLR": (13.1986, 77.7066, "Kempegowda International Airport"),
    "MAA": (12.9941, 80.1709, "Chennai International Airport"),
    "HYD": (17.2403, 78.4294, "Rajiv Gandhi International Airport"),
    "CCU": (22.6547, 88.4467, "Netaji Subhash Chandra Bose International Airport"),
    "GOI": (15.3800, 73.8312, "Goa International Airport"),
    "AMD": (23.0732, 72.6347, "Sardar Vallabhbhai Patel International Airport"),
    "PNQ": (18.5822, 73.9197, "Pune Airport"),
    "COK": (10.1556, 76.3910, "Cochin International Airport"),
    "TRV": (8.4821, 76.9204, "Trivandrum International Airport"),
    "JAI": (26.8242, 75.8122, "Jaipur International Airport"),
    "LKO": (26.7606, 80.8893, "Chaudhary Charan Singh International Airport"),
    "IXC": (30.6735, 76.7885, "Chandigarh International Airport"),
    "GAU": (26.1061, 91.5859, "Lokpriya Gopinath Bordoloi International Airport"),
    "PAT": (25.5913, 85.0877, "Jay Prakash Narayan International Airport"),
    "BBI": (20.2444, 85.8178, "Biju Patnaik International Airport"),
    "IXB": (26.6812, 88.3286, "Bagdogra International Airport"),
    "NAG": (21.0922, 79.0472, "Dr. Babasaheb Ambedkar International Airport"),
    "UDR": (24.6177, 73.7415, "Maharana Pratap Airport"),
    "IXR": (23.3143, 85.3214, "Birsa Munda Airport"),
    "SXR": (34.0056, 74.3805, "Sheikh ul-Alam International Airport"),
    "IXZ": (11.6410, 92.7297, "Veer Savarkar International Airport"),
    "VGA": (16.5304, 80.7968, "Vijayawada Airport"),
    "VTZ": (17.7211, 83.2245, "Visakhapatnam Airport"),
    "IXM": (9.8345, 78.0934, "Madurai Airport"),
    "TIR": (13.6325, 79.5434, "Tirupati Airport"),
    "HBX": (15.3617, 75.0849, "Hubli Airport"),
    "NDC": (19.1833, 77.3168, "Nanded Airport"),
    "IDR": (22.7215, 75.8011, "Devi Ahilyabai Holkar Airport"),
    "BHO": (23.2875, 77.3378, "Raja Bhoj Airport"),
    "JLR": (23.1778, 80.0520, "Jabalpur Airport"),
    "RPR": (21.1804, 81.7388, "Swami Vivekananda Airport"),
    "DIB": (27.4839, 95.0179, "Dibrugarh Airport"),
    "IMF": (24.7600, 93.8967, "Imphal Airport"),
    "DMU": (25.8839, 93.7714, "Dimapur Airport"),
    "AJL": (23.8400, 92.6197, "Lengpui Airport"),
    "SHL": (25.7023, 91.9787, "Shillong Airport"),
    "IXS": (24.9104, 92.9787, "Silchar Airport"),
    "IXA": (23.8860, 91.2404, "Agartala Airport"),
    "IXJ": (32.6886, 74.8379, "Jammu Airport"),
    "DHM": (32.1651, 76.2634, "Kangra Airport"),
    "DED": (30.1897, 78.1803, "Dehradun Airport"),
    "GAY": (24.7460, 84.9512, "Gaya Airport"),
    "VTZ": (17.7211, 83.2245, "Visakhapatnam Airport"),
    "IXL": (34.1359, 77.5465, "Kushok Bakula Rimpochee Airport"),
    "TCR": (8.7247, 78.0249, "Tuticorin Airport"),
    "BHU": (21.7522, 72.1852, "Bhavnagar Airport"),
    "BKB": (28.0700, 73.2075, "Nal Airport"),
    "RAJ": (22.3092, 70.7794, "Rajkot Airport"),
    "JGA": (22.4655, 70.0111, "Jamnagar Airport"),
    "UDR": (24.6177, 73.7415, "Maharana Pratap Airport"),
    "BOM": (19.0896, 72.8656, "Chhatrapati Shivaji Maharaj International Airport"),
    "GOI": (15.3800, 73.8312, "Goa International Airport"),
    "AMD": (23.0732, 72.6347, "Sardar Vallabhbhai Patel International Airport"),
    "BDQ": (22.3362, 73.2264, "Vadodara Airport"),
    "BHJ": (23.2875, 69.6701, "Bhuj Airport"),
    "JDH": (26.2518, 73.0484, "Jodhpur Airport"),
    "JAI": (26.8242, 75.8122, "Jaipur International Airport"),
    "IDR": (22.7215, 75.8011, "Devi Ahilyabai Holkar Airport"),
    "BHO": (23.2875, 77.3378, "Raja Bhoj Airport"),
    "RPR": (21.1804, 81.7388, "Swami Vivekananda Airport"),
    "NAG": (21.0922, 79.0472, "Dr. Babasaheb Ambedkar International Airport"),
    "PBD": (21.6487, 69.6573, "Porbandar Airport"),
    "IXE": (12.9613, 74.8896, "Mangalore Airport"),
    "HBX": (15.3617, 75.0849, "Hubli Airport"),
    "BLR": (13.1986, 77.7066, "Kempegowda International Airport"),
    "COK": (10.1556, 76.3910, "Cochin International Airport"),
    "TRV": (8.4821, 76.9204, "Trivandrum International Airport"),
    "MAA": (12.9941, 80.1709, "Chennai International Airport"),
    "IXM": (9.8345, 78.0934, "Madurai Airport"),
    "TIR": (13.6325, 79.5434, "Tirupati Airport"),
    "VTZ": (17.7211, 83.2245, "Visakhapatnam Airport"),
    "BOM": (19.0896, 72.8656, "Chhatrapati Shivaji Maharaj International Airport"),
    "GOI": (15.3800, 73.8312, "Goa International Airport"),
    "PNQ": (18.5822, 73.9197, "Pune Airport"),
    "NDC": (19.1833, 77.3168, "Nanded Airport")
}


G = nx.Graph()

# Adding nodes to the graph
for airport in airports:
    G.add_node(airport)

# Function to calculate distance between two coordinates
def calculate_distance(coord1, coord2):
    return geodesic(coord1, coord2).km

# Connect all nodes with a minimum spanning tree to ensure all nodes are connected
def add_spanning_tree_edges(graph, airports):
    coords = np.array([(data[0], data[1]) for data in airports.values()])
    nbrs = NearestNeighbors(n_neighbors=len(coords), algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    spanning_tree_edges = set()
    
    # Prim's algorithm to create a minimum spanning tree
    visited = set()
    edges = [(distances[0][i], 0, i) for i in range(1, len(coords))]
    visited.add(0)
    
    while edges:
        edges.sort(key=lambda x: x[0])
        dist, u, v = edges.pop(0)
        if v not in visited:
            visited.add(v)
            spanning_tree_edges.add((u, v))
            for i in range(len(coords)):
                if i not in visited:
                    edges.append((calculate_distance(coords[v], coords[i]), v, i))
    
    for u, v in spanning_tree_edges:
        airport_u = list(airports.keys())[u]
        airport_v = list(airports.keys())[v]
        distance = calculate_distance(coords[u], coords[v])
        graph.add_edge(airport_u, airport_v, weight=distance)

# Add edges from DEL, BOM, and BLR to ensure they are central hubs
def add_central_hub_edges(graph, airports, central_hubs, num_connections=10):
    nodes = list(graph.nodes)
    for hub in central_hubs:
        connected_airports = random.sample([node for node in nodes if node != hub], num_connections)
        for airport in connected_airports:
            if not graph.has_edge(hub, airport):
                distance = calculate_distance((airports[hub][0], airports[hub][1]),
                                              (airports[airport][0], airports[airport][1]))
                graph.add_edge(hub, airport, weight=distance)

# Add multiple routes to ensure at least three different routes between any two airports
def add_multiple_routes_edges(graph, airports, num_routes=3):
    nodes = list(graph.nodes)
    for node in nodes:
        connections = list(graph[node])
        while len(connections) < num_routes:
            target = random.choice(nodes)
            if target != node and not graph.has_edge(node, target):
                distance = calculate_distance((airports[node][0], airports[node][1]),
                                              (airports[target][0], airports[target][1]))
                graph.add_edge(node, target, weight=distance)
                connections.append(target)

# Add spanning tree edges
add_spanning_tree_edges(G, airports)

# Add central hub edges
central_hubs = ["DEL", "BOM", "BLR"]
add_central_hub_edges(G, airports, central_hubs)

# Add multiple routes to ensure at least three different routes between any two airports
add_multiple_routes_edges(G, airports, num_routes=3)

# Heuristic function for A* search (geographical distance to target)
def heuristic(node, goal):
    return calculate_distance((airports[node][0], airports[node][1]), (airports[goal][0], airports[goal][1]))

# Function to find the best route using A* search
def find_best_route(start, end):
    path = nx.astar_path(G, start, end, heuristic=heuristic, weight="weight")
    return [(airport, airports[airport]) for airport in path]

# Function to get weather data from OpenWeatherMap API
def get_weather_data(lat, lon):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_description = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]
        wind_deg = data["wind"]["deg"]
        return {
            "description": weather_description,
            "wind_speed": wind_speed,
            "wind_deg": wind_deg
        }
    else:
        return {
            "description": "N/A",
            "wind_speed": "N/A",
            "wind_deg": "N/A"
        }

# Function to get location description from OpenAI API
def get_location_description(location):
    prompt = f"Give me a description of {location} to tell the airplane passengers, I am the pilot. It should be in 50 words. Also add 20 words about latest event/celebration/current going on that place."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return completion.choices[0].message.content

def get_edges_for_route(G, start, end):
    # Find all simple paths from start to end
    paths = list(nx.all_simple_paths(G, source=start, target=end))
    
    # Extract edges from these paths
    edges = []
    for path in paths:
        for i in range(len(path) - 1):
            edges.append((path[i], path[i+1]))
    return edges

# Function to get the second best path using modified Dijkstra's algorithm
def find_second_best_route(start, end):
    first_path = nx.dijkstra_path(G, start, end, weight="weight")
    if len(first_path) < 2:
        return None  # No second path available if there's only one node or no path

    # Removing each edge in the first shortest path one by one
    second_path = None
    second_path_length = float('inf')
    for i in range(len(first_path) - 1):
        u, v = first_path[i], first_path[i+1]
        original_weight = G[u][v]['weight']
        G.remove_edge(u, v)
        try:
            path = nx.dijkstra_path(G, start, end, weight="weight")
            path_length = nx.dijkstra_path_length(G, start, end, weight="weight")
            if path_length < second_path_length:
                second_path = path
                second_path_length = path_length
        except nx.NetworkXNoPath:
            pass
        G.add_edge(u, v, weight=original_weight)  # Restore the edge

    if second_path:
        return [(airport, airports[airport]) for airport in second_path]
    else:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.json  # Get the JSON data
        start_airport = data["start"]
        end_airport = data["end"]
        best_route = find_best_route(start_airport, end_airport)
        second_best_route = find_second_best_route(start_airport, end_airport)
        
        # Fetch weather data and description for each airport in the best route
        route_with_weather_and_description = []
        for airport, (lat, lon, name) in best_route:
            weather = get_weather_data(lat, lon)
            description = get_location_description(name)
            route_with_weather_and_description.append({
                "airport": airport,
                "name": name,
                "lat": lat,
                "lon": lon,
                "weather": weather,
                "description": description
            })

                # Fetch weather data and description for each airport in the second-best route
        second_route_with_weather_and_description = []
        if second_best_route:
            for airport, (lat, lon, name) in second_best_route:
                weather = get_weather_data(lat, lon)
                description = get_location_description(name)
                second_route_with_weather_and_description.append({
                    "airport": airport,
                    "name": name,
                    "lat": lat,
                    "lon": lon,
                    "weather": weather,
                    "description": description
                })

        # Get all edges in the graph
        all_edges = [(u, v) for u, v in G.edges()]
        
        return jsonify({
            "best_route": route_with_weather_and_description,
            "all_edges": all_edges,
            "second_best_route": second_route_with_weather_and_description
        })
    
    return render_template("index.html", airports=airports)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)