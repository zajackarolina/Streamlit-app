from geopy.geocoders import Nominatim
import time
import random
import numpy as np
import matplotlib.pyplot as plt

def get_coordinates_osm(places):
    geolocator = Nominatim(user_agent="geoapi_explorer")  #  user_agent
    coords = []
    
    for place in places:
        location = geolocator.geocode(place)
        if location:
            coords.append((location.longitude, location.latitude))
            print(f"{place}: {location.latitude}, {location.longitude}")
        else:
            print(f"Nie znaleziono: {place}")
        time.sleep(1) 
    return coords


places = [
    "Wawel, Kraków",
    "Kościół Mariacki, Kraków",
    "Barbakan, Kraków",
    "Kazimierz, Kraków",
    "Kopiec Kościuszki, Kraków",
    "Kościół św. Anny, Kraków",
    "Sukiennice, Kraków",
    "Plac Wolnica, Kraków",
    "Rynek Podgórski, Kraków",
    "Wieża Ratuszowa, Kraków",
    "Kościół św. Wojciecha, Kraków",
    "Collegium Maius, Kraków",
    "Brama Floriańska, Kraków",
    "Smok Wawelski, Kraków",
    "Szeroka, Kraków",
    "Józefa 12, Kraków",
    "Stary cmentarz żydowski, Kraków",
    "Kładka Ojca Bernatka, Kraków",
    "Rynek Podgórski, Kraków",
    "Kościół św. Józefa, Kraków",
    "Plac Bohaterów Getta, Kraków",
    "Apteka Pod Orłem, Kraków",
    "Plac Matejki, Kraków",
    "Stary Kleparz, Kraków",
    "Ogród Botaniczny Uniwersytetu Jagiellońskiego, Kraków",
    "Zakrzówek, Kraków",
    "Park Jordana, Kraków",
    "Fort Benedykt, Kraków",
    "Fabryka Emalia Oskara Schindlera, Kraków",
    "Nadwiślańska, Kraków"
    
]

visit_times = {
    "Wawel, Kraków": 120,
    "Kościół Mariacki, Kraków": 40,
    "Barbakan, Kraków": 10,
    "Kazimierz, Kraków": 30,
    "Kopiec Kościuszki, Kraków": 90,
    "Kościół św. Anny, Kraków": 10,
    "Sukiennice, Kraków": 30,
    "Plac Wolnica, Kraków": 10,
    "Rynek Podgórski, Kraków": 20,
    "Wieża Ratuszowa, Kraków": 20,
    "Kościół św. Wojciecha, Kraków": 10,
    "Collegium Maius, Kraków": 30,
    "Brama Floriańska, Kraków": 10,
    "Smok Wawelski, Kraków": 10,
    "Szeroka, Kraków": 20,
    "Józefa 12, Kraków": 10,
    "Stary cmentarz żydowski, Kraków": 30,
    "Kładka Ojca Bernatka, Kraków": 15,
    "Kościół św. Józefa, Kraków": 10,
    "Plac Bohaterów Getta, Kraków": 10,
    "Apteka Pod Orłem, Kraków": 20,
    "Plac Matejki, Kraków": 10,
    "Stary Kleparz, Kraków": 15,
    "Ogród Botaniczny UJ, Kraków": 60,
    "Zakrzówek, Kraków": 90,
    "Park Jordana, Kraków": 40,
    "Fort Benedykt, Kraków": 30,
    "Fabryka Emalia Oskara Schindlera, Kraków": 90,
    "Nadwiślańska, Kraków": 10
}


coords = get_coordinates_osm(places)
print(coords)
len(places)

import requests


def get_distance_matrix_osrm(coords, mode="foot"):
    coord_str = ";".join([f"{c[0]},{c[1]}" for c in coords])
    url = f"https://routing.openstreetmap.de/routed-{mode}/table/v1/foot/{coord_str}?annotations=distance,duration"
    r = requests.get(url)
    data = r.json()
    distances_km = np.array(data["distances"]) / 1000  # km
    durations_min = np.array(data["durations"]) / 60   # min
    return distances_km, durations_min

distance_matrix, durations = get_distance_matrix_osrm(coords, mode="foot")
print(distance_matrix)
print(durations)
print("Liczba miejsc:", len(places))
print("Liczba współrzędnych:", len(coords))

def min_to_mmss(minutes):
    m = int(minutes)
    s = int(round((minutes - m) * 60))
    return f"{m:02d}:{s:02d}"

def h_to_hm(h):
    """Konwertuje godziny dziesiętne na format HH:MM"""
    hours = int(h)
    minutes = int(round((h - hours) * 60))
    return f"{hours:02d}:{minutes:02d}"

def build_visit_plan(best_path, places, visit_times, durations, start_hour):
    plan = []
    current_time = start_hour*60

    for i, idx in enumerate(best_path):
        place = places[idx]
        visit = visit_times.get(place, 30)

        if i > 0:
            walk = durations[best_path[i-1]][idx]
            current_time += walk
        else:
            walk = 0

        start = current_time
        end = start + visit

        plan.append({
            "place": place,
            "walk_min": min_to_mmss(walk),
            "visit_min": visit,
            "start_h": h_to_hm(start/60),
            "end_h": h_to_hm(end/60)
        })

        current_time = end
    if len(best_path) > 1:
        last_idx = best_path[-1]
        start_idx = best_path[0]
        walk_back = durations[last_idx][start_idx]
        start = current_time
        end = start + walk_back

        plan.append({
            "place": places[start_idx] + " (powrót)",
            "walk_min": min_to_mmss(walk_back),
            "visit_min": 0,
            "start_h": h_to_hm(start / 60),
            "end_h": h_to_hm(end / 60)
        })

    return plan


def initialize_path(size, n_cities):
    population = []
    for _ in range(size):
        path = list(range(n_cities))
        random.shuffle(path)
        population.append(path)
    return population

# Długość trasy
def total_distance(path):
    distance = 0
    for i in range(len(path)):
        from_idx = path[i]
        to_idx = path[(i+1) % len(path)]  
        distance += distance_matrix[from_idx][to_idx] 
    return distance
    
# Selekcja ruletki
def roulette(population, distance):
    fitnesses = [1/d for d in distance]
    total = sum(fitnesses)
    pick = random.uniform(0, total)
    current = 0
    for i, fitness in enumerate(fitnesses):
        current += fitness
        if current >= pick:
            return population[i]

# Selekcja rankingowa
def ranking(population, distance):
    ranked_population = sorted(zip(distance, population), key=lambda x: x[0])
    ranks = [len(population) - i for i in range(len(population))]
    total_ranks = sum(ranks)
    pick = random.uniform(0, total_ranks)
    current = 0
    for i, (_, p) in enumerate(ranked_population):
        current += ranks[i]
        if current >= pick:
            return p

# Krzyżowanie
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end + 1] = parent1[start:end + 1]

    current_pos = 0
    for city in parent2:
        if city not in child:
            while child[current_pos] != -1:
                current_pos += 1
            child[current_pos] = city

    return child

# Mutacja
def mutate(path, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]

# Algorytm genetyczny z elitarnymi osobnikami
def genetic_algorithm(n_cities, mutation_rate, crossover_rate, population_size, generations, selection_method, elite_size=0.1):
    population = initialize_path(population_size, n_cities)
    best_history = []
    avg=[]
    max_history=[]
    for gen in range(generations):
        distances = [total_distance(p) for p in population]
        best = min(distances)
        best_history.append(best)
        avg.append(np.mean(distances))
        max_history.append(max(distances))
        sorted_population = [x for _, x in sorted(zip(distances, population))]
        #print(f" Pokolenie {gen} ")
        #for i, path in enumerate(population):
         #   print(f"Osobnik {i}: {[places[i] for i in path]}")

        elite_count = int(elite_size * population_size)
        elite_population = sorted_population[:elite_count]
        new_population = elite_population[:]  # Zachowujemy elite

        select = ranking if selection_method == "ranking" else roulette

        while len(new_population) < population_size:
            p1 = select(population, distances)
            p2 = select(population, distances)

            if random.random() < crossover_rate:
                child = crossover(p1, p2)
            else:
                child = p1[:]

            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
    
    final_distances = [total_distance(p) for p in population]
    best_index = final_distances.index(min(final_distances))
    best_path = population[best_index]
    
    return best_path, best_history, avg, max_history

def plot_results(cities, best_path, best_history, avg, max_history):
    x, y = zip(*cities) #coords
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red', label='Cities')

    best_x = [cities[best_path[i]][0] for i in range(len(best_path))] + [cities[best_path[0]][0]]
    best_y = [cities[best_path[i]][1] for i in range(len(best_path))] + [cities[best_path[0]][1]]
    plt.plot(best_x, best_y, color='blue', label='Best Path', linewidth=2)
    
    plt.title("Best Path with Cities")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(best_history, label='Best Distance')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.title('Best Distance vs Generation')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(avg, label='Mean distance')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.title('Mean Distance vs Generation')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_history, label='Max Distance')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.title('Max Distance vs Generation')
    plt.legend()
    plt.show()

# algorytm
best_path, best_history, avg, max_history = genetic_algorithm(
    n_cities=len(places),
    mutation_rate=0.05,
    crossover_rate=0.98,
    population_size=100,
    generations=200,
    selection_method="ranking",  
    elite_size=0.1  
)
print(coords[best_path[1]])
# Wizualizacja wyników
plot_results(coords, best_path, best_history, avg, max_history)
print(min(best_history))
print(best_path)
# base_url = "https://www.google.com/maps/dir/"
# ordered_path = "/".join([f"{coords[i][0]},{coords[i][1]}" for i in best_path +[best_path[0]]])
# url = base_url + ordered_path + "/data=!3m1!4b1!4m2!4m1!3e2" 
# print("Otwórz trase na mapie:", url)



def generate_osrm_link(coords, best_path):
    # Ustawiamy pierwszy punkt jako centrum mapy
    center_lat, center_lon = coords[best_path[0]][1], coords[best_path[0]][0]
    #base_url = f"https://map.project-osrm.org/?z=14&center={center_lat},{center_lon}"
    base_url=f"https://www.google.com/maps/dir/"
    # Dodajemy wszystkie punkty w kolejności best_path
    locs = "".join([f"{coords[i][1]},{coords[i][0]}/" for i in best_path + [best_path[0]]])
    fix="".join(f"data=!3m1!4b1!4m2!4m1!3e2")
    return base_url + locs + fix

link = generate_osrm_link(coords, best_path)
print(link)

print(len(best_path))

import folium
import requests

def create_osrm_foot_map(coords, best_path):
    # Dodajemy powrót do punktu startowego
    ordered_coords = [coords[i] for i in best_path] + [coords[best_path[0]]]
    coord_str = ";".join([f"{c[0]},{c[1]}" for c in ordered_coords])
    
    # OSRM foot profile
    url = f"https://routing.openstreetmap.de/routed-foot/route/v1/foot/{coord_str}?overview=full&geometries=geojson"
    r = requests.get(url).json()
    
    geometry = r['routes'][0]['geometry']['coordinates']
    center = [ordered_coords[0][1], ordered_coords[0][0]]
    m = folium.Map(location=center, zoom_start=14)
    
    # Rysowanie trasy
    folium.PolyLine(
        locations=[[lat, lon] for lon, lat in geometry],
        color='blue',
        weight=5,
        opacity=0.7
    ).add_to(m)
    
    # Dodawanie markerów z popupem
    for idx, point in enumerate(ordered_coords):
        # Obsługa powrotu
        if idx == len(ordered_coords) - 1:
            popup_place = places[best_path[0]] + " (powrót)"
        else:
            popup_place = places[best_path[idx]]
        
        folium.Marker(
            location=[point[1], point[0]],
            tooltip=f"{idx}",
            popup=f"{idx}. {popup_place}"
        ).add_to(m)
    
    distance_m = r['routes'][0]['distance']  # metry
    distance_km = distance_m / 1000
    duration_min = r['routes'][0]['duration'] / 60

    print(f"Długość trasy: {distance_km:.2f} km")
    print(f"Czas pieszy: {duration_min/60:.1f} h")
    return m, distance_km, duration_min


