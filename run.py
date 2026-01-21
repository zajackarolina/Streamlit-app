import streamlit as st
from streamlit_folium import st_folium
import app
from datetime import time
st.title("Optymalna piesza trasa po zabytkach Krakowa")

selected_places = st.multiselect("Wybierz zabytki do odwiedzenia:", app.places, default=app.places)
st.subheader("Wybierz godzinę rozpoczęcia zwiedzania")
start_time = st.time_input("Godzina startu", value=time(9, 0))
start_hour = start_time.hour + start_time.minute / 60 
# Inicjalizacja zmiennych w session_state
if "best_path" not in st.session_state:
    st.session_state.best_path = None
if "mapa" not in st.session_state:
    st.session_state.mapa = None
if "dist_km" not in st.session_state:
    st.session_state.dist_km = None
if "time_min" not in st.session_state:
    st.session_state.time_min = None
if "link" not in st.session_state:
    st.session_state.link = None
if len(selected_places) >= 2:
    selected_indices = [app.places.index(p) for p in selected_places]
    selected_coords = [app.coords[i] for i in selected_indices]

    # Przycisk do obliczenia
    if st.button("Oblicz trasę") or st.session_state.best_path is None:
        with st.spinner("Obliczam optymalną trasę..."):
            distance_matrix, durations = app.get_distance_matrix_osrm(selected_coords, mode="foot")
            best_path, best_history, avg, max_history = app.genetic_algorithm(
                n_cities=len(selected_places),
                mutation_rate=0.05,
                crossover_rate=0.9,
                population_size=200,
                generations=500,
                selection_method="roulette",  
                elite_size=0.1  
            )
            mapa,dist_km, time_min = app.create_osrm_foot_map(selected_coords, best_path)
            plan=app.build_visit_plan(best_path, selected_places, app.visit_times, durations, start_hour)
            total_visit = sum(p["visit_min"] for p in plan)
            total_time = time_min + total_visit
            link_=app.generate_osrm_link(selected_coords, best_path)
            print(link_)
            # Zapisanie do session_state
            st.session_state.best_path = best_path
            st.session_state.mapa = mapa
            st.session_state.dist_km = dist_km
            st.session_state.time_min = time_min
            st.session_state.link=link_
            st.session_state.plan = plan
            st.session_state.total_visit = total_visit
            st.session_state.total_time = total_time

    # Wyświetlenie zapisanych wyników
    if st.session_state.mapa is not None:
        st.success(f"Długość trasy: {st.session_state.dist_km:.2f} km")
        st.info(f" Szacowany czas pieszy: {app.h_to_hm(st.session_state.time_min/60)} godzin")
        st_folium(st.session_state.mapa, width=700, height=500)
        st.subheader("Plan zwiedzania")
        st.info(f" Szacowany czas zwiedzania: {app.h_to_hm(st.session_state.total_time/60)} godzin")
        table_data = []

        for i, p in enumerate(st.session_state.plan):
            table_data.append({
                "Lp.": i + 1,
                "Miejsce": p["place"],
                "Dojście [min]": p["walk_min"],
                "Zwiedzanie [min]": p["visit_min"],
                "Start [h]": p["start_h"],
                "Koniec [h]": p["end_h"] ,
            })

        st.table(table_data)

        st.info(f"Link do trasy na mapie: {st.session_state.link}")
else:
    st.warning("Wybierz co najmniej dwa miejsca, aby wyznaczyć trasę.")
