import streamlit as st
from streamlit_folium import st_folium
import app

st.title("🗺️ Optymalna piesza trasa po zabytkach Krakowa")

selected_places = st.multiselect("Wybierz zabytki do odwiedzenia:", app.places, default=app.places)

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
                crossover_rate=0.98,
                population_size=50,
                generations=100,
                selection_method="roulette",  
                elite_size=0.1  
            )
            mapa,dist_km, time_min = app.create_osrm_foot_map(selected_coords, best_path)
            link_=app.generate_osrm_link(selected_coords, best_path)
            print(link_)
            # Zapisanie do session_state
            st.session_state.best_path = best_path
            st.session_state.mapa = mapa
            st.session_state.dist_km = dist_km
            st.session_state.time_min = time_min
            st.session_state.link=link_
    # Wyświetlenie zapisanych wyników
    if st.session_state.mapa is not None:
        st.success(f"✅ Długość trasy: {st.session_state.dist_km:.2f} km")
        st.info(f"🚶 Szacowany czas pieszy: {st.session_state.time_min/60:.2f} godzin")
        st.info(f"Link do trasy na mapie: {st.session_state.link}")
        st_folium(st.session_state.mapa, width=700, height=500)
else:
    st.warning("Wybierz co najmniej dwa miejsca, aby wyznaczyć trasę.")
