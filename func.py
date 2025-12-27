import math
import os
import time
import heapq
from datetime import datetime, date, time as dtime
from datetime import timedelta
import pandas as pd
import numpy as np
import contextily as ctx
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import matplotlib.ticker as mticker
from IPython.display import HTML
from itertools import combinations
from typing import Dict, List, Tuple, Set
from shapely.geometry import shape
import osmnx as ox
import networkx as nx
from scipy.spatial import cKDTree
import heapq
import pickle
from tqdm import tqdm
import pymoo
import itertools
import geopandas as gpd
from shapely.geometry import LineString, Point
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.util.ref_dirs import get_reference_directions
import cartopy.crs as ccrs
# Fatores de emissão de CO2 em gramas por metro
EMISSION_FACTORS = {
    'transit': 0.04, 
    'bus': 0.1099,
    'walk': 0.0
}
# Velocidade média em metros por segundo
AVERAGE_SPEED_WALK = 1.4 

# Tipo de rota
ROUTE_TYPE = {
    0: 'metro',
    1: 'bus', 
    2: 'walk'
}
# Funções auxiliares
# Cálculo da distância Haversine entre dois pontos geográficos
def HaversineDistance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6378100  # Raio da Terra em metros
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    return distance

# Calculo do CO2 emitido com base na distância e no modo de transporte
def EmissionCalculation(distance: float, mode: str) -> float:
    factor = EMISSION_FACTORS.get(mode)
    return distance * factor

# Conversão de tempos
def HMStoSeconds(hms: str) -> int:
    try:
        parts = hms.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        return 0

def SecondsToHMS(seconds: int) -> str:
    return str(timedelta(seconds=seconds))

# Obtenção dos serviços ativos numa data específica (YYYYMMDD)
def GetActiveServices(date: str, calendar: pd.DataFrame, calendar_dates: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    activeservice = set()
    dayofweek = pd.to_datetime(date).dayofweek
    year = pd.to_datetime(date).year
    month = pd.to_datetime(date).month
    day = pd.to_datetime(date).day

    for _, row in calendar.iterrows():
        start_date = pd.to_datetime(str(row['start_date']), format='%Y%m%d')
        end_date = pd.to_datetime(str(row['end_date']), format='%Y%m%d')
        if start_date <= pd.to_datetime(date) <= end_date:
            if ((dayofweek == 0 and row['monday'] == 1) or
                (dayofweek == 1 and row['tuesday'] == 1) or
                (dayofweek == 2 and row['wednesday'] == 1) or
                (dayofweek == 3 and row['thursday'] == 1) or
                (dayofweek == 4 and row['friday'] == 1) or
                (dayofweek == 5 and row['saturday'] == 1) or
                (dayofweek == 6 and row['sunday'] == 1)):
                activeservice.add(row['service_id'])
    exceptions = calendar_dates[calendar_dates['date'] == int(date)]
    for _, row in exceptions.iterrows():
        if row['exception_type'] == 1:
            activeservice.add(row['service_id'])
        elif row['exception_type'] == 2 and row['service_id'] in activeservice:
            activeservice.remove(row['service_id'])
    active_trips = trips[trips['service_id'].isin(activeservice)]
    return active_trips

# Encontrar o nó mais próximo no grafo para uma dada coordenada
def NearestGraphNode(graph, coord):
    target_y, target_x = coord
    best_node = None
    min_dist = float('inf')
    
    # Itera sobre todos os nós (pode ser lento em grafos muito grandes, mas funciona)
    for node, data in graph.nodes(data=True):
        # Tenta apanhar y/x ou lat/lon
        ny = data.get('y', data.get('lat'))
        nx = data.get('x', data.get('lon'))
        
        if ny is None or nx is None: continue
        
        # Distância Euclidiana (aproximação rápida)
        dist = (ny - target_y)**2 + (nx - target_x)**2
        
        if dist < min_dist:
            min_dist = dist
            best_node = node
            
    return best_node

# Random Scenario Creation
def CreateRandomScenario(G, difficulty='medium', max_attempts=100000):
    difficulty_map = {
        'low': (500, 5000),    # 500m a 5km
        'medium': (5000, 10000),  # 5km a 10km
        'high': (10000, 20000)    # 10km a 20km
    }
    
    if difficulty not in difficulty_map:
        raise ValueError("Dificuldade inválida, escolha entre 'low', 'medium' ou 'high'.")
    
    min_dist, max_dist = difficulty_map[difficulty]
    
    # Tentativas para encontrar um par válido
    for _ in range(max_attempts):
        start_node = random.choice(list(G.nodes))
        current_node = start_node
        
        steps = 0
        max_steps = 500 
        
        while steps < max_steps:
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break # Beco sem saída, tentar outro start_node
            
            next_node = random.choice(neighbors)
            current_node = next_node
            steps += 1
            
            # --- CORREÇÃO AQUI ---
            # Aceder às coordenadas através do dicionário de nós do grafo
            # OSMnx armazena latitude como 'y' e longitude como 'x'
            try:
                lat1 = G.nodes[start_node]['y']
                lon1 = G.nodes[start_node]['x']
                lat2 = G.nodes[current_node]['y']
                lon2 = G.nodes[current_node]['x']
                
                dist_real = HaversineDistance(lat1, lon1, lat2, lon2)
            except KeyError:
                # Caso o nó não tenha coordenadas (raro em OSMnx, mas possível)
                break 
            # ---------------------
            
            if min_dist <= dist_real <= max_dist:
                # Retorna os IDs dos nós encontrados
                return start_node, current_node
            
            if dist_real > max_dist:
                break
                
    raise ValueError(f"Não foi possível encontrar cenário '{difficulty}' após {max_attempts} tentativas.")

# Calcula o custo real (Viagem + Espera) e CO2 para atravessar de u para v numa hora específica.
def GetDynamicEdgeCost(graph, u, v, current_time, active_bus_ids, active_transit_ids, weight='time'):

    edge_data = graph.get_edge_data(u, v)
    if 0 in edge_data: edge_data = edge_data[0] 
    
    base_travel_time = edge_data.get(weight, float('inf'))
    distance = edge_data.get('length', 0) 
    mode = edge_data.get('mode', 'walk')
    
    wait_time = 0
    co2_emission = 0
    valid_edge = True
    
    # Lógica de Caminhada
    if mode == 'walk':
        # CO2 é zero 
        return base_travel_time, 0, True 

    # Lógica de Transportes (Bus/Transit) 
    elif mode in ['bus', 'transit']:
        schedules = graph.nodes[v].get('schedules', []) 
        
        best_departure = float('inf')
        found_service = False
        
        for schedule in schedules:
            trip_id = schedule['trip_id']
            departure_time = schedule['departure_time']
            
            # Verificar se o serviço está ativo hoje
            is_active = False
            if mode == 'bus' and trip_id in active_bus_ids:
                is_active = True
            elif mode == 'transit' and trip_id in active_transit_ids:
                is_active = True
            
            # Verificar se serve para a hora atual
            if is_active and departure_time >= current_time:
                if departure_time < best_departure:
                    best_departure = departure_time
                    found_service = True
        
        if found_service:
            wait_time = best_departure - current_time
            
            # Calcular CO2
            if mode == 'bus':
                # 109.9 g/km -> 0.1099 g/m
                co2_emission = EMISSION_FACTORS['bus'] * distance 
            else: # transit
                # 40 g/km -> 0.04 g/m
                co2_emission = EMISSION_FACTORS['transit'] * distance
                
        else:
            valid_edge = False # Não há mais transportes hoje

    total_time_cost = base_travel_time + wait_time
    return total_time_cost, co2_emission, valid_edge

def PlotPath(Graph, path, title="Rota Calculada", transformer=None):
    if not path:
        print("Caminho vazio, nada para plotar.")
        return

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # --- FUNÇÃO AUXILIAR PARA OBTER COORDENADAS ---
    def get_coords(n):
        # Se for tupla de floats, assume (lat, lon)
        if isinstance(n, tuple) and len(n) == 2 and isinstance(n[0], (int, float)):
            return n
        # Se for ID (string ou int), busca no grafo
        try:
            node_data = Graph.nodes[n]
            # Tenta chaves 'y'/'x' ou 'lat'/'lon'
            lat = node_data.get('y', node_data.get('lat'))
            lon = node_data.get('x', node_data.get('lon'))
            return (lat, lon)
        except:
            return None # Retorna None se falhar

    # Definir cores por modo
    mode_colors = {
        'walk': 'green', 
        'bus': 'blue', 
        'transit': 'red', 
        'transfer_walk': 'yellow'
    }
    default_color = 'gray'

    segments = []
    colors = []
    
    # Coordenadas para ajustar o zoom
    lats = []
    lons = []

    # Iterar pelos pares de nós no caminho
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        c_u = get_coords(u)
        c_v = get_coords(v)
        
        if c_u is None or c_v is None:
            continue
            
        lats.extend([c_u[0], c_v[0]])
        lons.extend([c_u[1], c_v[1]])

        # Inverter para (lon, lat) para plotagem X,Y
        p1 = (c_u[1], c_u[0]) 
        p2 = (c_v[1], c_v[0])
        
        # Transformação de projeção (se houver transformer, ex: Web Mercator)
        if transformer:
            try:
                p1 = transformer.transform(*p1)
                p2 = transformer.transform(*p2)
            except:
                pass # Ignora erros de transformação pontuais

        segments.append([p1, p2])
        
        # Obter cor baseada no modo de transporte da aresta
        edge_data = Graph.get_edge_data(u, v)
        color = default_color
        if edge_data:
            # MultiDiGraph pode ter múltiplas arestas, pegamos a primeira (chave 0)
            if 0 in edge_data: 
                data = edge_data[0]
            else:
                # Se não tiver chave 0, pega qualquer uma (padrão dict)
                data = list(edge_data.values())[0]
            
            mode = data.get('mode', 'walk')
            color = mode_colors.get(mode, default_color)
        
        colors.append(color)

    # Criar a coleção de linhas
    lc = LineCollection(segments, colors=colors, linewidths=3, alpha=0.8)
    ax.add_collection(lc)
    
    # Ajustar limites do gráfico
    if segments:
        # Se usou transformer, os limites são nas novas coordenadas
        if transformer:
            # Recalcula limites baseados nos segmentos transformados
            xs = [p[0] for seg in segments for p in seg]
            ys = [p[1] for seg in segments for p in seg]
            ax.set_xlim(min(xs), max(xs))
            ax.set_ylim(min(ys), max(ys))
        else:
            # Limites geográficos padrão
            if lons and lats:
                ax.set_xlim(min(lons) - 0.002, max(lons) + 0.002)
                ax.set_ylim(min(lats) - 0.002, max(lats) + 0.002)

    # Adicionar mapa de fundo (se contextily estiver disponível e não houver transformer manual conflitante)
    # Geralmente contextily espera WebMercator (EPSG:3857). 
    # Se os dados estiverem em lat/lon (WGS84), use crs="EPSG:4326" no cx.add_basemap
    try:
        if not transformer:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)
    except:
        pass # Ignora se não conseguir baixar o mapa

    # Adicionar pontos de Início e Fim
    start_c = get_coords(path[0])
    end_c = get_coords(path[-1])
    
    if start_c and end_c:
        p_start = (start_c[1], start_c[0])
        p_end = (end_c[1], end_c[0])
        
        if transformer:
            p_start = transformer.transform(*p_start)
            p_end = transformer.transform(*p_end)
            
        ax.scatter(p_start[0], p_start[1], c='green', s=100, label='Início', zorder=5, edgecolors='black')
        ax.scatter(p_end[0], p_end[1], c='red', s=100, label='Fim', zorder=5, edgecolors='black')

    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.show()

# Função para avaliar o caminho encontrado
def EvaluatePath(G, path):
    total_time = 0
    total_co2 = 0
    total_walk_dist = 0
    transfers = 0
    last_trip_id = None
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = G.get_edge_data(u, v)
        
        if edge_data is None:
            continue
        
        # Handle MultiGraph by getting the first edge data
        if isinstance(edge_data, dict):
            # Get the first key (could be 0 or any other key)
            first_key = next(iter(edge_data))
            data = edge_data[first_key]
        else:
            data = edge_data
        
        total_time += data.get('time', 0) # Tempo em segundos
        total_co2 += data.get('co2', 0)
        
        # Contar transferências
        current_trip_id = data.get('trip_id')
        if current_trip_id and last_trip_id and current_trip_id != last_trip_id:
            transfers += 1
        if current_trip_id:
            last_trip_id = current_trip_id
        
        # Distância a pé
        if data.get('mode') == 'walk':
            total_walk_dist += data.get('distance', 0)

    # Calorias aproximadas
    calories = total_walk_dist * 0.05
    
    return {
        'time': total_time,
        'co2': total_co2,
        'transfers': transfers,
        'walk_dist': total_walk_dist,
        'calories': calories
    }
# Algoritmo A* 
def AStarSearchVisual(graph: nx.MultiGraph, start_coords, end_coords, weight='time'):
    
    start_node = NearestGraphNode(graph, start_coords)
    end_node = NearestGraphNode(graph, end_coords)

    print(f"A procurar caminho de {start_node} para {end_node}...")

    # Obter coordenadas do destino para a Heurística
    try:
        end_data = graph.nodes[end_node]
        end_y = end_data.get('y', end_data.get('lat'))
        end_x = end_data.get('x', end_data.get('lon'))
    except KeyError:
        print("Erro: O nó de destino não tem coordenadas.")
        return [], []

    # Função Heurística Local (Distância Real até ao destino)
    def heuristic(node_id):
        try:
            node_data = graph.nodes[node_id]
            ny = node_data.get('y', node_data.get('lat'))
            nx = node_data.get('x', node_data.get('lon'))

            dist = HaversineDistance(ny, nx, end_y, end_x)
            if weight == 'time':
                return dist / 1.1 # 1.1 m/s (~4km/h) como velocidade base conservadora a pé
            return dist
        except:
            return float('inf')

    # Inicialização A*
    open_set = []
    counter = 0  # Contador único para evitar comparação de nós
    heapq.heappush(open_set, (0, counter, start_node))
    
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node)}
    
    visited_edges = []
    nodes_visited_count = 0

    while open_set:
        current = heapq.heappop(open_set)[2]  # Agora o nó está na posição 2
        nodes_visited_count += 1

        if current == end_node:
            print(f"Caminho encontrado! Nós explorados: {nodes_visited_count}")
            # Reconstruir caminho
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            path.reverse()
            return path, visited_edges

        # Explorar vizinhos
        for neighbor in graph.neighbors(current):

            all_edges = graph.get_edge_data(current, neighbor)

            min_edge_weight = float('inf')
            
            for key, attr in all_edges.items():
                w = attr.get(weight, float('inf'))
                if w < min_edge_weight:
                    min_edge_weight = w
            
            edge_weight = min_edge_weight

            if edge_weight == float('inf'):
                continue

            tentative_g_score = g_score[current] + edge_weight

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                
                counter += 1  # Incrementar contador para manter ordem de inserção
                heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                
                # Adicionar à lista de visualização
                visited_edges.append((current, neighbor))

    print("Caminho não encontrado.")
    return [], visited_edges

# Algoritmo Dijkstra
def DijkstraSearchVisual(graph: nx.MultiGraph, start_coords, end_coords, weight='time'):

    start_node = NearestGraphNode(graph, start_coords)
    end_node = NearestGraphNode(graph, end_coords)

    print(f"Dijkstra: A procurar de {start_node} para {end_node}...")

    # Inicialização
    open_set = []
    counter = 0  # Contador único para evitar comparação de nós
    heapq.heappush(open_set, (0, counter, start_node))
    
    came_from = {}
    g_score = {start_node: 0}
    
    visited_edges = []
    nodes_visited_count = 0

    while open_set:
        current_cost, _, current = heapq.heappop(open_set)  # Extrair nó da posição 2
        nodes_visited_count += 1

        # Encontrou o destino
        if current == end_node:
            print(f"Dijkstra concluído! Nós explorados: {nodes_visited_count}")
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            path.reverse()
            return path, visited_edges
        
        if current_cost > g_score.get(current, float('inf')):
            continue

        for neighbor in graph.neighbors(current):

            all_edges = graph.get_edge_data(current, neighbor)
            
            # Encontrar o menor peso entre as múltiplas arestas disponíveis
            min_edge_weight = float('inf')
            
            for key, attr in all_edges.items():
                w = attr.get(weight, float('inf'))
                if w < min_edge_weight:
                    min_edge_weight = w
            
            edge_weight = min_edge_weight

            # Se não houver peso válido, ignora
            if edge_weight == float('inf'):
                continue

            tentative_g_score = g_score[current] + edge_weight

            # Relaxamento da Aresta
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                
                # No Dijkstra, a prioridade é puramente o custo acumulado (g)
                counter += 1  # Incrementar contador
                heapq.heappush(open_set, (tentative_g_score, counter, neighbor))
                
                # Guardar aresta para a animação
                visited_edges.append((current, neighbor))
                
    print("Dijkstra: Caminho não encontrado.")
    return [], visited_edges 

# Animação da Busca A* vs Dijkstra
def AnimateBattle(G, visit_a, path_a, visit_d, path_d, start, end, skip=20):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
    
    # 1. Definição das cores por modo de transporte
    mode_colors = {
        'walk': 'lime', 
        'bus': 'cyan', 
        'transit': 'red', 
    }
    default_color = 'white'

    configs = [
        (ax1, "A*", visit_a, path_a, 'lightblue'),
        (ax2, "Dijkstra", visit_d, path_d, 'lightgreen'),
    ]
    
    exploring_lines = []
    final_collections = []

    # --- FUNÇÃO AUXILIAR PARA OBTER COORDENADAS ---
    def get_coords(n):
        # Se for tupla de floats, já é (lat, lon)
        if isinstance(n, tuple) and len(n) == 2 and isinstance(n[0], (int, float)):
            return n
        # Se for ID (string ou int), buscar no grafo
        try:
            node_data = G.nodes[n]
            # Tenta chaves 'y'/'x' ou 'lat'/'lon'
            lat = node_data.get('y', node_data.get('lat'))
            lon = node_data.get('x', node_data.get('lon'))
            return (lat, lon)
        except:
            return (0, 0) # Fallback para evitar erros

    # Coletar todas as coordenadas visitadas para ajustar o zoom (Dijkstra cobre mais área)
    # Usamos um set para evitar duplicados e processar mais rápido
    all_nodes_visited = set()
    for u, v in visit_d:
        all_nodes_visited.add(u)
        all_nodes_visited.add(v)
    
    # Converter nós para coordenadas
    all_coords = [get_coords(n) for n in all_nodes_visited]
    
    # Separar lats e lons, filtrando inválidos (0,0) se necessário
    lats = [c[0] for c in all_coords if c != (0,0)]
    lons = [c[1] for c in all_coords if c != (0,0)]
    
    # Se a lista estiver vazia (caso raro), usa start/end
    if not lats:
        lats = [start[0], end[0]]
        lons = [start[1], end[1]]

    # Configuração dos eixos
    for ax, title, visit, path, color in configs:
        ax.set_facecolor('black')
        ax.set_title(title, color='white', fontsize=14)
        ax.axis('off')
        
        # Ajustar limites com margem
        ax.set_xlim(min(lons)-0.005, max(lons)+0.005)
        ax.set_ylim(min(lats)-0.005, max(lats)+0.005)
        
        # Start/End points
        ax.scatter(start[1], start[0], c='white', s=50, zorder=10)
        ax.scatter(end[1], end[0], c='red', s=50 , zorder=10)
        
        # Linha de exploração
        ln, = ax.plot([], [], color=color, alpha=0.4, linewidth=1)
        exploring_lines.append(ln)
        
        # Linha Final
        lc = LineCollection([], linewidths=2, alpha=0.9)
        ax.add_collection(lc)
        final_collections.append(lc)

    def update(frame):
        idx = frame * skip
        artists = []
        
        for i, (_, _, visit, path, _) in enumerate(configs):
            # 1. Atualizar Animação de Exploração
            if idx < len(visit):
                xs, ys = [], []
                # Construir segmentos usando a função get_coords
                for u, v in visit[:idx+skip]: 
                    cu = get_coords(u)
                    cv = get_coords(v)
                    xs.extend([cu[1], cv[1], None])
                    ys.extend([cu[0], cv[0], None])
                exploring_lines[i].set_data(xs, ys)
            
            # 2. Desenhar Caminho Final Colorido
            if idx >= len(visit) and path:
                segments = []
                colors = []
                
                for k in range(len(path) - 1):
                    u = path[k]
                    v = path[k+1]
                    
                    edge_data = G.get_edge_data(u, v)
                    if edge_data is not None:
                        if 0 in edge_data: edge_data = edge_data[0]
                        mode = edge_data.get('mode', 'walk')
                        color = mode_colors.get(mode, default_color)
                        
                        cu = get_coords(u)
                        cv = get_coords(v)
                        
                        # (lon, lat) para o plot
                        segments.append([(cu[1], cu[0]), (cv[1], cv[0])])
                        colors.append(color)
                
                final_collections[i].set_segments(segments)
                final_collections[i].set_color(colors)
            
            artists.append(exploring_lines[i])
            artists.append(final_collections[i])
        
        return artists
    
    max_len = max(len(visit_a), len(visit_d))
    frames = (max_len // skip) + 30 
    
    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)
    plt.close()
    return anim

# A* Dinâmico com horários e espera
def AStarDynamicSearchVisual(graph: nx.Graph, start, end, hour, day, weight='time', 
                      trips_bus=None, calendar_bus=None, calendar_dates_bus=None, 
                      trips_transit=None, calendar_transit=None, calendar_dates_transit=None):
    
    print(f"Início A* Dinâmico às {SecondsToHMS(hour)} no dia {day}...")
    
    # 1. Resolver nós mais próximos se não existirem no grafo
    if start not in graph: start = NearestGraphNode(graph, start)
    if end not in graph: end = NearestGraphNode(graph, end)

    # --- FUNÇÃO AUXILIAR PARA OBTER COORDENADAS (CORREÇÃO) ---
    def get_coords(n):
        # Se for tupla de floats, assume que já é (lat, lon)
        if isinstance(n, tuple) and len(n) == 2 and isinstance(n[0], (int, float)):
            return n
        # Se for ID (string ou int), buscar atributos no grafo
        try:
            node_data = graph.nodes[n]
            # Tenta chaves 'y'/'x' (osmnx) ou 'lat'/'lon'
            lat = node_data.get('y', node_data.get('lat'))
            lon = node_data.get('x', node_data.get('lon'))
            return (lat, lon)
        except KeyError:
            # Caso extremo de falha, retorna 0,0 para não quebrar (ou trate o erro)
            return (0.0, 0.0)

    # Coordenadas fixas do destino para a heurística
    end_coords = get_coords(end)
    start_coords = get_coords(start)

    # 2. Preparar serviços ativos
    print("A filtrar serviços ativos para o dia...")
    active_bus_ids = set()
    active_transit_ids = set()

    if trips_bus is not None:
        active_bus_df = GetActiveServices(f"{day}", calendar_bus, calendar_dates_bus, trips_bus)
        active_bus_ids = set(active_bus_df['trip_id'].values)
    
    if trips_transit is not None:
        active_transit_df = GetActiveServices(f"{day}", calendar_transit, calendar_dates_transit, trips_transit)
        active_transit_ids = set(active_transit_df['trip_id'].values)
        
    print(f"Serviços prontos. Bus: {len(active_bus_ids)}, Transit: {len(active_transit_ids)}")

    # 3. Inicialização do A*
    open_set = []
    # (f_score, tie_breaker_counter, node_id) - contador ajuda no desempate do heapq
    counter = itertools.count() 
    heapq.heappush(open_set, (0, next(counter), start)) 
    
    came_from = {}
    g_score = {start: 0}
    
    # Heurística inicial (usando get_coords)
    f_score = {start: HaversineDistance(start_coords[0], start_coords[1], end_coords[0], end_coords[1])}
    
    visited_edges = []
    nodes_expanded = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)
        nodes_expanded += 1

        if current == end:
            print(f"Destino alcançado! Nós expandidos: {nodes_expanded}")
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, visited_edges

        # Hora atual neste nó
        current_time_at_node = hour + g_score[current]

        for neighbor in graph.neighbors(current):
            
            custo_tempo_total, custo_co2, valido = GetDynamicEdgeCost(
                graph, current, neighbor, current_time_at_node,
                active_bus_ids, active_transit_ids, weight
            )
            
            if not valido:
                continue

            tentative_g_score = g_score[current] + custo_tempo_total

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                
                # Calcular Heurística com coordenadas corretas
                neigh_coords = get_coords(neighbor)
                h_score = HaversineDistance(neigh_coords[0], neigh_coords[1], end_coords[0], end_coords[1])
                
                f_score[neighbor] = tentative_g_score + h_score
                
                heapq.heappush(open_set, (f_score[neighbor], next(counter), neighbor))
                visited_edges.append((current, neighbor))
                
    print("Caminho não encontrado.")
    return [], visited_edges

# Funções auxiliares para MOEA/D
def GetCoords(G, n):
    if isinstance(n, tuple) and len(n) == 2 and isinstance(n[0], (int, float)):
        return n
    try:
        node_data = G.nodes[n]
        lat = node_data.get('y', node_data.get('lat'))
        lon = node_data.get('x', node_data.get('lon'))
        if lat is not None and lon is not None:
            return (lat, lon)
    except:
        pass
    return (0.0, 0.0)

def MultiObjectiveHeuristic(coords_curr, coords_dest, w_time, w_co2):
    dist = HaversineDistance(coords_curr[0], coords_curr[1], coords_dest[0], coords_dest[1])
    max_speed = 13.8 # ~50km/h
    est_time = dist / max_speed
    est_co2 = dist * 0.0 
    return (w_time * est_time) + (w_co2 * est_co2)

# Inicialização inspirada no MOEA/D com A* (com diferentes pesos)
def InitMOEAD(G, source, target, start_time, pop_size=10):
    print(f"Gerando {pop_size} caminhos iniciais com pesos variados...")
    paths = []
    
    weights = []
    if pop_size > 1:
        for i in range(pop_size):
            alpha = i / (pop_size - 1)
            weights.append((1 - alpha, alpha)) 
    else:
        weights.append((1.0, 0.0))

    target_coords = GetCoords(G, target)

    for w_time, w_co2 in weights:
        pq = []
        heapq.heappush(pq, (0, 0, start_time, source, [source]))
        visited = {}
        path_found = None
        max_iter = 5000 
        iter_count = 0
        
        while pq and iter_count < max_iter:
            iter_count += 1
            _, g, curr_time, u, path = heapq.heappop(pq)
            
            if u == target:
                path_found = path
                break
            
            if u in visited and visited[u] < g:
                continue
            visited[u] = g
            
            for v in G.neighbors(u):
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    d = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
                    length = d.get('length', 100)
                    mode = d.get('mode', 'walk')
                    
                    speed = 1.4
                    co2_factor = 0.0
                    if mode == 'bus': speed = 8.3; co2_factor = 0.1
                    elif mode == 'transit': speed = 10.0; co2_factor = 0.05
                    
                    edge_time = length / speed
                    edge_co2 = length * co2_factor
                    
                    if mode in ['bus', 'transit']:
                        edge_time += 300 
                    
                    step_cost = (w_time * edge_time) + (w_co2 * edge_co2)
                    new_g = g + step_cost
                    
                    new_time = curr_time + timedelta(seconds=float(edge_time))
                    
                    v_coords = GetCoords(G, v)
                    h = MultiObjectiveHeuristic(v_coords, target_coords, w_time, w_co2)
                    
                    if v not in visited or new_g < visited[v]:
                         heapq.heappush(pq, (new_g + h, new_g, new_time, v, path + [v]))
        
        if path_found:
            paths.append(path_found)
        else:
            try:
                paths.append(nx.shortest_path(G, source, target))
            except:
                pass

    unique_paths = []
    seen = set()
    for p in paths:
        t_p = tuple(p)
        if t_p not in seen:
            unique_paths.append(p)
            seen.add(t_p)
            
    print(f"Inicialização concluída. {len(unique_paths)} caminhos únicos gerados.")
    return unique_paths

# Definição do problema
class RoutingProblem(ElementwiseProblem):
    def __init__(self, G, start_node, end_node, **kwargs):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, **kwargs)
        self.G = G
        self.start_node = start_node
        self.end_node = end_node

    def _evaluate(self, x, out, *args, **kwargs):
        path = x[0]
        PENALTY = 1e6 
        
        if path is None or len(path) == 0:
            out["F"] = [PENALTY, PENALTY]
            return

        if path[0] != self.start_node or path[-1] != self.end_node:
            out["F"] = [PENALTY, PENALTY]
            return

        try:
            stats = EvaluatePath(self.G, path)
            out["F"] = [float(stats['time']), float(stats['co2'])]
        except:
            out["F"] = [PENALTY, PENALTY]

# Sampling
class GraphSampling(Sampling):
    def __init__(self, G, start_node, end_node, start_time):
        super().__init__()
        self.G = G
        self.start_node = start_node
        self.end_node = end_node
        self.start_time = start_time

    def _do(self, problem, n_samples, **kwargs):
        initial_paths = InitMOEAD(
            self.G, self.start_node, self.end_node, self.start_time, pop_size=n_samples
        )
        
        if not initial_paths:
             try:
                initial_paths = [nx.shortest_path(self.G, self.start_node, self.end_node)]
             except:
                initial_paths = [[]] 
        
        while len(initial_paths) < n_samples:
            initial_paths.append(initial_paths[0]) 
            
        initial_paths = initial_paths[:n_samples]
        
        X = np.empty((n_samples, 1), dtype=object)
        for i in range(n_samples):
            X[i, 0] = initial_paths[i]
        
        return X

class PathCrossover(Crossover):
    def __init__(self, prob=0.9):
        super().__init__(2, 2, prob=prob)

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape
        Y = np.empty_like(X)
        for i in range(n_matings):
            p_a, p_b = X[0, i, 0], X[1, i, 0]
            if p_a is None or p_b is None:
                Y[0, i, 0], Y[1, i, 0] = p_a, p_b
                continue
            
            try:
                # Cruzamento simples em nó comum
                commons = list(set(p_a[1:-1]) & set(p_b[1:-1]))
                if commons:
                    cut = np.random.choice(commons)
                    idx_a, idx_b = p_a.index(cut), p_b.index(cut)
                    Y[0, i, 0] = p_a[:idx_a] + p_b[idx_b:]
                    Y[1, i, 0] = p_b[:idx_b] + p_a[idx_a:]
                else:
                    Y[0, i, 0], Y[1, i, 0] = p_a, p_b
            except:
                Y[0, i, 0], Y[1, i, 0] = p_a, p_b
        return Y
    
class PathMutation(Mutation):
    def __init__(self, G, prob=0.2):
        super().__init__(prob=1.0) 
        self.G = G
        self.real_prob = prob 

    def _do(self, problem, X, **kwargs):
        # X é uma matriz (n_indivíduos x 1) de objetos
        for i in range(len(X)):
            # AQUI aplicamos a probabilidade real
            if np.random.random() < self.real_prob:
                path = X[i, 0]
                
                # Proteção contra caminhos inválidos ou muito curtos
                if path is None or len(path) < 5:
                    continue
                
                try:
                    # Seleciona dois índices aleatórios no caminho
                    idxs = sorted(np.random.choice(range(len(path)), 2, replace=False))
                    u, v = path[idxs[0]], path[idxs[1]]
                    
                    subpath = nx.shortest_path(self.G, u, v, weight='length')
                    # Se encontrou e faz sentido, aplica
                    if len(subpath) > 0:
                        # Reconstrói: Inicio + Novo Trecho + Fim
                        new_path = path[:idxs[0]] + subpath + path[idxs[1]+1:]
                        X[i, 0] = new_path
                except:
                    pass 
        return X

def print_detailed_itinerary(G, path):
    """
    Imprime um itinerário passo-a-passo baseando-se nos nós do caminho
    e nos atributos das arestas do grafo (modo de transporte, nome da rota, etc).
    """
    if not path or len(path) < 2:
        print("Caminho vazio ou inválido.")
        return

    print("\n" + "="*40)
    print("📍 DETALHES DO ITINERÁRIO")
    print("="*40)

    # Inicialização
    current_mode = None
    current_route_name = None
    segment_start_node = path[0]
    segment_time = 0
    segment_co2 = 0
    
    # Função auxiliar para obter nome legível do nó (se existir atributo 'name' ou 'label')
    def get_node_name(n):
        return G.nodes[n].get('name', str(n)) # Retorna ID se não houver nome

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        # Obter dados da aresta (Assume MultiDiGraph, pega a primeira aresta [0])
        # Se o teu grafo não for MultiGraph, usa apenas G[u][v]
        edge_data = G[u][v][0] if G.is_multigraph() else G[u][v]
        
        # Extrair atributos (ajusta as chaves conforme o teu grafo: 'type', 'mode', etc.)
        # Exemplo: 'mode' pode ser 'walk', 'bus', 'subway', 'car'
        mode = edge_data.get('mode', 'walk') 
        route_name = edge_data.get('route_name', '') # Ex: '205', 'Linha Amarela'
        
        time_cost = edge_data.get('time', 0) # Ajusta chave se for 'weight' ou 'duration'
        co2_cost = edge_data.get('co2', 0)

        # Lógica de Agrupamento:
        # Se o modo mudou OU se o nome da rota mudou (ex: trocou de autocarro), imprime o anterior
        mode_changed = (mode != current_mode)
        route_changed = (route_name != current_route_name)
        
        if i > 0 and (mode_changed or route_changed):
            # Imprimir segmento anterior acumulado
            print_segment(current_mode, current_route_name, segment_start_node, u, segment_time, segment_co2, get_node_name)
            
            # Resetar para o novo segmento
            segment_start_node = u
            segment_time = 0
            segment_co2 = 0

        # Acumular custos e atualizar estado atual
        current_mode = mode
        current_route_name = route_name
        segment_time += time_cost
        segment_co2 += co2_cost

    # Imprimir o último segmento pendente
    print_segment(current_mode, current_route_name, segment_start_node, path[-1], segment_time, segment_co2, get_node_name)
    print("="*40 + "\n")

def print_segment(mode, route_name, start_node, end_node, time, co2, name_func):
    """Formata a impressão de um segmento de viagem."""
    start_name = name_func(start_node)
    end_name = name_func(end_node)
    
    # Ícones para embelezar
    icon = "🚶"
    action = "Caminhar"
    
    if mode in ['bus', 'autocarro']:
        icon = "🚌"
        action = f"Apanhar Autocarro {route_name}"
    elif mode in ['subway', 'metro']:
        icon = "🚇"
        action = f"Apanhar Metro {route_name}"
    elif mode in ['car', 'carro']:
        icon = "🚗"
        action = "Conduzir"
    elif mode in ['train', 'comboio']:
        icon = "🚆"
        action = f"Apanhar Comboio {route_name}"
    
    print(f"{icon} {action}")
    print(f"   De:   {start_name}")
    print(f"   Até:  {end_name}")
    print(f"   Dura: {time:.1f}s | CO2: {co2:.2f}g")
    print("   ---")