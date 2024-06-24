import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import folium
from folium.plugins import TimestampedGeoJson, MarkerCluster
from tqdm import tqdm
import random

print("程序開始運行...")

# 讀取數據
youbike_data = pd.read_csv(r'youbike_data.csv')
mrt_od_data = pd.read_csv(r'MRT_OD.csv')
mrt_gps_data = pd.read_csv(r'MRT_GPS.csv')

# 整合站點數據
youbike_data_grouped = youbike_data.groupby('sna').agg({
    'sno': 'first',
    'tot': 'first',
    'sbi': 'mean',
    'bemp': 'mean',
    'lat': 'mean',
    'lng': 'mean',
    'mday': 'first',
    'sarea': 'first',
    'ar': 'first',
    'sareaen': 'first',
    'snaen': 'first',
    'aren': 'first'
}).reset_index()

# 資料前處理
youbike_data_grouped['mday'] = pd.to_datetime(youbike_data_grouped['mday'])
youbike_data_grouped['hour'] = youbike_data_grouped['mday'].dt.hour
youbike_data_grouped['weekday'] = youbike_data_grouped['mday'].dt.weekday

# 添加高峰時間標記
youbike_data_grouped['is_peak'] = youbike_data_grouped['hour'].apply(lambda x: 1 if 7 <= x <= 10 or 18 <= x <= 20 else 0)

# 特徵工程
features = ['hour', 'weekday', 'is_peak', 'sno', 'tot']
X = youbike_data_grouped[features]
y = youbike_data_grouped['sbi']

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 預測自行車需求
days = pd.date_range(start='2024-05-21', end='2024-05-24', freq='D')
hours = np.arange(0, 24, 3)  # 每三個小時進行一次預測
stations = youbike_data_grouped['sno'].unique()

def fitness_function(schedule, youbike_data, model):
    total_cost = 0
    for task in schedule:
        timestamp, station, operation, bike_count = task
        predicted_sbi = model.predict(pd.DataFrame({
            'hour': [timestamp.hour],
            'weekday': [timestamp.weekday()],
            'is_peak': [1 if 7 <= timestamp.hour <= 10 or 18 <= timestamp.hour <= 20 else 0],
            'sno': [station],
            'tot': [youbike_data[youbike_data['sno'] == station]['tot'].iloc[0]]
        }))
        if operation == 'add':
            total_cost += max(0, 5 - predicted_sbi[0]) * bike_count
        elif operation == 'remove':
            total_cost += max(0, predicted_sbi[0] - 15) * bike_count
    return total_cost

def initialize_population(youbike_data, days, hours, stations, pop_size=20):
    population = []
    for _ in range(pop_size):
        schedule = []
        for day in days:
            for hour in hours:
                for station in stations:
                    if station in [1441, 1749]:
                        continue
                    operation = random.choice(['add', 'remove'])
                    bike_count = random.randint(1, 50)
                    timestamp = pd.Timestamp(day) + pd.Timedelta(hours=hour)
                    schedule.append((timestamp, station, operation, bike_count))
        population.append(schedule)
    return population

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(schedule, mutation_rate=0.01):
    for i in range(len(schedule)):
        if random.random() < mutation_rate:
            timestamp, station, operation, bike_count = schedule[i]
            operation = random.choice(['add', 'remove'])
            bike_count = random.randint(1, 50)
            schedule[i] = (timestamp, station, operation, bike_count)
    return schedule

def select_parents(population, fitness_scores):
    parents = random.choices(population, weights=fitness_scores, k=2)
    return parents[0], parents[1]

def local_search(schedule, youbike_data, model, max_iter=100):
    best_schedule = schedule
    best_cost = fitness_function(schedule, youbike_data, model)
    for _ in range(max_iter):
        new_schedule = mutate(schedule.copy(), mutation_rate=0.1)
        new_cost = fitness_function(new_schedule, youbike_data, model)
        if new_cost < best_cost:
            best_schedule = new_schedule
            best_cost = new_cost
    return best_schedule

# 初始化人口
population = initialize_population(youbike_data_grouped, days, hours, stations, pop_size=20)

# 遺傳算法參數
generations = 10
mutation_rate = 0.01

for generation in tqdm(range(generations), desc="進行遺傳算法"):
    fitness_scores = [fitness_function(schedule, youbike_data_grouped, model) for schedule in population]
    new_population = []
    for _ in range(len(population) // 2):
        parent1, parent2 = select_parents(population, fitness_scores)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        new_population.append(local_search(child1, youbike_data_grouped, model))
        new_population.append(local_search(child2, youbike_data_grouped, model))
    population = new_population

# 選擇最佳調度方案
best_schedule = min(population, key=lambda schedule: fitness_function(schedule, youbike_data_grouped, model))

# 打印最佳調度方案
for task in best_schedule:
    print(f"時間: {task[0]}, 站點: {task[1]}, 操作: {task[2]}, 自行車數量: {task[3]}")

# 創建地圖
map_osm = folium.Map(location=[youbike_data_grouped['lat'].mean(), youbike_data_grouped['lng'].mean()], zoom_start=12)

# 添加站點標記
marker_cluster = MarkerCluster().add_to(map_osm)
for idx, row in youbike_data_grouped.iterrows():
    folium.Marker(
        location=[row['lat'], 'lng']],
        popup=f"站點: {row['sna']}, 自行車數量: {round(row['sbi'])}, 空位數量: {round(row['bemp'])}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# 構建調度車路徑的 GeoJSON 數據
features = []
car_icon_path = 'car.png'
colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'brown', 'pink', 'grey', 'black']
for vehicle_num, task in enumerate(best_schedule, start=1):
    timestamp, station, operation, bike_count = task
    station_info = youbike_data_grouped[youbike_data_grouped['sno'] == station]
    if not station_info.empty:
        lat, lng = station_info.iloc[0]['lat'], station_info.iloc[0]['lng']
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [lng, lat],
            },
            'properties': {
                'time': timestamp.isoformat(),
                'style': {'color': colors[(vehicle_num - 1) % len(colors)]},
                'icon': 'icon',
                'iconstyle': {
                    'iconUrl': car_icon_path,
                    'iconSize': [32, 32]
                },
                'popup': f"時間: {timestamp}, 調度車: {vehicle_num}, 操作: {operation}, 數量: {bike_count}"
            }
        }
        features.append(feature)

# 添加時間控制器
TimestampedGeoJson({
    'type': 'FeatureCollection',
    'features': features,
}, period='PT3H', add_last_point=True, auto_play=True, loop=True).add_to(map_osm)

# 保存地圖
map_osm.save('youbike_dispatch_map_with_time_control.html')

print("程序運行結束。")
