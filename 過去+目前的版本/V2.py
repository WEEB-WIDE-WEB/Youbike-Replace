import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import folium
from folium.plugins import TimestampedGeoJson, MarkerCluster
from tqdm import tqdm  # 进度条显示

print("程序開始運行...")

# 讀取 YouBike 數據，請替換為您的實際文件路徑
print("讀取 YouBike 數據...")
youbike_data = pd.read_csv(r'C:\Users\henry\PycharmProjects\AndYoubike\youbike_data.csv')

# 讀取 MRT_OD 數據
print("讀取 MRT_OD 數據...")
mrt_od_data = pd.read_csv(r'C:\Users\henry\PycharmProjects\AndYoubike\MRT_OD.csv')

# 讀取 MRT_GPS 數據
print("讀取 MRT_GPS 數據...")
mrt_gps_data = pd.read_csv(r'C:\Users\henry\PycharmProjects\AndYoubike\MRT_GPS.csv')

# 整合相同站點名稱的數據，只保留時間、車輛數等
print("整合站點數據...")
youbike_data_grouped = youbike_data.groupby('sna').agg({
    'sno': 'first',       # 取第一個編號
    'tot': 'first',       # 總停車位數量取第一個
    'sbi': 'mean',        # 可借車輛數量取平均值
    'bemp': 'mean',       # 可還空位數量取平均值
    'lat': 'first',       # 緯度取第一個
    'lng': 'first',       # 經度取第一個
    'mday': 'first'       # 取第一個日期時間
}).reset_index()

# 資料前處理
print("資料前處理...")
youbike_data_grouped['mday'] = pd.to_datetime(youbike_data_grouped['mday'])
youbike_data_grouped['hour'] = youbike_data_grouped['mday'].dt.hour
youbike_data_grouped['weekday'] = youbike_data_grouped['mday'].dt.weekday

# 添加高峰時間標記
youbike_data_grouped['is_peak'] = youbike_data_grouped['hour'].apply(lambda x: 1 if 7 <= x <= 10 or 18 <= x <= 20 else 0)

# 特徵工程
print("特徵工程...")
features = ['hour', 'weekday', 'is_peak', 'sno', 'tot']
X = youbike_data_grouped[features]
y = youbike_data_grouped['sbi']

# 分割數據
print("分割數據...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型
print("訓練模型...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 預測一天中每小時的自行車需求
print("預測自行車需求...")
hours = np.arange(0, 24)
stations = youbike_data_grouped['sno'].unique()
predictions = []

for hour in tqdm(hours, desc="預測每小時自行車需求"):
    for station in stations:
        is_peak = 1 if 7 <= hour <= 10 or 18 <= hour <= 20 else 0
        prediction = model.predict(pd.DataFrame({
            'hour': [hour],
            'weekday': [0],  # 假設是星期一，您可以根據實際情況調整
            'is_peak': [is_peak],
            'sno': [station],
            'tot': [youbike_data_grouped[youbike_data_grouped['sno'] == station]['tot'].iloc[0]]
        }))
        predictions.append([hour, station, prediction[0]])

# 轉換為 DataFrame
print("整理預測結果...")
predictions_df = pd.DataFrame(predictions, columns=['hour', 'station', 'predicted_sbi'])

# 計算一天內需要調配的自行車數量
print("計算調度計劃...")
total_bikes = 30000
dispatch_vehicles = 5
max_bikes_per_vehicle = 30
extra_bikes_per_station = 10

# 簡單的調度策略：平均分配車輛並盡量使每個站點的自行車數量平衡
dispatch_schedule = []

for vehicle in tqdm(range(dispatch_vehicles), desc="分配調度車輛"):
    vehicle_schedule = []
    current_bikes = max_bikes_per_vehicle
    for hour in hours:
        stations_needing_bikes = predictions_df[(predictions_df['hour'] == hour) & (predictions_df['predicted_sbi'] < 5)]
        stations_with_extra_bikes = predictions_df[(predictions_df['hour'] == hour) & (predictions_df['predicted_sbi'] > 15)]
        
        for idx, station in stations_needing_bikes.iterrows():
            if current_bikes > 0:
                bikes_needed = round(5 - station['predicted_sbi'])
                if bikes_needed > 0:
                    timestamp = pd.Timestamp.now().replace(hour=hour, minute=0, second=0, microsecond=0)
                    if bikes_needed > current_bikes:
                        bikes_needed = current_bikes
                    vehicle_schedule.append((timestamp, station['station'], 'add', bikes_needed))
                    current_bikes -= bikes_needed

        for idx, station in stations_with_extra_bikes.iterrows():
            bikes_to_remove = round(station['predicted_sbi'] - 15)
            if bikes_to_remove > 0:
                timestamp = pd.Timestamp.now().replace(hour=hour, minute=0, second=0, microsecond=0)
                if bikes_to_remove + current_bikes > max_bikes_per_vehicle:
                    bikes_to_remove = max_bikes_per_vehicle - current_bikes
                vehicle_schedule.append((timestamp, station['station'], 'remove', bikes_to_remove))
                current_bikes += bikes_to_remove

    dispatch_schedule.append(vehicle_schedule)

# 添加捷運站出口放置10台不插入bump的ubike
extra_bike_schedule = []
for station in tqdm(stations, desc="添加捷運站旁的自行車"):
    if '捷運站' in youbike_data_grouped[youbike_data_grouped['sno'] == station]['sna'].values[0]:
        for hour in hours:
            timestamp = pd.Timestamp.now().replace(hour=hour, minute=0, second=0, microsecond=0)
            extra_bike_schedule.append((timestamp, station, 'add beside', extra_bikes_per_station))

# 創建地圖
print("創建地圖...")
map_osm = folium.Map(location=[youbike_data_grouped['lat'].mean(), youbike_data_grouped['lng'].mean()], zoom_start=12)

# 添加站點標記
print("添加站點標記...")
marker_cluster = MarkerCluster().add_to(map_osm)
for idx, row in youbike_data_grouped.iterrows():
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=f"站點: {row['sna']}, 自行車數量: {row['sbi']}, 空位數量: {row['bemp']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# 構建調度車路徑的 GeoJSON 數據
print("構建調度車路徑的 GeoJSON 數據...")
features = []
colors = ['red', 'green', 'blue', 'purple', 'orange']
for vehicle_num, schedule in enumerate(dispatch_schedule):
    for task in schedule:
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
                    'style': {'color': colors[vehicle_num]},
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': colors[vehicle_num],
                        'fillOpacity': 0.6,
                        'stroke': 'true',
                        'radius': 7
                    },
                    'popup': f"時間: {timestamp}, 操作: {operation}, 數量: {bike_count}"
                }
            }
            features.append(feature)

# 添加捷運站旁邊的自行車放置操作到 GeoJSON 數據
for task in extra_bike_schedule:
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
                'style': {'color': 'black'},
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': 'black',
                    'fillOpacity': 0.6,
                    'stroke': 'true',
                    'radius': 7
                },
                'popup': f"時間: {timestamp}, 操作: {operation}, 數量: {bike_count}"
            }
        }
        features.append(feature)

# 添加時間控制器
print("添加時間控制器...")
TimestampedGeoJson({
    'type': 'FeatureCollection',
    'features': features,
}, period='PT1H', add_last_point=True, auto_play=True, loop=True).add_to(map_osm)

# 保存地圖
print("保存地圖...")
map_osm.save('youbike_dispatch_map_with_time_control.html')

print("程序運行結束。")
