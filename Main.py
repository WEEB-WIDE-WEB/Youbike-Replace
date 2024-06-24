import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import folium
from folium.plugins import MarkerCluster
from tqdm import tqdm

print("程序開始運行...")

# 讀取 YouBike 數據，請替換為您的實際文件路徑
print("讀取 YouBike 數據...")
youbike_data = pd.read_csv(r'C:\Users\henry\PycharmProjects\AndYoubike\youbike_data.csv')

# 整合相同站點名稱的數據，只保留時間、車輛數等
print("整合站點數據...")
youbike_data_grouped = youbike_data.groupby('sna').agg({
    'sno': 'first',       # 取第一個編號
    'tot': 'mean',        # 平均總停車位數量
    'sbi': 'mean',        # 平均可借車輛數量
    'bemp': 'mean',       # 平均可還空位數量
    'lat': 'mean',        # 緯度取平均值
    'lng': 'mean',        # 經度取平均值
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

# 創建地圖
print("創建地圖...")
map_osm = folium.Map(location=[youbike_data_grouped['lat'].mean(), youbike_data_grouped['lng'].mean()], zoom_start=12)

# 添加站點標記
print("添加站點標記...")
for idx, row in youbike_data_grouped.iterrows():
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=f"站點: {row['sna']}, 自行車數量: {row['sbi']}, 空位數量: {row['bemp']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(map_osm)

# 添加調度車的路徑和標記
print("添加調度車的路徑和標記...")
colors = ['red', 'green', 'blue', 'purple', 'orange']
car_icon_path = r'C:\Users\henry\PycharmProjects\AndYoubike\car.png'

for vehicle_num, schedule in enumerate(dispatch_schedule):
    for task in schedule:
        timestamp, station, operation, bike_count = task
        station_info = youbike_data_grouped[youbike_data_grouped['sno'] == station]
        if not station_info.empty:
            lat, lng = station_info.iloc[0]['lat'], station_info.iloc[0]['lng']
            icon_url = car_icon_path if operation == 'add' else car_icon_path
            folium.Marker(
                location=[lat, lng],
                popup=f"時間: {timestamp}, 操作: {operation}, 數量: {bike_count}",
                icon=folium.CustomIcon(icon_url, icon_size=(30, 30))
            ).add_to(map_osm)

# 保存地圖
print("保存地圖...")
map_osm.save('youbike_dispatch_map_with_time_control.html')

print("程序運行結束。")
