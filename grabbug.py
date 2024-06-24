import requests
import pandas as pd
import schedule
import time
from datetime import datetime
import os
import io

# 定義CSV文件的URL
csv_url = "https://data.ntpc.gov.tw/api/datasets/71cd1490-a2df-4198-bef1-318479775e8a/csv/file"
# 定義JSON文件的URL
json_url = "https://data.ntpc.gov.tw/api/datasets/71CD1490-A2DF-4198-BEF1-318479775E8A/json?page=0&size=1000"
# 定義保存數據的文件路徑
data_file = r"C:\Users\henry\PycharmProjects\youbikeproject\ntpc_youbike_data.csv"

# 定義收集數據的函數
def collect_data():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(csv_url, headers=headers)
        response.raise_for_status()

        # 嘗試將CSV數據讀入DataFrame
        df = pd.read_csv(io.StringIO(response.text))

        # 添加當前時間
        df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 檢查DataFrame是否為空
        if df.empty:
            print("No data fetched from CSV")
            return
    except requests.exceptions.RequestException as e:
        print(f"Error fetching CSV data: {e}")
        print("Trying to fetch data from JSON URL...")
        try:
            response = requests.get(json_url, headers=headers)
            response.raise_for_status()

            # 嘗試將JSON數據讀入DataFrame
            data = response.json()
            df = pd.DataFrame(data)

            # 添加當前時間
            df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 檢查DataFrame是否為空
            if df.empty:
                print("No data fetched from JSON")
                return
        except requests.exceptions.RequestException as e:
            print(f"Error fetching JSON data: {e}")
            return

    # 檢查CSV文件是否存在
    if not os.path.exists(data_file):
        df.to_csv(data_file, mode='w', header=True, index=False, encoding='utf-8-sig')
        print(f"CSV file created and data collected at {df['timestamp'].iloc[0]}")
    else:
        df.to_csv(data_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"Data collected at {df['timestamp'].iloc[0]}")

# 每5分鐘收集一次數據
schedule.every(5).minutes.do(collect_data)

print("Scheduler started. Collecting data every 5 minutes...")

# 初始測試數據收集
collect_data()

# 啟動調度器
while True:
    schedule.run_pending()
    time.sleep(1)
