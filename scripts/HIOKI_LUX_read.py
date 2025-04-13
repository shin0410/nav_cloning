import serial
import time
import csv
from datetime import datetime

# シリアルポートの設定
PORT = '/dev/ttyACM0'  # ポート名を確認済み
BAUD_RATE = 38400
TIMEOUT = 10  # タイムアウトを10秒に延長

# シリアルポートを開く
try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit(1)

def send_command(command, retry=3):
    """コマンドをデバイスに送信し、応答を取得する"""
    for _ in range(retry):
        ser.write((command + '\r\n').encode())
        time.sleep(1)  # コマンド送信後に1秒待つ
        response = ser.read_all().decode().strip()
        if response:
            print(f"Sent: {command}, Received: {response}")  # デバッグ出力
            return response
        else:
            print(f"Sent: {command}, No response received, retrying...")  # 応答がない場合再試行
            time.sleep(2)  # 再試行前に2秒待つ
    return None

# デバイスの初期化
print("Initializing device...")
init_response = send_command(':SYST:INIT')
if not init_response:
    print("Failed to initialize device.")
    ser.close()
    exit(1)
time.sleep(3)  # 初期化後に3秒待つ

# デバイスモデルの確認
model = send_command('QPID')
print(f"Device Model: {model}")

# ゼロ調整
zero_adjustment = send_command(':0ADJUST')
if zero_adjustment == "CAP ERR":
    print("Zero Adjustment failed: Ensure the sensor cap is properly attached.")
elif zero_adjustment == "OK":
    print("Zero Adjustment successful.")
else:
    print(f"Zero Adjustment response: {zero_adjustment}")

# キャップを外して再度測定
input("Remove the sensor cap and press Enter to continue...")

# 日付時刻を付与したログファイル名を準備
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
logfile = f'lux_log_{timestamp_str}.csv'

# ログファイルの初期化（ヘッダー書き込み）
with open(logfile, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Lux'])

# 継続モニタリング
try:
    while True:
        lux_value = send_command(':MEAS?')
        if lux_value:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(logfile, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, lux_value])
            print(f"Logged at {timestamp}: {lux_value}")
        else:
            print("Failed to get Lux Value.")
        
        time.sleep(1)  # 1秒待つ
except KeyboardInterrupt:
    print("Monitoring stopped by user.")

# シリアルポートを閉じる
ser.close()



