import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from statsmodels.tsa.seasonal import STL


# STL anomaly score
def stl_anomaly_score(signal, period=1440):
    if len(signal) < period * 2:
        period = max(2, len(signal) // 2)
    stl = STL(signal, period=period, robust=True)
    res = stl.fit()
    return np.abs(res.resid)


def main(csv_file):
    try:
        df = pd.read_csv(
            csv_file, 
            header=None, 
            skiprows=1, 
            names=['event_timestamp', 'io_id', 'value', 'value_min', 'value_max']
        )
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return
    
    df = df[['event_timestamp', 'io_id', 'value']]
    df['event_timestamp'] = pd.to_numeric(df['event_timestamp'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna()
    
    time_unit = 's'
    if df['event_timestamp'].max() > 1e11:
        time_unit = 'ms'
    df['datetime'] = pd.to_datetime(df['event_timestamp'], unit=time_unit)
    df = df.sort_values(by='datetime')
    
    unique_sensors = df['io_id'].unique()
    num_sensors = len(unique_sensors)
    if num_sensors == 0:
        return
        
    fig, axes = plt.subplots(num_sensors, 2, figsize=(16, 5 * num_sensors))
    if num_sensors == 1:
        axes = np.array([axes])
        
    for i, sensor_id in enumerate(unique_sensors):
        sensor_data = df[df['io_id'] == sensor_id].copy()
        

        # Настройка периода:

        # Весь ряд
        # start_time = sensor_data['datetime'].iloc[0]
        # end_time = sensor_data['datetime'].iloc[-1]
        
        # Сутки
        target_date = '2024-07-15'
        start_time = pd.Timestamp(f'{target_date} 00:00:00')
        end_time   = pd.Timestamp(f'{target_date} 23:59:59')


        
        sensor_data = sensor_data[
            (sensor_data['datetime'] >= start_time) & 
            (sensor_data['datetime'] <= end_time)
        ].copy()
        
        if len(sensor_data) < 10:
            print(f"Датчик {sensor_id}: слишком мало данных, пропускаю.")
            continue

        values = sensor_data['value'].values
        short_id = str(sensor_id)[:6]
        
        stl_period = min(1440, max(2, len(values) // 2))
        
        s_scores = stl_anomaly_score(values, period=stl_period)
        sensor_data['STL_Score'] = s_scores
        

        # Настройка порога (по всем точкам)
        s_thresh = np.mean(s_scores) + 3.5 * np.std(s_scores)
        
        s_anomalies = sensor_data[sensor_data['STL_Score'] > s_thresh]
        
        # График 1: сигнал 
        ax1 = axes[i, 0]
        ax1.plot(sensor_data['datetime'], sensor_data['value'], color='#d97523', linewidth=1.2)
        ax1.set_title(f'Датчик {short_id}: STL (Сигнал)')
        ax1.set_ylabel('Температура (°C)')
        ax1.plot(s_anomalies['datetime'], s_anomalies['value'], 'ro', markersize=4)
        ax1.grid(True, alpha=0.2)
        
        # График 2: карта аномалий
        ax2 = axes[i, 1]
        ax2.plot(sensor_data['datetime'], sensor_data['STL_Score'], 
                 color='#4a90e2', linewidth=1, label='STL Score (|Residual|)')
        ax2.plot(s_anomalies['datetime'], s_anomalies['STL_Score'], 
                 'ro', markersize=5, label='Аномалия')
        ax2.axhline(y=s_thresh, color='red', linestyle='--', alpha=0.7, 
                    label=f'Порог: {s_thresh:.2f}')
        ax2.set_title(f'Датчик {short_id}: STL (Карта аномалий)')
        ax2.set_ylabel('Остаток')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.2)
        
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'freezer_temperature.csv')
    main(csv_path)