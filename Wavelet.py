import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pywt


# WAVELET anomaly score (для порога)
def wavelet_anomaly_score(signal, wavelet='haar', level=3, window_size=64, step=16):
    scores = np.zeros(len(signal))
    for start in range(0, len(signal) - window_size + 1, step):
        end = start + window_size
        window = signal[start:end]
        coeffs = pywt.wavedec(window, wavelet, level=level)
        cD1, cD3 = coeffs[1], coeffs[3]
        energy = np.sum(cD1**2) + np.sum(cD3**2)
        scores[start:end] = energy
    return scores


# Anomaly score только по cD1 
def wavelet_energy_cd1(signal, wavelet='haar', level=3, window_size=64, step=16):
    scores = np.zeros(len(signal))
    for start in range(0, len(signal) - window_size + 1, step):
        end = start + window_size
        window = signal[start:end]
        coeffs = pywt.wavedec(window, wavelet, level=level)
        cD1 = coeffs[1]
        scores[start:end] = np.sum(cD1**2)
    return scores


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
        
    # Теперь 3 графика в строке: сигнал, общий скор (карта), энергия cD1
    fig, axes = plt.subplots(num_sensors, 3, figsize=(21, 5 * num_sensors))
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
        
        if len(sensor_data) < 64:
            print(f"Датчик {sensor_id}: меньше 64 точек, вейвлет не применить.")
            continue

        values = sensor_data['value'].values
        short_id = str(sensor_id)[:6]
        
        # Общий score (для поиска аномалий)
        w_scores = wavelet_anomaly_score(values)
        sensor_data['Wavelet_Score'] = w_scores
        
        # Отдельный score по cD1 (для признака)
        e_cd1 = wavelet_energy_cd1(values)
        sensor_data['Energy_cD1'] = e_cd1
        
        # Настройка порога (по всем точкам) 
        w_thresh = np.mean(w_scores) + 3.5 * np.std(w_scores)
        w_anomalies = sensor_data[sensor_data['Wavelet_Score'] > w_thresh]
        
        # График 1: сигнал 
        ax1 = axes[i, 0]
        ax1.plot(sensor_data['datetime'], sensor_data['value'], color='#d97523', linewidth=1.2)
        ax1.set_title(f'Датчик {short_id}: Wavelet (Сигнал)')
        ax1.set_ylabel('Температура (°C)')
        ax1.plot(w_anomalies['datetime'], w_anomalies['value'], 'ro', markersize=4)
        ax1.grid(True, alpha=0.2)
        
        # График 2: общая карта аномалий (cD1 + cD3)
        ax2 = axes[i, 1]
        ax2.plot(sensor_data['datetime'], sensor_data['Wavelet_Score'], 
                 color='purple', linewidth=1, label='Wavelet Score')
        ax2.plot(w_anomalies['datetime'], w_anomalies['Wavelet_Score'], 
                 'ro', markersize=5, label='Аномалия')
        ax2.axhline(y=w_thresh, color='red', linestyle='--', alpha=0.7, 
                    label=f'Порог: {w_thresh:.0f}')
        ax2.set_title(f'Датчик {short_id}: Wavelet (Карта аномалий)')
        ax2.set_ylabel('Энергия cD1+cD3')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.2)
        
        # График 3: только энергия cD1
        ax3 = axes[i, 2]
        ax3.plot(sensor_data['datetime'], sensor_data['Energy_cD1'], 
                 color='teal', linewidth=1, label='Energy cD1')
        # Подсвечиваем те же аномальные точки 
        ax3.plot(w_anomalies['datetime'], w_anomalies['Energy_cD1'], 
                 'ro', markersize=4, label='Аномалия')
        ax3.set_title(f'Датчик {short_id}: Energy cD1 (резкие скачки)')
        ax3.set_ylabel('Энергия cD1')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.2)
        
        # Форматирование времени для всех трех графиков
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'freezer_temperature.csv')
    main(csv_path)
