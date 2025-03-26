"""
音声処理のためのユーティリティ関数
"""
import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

def load_audio(file_path, sr=None):
    """
    音声ファイルを読み込む
    
    Args:
        file_path: 音声ファイルのパス
        sr: サンプリングレート（Noneの場合は元のレートを維持）
        
    Returns:
        y: 音声データ
        sr: サンプリングレート
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    except Exception as e:
        print(f"音声ファイル {file_path} の読み込みに失敗しました: {e}")
        return None, None

def trim_silence(y, top_db=60):
    """
    無音区間を除去する
    
    Args:
        y: 音声データ
        top_db: 無音と判定するdBしきい値
        
    Returns:
        y_trimmed: 無音区間を除去した音声データ
    """
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed
    except Exception as e:
        print(f"無音区間の除去に失敗しました: {e}")
        return y

def plot_waveform(y, sr, title='Waveform', output_file=None):
    """
    波形をプロットする
    
    Args:
        y: 音声データ
        sr: サンプリングレート
        title: プロットのタイトル
        output_file: 出力ファイルパス（Noneの場合は表示のみ）
    """
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_spectrogram(y, sr, title='Spectrogram', output_file=None):
    """
    スペクトログラムをプロットする
    
    Args:
        y: 音声データ
        sr: サンプリングレート
        title: プロットのタイトル
        output_file: 出力ファイルパス（Noneの場合は表示のみ）
    """
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def augment_pitch(y, sr, n_steps=2):
    """
    ピッチシフトによるデータ拡張
    
    Args:
        y: 音声データ
        sr: サンプリングレート
        n_steps: ピッチシフトの量（半音単位）
        
    Returns:
        音声データのピッチを変更したもの
    """
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def augment_speed(y, speed_factor=1.2):
    """
    再生速度変更によるデータ拡張
    
    Args:
        y: 音声データ
        speed_factor: 速度変更の係数（>1で速く、<1で遅く）
        
    Returns:
        音声データの速度を変更したもの
    """
    indices = np.round(np.arange(0, len(y), speed_factor)).astype(int)
    indices = indices[indices < len(y)]
    return y[indices]

def add_noise(y, noise_level=0.005):
    """
    ノイズ追加によるデータ拡張
    
    Args:
        y: 音声データ
        noise_level: ノイズの強さ
        
    Returns:
        ノイズを追加した音声データ
    """
    noise = np.random.randn(len(y)) * noise_level
    return y + noise

def save_audio(y, sr, output_file):
    """
    音声データをファイルに保存する
    
    Args:
        y: 音声データ
        sr: サンプリングレート
        output_file: 出力ファイルパス
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sf.write(output_file, y, sr)
    print(f"音声ファイルを {output_file} に保存しました") 