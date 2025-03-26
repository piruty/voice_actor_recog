"""
音声ファイルから特徴量を抽出するモジュール
"""
import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import pickle
from pathlib import Path

def extract_mfcc(file_path, n_mfcc=40):
    """
    音声ファイルからMFCC特徴量を抽出する
    
    Args:
        file_path: 音声ファイルのパス
        n_mfcc: MFCCの次元数
        
    Returns:
        MFCC特徴量の平均値と標準偏差
    """
    try:
        # 音声の読み込み
        y, sr = librosa.load(file_path, sr=None)
        
        # 無音区間の除去
        y, _ = librosa.effects.trim(y)
        
        # MFCCの抽出
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # 統計量の計算
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        return np.concatenate([mfcc_mean, mfcc_std])
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_features_from_dir(data_dir, output_file=None):
    """
    指定したディレクトリ内の音声ファイルから特徴量を抽出し、DataFrame形式で返す
    
    Args:
        data_dir: 音声ファイルが含まれるディレクトリのパス
        output_file: 特徴量を保存するファイルパス（オプション）
        
    Returns:
        特徴量が格納されたDataFrame
    """
    features = []
    labels = []
    file_paths = []
    
    # 話者ディレクトリを走査
    for speaker_dir in tqdm(os.listdir(data_dir)):
        speaker_path = os.path.join(data_dir, speaker_dir)
        
        # スピーカーディレクトリ以外を無視
        if not os.path.isdir(speaker_path) or speaker_dir.startswith('.'):
            continue
            
        # スピーカーディレクトリ内の音声ファイルを処理
        for file in os.listdir(speaker_path):
            if not file.endswith(('.wav', '.mp3')):
                continue
                
            file_path = os.path.join(speaker_path, file)
            feature_vector = extract_mfcc(file_path)
            
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(speaker_dir)
                file_paths.append(file_path)
    
    # DataFrameの作成
    X = pd.DataFrame(features)
    df = pd.DataFrame({
        'file_path': file_paths,
        'speaker': labels
    })
    
    # 特徴量のカラム名を設定
    for i in range(X.shape[1]):
        df[f'feature_{i}'] = X[i]
    
    # 特徴量を保存
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_pickle(output_file)
        print(f"特徴量を {output_file} に保存しました")
    
    return df

if __name__ == "__main__":
    # カレントディレクトリからの相対パスで data ディレクトリを指定
    data_dir = Path(__file__).resolve().parents[2] / "data"
    output_file = Path(__file__).resolve().parents[2] / "data" / "features.pkl"
    
    print(f"データディレクトリ: {data_dir}")
    print(f"出力ファイル: {output_file}")
    
    # 特徴量の抽出
    features_df = extract_features_from_dir(data_dir, output_file)
    print(f"抽出された特徴量: {features_df.shape}")
    print(features_df.head()) 