"""
学習済みモデルを使用して、新しい音声データから話者を予測するモジュール
"""
import os
import sys
import argparse
import pickle
from pathlib import Path
import numpy as np

# パッケージのインポートパスを追加
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.features.extract_features import extract_mfcc

def load_model(model_dir):
    """
    モデルとスケーラーを読み込む
    
    Args:
        model_dir: モデルが保存されたディレクトリ
        
    Returns:
        model: 学習済みモデル
        scaler: 特徴量スケーラー
    """
    model_path = Path(model_dir) / "model.pkl"
    scaler_path = Path(model_dir) / "scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("モデルファイルが見つかりません。")
        print("まず src/models/train_model.py を実行してください。")
        sys.exit(1)
    
    # モデルの読み込み
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # スケーラーの読み込み
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict_speaker(audio_file, model, scaler):
    """
    音声ファイルから話者を予測する
    
    Args:
        audio_file: 音声ファイルのパス
        model: 学習済みモデル
        scaler: 特徴量スケーラー
        
    Returns:
        predicted_speaker: 予測された話者
        probability: 予測確率
    """
    # 特徴量の抽出
    features = extract_mfcc(audio_file)
    
    if features is None:
        print(f"音声ファイル {audio_file} の特徴量抽出に失敗しました。")
        return None, None
    
    # 特徴量のスケーリング
    features = features.reshape(1, -1)
    features = scaler.transform(features)
    
    # 予測
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    max_prob = max(probabilities)
    
    return prediction, max_prob

def main():
    """
    メイン処理
    """
    parser = argparse.ArgumentParser(description="音声ファイルから話者を予測します")
    parser.add_argument("--input", "-i", required=True, help="入力音声ファイルのパス")
    parser.add_argument("--model_dir", "-m", default=None, help="モデルディレクトリのパス")
    args = parser.parse_args()
    
    # 入力ファイルの確認
    if not os.path.exists(args.input):
        print(f"音声ファイル {args.input} が見つかりません。")
        sys.exit(1)
    
    # モデルディレクトリの設定
    if args.model_dir is None:
        model_dir = project_root / "models"
    else:
        model_dir = Path(args.model_dir)
    
    # モデルの読み込み
    model, scaler = load_model(model_dir)
    
    # 話者の予測
    speaker, probability = predict_speaker(args.input, model, scaler)
    
    if speaker is not None:
        print(f"\n予測結果:")
        print(f"音声ファイル: {args.input}")
        print(f"予測された話者: {speaker}")
        print(f"予測確率: {probability:.4f}")
    else:
        print("予測に失敗しました。")

if __name__ == "__main__":
    main() 