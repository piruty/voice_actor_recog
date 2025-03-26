"""
特徴量から話者認識モデルを訓練するモジュール
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(features_file):
    """
    特徴量ファイルからトレーニングデータとテストデータを準備する
    
    Args:
        features_file: 特徴量ファイルのパス
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # 特徴量の読み込み
    df = pd.read_pickle(features_file)
    
    # 特徴量の列だけを取得
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['speaker'].values
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特徴量のスケーリング
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_random_forest(X_train, y_train):
    """
    ランダムフォレストモデルを訓練する
    
    Args:
        X_train: トレーニングデータ
        y_train: トレーニングラベル
        
    Returns:
        訓練済みモデル
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    モデルの評価を行う
    
    Args:
        model: 訓練済みモデル
        X_test: テストデータ
        y_test: テストラベル
        
    Returns:
        y_pred: 予測結果
    """
    y_pred = model.predict(X_test)
    
    # 結果の表示
    print("\n分類レポート:")
    print(classification_report(y_test, y_pred))
    
    # 混同行列の表示
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), 
                yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # 保存先ディレクトリを作成
    save_dir = Path(__file__).resolve().parents[2] / "results"
    save_dir.mkdir(exist_ok=True)
    
    # 混同行列を保存
    plt.savefig(save_dir / "confusion_matrix.png")
    print(f"混同行列を {save_dir / 'confusion_matrix.png'} に保存しました")
    
    return y_pred

def save_model(model, scaler, output_dir):
    """
    モデルとスケーラーを保存する
    
    Args:
        model: 訓練済みモデル
        scaler: 特徴量スケーラー
        output_dir: 出力ディレクトリ
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデルの保存
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # スケーラーの保存
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"モデルとスケーラーを {output_dir} に保存しました")

if __name__ == "__main__":
    # 特徴量ファイルのパス
    project_root = Path(__file__).resolve().parents[2]
    features_file = project_root / "data" / "features.pkl"
    model_dir = project_root / "models"
    
    if not os.path.exists(features_file):
        print(f"特徴量ファイル {features_file} が見つかりません。")
        print("まず src/features/extract_features.py を実行してください。")
        exit(1)
    
    print(f"特徴量ファイル: {features_file}")
    print(f"モデル保存先: {model_dir}")
    
    # データの準備
    X_train, X_test, y_train, y_test, scaler = prepare_data(features_file)
    print(f"トレーニングデータ: {X_train.shape}, テストデータ: {X_test.shape}")
    
    # モデルのトレーニング
    model = train_random_forest(X_train, y_train)
    
    # モデルの評価
    evaluate_model(model, X_test, y_test)
    
    # モデルの保存
    save_model(model, scaler, model_dir) 