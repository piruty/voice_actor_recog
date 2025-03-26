# 話者認識システム

音声データから話者を識別するシステムです。

## プロジェクト構成

```
.
├── data/           # 音声データ
├── src/            # ソースコード
│   ├── features/   # 特徴量抽出
│   ├── models/     # モデル実装
│   └── utils/      # ユーティリティ関数
├── tests/          # テストコード
└── notebooks/      # 実験用Jupyter notebooks
```

## 環境セットアップ

このプロジェクトは `uv` を使用してPython環境を管理します。

```bash
# uvのインストール（初回のみ）
pip install uv

# 仮想環境の作成とパッケージのインストール
uv venv
uv pip install -r requirements.txt
```

## 使用方法

```bash
# 特徴量抽出
python src/features/extract_features.py

# モデルのトレーニング
python src/models/train_model.py

# 話者認識の実行
python src/predict.py --input <audio_file>
```

## データセット

`data/` ディレクトリには以下の話者のデータが含まれています:
- speaker1
- speaker2
- speaker3
- speaker4 