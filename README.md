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

kaggleに登録されている日本人話者データセットを使用
https://www.kaggle.com/datasets/nguyenthanhlim/japanese-non-verbal-verbal-voice-jvnv/data
これをvoiceディレクトリに保存し、以下のコマンドを実行してdataディレクトリ内に音声データを移動する

```bash
# 話者用のディレクトリを作成
mkdir -p data/speaker1 data/speaker2 data/speaker3 data/speaker4

# 音声ファイルをコピー
find voice/jvnv_v1/F1 -name "*.wav" -type f -exec cp {} data/speaker1/ \;
find voice/jvnv_v1/F2 -name "*.wav" -type f -exec cp {} data/speaker2/ \;
find voice/jvnv_v1/M1 -name "*.wav" -type f -exec cp {} data/speaker3/ \;
find voice/jvnv_v1/M2 -name "*.wav" -type f -exec cp {} data/speaker4/ \;
```

`data/` ディレクトリには以下の話者のデータが含まれています:
- speaker1
- speaker2
- speaker3
- speaker4 