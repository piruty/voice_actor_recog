#!/bin/bash

echo "話者認識システムの環境セットアップを開始します..."

# uvがインストールされているか確認
if ! command -v uv &> /dev/null; then
    echo "uvがインストールされていません。インストールします..."
    pip install uv
fi

# 仮想環境の作成
echo "Python仮想環境を作成しています..."
uv venv

# 仮想環境のアクティベート方法（環境によって異なる）
echo "仮想環境をアクティベートするには以下のコマンドを実行してください:"
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "  source .venv/bin/activate"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "  .venv\\Scripts\\activate"
fi

# 仮想環境のアクティベートを試行
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source .venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    .venv\\Scripts\\activate
fi

# 必要なパッケージのインストール
echo "必要なパッケージをインストールしています..."
uv pip install -r requirements.txt

# requirementsからだとうまく入れられなかったので、個別でインストール
uv add seaborn torch

echo "必要なディレクトリを作成しています..."
mkdir -p models results

echo "環境セットアップが完了しました！"
echo ""
echo "次のステップ:"
echo "1. まず特徴量を抽出します:"
echo "   python src/features/extract_features.py"
echo ""
echo "2. モデルを訓練します:"
echo "   python src/models/train_model.py"
echo ""
echo "3. 音声ファイルから話者を予測します:"
echo "   python src/predict.py --input <音声ファイルのパス>"
echo ""
echo "サンプルノートブックを試すには:"
echo "   jupyter notebook notebooks/speaker_recognition_example.ipynb" 