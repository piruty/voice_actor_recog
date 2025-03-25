# voice-actor-recog
音声データからメル周波数ケプストラム係数(mfcc)を抽出し識別・認識を行う学習機

## Files
* NN.py: ニューラルネットモデル
* train_mfcc_data.py: docフォルダ内のtrain.txt, test.txtとvoicesの音声データから、mfccを抽出しニューラルネットに学習させるスクリプト
* recog_mfcc_data.py: 学習済みモデル（NN.xml）を読み込みdocフォルダ内のrecog.txtとvoicesの音声データから、mfccを抽出し識別させるスクリプト
* make_mfcc_data.py: 指定された音声データからmfccを抽出するスクリプト
* make_train_data.py: 指定したフォルダ内のデータと階層構造から、train.txt, test.txt, labels.txtを作成するスクリプト
* make_train_data.py: 指定したフォルダ内のデータから、recog.txtを作成するスクリプト

* convert_mov_to_wav.sh: 動画データから音声データを抽出するシェルスクリプト
* split_wav.sh: 音声データを1秒ずつに分割し、それぞれをフォルダごとに分類して格納するシェルスクリプト

## Requirements
* Numpy
* Scipy
* Scikits Talkbox
* PyBrain
* progressbar2

(requirements.txt参照)

* ffmpeg (convert_mov_to_wav.shが使用)
* sox (split_wav.shが使用)

## LICENSE
MIT License
