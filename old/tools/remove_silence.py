import sox
import csv


with open('./files/list.csv') as f:
  reader = csv.DictReader(f)

  for line in reader:
    voice_file = f'./voice/raw/{line["id"]}.wav'
    tfm = sox.Transformer()
    tfm.silence().build(voice_file, f'./voice/remove_silence/{line["id"]}.wav')
