from math import floor
import sox
import csv


with open('./files/list.csv') as f:
  reader = csv.DictReader(f)

  for line in reader:
    voice_file = f'./voice/remove_silence/{line["id"]}.wav'

    duration = sox.file_info.duration(voice_file)
    for index in range(0, floor(duration)):
      if index + 1 > duration:
        break
      tfm = sox.Transformer()
      tfm.trim(index, index+1).build(voice_file, f'./voice/split/{line["id"]}_{index}.wav')

