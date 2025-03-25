import youtube_dl
import csv


results = []
with open('./files/list.csv', 'r') as f:
  reader = csv.DictReader(f)

  options = {
    'format': 'bestaudio/best',
    'outtmpl': 'voice/raw/%(id)s.%(ext)s',
    'postprocessors': [
      {
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192'
      },
      {
        'key': 'FFmpegMetadata'
      }
    ]
  }

  with youtube_dl.YoutubeDL(options) as ydl:
    for line in reader:
      ydl.download(f"https://www.youtube.com/watch?v={line['id']}")
