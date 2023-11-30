from mutagen.mp3 import MP3 as mp3
import pygame
import time


class mp3player:
    def __init__(self, sounddirectory="./sound/"):
        if not sounddirectory[-1] == "/":  # "/"が含まれていなければ追加
            sounddirectory = sounddirectory + "/"

        self.sounddirectory = sounddirectory

    def play_mp3(self, filename, sounddelay=0.25):
        if not filename[-4:] == ".mp3":  # ".mp3"が含まれていなければ追加
            filename = filename + ".mp3"

        filepath = self.sounddirectory + filename

        pygame.mixer.init()  # 初期化
        pygame.mixer.music.load(filepath)  # 音源を読み込み
        mp3_length = mp3(filepath).info.length  # 音源の長さ取得
        pygame.mixer.music.play(1)  # 再生開始。1の部分を変えるとn回再生
        time.sleep(mp3_length + sounddelay)  # 再生開始後、音源の長さだけ待つ
        pygame.mixer.music.stop()  # 音源の長さ待ったら再生停止
