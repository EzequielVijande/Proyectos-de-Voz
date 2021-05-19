import pyaudio
import wave
import sys

from matplotlib import pyplot as plt
from matplotlib import animation
import time

wf = wave.open("hello.wav", 'rb')
fs = wf.getframerate()

CHUNK = int(0.256*fs)

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=fs,
                output=True)

fig = plt.figure()
data_im = plt.imread('silence.jpeg')
im = plt.imshow(data_im)
im.set_data(data_im)
plt.draw()
plt.pause(1)


data = wf.readframes(CHUNK)

while data != b'':
    data_im = plt.imread('r.jpg') 
    im.set_data(data_im) 
    plt.draw()
    plt.pause(0.000001)
    stream.write(data)
    data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()

