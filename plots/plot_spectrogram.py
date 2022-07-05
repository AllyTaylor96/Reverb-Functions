"""
Command line function to plot spectrogram of given audio file.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import argparse
import librosa
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Audio file spectrogram generator.')
parser.add_argument('audio_path',
                    help='Path to audio file to be plotted', type=str)
parser.add_argument('out_path',
                    help='Path for output waveform to be saved to', type=str)
args = parser.parse_args()


waveform, sr = torchaudio.load(args.audio_path)
audio_path = Path(args.audio_path)

# set up spectrogram transform, these values can be amended as seen fit
spectrogram = T.Spectrogram(
    n_fft=1024,
    win_length=None,
    hop_length=512,
    center=True,
    pad_mode="reflect",
    power=2.0,
)

# generate spectrogram
spec = spectrogram(waveform)

# plot spectrogram, adapted from torchaudio docs
fig, axs = plt.subplots(1, 1)
axs.set_title("{}: Spectrogram (db)".format(audio_path.stem))
axs.set_ylabel("Frequency Bin")
axs.set_xlabel("Frame")
im = axs.imshow(librosa.power_to_db(spec[0]), origin="lower", aspect="auto")
fig.colorbar(im, ax=axs)
plt.savefig(args.out_path + '/{}_spectrogram.png'.format(audio_path.stem))
