"""
Command line function to plot waveform of given audio file.
"""

import torch
import torchaudio
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Audio file waveform generator.')
parser.add_argument('audio_path',
                    help='Path to audio file to be plotted', type=str)
parser.add_argument('out_path',
                    help='Path for output waveform to be saved to', type=str)
args = parser.parse_args()


waveform, sr = torchaudio.load(args.audio_path)
audio_path = Path(args.audio_path)

# below adapted from torchaudio documentation
waveform = waveform.numpy()
num_channels, num_frames = waveform.shape
time_axis = torch.arange(0, num_frames) / sr
figure, axes = plt.subplots(num_channels, 1)
axes.plot(time_axis, waveform[0], linewidth=1)
axes.grid(True)
figure.suptitle(audio_path.stem)
plt.savefig(args.out_path + '/{}_waveform.png'.format(audio_path.stem))
