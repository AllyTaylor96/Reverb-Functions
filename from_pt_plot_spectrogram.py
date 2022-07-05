"""
Command line function to plot spectrogram of given saved spectrogram file.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import argparse
import librosa
from pathlib import Path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Pt file spectrogram generator.')
parser.add_argument('pt_path',
                    help='Path to .pt file to be plotted', type=str)
parser.add_argument('out_path',
                    help='Path for output graph to be saved to', type=str)
args = parser.parse_args()


spec = torch.load(args.pt_path)
pt_path = Path(args.pt_path)


# plot spectrogram, adapted from torchaudio docs
fig, axs = plt.subplots(1, 1)
axs.set_title("{}: Spectrogram (db)".format(pt_path.stem))
axs.set_ylabel("Frequency Bin")
axs.set_xlabel("Frame")
im = axs.imshow(librosa.power_to_db(spec[0]), origin="lower", aspect="auto")
fig.colorbar(im, ax=axs)
plt.savefig(args.out_path + '/{}_spectrogram.png'.format(pt_path.stem))
