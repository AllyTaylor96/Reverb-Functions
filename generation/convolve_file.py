"""
Command line program to convolve an individual audio file with a
selected impulse response.
"""

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import argparse

# set up parser
parser = argparse.ArgumentParser(description='Audio file convolver.')
parser.add_argument('audio_path', 
                    help='Path to audio file that will be convolved', type=str)
parser.add_argument('ir_path', 
                    help='Path to IR that will be convolved', type=str)
parser.add_argument('out_filepath', 
                    help='Filepath for file to be saved to', type=str)
parser.add_argument('desired_sr', 
                    help='Desired sample rate for out file', type=int, 
                    default=22000)
args = parser.parse_args()
desired_sr = args.desired_sr

# load in given files and resample to desired sample rate
audio_wave, audio_sr = torchaudio.load(args.audio_path)
audio_wave = AF.resample(audio_wave, audio_sr, desired_sr)
ir_wave, ir_sr = torchaudio.load(args.ir_path)
ir_wave = AF.resample(ir_wave, ir_sr, desired_sr)

"""
Process the files appropriately - impulse response is made mono, normalized
and flipped while audio file is padded. They are then convolved and saved.
"""

mono_ir = torch.mean(ir_wave, dim=0).unsqueeze(0)
norm_ir = mono_ir / torch.norm(mono_ir, p=2)
ir_wave = torch.flip(norm_ir, [1])

audio_wave_pad = F.pad(audio_wave, (ir_wave.shape[1] - 1, 0))
audio_wave = F.pad(audio_wave_pad, (0, ir_wave.shape[1] - 1))

conv_wave = F.conv1d(audio_wave[None, ...], ir_wave[None, ...])[0]

torchaudio.save(args.out_filepath, conv_wave, desired_sr)
