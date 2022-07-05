"""
Command line program to inspect a given audio file.
"""

import torchaudio
import os
import argparse

# set up parser
parser = argparse.ArgumentParser(description='Audio file inspector.')
parser.add_argument('audio_path',
                    help='Path to audio file to be inspected', type=str)
args = parser.parse_args()

# inspect file and print to terminal
print('-' * 20)
print('Source: {}'.format(args.audio_path))
print('-' * 20)
print('File size: {} bytes, {} MB'.format(os.path.getsize(args.audio_path),
                                          float(os.path.getsize(args.audio_path)/1000000)))
print('{}'.format(torchaudio.info(args.audio_path)))


