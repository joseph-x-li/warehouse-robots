#! /usr/bin/python3.6

import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--runname', type=str)
args = parser.parse_args()

with open("_hold.npy", 'rb') as f:
    frames = np.load(f).astype(np.uint8) * 255

print(frames.shape)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# out, fourcc, fps, WIDTHxHEIGHT, COLOR?
writer = cv2.VideoWriter(f"recordings/{args.runname}.mp4", fourcc, 30, frames[0].shape[::-1], False)

for frame in frames:
    writer.write(frame)

writer.release()
