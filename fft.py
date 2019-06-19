#!/usr/bin/env python3
"""
Plot spectrograms of the data

See: https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.specgram.html
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from decoding import decode
from watch_data_pb2 import SensorData


def hide_border(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_fft(data, title, units, freq=50.0, plot=True, names=["x", "y", "z"],
        NFFT=30*50, noverlap=(30-1)*50, modes=["psd", "magnitude", "angle", "phase"]):
    """ Plot single FFT, by default an FFT for each of x/y/z """
    t = np.arange(0.0, len(data)*1.0/freq, 1.0/freq)

    fig, ax = plt.subplots(nrows=len(modes)+1, ncols=len(names),
        sharex=True, figsize=(15, 3*(len(modes)+1)))
    plt.suptitle(title)
    fig.canvas.set_window_title(title)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
        wspace=0.2, hspace=0.1)

    for i in range(len(names)):
        x = np.array([v[i] for v in data])
        ax[0][i].plot(t, x)
        ax[0][i].title.set_text(names[i])
        ax[0][i].margins(x=0)

        # Only show y-label on left, otherwise color bar overlaps
        if i == 0:
            ax[0][i].set_ylabel(units)

        # Align with plots below but make additional axes blank
        divider = make_axes_locatable(ax[0][i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cax.grid(False)
        cax.set_xticks([])
        cax.set_yticks([])
        hide_border(cax)

        for j, mode in enumerate(modes):
            Pxx, freqs, bins, im = ax[j+1][i].specgram(x, NFFT=NFFT, Fs=freq,
                noverlap=noverlap, mode=mode)
            hide_border(ax[j+1][i])
            ax[j+1][i].margins(x=0)

            # Only bottom one
            if j == len(modes)-1:
                ax[j+1][i].set_xlabel('seconds')

            # Only left one
            if i == 0:
                ax[j+1][i].set_ylabel(mode)

            # See: https://stackoverflow.com/a/49037495
            divider = make_axes_locatable(ax[j+1][i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

    if plot:
        plt.savefig("Plots - "+title+".png", dpi=100,
            bbox_inches="tight", pad_inches=0)


def plot_data(messages, max_len=None, sort=False):
    """ Sort messages on timestamp, plot max_len samples for FFTs, freq in Hz """
    # Sort since when saving to a file on the watch, they may be out of order
    if sort:
        messages.sort(key=lambda x: x.epoch)

    # Get max_len of data
    raw_accel = []
    accel_i = 0
    user_accel = []
    grav = []
    attitude = []
    rot_rate = []
    motion_i = 0

    for msg in messages:
        if msg.message_type == SensorData.MESSAGE_TYPE_ACCELEROMETER:
            if max_len is None or accel_i <= max_len:
                raw_accel.append((msg.raw_accel_x, msg.raw_accel_y, msg.raw_accel_z))
                accel_i += 1
        elif msg.message_type == SensorData.MESSAGE_TYPE_DEVICE_MOTION:
            if max_len is None or motion_i <= max_len:
                user_accel.append((msg.user_accel_x, msg.user_accel_y, msg.user_accel_z))
                grav.append((msg.grav_x, msg.grav_y, msg.grav_z))
                rot_rate.append((msg.rot_rate_x, msg.rot_rate_y, msg.rot_rate_z))
                attitude.append((msg.roll, msg.pitch, msg.yaw))
                motion_i += 1

        if max_len is not None and accel_i > max_len and motion_i > max_len:
            break

    plot_fft(raw_accel, "Raw Acceleration", "g's")
    plot_fft(user_accel, "User Acceleration", "g's")
    plot_fft(grav, "Gravity", "g's")
    plot_fft(rot_rate, "Rotation Rates", "rad/s")
    plot_fft(attitude, "Attitude", "rad", names=["roll", "pitch", "yaw"])

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./fft.py input.pb")
        exit(1)

    input_fn = sys.argv[1]

    if not os.path.exists(input_fn):
        print("Error: input file does not exist:", input_fn)
        exit(1)

    plot_data(decode(input_fn, SensorData))
