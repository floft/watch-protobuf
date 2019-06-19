#!/usr/bin/env python3
"""
Plot spectrograms of the data

See: https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.specgram.html
"""
import numpy as np
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from decoding import decode
from watch_data_pb2 import SensorData

FLAGS = flags.FLAGS

flags.DEFINE_string("input", None, "Input protobuf file")
flags.DEFINE_boolean("save", False, "If not animating, save the figures to files")
flags.DEFINE_boolean("sort", True, "Sort protobuf messages")
flags.DEFINE_float("freq", 50.0, "Sampling frequency in Hz of accelerometers, etc.")
flags.DEFINE_enum("animate", "none", ["none", "raw_accel", "user_accel", "grav", "rot_rate", "attitude"], "Animate spectrogram rather than plot, if any other than \"none\"")
flags.DEFINE_integer("nfft", 128, "NFFT for spectrogram, samples per FFT block")
flags.DEFINE_integer("noverlap", 118, "noverlap for spectrogram, overlap between subsequent windows for FFT")
flags.DEFINE_integer("animate_start", 0, "Sample to start animation at")
flags.DEFINE_integer("animate_length", 120*50, "Length in samples of window for each animation time step")
flags.DEFINE_integer("animate_update", 60*50, "Shift in samples from one animation time step to the next")
flags.DEFINE_boolean("psd", True, "Plot the power spectral density spectrogram")
flags.DEFINE_boolean("magnitude", False, "Plot the magnitude spectrogram")
flags.DEFINE_boolean("angle", False, "Plot the angle spectrogram")
flags.DEFINE_boolean("phase", False, "Plot the phase spectrogram")

flags.mark_flag_as_required("input")


def hide_border(ax):
    """ Hide the four borders of a plot """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def get_modes():
    """ Get which modes we want to plot from the command-line arguments """
    modes = []

    if FLAGS.psd:
        modes.append("psd")
    if FLAGS.magnitude:
        modes.append("magnitude")
    if FLAGS.angle:
        modes.append("angle")
    if FLAGS.phase:
        modes.append("phase")

    return modes


def _fft_create(title, names, modes):
    """ Create the FFT figure """
    fig, axes = plt.subplots(nrows=len(modes)+1, ncols=len(names),
        sharex=True, figsize=(15, 2*(len(modes)+1)))
    plt.suptitle(title)
    fig.canvas.set_window_title(title)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9,
        wspace=0.2, hspace=0.1)

    return fig, axes


def _fft_plot(data, units, names, modes, axes, fig, t):
    """ Fill in the FFT figure with the data and spectrograms """
    plot_list = []
    additional_axes = []

    # For some reason sometimes t ends up being one more sample than data
    t = t[:len(data)]

    for i in range(len(names)):
        y = np.array([v[i] for v in data])
        plot_list.append(axes[0][i].plot(t, y))
        axes[0][i].title.set_text(names[i])
        axes[0][i].margins(x=0)

        # Only show y-label on left, otherwise color bar overlaps
        if i == 0:
            axes[0][i].set_ylabel(units)

        # Align with plots below but make additional axes blank
        divider = make_axes_locatable(axes[0][i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cax.grid(False)
        cax.set_xticks([])
        cax.set_yticks([])
        hide_border(cax)
        additional_axes.append(cax)

        for j, mode in enumerate(modes):
            Pxx, freqs, bins, im = axes[j+1][i].specgram(y, NFFT=FLAGS.nfft,
                Fs=FLAGS.freq, noverlap=FLAGS.noverlap, mode=mode,
                xextent=(t[0], t[-1]))
            plot_list.append(im)
            hide_border(axes[j+1][i])
            axes[j+1][i].margins(x=0)

            # Only bottom one
            if j == len(modes)-1:
                axes[j+1][i].set_xlabel('seconds')

            # Only left one
            if i == 0:
                axes[j+1][i].set_ylabel(mode)

            # See: https://stackoverflow.com/a/49037495
            divider = make_axes_locatable(axes[j+1][i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            additional_axes.append(cax)

    return plot_list, additional_axes


def plot_fft(data, title, units, names=["x", "y", "z"]):
    """ Plot single FFT, by default an FFT for each of x/y/z """
    t = np.arange(0.0, len(data)/FLAGS.freq, 1/FLAGS.freq)

    modes = get_modes()
    fig, axes = _fft_create(title, names, modes)
    _fft_plot(data, units, names, modes, axes, fig, t)

    if FLAGS.save:
        plt.savefig("Plots - "+title+".png", dpi=100,
            bbox_inches="tight", pad_inches=0)


def animate_fft(data, title, units, names=["x", "y", "z"]):
    """ Animate the spectrogram, based on Steve's animate code """
    start = FLAGS.animate_start
    length = FLAGS.animate_length
    update = FLAGS.animate_update
    num_samples = len(data)
    modes = get_modes()
    fig, axes = _fft_create(title, names, modes)

    # Otherwise the colorbars aren't ever cleared
    additional_axes = []

    def animate_init():
        pass

    def animate_update(frame):
        nonlocal additional_axes

        # 2D since we have both rows and columns
        for ax1 in axes:
            for ax2 in ax1:
                ax2.cla()
        for ax in additional_axes:
            ax.remove()

        ani_start = (start + frame * update) % num_samples
        ani_stop = min(ani_start + length, num_samples)

        t = np.arange(ani_start/FLAGS.freq, ani_stop/FLAGS.freq, 1/FLAGS.freq)
        y = data[ani_start:ani_stop]

        plot_frames, new_additional_axes = _fft_plot(y, units, names, modes,
            axes, fig, t)
        additional_axes = new_additional_axes

        return plot_frames

    ani = FuncAnimation(fig, animate_update, init_func=animate_init,
        frames=range(int(num_samples / update)), interval=10)

    return ani


def plot_data(messages, max_len=None):
    """ Sort messages on timestamp, plot max_len samples for FFTs """
    # Sort since when saving to a file on the watch, they may be out of order
    if FLAGS.sort:
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

    if FLAGS.animate != "none":
        # If we don't keep the returned value, it won't animate
        if FLAGS.animate == "raw_accel":
            ani = animate_fft(raw_accel, "Raw Acceleration", "g's")
        elif FLAGS.animate == "user_accel":
            ani = animate_fft(user_accel, "User Acceleration", "g's")
        elif FLAGS.animate == "grav":
            ani = animate_fft(grav, "Gravity", "g's")
        elif FLAGS.animate == "rot_rate":
            ani = animate_fft(rot_rate, "Rotation Rates", "rad/s")
        elif FLAGS.animate == "attitude":
            ani = animate_fft(attitude, "Attitude", "rad",
                names=["roll", "pitch", "yaw"])

        plt.show()
    else:
        plot_fft(raw_accel, "Raw Acceleration", "g's")
        plot_fft(user_accel, "User Acceleration", "g's")
        plot_fft(grav, "Gravity", "g's")
        plot_fft(rot_rate, "Rotation Rates", "rad/s")
        plot_fft(attitude, "Attitude", "rad", names=["roll", "pitch", "yaw"])

        plt.show()


def main(argv):
    plot_data(decode(FLAGS.input, SensorData))


if __name__ == "__main__":
    app.run(main)
