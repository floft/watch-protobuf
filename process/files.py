"""
Handle files
"""
import pathlib


def get_watch_files(dir_name, watch_number, responses=False):
    """
    For a directory FLAGS.dir, we should get all the *.pb files in any subfolder
    that has the name watchXYZ for XYZ=watch_number with leading zeros to make
    it exactly 3 characters. For example, if we have subdirectories week{1,2,3},
    then we would match:
        dir/week1/watch001/sensor_data_*.pb
        dir/week2/watch001/sensor_data_*.pb
        dir/week3/watch001/sensor_data_*.pb

    If responses=True, then we instead match responses_*.pb in each directory.
    """
    name = "watch%03d"%watch_number

    if responses:
        prefix = "responses_"
    else:
        prefix = "sensor_data_"

    files = list(pathlib.Path(dir_name).glob("**/%s/%s*"%(name, prefix)))

    # Sort so we get them in time ascending order
    files.sort()

    # Return as a string rather than a PosixPath so we can directly pass
    # to open(..., 'b')
    return [str(x) for x in files]
