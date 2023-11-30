import time
import numpy as np
import cv2
import subprocess
from flirpy.camera.boson import Boson
import syslog
import traceback

# https://github.com/LJMUAstroecology/flirpy
# これを参照


def U16to8_Normalization(RawIMG, DisableEncoding=False):
    """
    PROCESSES
    1. Converting RawIMG (16 bit) into Raw_U8 (8 bit).
       (WARNING: Convertion is not exactly inaccurate.)
    2. Rescaling to 0 - 255.
    3. Resizing to double size.
    """

    Raw_U8 = np.sqrt(RawIMG)  # Process 1
    Normal_U8 = (
        (Raw_U8 - np.min(Raw_U8)) / (np.max(Raw_U8) - np.min(Raw_U8)) * 255
    )  # Process 2
    Normal_U8 = cv2.resize(
        Normal_U8, dsize=(Normal_U8.shape[1] * 2, Normal_U8.shape[0] * 2)
    )  # Process 3

    # For Calibration
    if DisableEncoding:
        return Normal_U8

    Img2Show = cv2.imencode(".png", Normal_U8)[1].tobytes()
    return Img2Show


def boson_reset(boson_camera_id_text):
    try:
        try:
            command = f"sudo sh -c 'echo -n {boson_camera_id_text} > /sys/bus/usb/drivers/usb/unbind'"
            command_output = subprocess.run(
                [command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                encoding="utf8",
            )
        except Exception as e:
            traceback.print_exc()
            syslog.syslog(f"ERROR: {e}")
        try:
            command = f"sudo sh -c 'echo -n {boson_camera_id_text} > /sys/bus/usb/drivers/usb/bind'"
            command_output = subprocess.run(
                [command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                encoding="utf8",
            )
        except Exception as e:
            traceback.print_exc()
            syslog.syslog(f"ERROR: {e}")
        if command_output.stdout != "":
            print(command_output.stdout.rstrip())
        if command_output.stderr != "":
            print(command_output.stderr.rstrip())
    except Exception as e:
        traceback.print_exc()
        syslog.syslog(f"ERROR: {e} boson_reset error")


class thermo_camera_img:
    def __init__(self):
        self.bosonCap = Boson()

    def take_thermo_ir(self):
        return self.bosonCap.grab()

    def ir_conv_temp(self, the_ir_16bit):
        the_temp_8bit = U16to8_Normalization(the_ir_16bit, DisableEncoding=True)
        the_temp_8bit = np.trunc(the_temp_8bit)
        the_temp_8bit = the_temp_8bit.astype(np.uint8)
        return the_temp_8bit
