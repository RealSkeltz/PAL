# /Users/jscheltema/Documents/Personal/PAL/controls/resources/headset_controls.py
import subprocess
import threading


def listen_for_headset_button(callback):
    process = subprocess.Popen(
        ["log", "stream", "--predicate",
         'subsystem == "com.apple.bluetooth"',
         "--level", "debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )

    for line in process.stdout:
        if "AVRCP Play" in line or "AVRCP Pause" in line:
            print('TRIGGER SET')
            callback()


def start_headset_listener(callback):
    threading.Thread(
        target=listen_for_headset_button,
        args=(callback,),
        daemon=True
    ).start()