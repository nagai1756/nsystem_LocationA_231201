# -*- coding: utf-8 -*-
import subprocess
import sys
import syslog


def syslog_reset():
    try:
        subprocess.run(["sudo chmod 777 /var/log/syslog"], shell=True)
    except Exception as e:
        syslog.syslog(f"ERROR: {e} chmod /var/log/syslog failed")
    try:
        with open("/var/log/syslog", "wb"):
            print("syslog cleared")
        syslog.syslog("INFO: syslog cleared")
    except Exception as e:
        print("Failed to clear", e)
        syslog.syslog("ERROR: syslog reset failed")


try:
    if sys.argv[1] == "reset":
        syslog_reset()
except:
    pass
