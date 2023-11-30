import os
import syslog
import traceback
import paramiko
from scp import SCPClient
import subprocess
import sys
import datetime
import time


def delete(system_file, location):
    uploaded_dir_path = f"/home/awazon/nsystem/uploaded_data"
    print(f"uploaded_directory path : {uploaded_dir_path}")
    config = {
        "host": "108.166.181.202",
        "port": 22,
        "username": "mediauser",
        "password": "bandbtech@@123",
    }
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        client.connect(
            config["host"],
            port=config["port"],
            username=config["username"],
            password=config["password"],
            timeout=300,
        )
        print("delete_process server connected")
        syslog.syslog("INFO: delete_process server connected")
        connected = True
    except:
        syslog.syslog("ERROR: couldn't connect to server")
        connected = False
    if connected:
        try:
            server_dir = f"/home/mediauser/device_data"
            command = (
                f'find {server_dir} -daystart -mtime +7 -path "*{location}/*" -exec rm -rfv '
                + "{} \;"
            )
            syslog.syslog(f"INFO: try '{command}'")
            stdin, stdout, stderr = client.exec_command(command)
            output_stdout = stdout.readlines()
            output_stderr = stderr.readlines()
            if output_stdout != []:
                print(f"INFO: {output_stdout}")
                syslog.syslog(f"INFO: {output_stdout}")
            if output_stderr != []:
                print(f"INFO: {output_stderr}")
                syslog.syslog(f"INFO: {output_stderr}")
        except Exception as e:
            traceback.print_exc()
            syslog.syslog(f"ERROR: {e} server data_delete failed")
    try:
        command = (
            f"find {uploaded_dir_path} -daystart -mindepth 1 -mtime +7 -exec rm -rfv"
            + " {} \;"
        )
        result = subprocess.run(
            [command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding="utf8",
        )
        output_stdout = result.stdout.rstrip()
        output_stderr = result.stderr.rstrip()
        if output_stdout != "":
            print(output_stdout)
            syslog.syslog(f"INFO: delete_stdout {output_stdout}")
        if output_stderr != "":
            print(output_stderr)
            syslog.syslog(f"ERROR: delete_stderr {output_stderr}")
        for date_dir in sorted(os.listdir(uploaded_dir_path)):
            date_dir_path = os.path.join(uploaded_dir_path, date_dir)
            for sub_dir in sorted(os.listdir(date_dir_path)):
                sub_dir_path = os.path.join(date_dir_path, sub_dir)
                if os.path.isdir(sub_dir_path) and not os.listdir(sub_dir_path):
                    print(sub_dir_path)
                    os.rmdir(sub_dir_path)
            if not os.listdir(date_dir_path):
                os.rmdir(date_dir_path)
    except Exception as e:
        traceback.print_exc()
        syslog.syslog(f"ERROR: {e} jetson data_delete failed")
    print("delete_process finished")


if __name__ == "__main__":
    if sys.argv[1] == "delete":
        delete("/home/awazon/nsystem/main5.py", "")
    elif sys.argv[1] == "delete_bashrc":
        date = datetime.datetime.now().strftime("%y%m%d")
        location = sys.argv[2]
        print("delete", location)
        while True:
            time.sleep(1)
            now = datetime.datetime.now()
            if now.strftime("%y%m%d") != date:
                delete("/home/awazon/nsystem/main7.py", location)
                date = now.strftime("%y%m%d")
