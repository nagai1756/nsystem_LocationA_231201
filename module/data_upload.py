# date : 2023/9/7
# writer : Akira Hoshino, Chitose Kitawaki, Sho Nagai
# upload data of device to KJ sever
import paramiko
from scp import SCPClient
import glob as gl
import datetime
import os
import subprocess
import syslog
import shutil
import requests
import traceback
import sys
import time
from module import syslog_reset
import csv

def record_log(date, system_file):
    try:
        print("record_log work")
        syslog.syslog("record_log work")
        output = subprocess.run(
            ["tegrastats | head -n 1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding="utf8",
        )
        syslog.syslog(f"INFO: {output.stdout.rstrip()}")
        syslog_output_file = f"{os.path.dirname(os.path.abspath(system_file))}/data/{date}/{date}_syslog.txt"
        syslog_output = subprocess.run(
            ["cat /var/log/syslog"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding="utf8",
        )
        with open(syslog_output_file, "w") as file:
            file.write(syslog_output.stdout.rstrip())
        answer_count_output = subprocess.run(
            [f'cat {syslog_output_file} | grep -ac "INFO: answered"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding="utf8",
        )
        answer_count = answer_count_output.stdout.rstrip()
        mask_glass_count_output = subprocess.run(
            [f'cat {syslog_output_file} | grep -ac "INFO: MASK_GLASSES answered"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding="utf8",
        )
        mask_glass_count = mask_glass_count_output.stdout.rstrip()
        thermo_skipped_count_output = subprocess.run(
            [f'cat {syslog_output_file} | grep -ac "INFO: thermo_skipped answered"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding="utf8",
        )
        thermo_skipped_count = thermo_skipped_count_output.stdout.rstrip()
        dmesg_output_file = f"{os.path.dirname(os.path.abspath(system_file))}/data/{date}/{date}_dmesg.txt"
        dmesg_output = subprocess.run(
            ["sudo dmesg -c"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding="utf8",
        )
        with open(dmesg_output_file, "w") as file:
            file.write(dmesg_output.stdout.rstrip())
        return answer_count,mask_glass_count,thermo_skipped_count
    except Exception as e:
        traceback.print_exc()
        print(e)
        syslog.syslog(f"ERROR: {e}")


def make_dir_in_date(date=None):
    if date == None:
        date = datetime.datetime.now().strftime("%y%m%d")
    try:
        os.mkdir("data")
    except:
        pass
    if not os.path.isdir(f"data/{date}"):
        os.mkdir(f"data/{date}")
        print(f"data/{date}", "directory maked")
    else:
        print(f"data/{date}", "directory exists")
    try:
        os.mkdir(f"data/{date}")
    except:
        pass
    folders = [
        "csv_data",
        "visible_images",
        "thermo_images",
        "visible_affines",
        "thermo_affines",
        "thermo_temp",
        "thermo_affine_temp",
    ]
    for name in folders:
        try:
            os.mkdir(f"data/{date}/{name}")
        except:
            pass


def upload(
    date, system_file, location, rm_date_dir=False, rm_child_dir=False, send_notify=True
):
    # === PATH ===
    line_notify_token = "ICh4u9kyRTju2ulAaynYUKDlbommMcGWjZkc5OHpEGX"  # 鹿島グループのトークン
    if send_notify == False:
        line_notify_token = "dSZkCDIGHz1ueNPxsZB4DXK2cQlS8KF3FTx6Z4U2swn"  # 自分用のトークン
    line_notify_api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    headers2 = {"Authorization": f"Bearer dSZkCDIGHz1ueNPxsZB4DXK2cQlS8KF3FTx6Z4U2swn"}
    response = requests.post(
        line_notify_api, headers=headers2, data={"message": f"{location} upload start"}
    )
    # Get local data directory path and upload server directory path
    data_dir_path = f"{os.path.dirname(os.path.abspath(system_file))}/data"
    upload_dir_path = f"/home/mediauser/device_data/{location}"
    data_backup_dir = f"{os.path.dirname(os.path.abspath(system_file))}/uploaded_data"
    print(f"data directory : {data_dir_path}")
    print(f"upload server directory : {upload_dir_path}")
    # === SFTP setup ===
    config = {
        "host": "108.166.181.202",
        "port": 22,
        "username": "mediauser",
        "password": "bandbtech@@123",
    }
    # === MAIN ===
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
        print("upload_process server connected")
        syslog.syslog("INFO: upload_process server connected")
        connected = True
    except:
        data = {"message": f"\ncouldn't connect to server."}
        requests.post(line_notify_api, headers=headers, data=data)
        connected = False
        syslog.syslog("ERROR: couldn't connect to server.")
    # === START SFTP SESSION ===
    if connected:
        try:
            sftp_con = SCPClient(client.get_transport(), socket_timeout=15.0)
            # make dir on sever
            stdin, stdout, stderr = client.exec_command(f"mkdir -p {upload_dir_path}")
            try:
                os.mkdir(f"{data_dir_path}/{date}")
            except:
                pass
            try:
                os.mkdir(data_backup_dir)
            except:
                pass
            upload_data_size = 0
            file_count = 0  # 保存ファイル数
            count = 0  # アップロード済みファイル数
            upload_error_count = 0
            upload_error_text = ""
            hash_error_count = 0
            hash_error_text = ""
            start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            os_walk = sorted(os.walk(data_dir_path))
            uploaded_os_walk = sorted(os.walk(data_backup_dir))
            syslog.syslog(f"INFO: data_directory info {os_walk}")
            syslog.syslog(f"INFO: uploaded_data_directory info {uploaded_os_walk}")
            answer_count, mask_glass_count,thermo_skipped_count = record_log(date, system_file)
            os_walk = sorted(os.walk(data_dir_path))
            uploaded_os_walk = sorted(os.walk(data_backup_dir))
            if send_notify:
                syslog_reset.syslog_reset()
            for root, _, files in os_walk:
                try:
                    os.mkdir(root.replace(data_dir_path, data_backup_dir))
                except:
                    pass
                try:
                    upload_path = root.replace(data_dir_path, upload_dir_path)
                    client.exec_command(f"mkdir -p {upload_path}")
                except:
                    pass
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    if not file.startswith("."):
                        file_count += 1
                        try:
                            upload_path = file_path.replace(
                                data_dir_path, upload_dir_path
                            )
                            sftp_con.put(
                                file_path,
                                upload_path,
                            )
                            print(f"{file} uploaded")
                            count += 1
                            try:
                                upload_data_size += os.path.getsize(file_path)
                            except:
                                print(f"{file_path} get_data_size failed")
                            try:
                                os.remove(
                                    file_path.replace(data_dir_path, data_backup_dir)
                                )
                            except:
                                pass
                            try:
                                output = subprocess.run(
                                    [f"md5sum {file_path}"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    shell=True,
                                    encoding="utf8",
                                )
                                local_file_hash = (
                                    output.stdout.rstrip()
                                    .replace(" ", "")
                                    .replace("\t", "")
                                    .replace("\n", "")
                                    .replace(file_path, "")
                                )
                                print(local_file_hash)
                                stdin, stdout, stderr = client.exec_command(
                                    f"md5sum {upload_path}"
                                )
                                output = stdout.readline()
                                server_file_hash = (
                                    output.replace(" ", "")
                                    .replace("\t", "")
                                    .replace("\n", "")
                                    .replace(upload_path, "")
                                )
                                print(server_file_hash)
                                # server_file_hash = "abcde"  # テスト用
                                if server_file_hash != local_file_hash:
                                    print(f"server_file_hash : {server_file_hash}")
                                    print(f"local_file_hash : {local_file_hash}")
                                    print("different hash.")
                                    hash_error_count += 1
                                    hash_error_text += f"{file}\n"
                            except Exception as e:
                                traceback.print_exc()
                                syslog.syslog(f"ERROR: {e} check_hash error")
                                print("check_hash error")
                            try:
                                shutil.move(
                                    file_path,
                                    root.replace(data_dir_path, data_backup_dir),
                                )
                            except Exception as e:
                                syslog.syslog(
                                    f"ERROR: {e} {file_path} couldn't move to {root.replace(data_dir_path,data_backup_dir)}"
                                )
                                print(
                                    f"ERROR: {e} {file_path} couldn't move to {root.replace(data_dir_path,data_backup_dir)}"
                                )
                        except Exception as e:
                            traceback.print_exc()
                            upload_error_count += 1
                            syslog.syslog(
                                f"ERROR: {e} {file_path} couldn't upload to {upload_path}",
                            )
                            upload_error_text += f"{file}\n"
                    else:
                        os.remove(file_path)
                        print(f"{file} removed")
                if rm_child_dir:
                    file_list = os.listdir(root)
                    if file_list == []:
                        os.rmdir(root)

            end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if rm_date_dir:
                for date_dir in os.listdir(data_dir_path):
                    try:
                        os.rmdir(f"{data_dir_path}/{date_dir}")
                    except:
                        print(f"couldn't delete folder {data_dir_path}/{date_dir}")
        except Exception as e:
            syslog.syslog(f"ERROR: {e} upload something failed")
        try:
            size = ""
            size = f"{round(upload_data_size/1024/1024, 2)} MB"
            send_text = "\n=== UL notification ===\n"
            send_text += f"{location}\n"
            send_text += f"{count}/{file_count} files ({size})\n"
            try:
                server_command = f"find {upload_dir_path} -type f"
                stdin, stdout, stderr = client.exec_command(server_command)
                server_output_list = stdout.readlines()
                server_count = 0
                for file in server_output_list:
                    server_count += 1
                send_text += f"total: {server_count} files "
            except Exception as e:
                send_text += f"total: ?? files (ERROR: {e})"
                syslog.syslog(f"ERROR: {e} server_count failed")
            try:
                server_command = (
                    f"find {upload_dir_path} -maxdepth 0 -type d -exec du -chm "
                    + "{}"
                    + " \; | grep total"
                )
                stdin, stdout, stderr = client.exec_command(server_command)
                server_output_list = stdout.readlines()
                server_volume = server_output_list[0].replace("\ttotal\n", "")
                send_text += f"({server_volume} MB)\n"
            except Exception as e:
                send_text += f"(?? MB) ERROR: {e}\n"
                syslog.syslog(f"ERROR: {e} check server_volume failed")
            send_text += f"answers: {answer_count}\n"
            send_text += f"mask_glasses: {mask_glass_count}\n"
            send_text += f"thermo_skipped: {thermo_skipped_count}\n"
            if not upload_error_text == "":
                send_text += f"\n=== UL FAILED ===\n"
                send_text += f"{upload_error_count} files\n"
                send_text += upload_error_text
            if not hash_error_text == "":
                send_text += f"\n=== HASH ERROR ===\n"
                send_text += f"{hash_error_count} files\n"
                send_text += hash_error_text
            send_text = send_text.rstrip("\n")
            data = {"message": send_text}
            print(send_text)
            try:
                response = requests.post(line_notify_api, headers=headers, data=data)
                if response.status_code != 200:
                    syslog.syslog(
                        f"ERROR: data_upload notification couldn't send: {send_text}"
                    )
                else:
                    syslog.syslog(f"INFO: data_upload notification sent: {send_text}")
            except Exception as e:
                syslog.syslog(f"ERROR: {e} data_upload notification something failed")
            print("upload_process finished")
        except Exception as e:
            syslog.syslog(f"ERROR: {e} send_text something failed")
        client.close()


if __name__ == "__main__":
    if sys.argv[1] == "upload":
        date = datetime.datetime.now().strftime("%y%m%d")
        upload(date, "/home/awazon/nsystem/main7.py", "LocationC", True, True, False)
    elif sys.argv[1] == "upload_bashrc":
        date = datetime.datetime.now().strftime("%y%m%d")
        location = sys.argv[2]
        print("upload", location)
        while True:
            time.sleep(1)
            now = datetime.datetime.now()
            if now.strftime("%y%m%d") != date:
                upload(
                    date, "/home/awazon/nsystem/main7.py", location, True, True, False
                )
                date = now.strftime("%y%m%d")
