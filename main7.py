import csv
import os
import datetime
import subprocess
import time
import cv2
import numpy as np
from module import (
    fs_boson_camera_lib,
    fs_face_detection_trt_mix4380,
    fs_affine_convert_traiangle,
    fs_bbox_landmark,
    data_upload,
    data_delete,
)
from KJ_temp_estimate import simple_estimate_coretemp
import datetime
import pyautogui
import multiprocessing as mp
import smbus
import syslog
import simpleaudio
import traceback
import sys
import threading as td
import Jetson.GPIO as GPIO
import requests
import copy


class global_parameters:
    def __init__(self) -> None:
        self.glass_or_mask = False
        self.phase = 1
        self.top_gra_px = 250
        self.down_gra_px = 250
        self.top_gra_phase1 = None
        self.top_gra_phase2 = None
        self.top_gra_phase3 = None
        self.ansBtnSize = (250, 250)
        self.img_vis_cam = None
        self.camW = 0
        self.camH = 0
        self.img_vis_base = None
        self.img_vis_baseW = 0
        self.img_vis_baseH = 0
        self.img_vis_resized = None
        self.img_vis_resizedH = 0
        self.img_vis_resizedW = 0
        self.img_the_baseW = 0
        self.img_the_baseH = 0
        self.faceDetected = False
        self.mask_glass_detected = False
        self.keyboardClicked = False
        self.detectTimeSta = None
        self.vis_bbox = [[]]
        self.vis_ltX = 0
        self.vis_ltY = 0
        self.vis_rbX = 0
        self.vis_rbY = 0
        self.the_bbox = [[]]
        self.the_ltX = 0
        self.the_ltY = 0
        self.the_rbX = 0
        self.the_rbY = 0
        self.vis_landmarks = []
        self.the_landmarks = []
        self.ansTopPad = 500
        self.ans1 = None
        self.ans2 = None
        self.ans3 = None
        self.ans4 = None
        self.text_phase1_1 = None
        self.text_phase2_1 = None
        self.text_phase2_2 = None
        self.text_phase3_1 = None
        self.text_phase3_2 = None
        self.faceOutTimeSta = None
        self.stable = True
        self.stable_started = False
        self.stable_start_time = None
        self.visibles = None
        self.visible_affines = None
        self.thermos = None
        self.thermo_affines = None
        self.thermo_data = {"thermo_image": np.full((10, 10, 3), [0, 0, 0])}
        self.coretemp = None
        self.phase1 = np.full((dispH - self.down_gra_px, dispW, 3), [0, 0, 0], np.uint8)
        self.phase2 = np.full((dispH - self.down_gra_px, dispW, 3), [0, 0, 0], np.uint8)
        self.phase3 = np.full((dispH - self.down_gra_px, dispW, 3), [0, 0, 0], np.uint8)
        self.background_image = np.full((dispH, dispW, 3), [0, 0, 0], np.uint8)
        self.numberImages = []
        for i in range(60):
            text = str(i).zfill(2)
            image = cv2.imread(f"./images/numberImages/{text}.png", -1)
            h, w = image.shape[:2]
            image = cv2.resize(image, (int(w * 75 / h), 75))
            self.numberImages.append(image)
        self.yearImage = []
        for i in range(2023, 2031):
            image = cv2.imread(f"./images/numberImages/{i}.png", -1)
            h, w = image.shape[:2]
            image = cv2.resize(image, (int(w * 75 / h), 75))
            self.yearImage.append(image)
        haihun = cv2.imread("./images/-.png", -1)
        haihun_H, haihun_W = haihun.shape[:2]
        haihun = cv2.resize(haihun, (25, int(haihun_H * 25 / haihun_W)))
        self.numberImages.append(haihun)
        koron = cv2.imread("./images/コロン.png", -1)
        koron_H, koron_W = koron.shape[:2]
        koron = cv2.resize(koron, (int(koron_W * 50 / koron_H), 50))
        self.numberImages.append(koron)
        self.body_coretemp_images = {}
        body_coretemp_images_list = [
            file
            for file in sorted(os.listdir("./images/body_temp/"))
            if not file.startswith(".")
        ]
        for file in body_coretemp_images_list:
            path = f"./images/body_temp/{file}"
            body_temp_image = cv2.imread(path)
            h, w = body_temp_image.shape[:2]
            body_temp_image = cv2.resize(body_temp_image, (dispW, h * dispW // w))
            body_temp = file.replace(".png", "")
            self.body_coretemp_images[f"{body_temp}"] = body_temp_image
        mask_glasses_image = cv2.imread("./images/マスク メガネ を外して.png")
        h, w = mask_glasses_image.shape[:2]
        self.mask_glasses_image = cv2.resize(
            mask_glasses_image, (dispW, h * dispW // w)
        )
        face_far_image = cv2.imread("./images/かおを 近づけて ください.png")
        h, w = face_far_image.shape[:2]
        self.face_far_image = cv2.resize(face_far_image, (dispW, h * dispW // w))
        self.face_far = False
        self.light_off = False
        self.visCap_ID = 0
        self.boson_ID = 1
        self.boson_ERROR = ""
        boson_notConnected_image = cv2.imread("./images/Boson_notConnected_image.png")
        h, w = boson_notConnected_image.shape[:2]
        self.boson_notConnected_image = cv2.resize(
            boson_notConnected_image, (900, h * 900 // w)
        )
        boson_error_something_image = cv2.imread(
            "./images/Boson_error_something_image.png"
        )
        h, w = boson_error_something_image.shape[:2]
        self.boson_error_something_image = cv2.resize(
            boson_error_something_image, (900, h * 900 // w)
        )
        self.monolith_temp = None
        self.monolith_correct_temp = 40

    def global_parameters_reset(self):
        self.glass_or_mask = False
        self.phase = 1
        self.faceDetected = False
        self.keyboardClicked = False
        self.detectTimeSta = None
        self.faceIn = False
        self.faceOut = False
        self.endTime = None
        self.picCount = 0
        self.passedFlashOnTime = 0
        self.passedFlashOffTime = 0
        self.visibles = []
        self.visible_affines = []
        self.thermos = []
        self.thermo_affines = []
        self.vis_landmarks = []
        self.the_landmarks = []
        self.thermo_data = {
            "thermo_image": np.full((256, 320, 3), [0, 0, 0]),
        }
        self.the_bbox = [[]]
        self.vis_bbox = [[]]
        self.coretemp = None
        self.monolith_temp = None
        self.stable = True
        self.stable_started = False


class answer_data:
    def __init__(
        self,
        dispW,
        ansBtnSize,
        pad,
    ):
        self.answer_time = None
        self.answer = ""
        self.dispW = dispW
        self.ansBtnSize = ansBtnSize
        self.pad = pad
        self.ansBtnClicked = False

    # クリックした座標を取得し、ボタンに触れていたら回答データを更新する処理
    def click_pos(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            offsetY = g_paras.top_gra_px + self.pad
            judgeY = y >= offsetY and y <= offsetY + self.ansBtnSize[1]
            if judgeY:
                if (
                    x >= self.dispW // 8 - self.ansBtnSize[0] // 2
                    and x <= self.dispW // 8 + self.ansBtnSize[0] // 2
                ):
                    self.answered = True
                    self.answer = 1
                    self.ansBtnClicked = True
                elif (
                    x >= self.dispW // 8 * 3 - self.ansBtnSize[0] // 2
                    and x <= self.dispW // 8 * 3 + self.ansBtnSize[0] // 2
                ):
                    self.answered = True
                    self.answer = 2
                    self.ansBtnClicked = True
                elif (
                    x >= self.dispW // 8 * 5 - self.ansBtnSize[0] // 2
                    and x <= self.dispW // 8 * 5 + self.ansBtnSize[0] // 2
                ):
                    self.answered = True
                    self.answer = 3
                    self.ansBtnClicked = True
                elif (
                    x >= self.dispW // 8 * 7 - self.ansBtnSize[0] // 2
                    and x <= self.dispW // 8 * 7 + self.ansBtnSize[0] // 2
                ):
                    self.answered = True
                    self.answer = 4
                    self.ansBtnClicked = True
            if self.ansBtnClicked:
                self.answer_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
                print(self.answer)

    def answer_data_reset(self):
        self.ansBtnClicked = False
        self.answer_time = None
        self.answer = ""


# システムの状態を初期化する処理
def reset():
    g_paras.global_parameters_reset()
    answer_data_class.answer_data_reset()


def save_contents_csv(
    date,
    answer_time,
    answer,
    temp,
    humi,
    vis_bbox,
    vis_landmark,
    the_bbox,
    the_landmark,
    coretemp,
    monolith_temp,
):
    try:
        filename = f"./data/{date}/csv_data/{answer_time}.csv"
        with open(filename, "w") as file:
            writer = csv.writer(file)
            data = [
                ["answer_time", answer_time],
                ["answer", answer],
                ["environment_temprature", temp],
                ["environment_humidity", humi],
                ["visible_bboxpoint", vis_bbox],
                ["visible_landmark", vis_landmark],
                ["thermo_bboxpoint", the_bbox],
                ["thermo_landmark", the_landmark],
                ["coretemp", coretemp],
                ["monolith_temp", monolith_temp],
            ]
            writer.writerows(data)
    except Exception as e:
        traceback.print_exc()
        syslog.syslog(f"ERROR: {e} csv_save failed")


# バウンディングボックスを描画する関数
def draw_face_frame(img, color, ltX, ltY, rbX, rbY):
    thickness, length = 20, 70
    img[ltY : ltY + length, ltX : ltX + thickness] = color
    img[ltY : ltY + thickness, ltX : ltX + length] = color
    img[ltY : ltY + length, rbX - thickness : rbX] = color
    img[ltY : ltY + thickness, rbX - length : rbX] = color
    img[rbY - length : rbY, rbX - thickness : rbX] = color
    img[rbY - thickness : rbY, rbX - length : rbX] = color
    img[rbY - length : rbY, ltX : ltX + thickness] = color
    img[rbY - thickness : rbY, ltX : ltX + length] = color
    return img


def draw_the_face_frame(img, ltX, ltY, rbX, rbY):
    img[ltY:rbY, ltX : ltX + 5] = 255
    img[ltY : ltY + 5, ltX:rbX] = 255
    img[ltY:rbY, rbX - 5 : rbX] = 255
    img[rbY - 5 : rbY, ltX:rbX] = 255
    return img


# 顔が枠に収まっているか判定する関数
def face_in_fit(ltX, ltY, rbX, rbY):
    if rbX - ltX >= g_paras.img_vis_resizedW * 0.45:
        return True
    return False


# 選択肢
def draw_ans(img):
    dy = g_paras.ansBtnSize[1] // 2 + g_paras.ansTopPad
    dx = g_paras.img_vis_resizedW // 8
    con = g_paras.ans1[:, :, 3] > 0
    img[
        dy - g_paras.ansBtnSize[1] // 2 : dy + g_paras.ansBtnSize[1] // 2,
        dx - g_paras.ansBtnSize[0] // 2 : dx + g_paras.ansBtnSize[0] // 2,
    ][con] = g_paras.ans1[:, :, :3][con]
    dx = g_paras.img_vis_resizedW // 8 * 3
    con = g_paras.ans2[:, :, 3] > 10
    img[
        dy - g_paras.ansBtnSize[1] // 2 : dy + g_paras.ansBtnSize[1] // 2,
        dx - g_paras.ansBtnSize[0] // 2 : dx + g_paras.ansBtnSize[0] // 2,
    ][con] = g_paras.ans2[:, :, :3][con]
    dx = g_paras.img_vis_resizedW // 8 * 5
    con = g_paras.ans3[:, :, 3] > 10
    img[
        dy - g_paras.ansBtnSize[1] // 2 : dy + g_paras.ansBtnSize[1] // 2,
        dx - g_paras.ansBtnSize[0] // 2 : dx + g_paras.ansBtnSize[0] // 2,
    ][con] = g_paras.ans3[:, :, :3][con]
    dx = g_paras.img_vis_resizedW // 8 * 7
    con = g_paras.ans4[:, :, 3] > 10
    img[
        dy - g_paras.ansBtnSize[1] // 2 : dy + g_paras.ansBtnSize[1] // 2,
        dx - g_paras.ansBtnSize[0] // 2 : dx + g_paras.ansBtnSize[0] // 2,
    ][con] = g_paras.ans4[:, :, :3][con]
    return img


# リストに画像を追加する関数
def add_images():
    try:
        # 可視画像処理
        g_paras.visibles = {
            "src": g_paras.img_vis_cam,
            "type": "visible_image",
            "vis_BBoxPoints": [
                (g_paras.vis_ltX, g_paras.vis_ltY),
                (g_paras.vis_rbX, g_paras.vis_rbY),
            ],
        }
    except Exception as e:
        syslog.syslog(f"ERROR: {e} get_visible_data failed")
        print("get_visible_data failed", e)
    # 熱画像処理
    try:
        g_paras.thermos = {
            "src": g_paras.thermo_data["thermo_image"],
            "temp_data": g_paras.thermo_data["temp_data"],
            "type": "thermo_image",
            "the_BBoxPoints": [
                g_paras.the_ltX,
                g_paras.the_ltY,
                g_paras.the_rbX,
                g_paras.the_rbY,
            ],
        }
    except Exception as e:
        traceback.print_exc()
        syslog.syslog(f"ERROR: {e} get_thermo_data failed")
        print("get_thermo_data failed", e)
    try:
        print(
            g_paras.the_ltX,
            g_paras.the_ltY,
            g_paras.the_rbX,
            g_paras.the_rbY,
        )
        the_landmark = landmark_class.the_landmarks(
            g_paras.thermo_data["temp_affine"],
            [
                g_paras.the_ltX,
                g_paras.the_ltY,
                g_paras.the_rbX,
                g_paras.the_rbY,
            ],
            binary_on=True,
        )
        g_paras.the_landmarks = the_landmark.astype(int).tolist()
    except Exception as e:
        traceback.print_exc()
        syslog.syslog(f"ERROR: {e} get_thermo_landmark failed")
        print("get_thermo_landmark failed", e)
    try:
        thermo_affine_temp = affine_class.main(
            g_paras.thermo_data["temp_data"],
            the_landmark,
            binary_on=True,
        )
        thermal_min = np.nanmin(thermo_affine_temp)
        thermal_max = np.nanmax(thermo_affine_temp)
        thermo_affine = thermo_affine_temp - thermal_min
        thermo_affine = thermo_affine * (255 / (thermal_max - thermal_min))
        thermo_affine = thermo_affine.astype(np.uint8)
        g_paras.thermo_affines = {
            "src": thermo_affine,
            "temp": thermo_affine_temp,
            "type": "thermo_affine",
        }
    except Exception as e:
        traceback.print_exc()
        syslog.syslog(f"ERROR: {e} thermo_affine failed")
        print("thermo_affine failed", e)
    try:
        g_paras.coretemp = estimate_coretemp_class.estimate(
            g_paras.thermo_affines["temp"]
        )
        if (
            g_paras.the_ltX
            == g_paras.the_ltY
            == g_paras.the_rbX
            == g_paras.the_rbY
            == 0
        ):
            g_paras.coretemp = None
        print(g_paras.coretemp)
        syslog.syslog(f"INFO: core_temprature: {g_paras.coretemp} ")
    except Exception as e:
        traceback.print_exc()
        syslog.syslog(f"ERROR: {e} estimate_coretemp failed")
        print(f"estimate_coretemp failed {e}")


# ファイルに画像を出力する関数
def save_images(date, answer_time, visibles, thermos, visible_affines, thermo_affines):
    answer_time = answer_data_class.answer_time
    try:
        type = visibles["type"]
        filename = f"data/{date}/visible_images/{answer_time}_{type}.png"
        cv2.imwrite(filename, visibles["src"])
    except Exception as e:
        traceback.print_exc()
        syslog.syslog(f"ERROR: {e} visible_image couldn't save")
        print(f"{answer_time} visible_image couldn't save")
    try:
        type = visible_affines["type"]
        filename = f"data/{date}/visible_affines/{answer_time}_{type}.png"
        cv2.imwrite(filename, visible_affines["src"])
    except Exception as e:
        traceback.print_exc()
        syslog.syslog(f"ERROR: {e} visible_affine couldn't save")
        print(f"{answer_time} visible_affine couldn't save")
    # 熱画像保存
    try:
        type = thermos["type"]
        filename = f"data/{date}/thermo_images/{answer_time}_{type}"
        cv2.imwrite(f"{filename}.png", thermos["src"])
        np.save(
            f"data/{date}/thermo_temp/{answer_time}_thermo_temp.npy",
            thermos["temp_data"],
        )
    except Exception as e:
        syslog.syslog(f"ERROR: {answer_time} {e} thermo_image couldn't save")
        print(f"{answer_time} thermo_image couldn't save")
    try:
        type = thermo_affines["type"]
        filename = f"data/{date}/thermo_affines/{answer_time}_{type}"
        cv2.imwrite(
            f"{filename}.png",
            thermo_affines["src"],
        )
        np.save(
            f"data/{date}/thermo_affine_temp/{answer_time}_thermo_affine_temp.npy",
            thermo_affines["temp"],
        )
    except Exception as e:
        syslog.syslog(f"ERROR: {answer_time}{e} thermo_affine couldn't save")
        print(f"{answer_time} thermo_affine couldn't save")


def save(date, answer_data_class, g_paras):
    data_upload.make_dir_in_date(answer_data_class.answer_time.split("-")[0])
    humi, temp = None, None
    try:
        bus.write_byte_data(i2c_addr, 0xE0, 0x00)
        data = bus.read_i2c_block_data(i2c_addr, 0x00, 6)
        syslog.syslog(f"INFO: i2c_block_data {data}")
        # 温度計算
        temp_mlsb = (data[0] << 8) | data[1]
        temp = -45 + 175 * int(str(temp_mlsb), 10) / (pow(2, 16) - 1)
        syslog.syslog(f"INFO: temprature = {temp}")
        # 湿度計算
        humi_mlsb = (data[3] << 8) | data[4]
        humi = 100 * int(str(humi_mlsb), 10) / (pow(2, 16) - 1)
        syslog.syslog(f"INFO: humidity = {humi}")
    except Exception as e:
        traceback.print_exc()
        print(f"ERROR: {e} get temp_humi_data failed")
        syslog.syslog(f"ERROR: {e} get temp_humi_data failed")
    # 可視画像についてはリサイズした画像で顔検出しているので、座標の変換が必要
    g_paras.visibles["vis_BBoxPoints"] = np.array(g_paras.visibles["vis_BBoxPoints"])
    x_points = (
        g_paras.visibles["vis_BBoxPoints"][:, 0]
        + (g_paras.img_vis_baseW - g_paras.img_vis_resizedW) / 2
    )
    g_paras.visibles["vis_BBoxPoints"][:, 0] = (
        x_points * g_paras.camH / g_paras.img_vis_resizedH
    )
    y_points = g_paras.visibles["vis_BBoxPoints"][:, 1]
    g_paras.visibles["vis_BBoxPoints"][:, 1] = (
        y_points * g_paras.camH / g_paras.img_vis_resizedH
    )
    g_paras.visibles["vis_BBoxPoints"] = g_paras.visibles["vis_BBoxPoints"].tolist()
    points = []
    points.append(g_paras.camW - g_paras.visibles["vis_BBoxPoints"][1][0])
    points.append(g_paras.visibles["vis_BBoxPoints"][0][1])
    points.append(g_paras.camW - g_paras.visibles["vis_BBoxPoints"][0][0])
    points.append(g_paras.visibles["vis_BBoxPoints"][1][1])
    g_paras.visibles["vis_BBoxPoints"] = points
    # visible_landmark
    # 顔特徴点の座標抽出
    g_paras.vis_landmarks = landmark_class.vis_landmark(
        g_paras.visibles["src"],
        [
            points[0],
            points[1],
            points[2],
            points[3],
        ],
    )
    g_paras.vis_landmarks = g_paras.vis_landmarks.tolist()
    try:
        vis_affine = affine_class.main(g_paras.visibles["src"], g_paras.vis_landmarks)
        g_paras.visible_affines = {
            "src": vis_affine,
            "type": "visible_affine",
        }
    except Exception as e:
        traceback.print_exc()
        print("visible_affine failed")
        syslog.syslog(f"ERROR: {e} visible_affine failed")
    save_contents_csv(
        date,
        answer_data_class.answer_time,
        answer_data_class.answer,
        temp,
        humi,
        g_paras.visibles["vis_BBoxPoints"],
        g_paras.vis_landmarks,
        g_paras.thermos["the_BBoxPoints"],
        g_paras.the_landmarks,
        g_paras.coretemp,
        g_paras.monolith_temp,
    )
    save_images(
        date,
        answer_data_class.answer_time,
        g_paras.visibles,
        g_paras.thermos,
        g_paras.visible_affines,
        g_paras.thermo_affines,
    )
    syslog.syslog(f"INFO: save finished {answer_data_class.answer_time}")


# 画像のサイズを画面の横幅に合わせる関数
def img_width_resize(img):
    return img[
        0:,
        img.shape[1] // 2 - dispW // 2 : img.shape[1] // 2 + dispW // 2,
    ]


# サーモカメラから温度値データ(ttt)と白黒に変換した画像データ(the_image)を取得する関数
def get_thermo_data(monolith_correct_temp, queue=None):
    try:
        ir = boson_class.take_thermo_ir()
        if np.nanmin(ir) == 0:
            print("thermo_camera_noise_error")
            syslog.syslog("ERROR: thermo_camera_noise_error")
        print("=== ir_value ===")
        print(f"{ir}\n")
        ttt = (ir - 18092) / 117.87  # 温度値に変換
        monolith_set = monolith_correct_temp  # 黒体設定温度
        x_p, y_p = 30, 30
        print("=== monolith_info ===")
        monolith_ir = ir[x_p][y_p]
        print(f"monolith_ir: {monolith_ir}")
        monolith_temp = ttt[x_p][y_p]  # 黒体実測温度
        print(f"monolith_temp: {g_paras.monolith_temp}\n")
        syslog.syslog(f"INFO: monolith_ir {monolith_ir}")
        syslog.syslog(f"INFO: monolith_temp {monolith_temp}")
        correction_temp = monolith_set - monolith_temp
        if sys.argv[3] == "demo":
            ttt += correction_temp
        # テスト用
        # ttt = np.load("./images/231013-094223_thermo_temp.npy")
        # x_p, y_p = 30, 30
        # monolith_temp = ttt[x_p][y_p]
        ttt1 = copy.deepcopy(ttt)  # pythonにおけるコピーのやり方
        ttt[ttt <= 20] = 20
        ttt[ttt >= 36] = 36
        thermal_min = np.nanmin(ttt)
        thermal_max = np.nanmax(ttt)
        the_temp_int256 = ttt - thermal_min
        the_temp_int256 = the_temp_int256 / (thermal_max - thermal_min) * 255
        the_image = the_temp_int256.astype(np.uint8)
        the_image_W = the_image.shape[1]
        the_image_H = the_image.shape[0]
        print("boson_error:''")
        if queue != None:
            queue.put(
                {
                    "temp_data": ttt1,
                    "temp_affine": ttt,
                    "thermo_image": the_image,
                    "img_width_height": [the_image_W, the_image_H],
                    "monolith_temp": monolith_temp,
                    "boson_error": "",
                }
            )
        return {
            "temp_data": ttt1,
            "temp_affine": ttt,
            "thermo_image": the_image,
            "img_width_height": [the_image_W, the_image_H],
            "monolith_temp": monolith_temp,
            "boson_error": "",
        }
    except Exception as e:
        traceback.print_exc()
        print(f"boson_error:{e}")
        the_image_W = 320
        the_image_H = 256
        if queue != None:
            queue.put(
                {
                    "temp_data": np.full((the_image_H, the_image_W, 3), [0, 0, 0]),
                    "temp_affine": np.full((the_image_H, the_image_W, 3), [0, 0, 0]),
                    "thermo_image": np.full((the_image_H, the_image_W, 3), [0, 0, 0]),
                    "img_width_height": [the_image_W, the_image_H],
                    "monolith_temp": None,
                    "boson_error": f"{e}",
                }
            )
        return {
            "temp_data": np.full((the_image_H, the_image_W, 3), [0, 0, 0]),
            "temp_affine": np.full((the_image_H, the_image_W, 3), [0, 0, 0]),
            "thermo_image": np.full((the_image_H, the_image_W, 3), [0, 0, 0]),
            "img_width_height": [the_image_W, the_image_H],
            "monolith_temp": None,
            "boson_error": f"{e}",
        }


def visible_camera_setup():
    # カメラの設定
    visCap = cv2.VideoCapture(0)
    _, image = visCap.read()
    result = np.all(image[:, :, 0] == image[:, :, 1]) and np.all(
        image[:, :, 1] == image[:, :, 2]
    )
    if result == True:
        print("camera changed")
        visCap.release()
        time.sleep(1)
        visCap = cv2.VideoCapture(1)
        g_paras.visCap_ID = 1
        g_paras.boson_ID = 0
    print("visCap state is", visCap.isOpened())
    # 青学では以下の動画圧縮方式は機能しないためコメントアウトしている
    # visCap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("H", "2", "6", "4"))
    visCap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    visCap.set(cv2.CAP_PROP_FPS, 10)
    visCap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    visCap.set(cv2.CAP_PROP_FOCUS, 0)
    _, g_paras.img_vis_cam = visCap.read()
    g_paras.camH, g_paras.camW = g_paras.img_vis_cam.shape[:2]
    # 画面の高さから上部のグラデーションを引いた高さに合わせてリサイズ
    g_paras.img_vis_base = cv2.resize(
        g_paras.img_vis_cam,
        (
            g_paras.camW
            * (dispH - g_paras.top_gra_px - g_paras.down_gra_px)
            // g_paras.camH,
            dispH - g_paras.top_gra_px - g_paras.down_gra_px,
        ),
    )
    g_paras.img_vis_baseH, g_paras.img_vis_baseW = g_paras.img_vis_base.shape[:2]
    g_paras.img_vis_resized = img_width_resize(g_paras.img_vis_base)
    g_paras.img_vis_resizedH, g_paras.img_vis_resizedW = g_paras.img_vis_resized.shape[
        :2
    ]
    return visCap


# テキスト画像と可視画像の前処理関数
def images_setup():
    g_paras.text_phase1_1 = cv2.imread("./images/かおを 枠に あわせて.png", -1)
    fontSize = 180
    g_paras.text_phase1_1 = cv2.resize(
        g_paras.text_phase1_1,
        (
            g_paras.text_phase1_1.shape[1] * fontSize // g_paras.text_phase1_1.shape[0],
            fontSize,
        ),
    )
    g_paras.text_phase2_1 = cv2.imread("./images/いまの 調子を えらんで.png", -1)
    fontSize = 180
    g_paras.text_phase2_1 = cv2.resize(
        g_paras.text_phase2_1,
        (
            g_paras.text_phase2_1.shape[1] * fontSize // g_paras.text_phase2_1.shape[0],
            fontSize,
        ),
    )
    g_paras.text_phase2_2 = cv2.imread("./images/イラスト.png", -1)
    fontSize = 140
    g_paras.text_phase2_2 = cv2.resize(
        g_paras.text_phase2_2,
        (
            g_paras.text_phase2_2.shape[1] * fontSize // g_paras.text_phase2_2.shape[0],
            fontSize,
        ),
    )
    g_paras.text_phase3_1 = cv2.imread("./images/終了 です.png", -1)
    fontSize = 150
    g_paras.text_phase3_1 = cv2.resize(
        g_paras.text_phase3_1,
        (
            g_paras.text_phase3_1.shape[1] * fontSize // g_paras.text_phase3_1.shape[0],
            fontSize,
        ),
    )
    g_paras.text_phase3_2 = cv2.imread("./images/ご安全に！.png", -1)
    fontSize = 400
    g_paras.text_phase3_2 = cv2.resize(
        g_paras.text_phase3_2,
        (
            g_paras.text_phase3_2.shape[1] * fontSize // g_paras.text_phase3_2.shape[0],
            fontSize,
        ),
    )
    g_paras.ans1 = cv2.imread("./images/とても げんき.png", -1)
    g_paras.ans1 = cv2.resize(g_paras.ans1, g_paras.ansBtnSize)
    g_paras.ans2 = cv2.imread("./images/いつも どおり.png", -1)
    g_paras.ans2 = cv2.resize(g_paras.ans2, g_paras.ansBtnSize)
    g_paras.ans3 = cv2.imread("./images/すこし わるい.png", -1)
    g_paras.ans3 = cv2.resize(g_paras.ans3, g_paras.ansBtnSize)
    g_paras.ans4 = cv2.imread("./images/とても わるい.png", -1)
    g_paras.ans4 = cv2.resize(g_paras.ans4, g_paras.ansBtnSize)
    # 画面上部のグラデーションの設定
    g_paras.top_gra_phase1 = make_top_gradation(
        start_color=(0, 165, 255, 255), end_color=(0, int(50 * 165 / 255), 50, 255)
    )
    g_paras.top_gra_phase2 = make_top_gradation(
        start_color=(47, 255, 173, 255),
        end_color=(int(50 * 47 / 255), 50, int(50 * 173 / 255), 255),
    )
    g_paras.top_gra_phase3 = make_top_gradation(
        start_color=(255, 144, 30, 255),
        end_color=(50, int(50 * 144 / 255), int(50 * 30 / 255), 255),
    )
    h, w = g_paras.text_phase1_1.shape[:2]
    dy = (g_paras.top_gra_px - h) // 2
    dx = (g_paras.img_vis_resizedW - w) // 2
    g_paras.top_gra_phase1[dy : dy + h, dx : dx + w][
        g_paras.text_phase1_1[:, :, 3] != 0
    ] = (255, 255, 255, 255)

    h, w = g_paras.text_phase2_1.shape[:2]
    dy = (g_paras.top_gra_px - h) // 2
    dx = (g_paras.img_vis_resizedW - w) // 2
    g_paras.top_gra_phase2[dy : dy + h, dx : dx + w][
        g_paras.text_phase2_1[:, :, 3] != 0
    ] = (255, 255, 255, 255)

    h, w = g_paras.text_phase2_2.shape[:2]
    dy = (g_paras.top_gra_px - h) // 2
    dx = 950
    g_paras.top_gra_phase2[dy : dy + h, dx : dx + w][
        g_paras.text_phase2_2[:, :, 3] != 0
    ] = g_paras.text_phase2_2[g_paras.text_phase2_2[:, :, 3] != 0]

    h, w = g_paras.text_phase3_1.shape[:2]
    dy = (g_paras.top_gra_px - h) // 2
    dx = (g_paras.img_vis_resizedW - w) // 2
    g_paras.top_gra_phase3[dy : dy + h, dx : dx + w][
        g_paras.text_phase3_1[:, :, 3] != 0
    ] = (255, 255, 255, 255)

    g_paras.phase1[: g_paras.top_gra_px] = g_paras.top_gra_phase1[:, :, :3]
    g_paras.phase2[: g_paras.top_gra_px] = g_paras.top_gra_phase2[:, :, :3]
    g_paras.phase3[: g_paras.top_gra_px] = g_paras.top_gra_phase3[:, :, :3]


# 特徴点をプロットする関数
def draw_landmarks(img, landmarks):
    try:
        landmarks = landmarks.astype(np.uint8)
        for point in landmarks.tolist():
            center = (point[0], point[1])
            cv2.circle(img, center=center, radius=1, color=(255, 0, 0), thickness=2)
        return img
    except:
        print("can't draw_landmarks")


def make_top_gradation(start_color, end_color):
    top_gra_px = g_paras.top_gra_px
    gradient = np.linspace(start_color, end_color, num=top_gra_px).astype(np.uint8)
    gradation_image = np.zeros(
        (top_gra_px, g_paras.img_vis_resizedW, 4), dtype=np.uint8
    )
    gradation_image[:top_gra_px] = gradient[:, np.newaxis]
    return gradation_image


def make_face_mask():
    mask = np.full(
        (g_paras.img_vis_resizedH, g_paras.img_vis_resizedW, 3),
        [255, 255, 255],
        dtype=np.uint8,
    )
    mask_img = cv2.imread("./images/シルエット.jpg")
    h, w = mask_img.shape[:2]
    mask_img = cv2.resize(
        mask_img, (g_paras.img_vis_resizedW, h * g_paras.img_vis_resizedW // w)
    )
    h, w = mask_img.shape[:2]
    mask[-h:, :][mask_img[:, :, 0] < 100] = (0, 0, 0)
    return mask


def draw_date_bottom(dummy):
    date = datetime.datetime.now()
    yearNum = date.year
    year = g_paras.yearImage[yearNum - 2023]
    year_H, year_W = year.shape[:2]
    month = g_paras.numberImages[date.month]
    month_H, month_W = month.shape[:2]
    day = g_paras.numberImages[date.day]
    day_H, day_W = day.shape[:2]
    hour = g_paras.numberImages[date.hour]
    hour_H, hour_W = hour.shape[:2]
    min = g_paras.numberImages[date.minute]
    min_H, min_W = min.shape[:2]
    sec = g_paras.numberImages[date.second]
    sec_H, sec_W = sec.shape[:2]
    haihun = g_paras.numberImages[60]
    haihun_H, haihun_W = haihun.shape[:2]
    koron = g_paras.numberImages[61]
    koron_H, koron_W = koron.shape[:2]
    white = (255, 255, 255)
    number_place = -170
    haihun_place = -140
    koron_place = -155
    down_gra = np.full((g_paras.down_gra_px, dispW, 3), [128, 128, 128])
    down_gra[number_place : number_place + year_H, 20 : 20 + year_W][
        year[:, :, 3] > 80
    ] = white
    down_gra[haihun_place : haihun_place + haihun_H, 260 : 260 + haihun_W][
        haihun[:, :, 3] != 0
    ] = white
    down_gra[number_place : number_place + month_H, 300 : 300 + month_W][
        month[:, :, 3] != 0
    ] = white
    down_gra[haihun_place : haihun_place + haihun_H, 420 : 420 + haihun_W][
        haihun[:, :, 3] != 0
    ] = white
    down_gra[number_place : number_place + day_H, 455 : 455 + day_W][
        day[:, :, 3] != 0
    ] = white
    down_gra[number_place : number_place + hour_H, 660 : 660 + hour_W][
        hour[:, :, 3] != 0
    ] = white
    down_gra[koron_place : koron_place + koron_H, 775 : 775 + koron_W][
        koron[:, :, 3] != 0
    ] = white
    down_gra[number_place : number_place + min_H, 805 : 805 + min_W][
        min[:, :, 3] != 0
    ] = white
    down_gra[koron_place : koron_place + koron_H, 920 : 920 + koron_W][
        koron[:, :, 3] != 0
    ] = white
    down_gra[number_place : number_place + sec_H, 950 : 950 + sec_W][
        sec[:, :, 3] != 0
    ] = white
    g_paras.background_image[-g_paras.down_gra_px :] = down_gra


# それぞれのphaseの画面描画処理をまとめた関数
def draw_image():
    if g_paras.boson_ERROR != "":
        if "Boson not connected" in g_paras.boson_ERROR:
            error_image = g_paras.boson_notConnected_image
        else:
            error_image = g_paras.boson_error_something_image
        h, w = error_image.shape[:2]
        x = (dispW - w) // 2
        y = (g_paras.img_vis_resizedH - h) // 2
        g_paras.img_vis_resized[y : y + h, x : x + w] = error_image
    else:
        if g_paras.phase == 1 or g_paras.phase == 4:
            g_paras.img_vis_resized = cv2.addWeighted(
                g_paras.img_vis_resized, 1, mask, 0.5, 0
            )
            g_paras.background_image[: -g_paras.down_gra_px] = g_paras.phase1
            if play_wav_process.is_alive() or g_paras.phase == 4:
                g_paras.img_vis_resized = mask3
                g_paras.background_image[: -g_paras.down_gra_px] = g_paras.phase3
            if g_paras.the_bbox != [[]] and g_paras.the_bbox[0][0][6] != 0:
                h = g_paras.mask_glasses_image.shape[0]
                g_paras.img_vis_resized[50 : 50 + h] = g_paras.mask_glasses_image
            elif g_paras.face_far == True:
                h = g_paras.face_far_image.shape[0]
                g_paras.img_vis_resized[50 : 50 + h] = g_paras.face_far_image
            if g_paras.vis_bbox != [[]]:
                g_paras.background_image[:10, :10] = (255, 0, 0)
            if g_paras.the_bbox != [[]]:
                g_paras.background_image[:10, 11:20] = (0, 255, 0)
            temp_color = (0, 0, 0)
            try:
                if g_paras.monolith_temp < 38:
                    temp_color = (255, 0, 0)
                elif g_paras.monolith_temp < 40:
                    blue = 255 - int((g_paras.monolith_temp - 38) * (255 / 2))
                    green = int((g_paras.monolith_temp - 38) * (255 / 2))
                    temp_color = (blue, green, 0)
                elif g_paras.monolith_temp <= 42:
                    green = 255 - int((g_paras.monolith_temp - 40) * (255 / 2))
                    red = int((g_paras.monolith_temp - 40) * (255 / 2))
                    temp_color = (0, green, red)
                else:
                    temp_color = (0, 0, 255)
            except:
                temp_color = (255, 0, 0)
            g_paras.background_image[:10, -10:] = temp_color
        elif g_paras.phase == 2:
            g_paras.img_vis_resized = cv2.addWeighted(
                g_paras.img_vis_resized, 1, mask, 0.5, 0
            )
            try:
                image = None
                if g_paras.coretemp > 38:
                    image = g_paras.body_coretemp_images["38.0"]
                elif g_paras.coretemp < 34.5:
                    image = g_paras.body_coretemp_images["34.5"]
                else:
                    image = g_paras.body_coretemp_images[f"{g_paras.coretemp}"]
                h = image.shape[0]
            except:
                image = g_paras.body_coretemp_images[f"36.5"]
                h = image.shape[0]
            g_paras.img_vis_resized[50 : 50 + h] = image
        elif g_paras.phase == 3:
            g_paras.img_vis_resized = mask2
            g_paras.background_image[: -g_paras.down_gra_px] = g_paras.phase2
            g_paras.img_vis_resized = draw_ans(g_paras.img_vis_resized)


def play_wav(wav_obj):
    play_obj = wav_obj.play()
    play_obj.wait_done()


# main開始
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    location = sys.argv[2]
    # 前処理開始
    print("処理開始")
    syslog.syslog(f"INFO: {location} preprocess start")
    for i in range(len(sys.argv)):
        print(sys.argv[i], end=" ")
    print()
    estimate_coretemp_class = simple_estimate_coretemp.extimate_coretemp(
        load_model=True
    )
    try:
        i2c_addr = 0x45
        # SMBusモジュールの設定
        bus = smbus.SMBus(1)
        # i2c通信の設定
        bus.write_byte_data(i2c_addr, 0x21, 0x30)
        time.sleep(1)
        print("SHT31_check_OK")
        syslog.syslog("INFO: SHT31_check_OK")
    except Exception as e:
        syslog.syslog(f"ERROR: {e} SHT31_check failed")
        traceback.print_exc()
    try:
        os.mkdir("data")
        syslog.syslog("INFO: data directory maked")
    except:
        pass
    # data_upload.make_dir_in_date()
    dispW, dispH = pyautogui.size()
    dispH -= 50
    print(f"dispH, dispW = {dispH}, {dispW}")
    g_paras = global_parameters()
    visCap = visible_camera_setup()
    print(f"camH, camW = {g_paras.camH}, {g_paras.camW}")
    print(f"img_baseH, img_baseW = {g_paras.img_vis_baseH}, {g_paras.img_vis_baseW}")
    print(
        f"img_resizedH, img_resizedW = {g_paras.img_vis_resizedH}, {g_paras.img_vis_resizedW}"
    )
    syslog.syslog(f"INFO: camH, camW = {g_paras.camH}, {g_paras.camW}")
    syslog.syslog(
        f"INFO: img_baseH, img_baseW = {g_paras.img_vis_baseH}, {g_paras.img_vis_baseW}"
    )
    syslog.syslog(
        f"INFO: img_resizedH, img_resizedW = {g_paras.img_vis_resizedH}, {g_paras.img_vis_resizedW}"
    )
    images_setup()
    mask = make_face_mask()
    mask2 = np.full(
        (dispH - g_paras.top_gra_px - g_paras.down_gra_px, dispW, 3),
        [200, 200, 200],
        np.uint8,
    )
    mask3 = np.full(
        (dispH - g_paras.top_gra_px - g_paras.down_gra_px, dispW, 3),
        [200, 200, 200],
        np.uint8,
    )
    h, w = g_paras.text_phase3_2.shape[:2]
    mask_h = dispH - g_paras.top_gra_px - g_paras.down_gra_px
    mask3[(mask_h - h) // 2 : (mask_h + h) // 2, (dispW - w) // 2 : (dispW + w) // 2][
        g_paras.text_phase3_2[:, :, 0] > 90
    ] = (192, 112, 0)
    wav_obj2 = simpleaudio.WaveObject.from_wave_file("./sound/アンケート画面遷移時1.wav")
    wav_obj3 = simpleaudio.WaveObject.from_wave_file("./sound/終了時3.wav")
    play_wav_process = mp.Process(
        target=play_wav,
        args=("dummy",),
    )
    play_wav_process.daemon = True
    try:
        command = f"sudo udevadm info --query=path --name=/dev/video{g_paras.boson_ID}"
        print(command)
        command_output = subprocess.run(
            [command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            encoding="utf8",
        )
        print(command_output.stdout.rstrip())
        output_list = command_output.stdout.rstrip().split("/")
        target_text_id = len(output_list) - 4
        boson_camera_id_text = output_list[target_text_id]
    except:
        boson_camera_id_text = ""
    print(f"boson_reset_id : {boson_camera_id_text}")
    syslog.syslog(f"INFO: boson_reset_id : {boson_camera_id_text}")
    faceDetect_class = fs_face_detection_trt_mix4380.face_detect(
        engine_path="./model/yolov4_tiny_mix4380.trt"
    )
    answer_data_class = answer_data(
        dispW=dispW,
        ansBtnSize=g_paras.ansBtnSize,
        pad=g_paras.ansTopPad,
    )
    landmark_class = fs_bbox_landmark.bbox_landmark()
    affine_class = fs_affine_convert_traiangle.affine_convert(
        afsize=[120, 120], square=True
    )
    boson_class = fs_boson_camera_lib.thermo_camera_img()
    boson_error_queue = mp.Queue()
    time_thread = td.Thread(target=draw_date_bottom, args=("dummy",))
    time_thread.daemon = True
    thermo_data_queue = mp.Queue()
    get_thermo_data_process = mp.Process(
        target=get_thermo_data,
        args=(
            g_paras.monolith_correct_temp,
            thermo_data_queue,
        ),
    )
    get_thermo_data_process.daemon = True
    save_process = mp.Process(
        target=save,
        args=("dummy",),
    )
    sample_image = cv2.flip(cv2.imread("./images/231013-094223_visible_image.png"), 1)
    upload_process = mp.Process(
        target=data_upload.upload, args=("dummy")
    )  # 日付, パスを取得するための__file__, rm_date_dir, rm_child_dir, rm_file
    faceDetected_sleep_process = mp.Process(
        target=time.sleep,
        args=(0.7,),
    )
    faceDetected_sleep_process.daemon = True
    delete_process = mp.Process(
        target=data_delete.delete,
        args=(__file__, location),
    )
    print("処理完了")
    line_notify_token = "LmaVs3D4yimkQVGLWxLvuEA6Br4inNJzH1He6Hxwkul"
    line_notify_api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": f"\n{location} start!"}
    while True:
        try:
            response = requests.post(line_notify_api, headers=headers, data=data)
            if response.status_code == 200:
                syslog.syslog("INFO: START Notify Send")
                break
            else:
                raise Exception
        except Exception as e:
            print("START Notify FAILED")
            syslog.syslog(f"ERROR: START Notify FAILED {e}")
            time.sleep(1)
    syslog.syslog("INFO: SYSTEM START")
    cv2.imshow("form", g_paras.background_image)
    cv2.setWindowProperty("form", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("form", 0, 0)
    try:
        if sys.argv[4] == "show_thermo":
            cv2.imshow("Boson", g_paras.thermo_data["thermo_image"])
    except:
        pass
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(33, GPIO.OUT)
    time.sleep(1)
    now = datetime.datetime.now()
    date = now.strftime("%y%m%d")
    syslog.syslog(f"INFO: {location} system start!")
    while True:
        try:
            start = time.perf_counter()
            now = datetime.datetime.now()
            if date != now.strftime("%y%m%d"):
                upload_process = mp.Process(
                    target=data_upload.upload,
                    args=(date, __file__, location, True, True, True),
                )  # 日付, パスを取得するための__file__, rm_date_dir, rm_child_dir
                date = now.strftime("%y%m%d")
                upload_process.daemon = True
                upload_process.start()
                delete_process = mp.Process(
                    target=data_delete.delete,
                    args=(__file__, location),
                )
                delete_process.daemon = True
                delete_process.start()
            if g_paras.boson_ERROR != "" and not get_thermo_data_process.is_alive():
                reset()
                thermo_data_queue = mp.Queue()
                syslog.syslog("INFO: boson_reset work")
                fs_boson_camera_lib.boson_reset(boson_camera_id_text)
                time.sleep(1)
                g_paras.thermo_data = get_thermo_data(g_paras.monolith_correct_temp)
                g_paras.boson_ERROR = g_paras.thermo_data["boson_error"]
                print("boson_reset done")
            else:
                if (
                    len(sys.argv) > 1
                    and sys.argv[1] == "light_off"
                    and (now.hour >= 19 or now.hour <= 5)
                ):
                    g_paras.light_off = True
                else:
                    g_paras.light_off = False
                if not g_paras.light_off:
                    GPIO.output(33, 1)
                    ret, g_paras.img_vis_cam = visCap.read()
                    # g_paras.img_vis_cam = sample_image  # サンプル画像
                    g_paras.img_vis_base = cv2.resize(
                        g_paras.img_vis_cam,
                        (
                            g_paras.camW * g_paras.img_vis_baseH // g_paras.camH,
                            g_paras.img_vis_baseH,
                        ),
                        interpolation=cv2.INTER_NEAREST,  # 処理を軽くするため
                    )
                    g_paras.img_vis_base = cv2.flip(g_paras.img_vis_base, 1)
                    g_paras.img_vis_resized = img_width_resize(g_paras.img_vis_base)
                    if not g_paras.keyboardClicked:
                        if g_paras.phase == 1 and not play_wav_process.is_alive():
                            try:
                                g_paras.vis_bbox = faceDetect_class.detect(
                                    g_paras.img_vis_resized
                                )
                                g_paras.vis_ltX = int(
                                    g_paras.vis_bbox[0][0][0] * g_paras.img_vis_resizedW
                                )
                                g_paras.vis_ltY = int(
                                    g_paras.vis_bbox[0][0][1] * g_paras.img_vis_resizedH
                                )
                                g_paras.vis_rbX = int(
                                    g_paras.vis_bbox[0][0][2] * g_paras.img_vis_resizedW
                                )
                                g_paras.vis_rbY = int(
                                    g_paras.vis_bbox[0][0][3] * g_paras.img_vis_resizedH
                                )
                                face_width = g_paras.vis_rbX - g_paras.vis_ltX
                                if face_width < g_paras.img_vis_resizedW * 0.45:
                                    g_paras.face_far = True
                                else:
                                    g_paras.face_far = False
                            except:
                                g_paras.vis_bbox = [[]]
                                g_paras.face_far = False
                                g_paras.vis_ltX = 0
                                g_paras.vis_ltY = 0
                                g_paras.vis_rbX = 0
                                g_paras.vis_rbY = 0
                            if g_paras.vis_bbox != [[]]:
                                if (
                                    thermo_data_queue.empty()
                                    and not get_thermo_data_process.is_alive()
                                ):
                                    get_thermo_data_process = mp.Process(
                                        target=get_thermo_data,
                                        args=(
                                            g_paras.monolith_correct_temp,
                                            thermo_data_queue,
                                        ),
                                    )
                                    get_thermo_data_process.daemon = True
                                    get_thermo_data_process.start()
                                if not thermo_data_queue.empty():
                                    g_paras.thermo_data = thermo_data_queue.get()
                                    g_paras.boson_ERROR = g_paras.thermo_data[
                                        "boson_error"
                                    ]
                                    g_paras.monolith_temp = g_paras.thermo_data[
                                        "monolith_temp"
                                    ]
                                    try:
                                        if sys.argv[4] == "show_thermo":
                                            cv2.imshow(
                                                "Boson",
                                                g_paras.thermo_data["thermo_image"],
                                            )
                                    except:
                                        pass
                                    try:
                                        g_paras.the_bbox = faceDetect_class.detect(
                                            g_paras.thermo_data["thermo_image"]
                                        )
                                        the_bbox = [
                                            [
                                                bbox
                                                for bbox in g_paras.the_bbox[0]
                                                if (bbox[0] >= 0.2 and bbox[2] <= 0.8)
                                            ]
                                        ]  # 画面の端で検出された顔を省く
                                        g_paras.the_bbox = the_bbox
                                        if len(g_paras.the_bbox[0]) > 1:
                                            id = 0
                                            max = 0
                                            for i in range(len(g_paras.the_bbox[0])):
                                                a = (
                                                    g_paras.the_bbox[0][i][2]
                                                    - g_paras.the_bbox[0][i][0]
                                                )
                                                b = (
                                                    g_paras.the_bbox[0][i][3]
                                                    - g_paras.the_bbox[0][i][1]
                                                )
                                                if a + b > max:
                                                    id = i
                                                    max = a + b
                                            g_paras.the_bbox[0] = g_paras.the_bbox[0][
                                                id
                                            ]  # 一番大きいものを採用
                                        g_paras.the_ltX = int(
                                            g_paras.the_bbox[0][0][0]
                                            * g_paras.thermo_data["img_width_height"][0]
                                        )
                                        g_paras.the_ltY = int(
                                            g_paras.the_bbox[0][0][1]
                                            * g_paras.thermo_data["img_width_height"][1]
                                        )
                                        g_paras.the_rbX = int(
                                            g_paras.the_bbox[0][0][2]
                                            * g_paras.thermo_data["img_width_height"][0]
                                        )
                                        g_paras.the_rbY = int(
                                            g_paras.the_bbox[0][0][3]
                                            * g_paras.thermo_data["img_width_height"][1]
                                        )
                                    except:
                                        g_paras.the_bbox = [[]]
                                        g_paras.the_ltX = 0
                                        g_paras.the_ltY = 0
                                        g_paras.the_rbX = 0
                                        g_paras.the_rbY = 0
                            # try:
                            #     g_paras.the_bbox[0][0][6] = 1
                            # except:
                            #     pass
                            print("=== bbox info ===")
                            print(f"vis_bbox: {g_paras.vis_bbox}")
                            print(f"the_bbox: {g_paras.the_bbox}\n")
                            if g_paras.vis_bbox != [[]] or g_paras.the_bbox != [[]]:
                                syslog.syslog(
                                    f"INFO: vis_bbox {g_paras.vis_bbox}, the_bbox {g_paras.the_bbox}"
                                )
                                if g_paras.vis_bbox != [[]]:
                                    syslog.syslog("INFO: visible FACE detected")
                                if g_paras.the_bbox != [[]]:
                                    if g_paras.the_bbox[0][0][6] == 0:
                                        syslog.syslog(f"INFO: thermo FACE detected")
                                    elif g_paras.the_bbox[0][0][6] == 1:
                                        syslog.syslog(f"INFO: thermo MASK detected")
                                    elif g_paras.the_bbox[0][0][6] == 2:
                                        syslog.syslog(f"INFO: thermo GLASSES detected")
                                    elif g_paras.the_bbox[0][0][6] == 3:
                                        syslog.syslog(
                                            f"INFO: thermo MASK_AND_GLASSES detected"
                                        )
                            if face_in_fit(
                                g_paras.vis_ltX,
                                g_paras.vis_ltY,
                                g_paras.vis_rbX,
                                g_paras.vis_rbY,
                            ):
                                g_paras.img_vis_resized = draw_face_frame(
                                    g_paras.img_vis_resized,
                                    (255, 255, 255),
                                    g_paras.vis_ltX,
                                    g_paras.vis_ltY,
                                    g_paras.vis_rbX,
                                    g_paras.vis_rbY,
                                )
                                if not g_paras.faceDetected:
                                    g_paras.faceDetected = True
                                    detectTimeSta = time.perf_counter()
                                    syslog.syslog("INFO: Detect_Timer START")
                                detectTimeEnd = time.perf_counter()
                                passedDetectTime = detectTimeEnd - detectTimeSta
                                syslog.syslog(
                                    f"INFO: passedDetectTime is {passedDetectTime} s"
                                )
                                if (
                                    (
                                        passedDetectTime >= 0.5
                                        and g_paras.the_bbox != [[]]
                                        and g_paras.the_bbox[0][0][6] == 0
                                    )
                                    or (
                                        passedDetectTime >= 2.0
                                        and g_paras.the_bbox != [[]]
                                        and g_paras.the_bbox[0][0][6] != 0
                                    )
                                    or passedDetectTime >= 3.0
                                ):
                                    syslog.syslog("INFO: Detect_Timer FINISH")
                                    get_thermo_data_process.terminate()
                                    thermo_data_queue = mp.Queue()
                                    faceDetected_sleep_process = mp.Process(
                                        target=time.sleep,
                                        args=(0.7,),
                                    )
                                    faceDetected_sleep_process.daemon = True
                                    faceDetected_sleep_process.start()
                                    g_paras.faceDetected = False
                                    g_paras.phase = 2
                                    add_images()
                                    syslog.syslog(f"INFO: get data finished")
                            elif g_paras.vis_bbox == [[]]:
                                reset()
                        elif g_paras.phase == 2:
                            if not faceDetected_sleep_process.is_alive():
                                play_wav_process = mp.Process(
                                    target=play_wav,
                                    args=(wav_obj2,),
                                )
                                play_wav_process.daemon = True
                                play_wav_process.start()
                                g_paras.phase = 3
                        elif g_paras.phase == 3:
                            cv2.setMouseCallback("form", answer_data_class.click_pos)
                            if not g_paras.stable_started:
                                stable_start_time = time.perf_counter()
                                g_paras.stable_started = True
                            stable_passed_time = time.perf_counter() - stable_start_time
                            if stable_passed_time > 10:
                                syslog.syslog("INFO: notAnswer")
                                g_paras.phase = 1
                                g_paras.stable_started = False
                                thermo_data_queue = mp.Queue()
                            if answer_data_class.ansBtnClicked:
                                syslog.syslog("INFO: answered")
                                if (
                                    g_paras.the_bbox != [[]]
                                    and g_paras.the_bbox[0][0][6] != 0
                                ):
                                    syslog.syslog("INFO: MASK_GLASSES answered")
                                elif g_paras.the_bbox == [[]]:
                                    syslog.syslog("INFO: thermo_skipped answered")
                                g_paras.phase = 1
                                save_process = mp.Process(
                                    target=save,
                                    args=(date, answer_data_class, g_paras),
                                )
                                save_process.daemon = True
                                save_process.start()
                                if play_wav_process.is_alive():
                                    play_wav_process.terminate()
                                play_wav_process = mp.Process(
                                    target=play_wav,
                                    args=(wav_obj3,),
                                )
                                play_wav_process.daemon = True
                                play_wav_process.start()
                                g_paras.stable_started = False
                                reset()
                                thermo_data_queue = mp.Queue()
                    else:
                        syslog.syslog("INFO: KEYBOARD CLICKED")
                    draw_image()
                    g_paras.background_image[
                        g_paras.top_gra_px : -g_paras.down_gra_px
                    ] = g_paras.img_vis_resized
                    if not time_thread.is_alive():
                        time_thread = td.Thread(
                            target=draw_date_bottom, args=("dummy",)
                        )
                        time_thread.daemon = True
                        time_thread.start()
                else:
                    g_paras.background_image = np.full(
                        (dispH, dispW, 3), (0, 0, 0), dtype=np.uint8
                    )
                    GPIO.output(33, 0)
            if g_paras.phase != 1:
                syslog.syslog(f"INFO: now phase is {g_paras.phase}")
            cv2.imshow("form", g_paras.background_image)
            keyValue = cv2.waitKey(1) & 0xFF  # キーボード入力による画面遷移
            if keyValue == ord("q"):
                break
            elif keyValue >= ord("1") and keyValue <= ord("9"):
                answer_data_class.answer = ""
                g_paras.keyboardClicked = True
                g_paras.phase = keyValue - ord("0")
            elif keyValue == ord("r"):  # キーボード入力状態を解除
                reset()
            elif keyValue == ord("u"):
                if not upload_process.is_alive():
                    upload_process = mp.Process(
                        target=data_upload.upload,
                        args=(
                            now.strftime("%y%m%d"),
                            __file__,
                            location,
                            True,
                            True,
                            False,
                        ),
                    )  # 日付, パスを取得するための__file__, rm_date_dir, rm_child_dir, rm_file
                    upload_process.daemon = True
                    upload_process.start()
                    upload_image = cv2.imread("./images/upload.png", -1)
                    h, w = upload_image.shape[:2]
                    fontSize = 100
                    upload_image = cv2.resize(
                        upload_image, (w * fontSize // h, fontSize)
                    )
                    h, w = upload_image.shape[:2]
                    g_paras.img_vis_resized[100 : 100 + fontSize, -w:][
                        upload_image[:, :, 3] != 0
                    ] = upload_image[:, :, :3][upload_image[:, :, 3] != 0]
                    g_paras.background_image[
                        g_paras.top_gra_px : dispH - g_paras.down_gra_px
                    ] = g_paras.img_vis_resized
                    upload_start = time.perf_counter()
                    while time.perf_counter() - upload_start < 3:
                        cv2.imshow("form", g_paras.background_image)
                        keyValue = cv2.waitKey(1) & 0xFF  # キーボード入力による画面遷移
                        if keyValue == ord("q"):
                            break
                    reset()
                else:
                    print("upload_process is alive")
            if cv2.getWindowProperty("form", cv2.WND_PROP_AUTOSIZE) == -1:
                cv2.moveWindow("form", 0, 0)
            process_time = time.perf_counter() - start
            print(f"time:{process_time}")
            print(f"fps:{1/process_time}\n")
        except Exception as e:
            traceback.print_exc()
            syslog.syslog(f"ERROR: system finished ({e})")
            data = {"message": f"\n{location} finish! ({e})"}
            response = requests.post(line_notify_api, headers=headers, data=data)
            break
    GPIO.cleanup()
    visCap.release()
    cv2.destroyAllWindows()
    data = {"message": f"\n{location} finish!"}
    response = requests.post(line_notify_api, headers=headers, data=data)
