import numpy as np
import cv2
import dlib
import copy
from imutils import face_utils


class bbox_landmark:
    def __init__(self):
        # 可視画像顔検出器
        face_detector = dlib.get_frontal_face_detector()

        # 熱画像顔検出器
        ##tensorRT_yolotinyモデルで実行

        # 可視画像特徴点抽出機
        predictor_path_vis = "./model/shape_predictor_68_face_landmarks.dat"
        face_predictor_vis = dlib.shape_predictor(predictor_path_vis)

        # 熱画像特徴点抽出機
        predictor_path_the = "./model/landmarkModel_20210901210352.dat"
        face_predictor_the = dlib.shape_predictor(predictor_path_the)

        self.face_detector = face_detector
        self.face_predictor_vis = face_predictor_vis
        self.face_predictor_the = face_predictor_the

    # 可視画像顔検出
    def vis_bbox(self, thermal_data):
        # img_gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(thermal_data, 1)
        return faces

    # 可視画像特徴点抽出
    def vis_landmark(self, thermal_data, faces):
        bbox_dlib_type = dlib.rectangle(
            int(faces[0]), int(faces[1]), int(faces[2]), int(faces[3])
        )
        landmark = self.face_predictor_vis(thermal_data, bbox_dlib_type)
        landmark = face_utils.shape_to_np(landmark)
        return landmark

    # 熱画像特徴点抽出
    def the_landmarks(self, temp_data, bbox, binary_on=False):
        landmarks = []

        # 温度値データであれはbinary_on == True
        if binary_on:
            min_temp = 26
            max_temp = 37
            norm_data = copy.deepcopy(temp_data)

            if np.max(norm_data.flatten()) < max_temp:
                max_temp = np.max(norm_data.flatten())

            if np.min(norm_data.flatten()) > min_temp:
                min_temp = np.min(norm_data.flatten())

            # 最小値を0，最大値を255に変換
            norm_data = (norm_data - min_temp) / (max_temp - min_temp) * 255

            # 画像を3次元に変換
            norm_data_3dim = np.zeros(
                (norm_data.shape[0], norm_data.shape[1], 3), dtype="uint8"
            )
            for i in range(3):
                norm_data_3dim[:, :, i] = norm_data

        else:
            norm_data_3dim = temp_data

        # 特徴点を算出
        d = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        # print(d)
        shape = self.face_predictor_the(norm_data_3dim, d)
        result_pts = np.zeros((68, 2))
        for j in range(68):
            result_pts[j][0] = shape.part(j).x
            result_pts[j][1] = shape.part(j).y

        return result_pts
