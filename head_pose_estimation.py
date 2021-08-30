# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 03:00:36 2020

@author: hp
"""

import cv2
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
import argparse
from utils import params


def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d


def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    """
    Draw a 3D anotation box on the face for head pose estimation

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix
    rear_size : int, optional
        Size of rear box. The default is 300.
    rear_depth : int, optional
        The default is 0.
    front_size : int, optional
        Size of front box. The default is 500.
    front_depth : int, optional
        Front depth. The default is 400.
    color : tuple, optional
        The color with which to draw annotation box. The default is (255, 255, 0).
    line_width : int, optional
        line width of lines drawn. The default is 2.

    Returns
    -------
    None.

    """

    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    """
    Get the points to estimate head pose sideways    

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix

    Returns
    -------
    (x, y) : tuple
        Coordinates of line to estimate head pose

    """
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8]) // 2
    x = point_2d[2]

    return (x, y)


face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(params["cam_id"])
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX
# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

# Camera internals
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

# TODO ADD OBJECT DETECTION
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--prototxt', help='path to Caffe deploy prototxt file',
                default="model_objects/MobileNetSSD_deploy.prototxt.txt")
ap.add_argument('-m', '--model', help='path to the Caffe pre-trained model',
                default="model_objects/MobileNetSSD_deploy.caffemodel")
ap.add_argument('-c', '--confidence', type=float, default=0.2, help='minimum probability to filter weak detections')
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# TODO END OBJECT DETECTION

while True:
    ret, img = cap.read()
    if ret == True:

        objects = {}

        # TODO ADD OBJECT DETECTION
        frame = img
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] == "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                                             confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                objects[CLASSES[idx]] = (startX, startY, endX, endY)
        # TODO END OBJECT DETECTION

        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array([
                marks[30],  # Nose tip
                marks[8],  # Chin
                marks[36],  # Left eye left corner
                marks[45],  # Right eye right corne
                marks[48],  # Left Mouth corner
                marks[54]  # Right mouth corner
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                             translation_vector, camera_matrix, dist_coeffs)

            # draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix)

            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            cv2.line(img, p1, p2, (0, 255, 255), 2)
            cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)

            # for (x, y) in marks:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
            try:
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90

            try:
                m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1 / m)))
            except:
                ang2 = 90

            if True:
            #     -40 <= ang1 <= 40 and -40 <= ang2 <= 40:
            #     cv2.putText(img, 'Focus', (30, 30), font, 2, (255, 255, 128), 3)
            # else:

            # TODO STARTING YELLOW LINE
                drag_x = p1[0]
                drag_y = p1[1]

                p1 = (p1[0] - drag_x, p1[1] - drag_y)
                p2 = (p2[0] - drag_x, p2[1] - drag_y)

                if (p1[0] - p2[0]) != 0:  # to avoid division with 0
                    delta = (p1[1] - p2[1]) / (p1[0] - p2[0])
                else:
                    delta = 999

                saw = False

                for o in objects.keys():  # translate box
                    trans = (objects[o][0] - drag_x,  # startX
                             objects[o][1] - drag_y,  # startY
                             objects[o][2] - drag_x,  # endX
                             objects[o][3] - drag_y)  # endY

                    l_met = int(delta * trans[0])  # startX
                    r_met = int(delta * trans[2])  # endX

                    if delta == 0:
                        u_met = trans[1]
                        d_met = trans[3]
                    else:
                        u_met = int(trans[1] / delta)  # startY
                        d_met = int(trans[3] / delta)  # endY

                    l_seen = False
                    u_seen = False
                    r_seen = False
                    d_seen = False


                    def second_more_distant(x, a, b):
                        """
                        Return True if x is closer to a w.r.t. b
                        """
                        da = math.dist(x, a)
                        db = math.dist(x, b)
                        return da <= db


                    if trans[1] <= l_met <= trans[3] and second_more_distant((trans[0], l_met), p2, p1):
                        l_seen = True
                        cv2.circle(img, (trans[0] + drag_x, l_met + drag_y), 4, (0, 255, 255), -1)

                    if trans[1] <= r_met <= trans[3] and second_more_distant((trans[2], r_met), p2, p1):
                        r_seen = True
                        cv2.circle(img, (trans[2] + drag_x, r_met + drag_y), 4, (0, 255, 255), -1)

                    if trans[0] <= u_met <= trans[2] and second_more_distant((u_met, trans[1]), p2, p1):
                        u_seen = True
                        cv2.circle(img, (u_met + drag_x, trans[1] + drag_y), 4, (0, 255, 255), -1)

                    if trans[0] <= d_met <= trans[2] and second_more_distant((d_met, trans[2]), p2, p1):
                        d_seen = True
                        cv2.circle(img, (d_met + drag_x, trans[3] + drag_y), 4, (0, 255, 255), -1)

                    if l_seen or r_seen or u_seen or d_seen:
                        cv2.putText(img, 'FOCUS on ' + o, (30, 30), font, 2, (255, 255, 128), 3)
                        saw = True

            # TODO STARTING BLUE LINE
                    p1 = x1
                    p2 = x2

                    drag_x = p1[0]
                    drag_y = p1[1]

                    p1 = (p1[0] - drag_x, p1[1] - drag_y)
                    p2 = (p2[0] - drag_x, p2[1] - drag_y)

                    if (p1[0] - p2[0]) != 0:  # to avoid division with 0
                        delta = (p1[1] - p2[1]) / (p1[0] - p2[0])
                    else:
                        delta = 999

                    for o in objects.keys():  # translate box
                        trans = (objects[o][0] - drag_x,  # startX
                                 objects[o][1] - drag_y,  # startY
                                 objects[o][2] - drag_x,  # endX
                                 objects[o][3] - drag_y)  # endY

                        l_met = int(delta * trans[0])  # startX
                        r_met = int(delta * trans[2])  # endX

                        if delta == 0:
                            u_met = trans[1]
                            d_met = trans[3]
                        else:
                            u_met = int(trans[1] / delta)  # startY
                            d_met = int(trans[3] / delta)  # endY

                        l_seen = False
                        u_seen = False
                        r_seen = False
                        d_seen = False


                        def second_more_distant(x, a, b):
                            """
                            Return True if x is closer to a w.r.t. b
                            """
                            da = math.dist(x, a)
                            db = math.dist(x, b)
                            return da <= db


                        if trans[1] <= l_met <= trans[3] and second_more_distant((trans[0], l_met), p2, p1):
                            l_seen = True
                            cv2.circle(img, (trans[0] + drag_x, l_met + drag_y), 4, (255, 255, 0), -1)

                        if trans[1] <= r_met <= trans[3] and second_more_distant((trans[2], r_met), p2, p1):
                            r_seen = True
                            cv2.circle(img, (trans[2] + drag_x, r_met + drag_y), 4, (255, 255, 0), -1)

                        if trans[0] <= u_met <= trans[2] and second_more_distant((u_met, trans[1]), p2, p1):
                            u_seen = True
                            cv2.circle(img, (u_met + drag_x, trans[1] + drag_y), 4, (255, 255, 0), -1)

                        if trans[0] <= d_met <= trans[2] and second_more_distant((d_met, trans[2]), p2, p1):
                            d_seen = True
                            cv2.circle(img, (d_met + drag_x, trans[3] + drag_y), 4, (255, 255, 0), -1)

                        if l_seen or r_seen or u_seen or d_seen:
                            cv2.putText(img, 'FOCUS on ' + o, (30, 30), font, 2, (255, 255, 128), 3)
                            saw = True

                if not saw:
                    cv2.putText(img, 'NOT focus', (30, 30), font, 2, (255, 255, 128), 3)

            cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
