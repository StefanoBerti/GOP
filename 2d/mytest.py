import cv2
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import torch
import utils
import hopenet
import torchvision
from facenet_pytorch import MTCNN
import time
import numpy as np
from utils import params, second_more_distant
from depth import RealSense
# import yarp

# for yarp
# yarpview --name example
# yarp connect tcp://icubSim/cam/left/rgbImage:o /example

if __name__ == "__main__":

    # Connect to webcam
    # cap = cv2.VideoCapture(params["cam_id"])
    cap = RealSense()
    font = cv2.FONT_HERSHEY_SIMPLEX

    ############################
    # TODO TEST CONNECT GAZEBO #
    ############################
    # yarp.Network.init()
    # input_port = yarp.Port()

    # connect for arm control
    # props = yarp.Property()
    # props.put("device", "remote_controlboard")
    # props.put("local", "/client/head")
    # props.put("remote", "/icubSim/head")
    # headDriver = yarp.PolyDriver(props)
    # riPos = headDriver.viewIPositionControl()
    # riVel = headDriver.viewIVelocityControl()
    # riEnc = headDriver.viewIEncoders()
    #
    # # retrieve number of joints
    # jnts = riPos.getAxes()
    # print('left: Controlling', jnts, 'joints')
    #
    # # increase joint speed
    # sp = 100
    # riPos.setRefSpeed(0, sp)
    # riPos.setRefSpeed(1, sp)
    # riPos.setRefSpeed(2, sp)

    ##################################
    # INITIALIZE HEAD POSE ESTIMATOR #
    ##################################

    # Initialize tensor of ids
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device='cuda')

    # Normalizer
    transformations = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Loading ResNET50
    print("Loading head pose estimation model")
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    saved_state_dict = torch.load(params["PE_model_path"])
    model.load_state_dict(saved_state_dict)
    model.cuda()
    model.eval()

    ############################
    # INITIALIZE FACE DETECTOR #
    ############################

    # Loading customized face detector
    mtcnn = MTCNN(keep_all=True, device="cuda:0")

    #################################
    # INITIALIZE OBJECT RECOGNITION #
    #################################

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(params["OD_prototxt_path"],
                                   params["OD_model_path"])

    #############
    # MAIN LOOP #
    #############

    frame_num = 1
    start_time = time.time()

    while True:

        # get frame
        # ret, frame = cap.read()
        frame, depth = cap.read()

        #########################
        # OBJECT DETECTION PART #
        #########################

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        objects = {}
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            if confidence > params["object_min_conf"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] == "person":  # useless in our case
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
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, COLORS[idx], 2)

                objects[CLASSES[idx]] = (startX, startY, endX, endY)

        ####################
        # FACE RECOGNITION #
        ####################

        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Look for faces inside it
        boxes, confidences = mtcnn.detect(Image.fromarray(cv2_frame))

        # If at least one face is recognized
        if boxes is not None:
            for i, elem in enumerate(boxes):
                # Get x_min, y_min, x_max, y_max, conf
                x_min = elem[0]
                y_min = elem[1]
                x_max = elem[2]
                y_max = elem[3]
                confidence = confidences[i].item()

                tdx = None
                tdy = None
                x_ = None
                y_ = None

                if confidence > params["face_min_conf"]:  # threshold on face confidence
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)
                    x_min -= 2 * bbox_width / 4
                    x_max += 2 * bbox_width / 4
                    y_min -= 3 * bbox_height / 4
                    y_max += bbox_height / 4
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(frame.shape[1], x_max)
                    y_max = min(frame.shape[0], y_max)

                    # TODO additional information, remove it
                    confidence = "{:.2f}".format(confidence)
                    cv2.putText(frame, confidence, (int(boxes[0][0]), int(boxes[0][1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                color=(255, 255, 128))
                    cv2.line(frame, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][1])),
                             color=(255, 255, 128))
                    cv2.line(frame, (int(boxes[0][2]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])),
                             color=(255, 255, 128))
                    cv2.line(frame, (int(boxes[0][2]), int(boxes[0][3])), (int(boxes[0][0]), int(boxes[0][3])),
                             color=(255, 255, 128))
                    cv2.line(frame, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][0]), int(boxes[0][3])),
                             color=(255, 255, 128))

                    ########################
                    # HEAD POSE ESTIMATION #
                    ########################

                    # Crop image
                    img = cv2_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                    img = Image.fromarray(img)

                    # Transform
                    img = transformations(img)
                    img_shape = img.size()
                    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                    img = Variable(img).to(device='cuda')

                    # Get prediction
                    yaw, pitch, roll = model(img)
                    yaw_predicted = F.softmax(yaw, dim=1)
                    pitch_predicted = F.softmax(pitch, dim=1)
                    roll_predicted = F.softmax(roll, dim=1)

                    # Get continuous predictions in degrees.
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                    # TODO TEST AND REMOVE
                    print("YAW: ", yaw_predicted.item(), ", PITCH: ", pitch_predicted.item(), ", ROLL: ", roll_predicted.item())
                    # riPos.positionMove(0, pitch_predicted.item())  # up and down
                    # riPos.positionMove(1, roll_predicted.item())  # tilt left right
                    # riPos.positionMove(2, yaw_predicted.item())  # turn left right
                    # TODO END TEST

                    # Print new frame with cube and axis
                    # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted,
                    #                      (x_min + x_max) / 2, (y_min + y_max) / 2, size=bbox_width)
                    frame, tdx, tdy, x_, y_ = utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted,
                                                              tdx=(x_min + x_max) / 2,
                                                              tdy=(y_min + y_max) / 2,
                                                              size=1000)  # , size=bbox_height / 2)

                    # Plot expanded bounding box
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

                    ############################
                    # HEAD POSITION ESTIMATION #
                    ############################

                    depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

                    head_x = int((x_min + x_max) / 2)
                    head_y = int((y_min + y_max) / 2)
                    distance = depth[head_y][head_x] / 1500
                    print("x: ", head_x, ", y: ", head_y, ", z: ", distance)

                    cv2.circle(depth_colored, (head_x, head_y), 4, (255, 255, 0), -1)
                    cv2.imshow('depth', depth_colored)

                    ###################
                    # FOCUS ON OBJECT #
                    ###################

                    # rename
                    p1 = (tdx, tdy)
                    p2 = (x_, y_)
                    drag_x = p1[0]
                    drag_y = p1[1]

                    # compute offset point (to understand the direction of the sight)
                    off_x = int(((x_ - tdx) * 0.1) + tdx)
                    off_y = int(((y_ - tdy) * 0.1) + tdy)
                    # cv2.circle(frame, (off_x, off_y), 4, (0, 255, 255), -1)

                    # translate points
                    p1 = (p1[0] - drag_x, p1[1] - drag_y)
                    p2 = (p2[0] - drag_x, p2[1] - drag_y)
                    off_x = off_x - drag_x
                    off_y = off_y - drag_y

                    # Compute delta
                    if (p1[0] - p2[0]) != 0:  # to avoid division with 0
                        delta = (p1[1] - p2[1]) / (p1[0] - p2[0])
                    else:
                        delta = 999

                    # Loop over objects
                    saw = False
                    n_focus = 0
                    for o in objects.keys():  # translate box

                        # Translate object's box
                        trans = (int(objects[o][0] - drag_x),  # startX
                                 int(objects[o][1] - drag_y),  # startY
                                 int(objects[o][2] - drag_x),  # endX
                                 int(objects[o][3] - drag_y))  # endY

                        # Compute meeting points
                        l_met = int(delta * trans[0])  # startX
                        r_met = int(delta * trans[2])  # endX

                        # Watching vertically
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

                        drag_x = int(drag_x)
                        drag_y = int(drag_y)

                        off = (off_x, off_y)

                        if trans[1] <= l_met <= trans[3] and second_more_distant((trans[0], l_met), off, p1):
                            l_seen = True
                            cv2.circle(frame, (trans[0] + drag_x, l_met + drag_y), 4, (0, 255, 255), -1)

                        if trans[1] <= r_met <= trans[3] and second_more_distant((trans[2], r_met), off, p1):
                            r_seen = True
                            cv2.circle(frame, (trans[2] + drag_x, r_met + drag_y), 4, (0, 255, 255), -1)

                        if trans[0] <= u_met <= trans[2] and second_more_distant((u_met, trans[1]), off, p1):
                            u_seen = True
                            cv2.circle(frame, (u_met + drag_x, trans[1] + drag_y), 4, (0, 255, 255), -1)

                        if trans[0] <= d_met <= trans[2] and second_more_distant((d_met, trans[2]), off, p1):
                            d_seen = True
                            cv2.circle(frame, (d_met + drag_x, trans[3] + drag_y), 4, (0, 255, 255), -1)

                        if l_seen or r_seen or u_seen or d_seen:
                            n_focus += 1
                            # cv2.putText(frame, 'FOCUS on ' + o, (30, 30), font, 2, (255, 255, 128), 3)
                            cv2.putText(frame, 'FOCUS on ' + o, (20, 40 + n_focus*30), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        color=(255, 255, 128))
                            start_x = trans[0] + drag_x
                            end_x = trans[2] + drag_x
                            start_y = trans[1] + drag_y
                            end_y = trans[3] + drag_y
                            to_apply = frame[start_y:end_y, start_x:end_x]
                            white_rect = np.ones(to_apply.shape, dtype=np.uint8) * 255
                            res = cv2.addWeighted(to_apply, 0.5, white_rect, 0.5, 1.0)

                            if res is not None:
                                frame[start_y:end_y, start_x:end_x] = res
                            else:
                                print("Res is None")
                            saw = True

        # Print fps counter
        fps = frame_num / (time.time() - start_time)
        fps = "{:.2f}".format(fps)
        cv2.putText(frame, "FPS: " + str(fps), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 255, 128))
        frame_num += 1

        cv2.imshow('faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
