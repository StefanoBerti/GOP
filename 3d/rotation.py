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
from utils import params
from realsense import RealSense


class Rotation:
    def __init__(self, cam, debug_show=False):
        self.cam = cam
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.frame_num = 1
        self.start_time = time.time()
        self.debug_show = debug_show

        # Load face detector
        self.face_net = MTCNN(keep_all=True, device="cuda:0")

        # Load pose estimator
        self.idx_tensor = torch.FloatTensor([k for k in range(66)]).to(device='cuda')
        self.normalizer = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        self.pose_net = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        saved_state_dict = torch.load(params["PE_model_path"])
        self.pose_net.load_state_dict(saved_state_dict)
        self.pose_net.cuda()
        self.pose_net.eval()

    def get_rpy(self, frame, _):
        ####################
        # FACE RECOGNITION #
        ####################

        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, confidences = self.face_net.detect(Image.fromarray(cv2_frame))

        pose = None
        frame_num = 0
        start_time = time.time()

        # If at least one face is recognized
        if boxes is not None:
            for i, elem in enumerate(boxes):
                # Get x_min, y_min, x_max, y_max, conf
                x_min = elem[0]
                y_min = elem[1]
                x_max = elem[2]
                y_max = elem[3]
                confidence = confidences[i].item()

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

                    if self.debug_show:
                        confidence = "{:.2f}".format(confidence)
                        cv2.putText(frame, confidence, (int(boxes[0][0]), int(boxes[0][1])), cv2.FONT_HERSHEY_COMPLEX,
                                    1,
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
                    img = self.normalizer(img)
                    img_shape = img.size()
                    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                    img = Variable(img).to(device='cuda')

                    # Get prediction
                    yaw, pitch, roll = self.pose_net(img)
                    yaw_predicted = F.softmax(yaw, dim=1)
                    pitch_predicted = F.softmax(pitch, dim=1)
                    roll_predicted = F.softmax(roll, dim=1)

                    # Get continuous predictions in degrees.
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
                    roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

                    pose = (yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item())

                    if self.debug_show:
                        frame, tdx, tdy, x_, y_ = utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted,
                                                                  tdx=(x_min + x_max) / 2,
                                                                  tdy=(y_min + y_max) / 2,
                                                                  size=1000)  # , size=bbox_height / 2)
        if self.debug_show:
            # Print fps counter
            fps = frame_num / (time.time() - start_time)
            fps = "{:.2f}".format(fps)
            cv2.putText(frame, "FPS: " + str(fps), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 255, 128))
            frame_num += 1
            cv2.imshow('faces', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()

        return pose


if __name__ == "__main__":
    rs = RealSense()
    rot = Rotation(rs, debug_show=True)
