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

# TODO ############################################################################################################
# UNIRE MYTEST E HEAD POSE ESTIMATION
# TODO ############################################################################################################

if __name__ == "__main__":

    # Connect to webcam
    face_min_conf = 0.2
    webcam = 2
    cap = cv2.VideoCapture(webcam)

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
    saved_state_dict = torch.load("hopenet_robust_alpha1.pkl")
    model.load_state_dict(saved_state_dict)
    model.cuda()
    model.eval()

    # Loading customized face detector
    mtcnn = MTCNN(keep_all=True, device="cuda:0")

    # Main loop
    frame_num = 1
    while True:

        # Get frame and convert it
        ret, frame = cap.read()
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
                # conf = det.confidence
                confidence = confidences[i].item()

                if confidence > face_min_conf:  # threshold on face confidence
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
                    # Crop image
                    img = cv2_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                    img = Image.fromarray(img)

                    # TODO additional information, remove it
                    confidence = "{:.2f}".format(confidence)
                    cv2.putText(frame, confidence, (boxes[0][0], boxes[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color=3)
                    cv2.line(frame, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][1]), color=3)
                    cv2.line(frame, (boxes[0][2], boxes[0][1]), (boxes[0][2], boxes[0][3]), color=3)
                    cv2.line(frame, (boxes[0][2], boxes[0][3]), (boxes[0][0], boxes[0][3]), color=3)
                    cv2.line(frame, (boxes[0][0], boxes[0][1]), (boxes[0][0], boxes[0][3]), color=3)

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
                    print(yaw_predicted.item(), " ", pitch_predicted.item(), " ", roll_predicted.item())

                    # Print new frame with cube and axis
                    # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted,
                    #                      (x_min + x_max) / 2, (y_min + y_max) / 2, size=bbox_width)
                    utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min + x_max) / 2,
                                    tdy=(y_min + y_max) / 2, size=bbox_height / 2)
                    # Plot expanded bounding box
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

        cv2.imshow('faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_num += 1

    cap.release()
