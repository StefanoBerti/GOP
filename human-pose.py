import torch
import cv2
import posenet
from utils import params
from multiprocessing.connection import Client


def main():
    # Connect to master and net
    # print("Connecting to master...")
    # master_address = ('localhost', params["hpe-master-port"])
    # master_conn = Client(master_address, authkey=b'secret password')
    # print("Connected")
    # print("Connecting to net...")
    # net_address = ('localhost', params["hpe-net-port"])
    # net_conn = Client(net_address,  authkey=b'secret password')
    # print("Connected")

    # Load model
    model = posenet.load_model(params["model"])
    model = model.cuda()
    output_stride = model.output_stride

    # Set cam connection
    cap = cv2.VideoCapture(params["cam_id"])
    cap.set(3, params["cam_width"])
    cap.set(4, params["cam_height"])

    frame_count = 0
    # Main Loop
    while True:

        # Get image from webcam
        ret, img = cap.read()
        input_image, display_image, output_scale = posenet.read_cap(ret, img,
                                                                    scale_factor=params["scale_factor"],
                                                                    output_stride=output_stride)

        # Get prediction
        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=params["max_pose_detection"],
                min_pose_score=params["min_pose_score"])
        keypoint_coords *= output_scale

        # Send results to both master (to collect data) and net (to do inference)
        # master_conn.send(keypoint_coords)
        # net_conn.send(keypoint_coords)

        # Display estimated skeleton on human
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)
        cv2.imshow('posenet', overlay_image)

        # Necessary to work
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
