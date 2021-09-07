import numpy as np
import pyrealsense2 as rs
import cv2


class RealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)

        self.align = rs.align(rs.stream.color)

    def intrinsics(self):
        return self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def stop(self):
        self.pipeline.stop()


if __name__ == "__main__":
    camera = RealSense()

    while True:
        color_image, depth_image = camera.read()
        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        print(depth_image[int(depth_image.shape[1]/2)][int(depth_image.shape[0]/2)] / 1500)
        cv2.circle(depth_colored, (int(depth_image.shape[1]/2), int(depth_image.shape[0]/2)), 4, (255, 255, 0), -1)

        cv2.imshow('color', color_image)
        cv2.imshow('depth', depth_colored)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
