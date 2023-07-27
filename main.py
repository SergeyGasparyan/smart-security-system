# Import necessary libraries
import argparse
import os
import sys

import cv2
import numpy as np
import torch

from custom_utils import YOLOv5Detector, draw_boxes, get_config, xyxy2xywh
from deep_sort import build_tracker

# Check if CUDA is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


class SystemTracker:
    def __init__(self, args) -> None:
        """
        Initializes the SystemTracker object.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.

        Returns:
            None
        """
        self.args = args
        print("[+] Starting the algorithm...")

        # Initialize video capture from camera or file
        if args.cam != -1:
            print("[+] Opening the camera...")
            self.cap = cv2.VideoCapture(args.cam)
            print("[+] Done")
        elif args.input_path != "":
            print("[+] Loading the video")
            self.cap = cv2.VideoCapture(args.input_path)
            print("[+] Done")
        else:
            print("[-] Give the input path for the video or use the camera!")
            sys.exit()

        # Create a display window if required
        if args.display:
            cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Display", args.display_width, args.display_height)

        # Create a folder to save the results if required
        if args.save_path:
            print("[+] Creating folder to save the results")
            os.makedirs(args.save_path, exist_ok=True)
            self.save_video_path = os.path.join(args.save_path, "result.mp4")

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
            self.im_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.writer = cv2.VideoWriter(
                self.save_video_path,
                fourcc,
                self.cap.get(cv2.CAP_PROP_FPS),
                (self.im_width, self.im_height),
            )
            print("[+] Done")

        # Initialize YOLOv5 object detection model
        print("[+] Initializing YOLOv5 model")
        self.detector = YOLOv5Detector(args.weights, args.conf_thres, device)
        print("[+] Done")

        # Initialize DeepSORT object tracking model
        print("[+] Initializing DeepSORT model")
        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)
        use_cuda = device != "cpu"
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        print("[+] Done\n")

    def run(self):
        """
        Runs the tracking algorithm on the provided video stream or camera feed.

        Returns:
            None
        """
        idx_frame = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if idx_frame % self.args.frame_interval == 0:
                bounding_boxes, confs = self.detector.detect_people(frame)
                if len(bounding_boxes) > 0:
                    outputs = self.deepsort.update(
                        xyxy2xywh(bounding_boxes), confs, frame
                    )
                else:
                    outputs = torch.zeros((0, 5))
                last_output = outputs
            else:
                outputs = last_output

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                frame = draw_boxes(frame, bbox_xyxy, identities)

                # Add FPS information on output video
                text_scale = max(1, frame.shape[1] // 1600)
                cv2.putText(
                    frame,
                    f"frame: {idx_frame}",
                    (20, 20 + text_scale),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 0, 255),
                    thickness=2,
                )

            if self.args.display:
                cv2.imshow("Display", frame)
                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    break

            if self.args.save_path:
                self.writer.write(frame)

            idx_frame += 1
            print(f"Frame {idx_frame}")

        # Release video capture and writer resources
        self.cap.release()
        self.writer.release()
        print("Done")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="", help="source")
    parser.add_argument(
        "--save_path", type=str, default="output/", help="output folder"
    )
    parser.add_argument("--frame_interval", type=int, default=2)
    parser.add_argument("--fourcc", type=str, default="mp4v", help="output video codec")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--weights", type=str, default="yolov5s", help="model.pt path")
    parser.add_argument(
        "--conf-thres", type=float, default=0.45, help="object confidence threshold"
    )
    parser.add_argument(
        "--config_deepsort", type=str, default="./deep_sort/deep_sort.yaml"
    )

    args = parser.parse_args()

    # Initialize and run the SystemTracker
    cap_tracker = SystemTracker(args)
    cap_tracker.run()
