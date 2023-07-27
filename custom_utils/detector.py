import torch


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    
    return y


class YOLOv5Detector:
    def __init__(self, weights='yolov5s', confidence_threshold=0.45, device='cpu'):
        # Load YOLOv5 model from Torch Hub
        self.model = torch.hub.load('ultralytics/yolov5', weights, pretrained=True).to(device)
        self.confidence_threshold = confidence_threshold

    def detect_people(self, frame):
        # Perform object detection on the frame using YOLOv5
        # Run the detection
        results = self.model(frame)

        # Get bounding boxes, confidence scores, and class labels
        boxes = results.pred[0][:, :4].cpu().numpy()
        confidences = results.pred[0][:, 4].cpu().numpy()
        class_labels = results.pred[0][:, 5].cpu().numpy()

        # Filter detections to keep only person class (class 0)
        person_indices = class_labels == 0
        person_boxes = boxes[person_indices]
        person_confidences = confidences[person_indices]

        high_confidence_indices = person_confidences > self.confidence_threshold

        high_confidence_boxes = person_boxes[high_confidence_indices]
        high_confidences = person_confidences[high_confidence_indices]

        return high_confidence_boxes, high_confidences
