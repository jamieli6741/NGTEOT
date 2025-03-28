import cv2
import numpy as np
import torch
import imutils
import argparse
from segment_anything import sam_model_registry, SamPredictor
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
import os
import matplotlib.pyplot as plt

class DeepSORTWrapper:
    def __init__(self, model_path="mars-small128.pb", max_cosine_distance=0.2,
                 nn_budget=100, max_age=30, min_hits=3):
        """Initialize DeepSORT tracker with feature extractor and settings."""
        # Initialization process from https://learnopencv.com/understanding-multiple-object-tracking-using-deepsort/
        self.encoder = gdet.create_box_encoder(model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=max_age, n_init=min_hits)

        self.current_bbox = None
        self.track_id = None
        self.frame_count = 0
        self.initial_bbox = None

    def init(self, frame, bbox):
        """Initialize tracker with first frame and bounding box."""
        self.frame_count = 0
        x, y, w, h = bbox
        self.initial_bbox = bbox
        self.current_bbox = bbox

        # DeepSORT expects boxes in format [x1, y1, x2, y2]
        boxes_for_encoder = np.array([[x, y, x+w, y+h]])
        features = self.encoder(frame, boxes_for_encoder)

        bbox_tlwh = np.array([x, y, w, h])
        confidence = 1.0  # High confidence for manually selected region
        detection = Detection(bbox_tlwh, confidence, features[0])

        self.tracker.predict()
        self.tracker.update([detection])

        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                self.track_id = track.track_id
                break

        return True

    def update(self, frame):
        """Update tracker with new frame."""
        self.frame_count += 1

        self.tracker.predict()

        # Use current bbox to guide detection
        if self.current_bbox is not None:
            x, y, w, h = self.current_bbox

            # Extract features from detection area
            # We can use a slightly larger region around the current bbox for robustness
            search_margin = 0.2  # 20% search margin
            x_search = max(0, int(x - w * search_margin))
            y_search = max(0, int(y - h * search_margin))
            w_search = int(w * (1 + 2 * search_margin))
            h_search = int(h * (1 + 2 * search_margin))

            # Ensure search area is within frame boundaries
            w_search = min(w_search, frame.shape[1] - x_search)
            h_search = min(h_search, frame.shape[0] - y_search)

            # DeepSORT expects boxes in format [x1, y1, x2, y2]
            boxes_for_encoder = np.array([[x_search, y_search,
                                           x_search + w_search,
                                           y_search + h_search]])

            # Extract appearance features
            features = self.encoder(frame, boxes_for_encoder)

            # Create a Detection object
            bbox_tlwh = np.array([x_search, y_search, w_search, h_search])
            confidence = 0.9  # High confidence
            detection = Detection(bbox_tlwh, confidence, features[0])

            # Update tracker with this detection
            self.tracker.update([detection])

        # Find our target track
        best_track = None
        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                if self.track_id is None or track.track_id == self.track_id:
                    best_track = track
                    self.track_id = track.track_id
                    break

        # Movement Validation
        if best_track is not None:
            bbox = best_track.to_tlwh()  # Format: (top left x, top left y, width, height)

            # Calculate movement from previous position
            if self.current_bbox is not None:
                prev_x, prev_y, prev_w, prev_h = self.current_bbox
                new_x, new_y, new_w, new_h = bbox

                # Calculate center points
                prev_center_x = prev_x + prev_w/2
                prev_center_y = prev_y + prev_h/2
                new_center_x = new_x + new_w/2
                new_center_y = new_y + new_h/2

                # Calculate distance
                distance = np.sqrt((new_center_x - prev_center_x)**2 +
                                       (new_center_y - prev_center_y)**2)

                # If distance is too large, consider it a tracking error
                max_reasonable_distance = max(prev_w, prev_h) * 0.5
                if distance > max_reasonable_distance:
                    print(f"Suspicious movement detected: {distance:.2f} pixels")
                    return False, self.current_bbox 

            # Convert to OpenCV format (x, y, w, h)
            self.current_bbox = (int(bbox[0]), int(bbox[1]),
                                 int(bbox[2]), int(bbox[3]))
            return True, self.current_bbox
        else:
            # If track was lost, return the last known position
            return False, self.current_bbox

class TrackingEvaluator:
    """Evaluates tracking performance without ground truth."""
    def __init__(self):
        self.metrics_history = []
        self.template = None
        self.bbox_history = []
        self.initial_size = None
        self.frame_count = 0
        self.drift_detected = False
        self.consecutive_low_similarity = 0
        self.consecutive_threshold = 5  # Number of consecutive frames with low similarity to consider as drift
        self.drift_frame = None

    def initialize(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        self.template = frame[y:y+h, x:x+w].copy()
        self.bbox_history = [bbox]
        self.initial_size = (w, h)

        # Calculate initial histogram
        self.initial_hist = self.calculate_color_histogram(frame, bbox)

        # Store initial frame for reference
        self.initial_frame = frame.copy()

    def update(self, frame, bbox, success):
        self.frame_count += 1

        if not success:
            self.metrics_history.append({
                'tracking_success': False,
                'appearance_similarity': 0,
                'bbox_jitter': float('inf'),
                'aspect_ratio_consistency': float('inf'),
                'size_consistency': float('inf'),
                'drift_detected': True
            })
            return

        # Store bbox history
        self.bbox_history.append(bbox)

        # Calculate metrics
        similarity = self.calculate_appearance_similarity(frame, bbox)
        jitter = self.calculate_bbox_jitter()
        ar_consistency = self.calculate_aspect_ratio_consistency()
        size_consistency = self.calculate_size_consistency(bbox)

        # Drift detection logic
        if similarity < 0.4:
            self.consecutive_low_similarity += 1
        else:
            self.consecutive_low_similarity = 0

        # Mark as drift if we have several consecutive frames with low similarity
        drift_detected = self.drift_detected
        if self.consecutive_low_similarity >= self.consecutive_threshold and not self.drift_detected:
            self.drift_detected = True
            self.drift_frame = self.frame_count
            print(f"Drift detected at frame {self.frame_count} (similarity: {similarity:.3f})")
            drift_detected = True

        # Store metrics
        self.metrics_history.append({
            'tracking_success': True,
            'appearance_similarity': similarity,
            'bbox_jitter': jitter,
            'aspect_ratio_consistency': ar_consistency,
            'size_consistency': size_consistency,
            'drift_detected': drift_detected
        })

    def calculate_appearance_similarity(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]

        # Ensure coordinates are within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)

        # Skip if bounding box is invalid
        if w <= 0 or h <= 0:
            return 0

        current_region = frame[y:y+h, x:x+w]

        # Calculate histogram similarity to initial template
        try:
            hist_current = cv2.calcHist([cv2.cvtColor(current_region, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist_current, hist_current, 0, 1, cv2.NORM_MINMAX)

            similarity = cv2.compareHist(self.initial_hist, hist_current, cv2.HISTCMP_CORREL)
            return similarity
        except Exception:
            return 0  # Return low similarity on error

    def calculate_color_histogram(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]

        # Ensure coordinates are within frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)

        # Skip if bounding box is invalid
        if w <= 0 or h <= 0:
            return None

        roi = frame[y:y+h, x:x+w]

        # Calculate histogram in HSV space
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        return hist

    def calculate_bbox_jitter(self):
        if len(self.bbox_history) < 2:
            return 0

        prev_bbox = self.bbox_history[-2]
        curr_bbox = self.bbox_history[-1]

        prev_center = (prev_bbox[0] + prev_bbox[2]/2, prev_bbox[1] + prev_bbox[3]/2)
        curr_center = (curr_bbox[0] + curr_bbox[2]/2, curr_bbox[1] + curr_bbox[3]/2)

        distance = np.sqrt((prev_center[0] - curr_center[0])**2 + (prev_center[1] - curr_center[1])**2)
        return distance

    def calculate_aspect_ratio_consistency(self):
        if len(self.bbox_history) < 2:
            return 0

        # Get last few bounding boxes to reduce noise
        recent_boxes = self.bbox_history[-5:] if len(self.bbox_history) >= 5 else self.bbox_history

        aspect_ratios = [bbox[2] / max(bbox[3], 1) for bbox in recent_boxes]  # Avoid division by zero
        return np.std(aspect_ratios) if len(aspect_ratios) > 1 else 0

    def calculate_size_consistency(self, bbox):
        """Calculate how consistent the current box size is with the initial size"""
        x, y, w, h = bbox
        initial_w, initial_h = self.initial_size

        # Size ratio (current/initial)
        w_ratio = w / initial_w if initial_w > 0 else float('inf')
        h_ratio = h / initial_h if initial_h > 0 else float('inf')

        # Perfect consistency = 1.0, higher or lower indicates size change
        size_consistency = abs(1 - ((w_ratio + h_ratio) / 2))
        return size_consistency

    def get_overall_performance(self):
        if not self.metrics_history:
            return None

        # Count various metrics
        total_frames = len(self.metrics_history)
        failure_frames = sum(1 for m in self.metrics_history if not m['tracking_success'])
        drift_frames = sum(1 for m in self.metrics_history if m['drift_detected'])

        # Calculate effective tracking length (frames before drift)
        effective_tracking = self.drift_frame if self.drift_detected else total_frames
        effective_tracking_percent = effective_tracking / total_frames if total_frames > 0 else 0

        # Filter out infinity values for proper calculation
        valid_jitters = [m['bbox_jitter'] for m in self.metrics_history
                         if m['tracking_success'] and m['bbox_jitter'] != float('inf') and not m['drift_detected']]
        valid_similarities = [m['appearance_similarity'] for m in self.metrics_history
                              if m['tracking_success'] and not m['drift_detected']]

        # Calculate composite score - higher is better
        # This weights effective tracking length highly and penalizes drift
        avg_similarity = np.mean(valid_similarities) if valid_similarities else 0
        avg_jitter = np.mean(valid_jitters) if valid_jitters else float('inf')
        drift_percentage = (drift_frames / total_frames) * 100 if total_frames > 0 else 100
        failure_rate = (failure_frames / total_frames) * 100 if total_frames > 0 else 100

        # === 配置化权重：你可以在这里增删指标，调整权重 ===
        metric_weights = {
            'effective_tracking_percentage': 0.5,  # 越高越好
            'avg_appearance_similarity': 0.3,  # 越高越好
            'drift_percentage': -0.2,  # 越低越好（负相关）
            # 'avg_bbox_jitter': -0.1,                  # 可选指标
            # 'failure_rate': -0.2                      # 可选指标
        }

        # 组合得分：每个指标先归一化（百分比/100），再按权重累加
        composite_score = 0
        for metric_name, weight in metric_weights.items():
            if metric_name == 'effective_tracking_percentage':
                value = effective_tracking_percent  # 已经是0~1
            elif metric_name == 'avg_appearance_similarity':
                value = avg_similarity  # 一般也是0~1
            elif metric_name == 'drift_percentage':
                value = drift_percentage / 100
            elif metric_name == 'failure_rate':
                value = failure_rate / 100
            elif metric_name == 'avg_bbox_jitter':
                value = avg_jitter / 100  # 你可以根据最大期望 jitter 做归一
            else:
                value = 0  # 不识别的字段跳过
            composite_score += weight * value

        avg_metrics = {
            'avg_appearance_similarity': avg_similarity,
            'avg_bbox_jitter': avg_jitter,
            'drift_frames': drift_frames,
            'drift_percentage': drift_percentage,
            'effective_tracking': effective_tracking,
            'effective_tracking_percentage': effective_tracking_percent * 100,
            'failure_rate': failure_rate,
            'composite_score': composite_score
        }

        return avg_metrics

def run_tracking_comparison(video_path, trackers_list, compress_rate=5, compute="cpu", deepsort_model="mars-small128.pb"):
    """
    Run tracking comparison across multiple trackers.

    Args:
        video_path: Path to the input video
        trackers_list: List of tracker names to compare
        compress_rate: Frame compression rate
        compute: Computation device (cpu or cuda)
        deepsort_model: Path to DeepSORT model

    Returns:
        Dictionary of metrics for each tracker
    """
    # Setup device
    if compute == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = "cuda"
        print("Using CUDA for processing")
    else:
        device = "cpu"
        print("Using CPU for processing")

    # Define available trackers
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.legacy.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.legacy.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.legacy.TrackerTLD_create,
        "medianflow": cv2.legacy.TrackerMedianFlow_create,
        "mosse": cv2.legacy.TrackerMOSSE_create,
        "deepsort": lambda: DeepSORTWrapper(model_path=deepsort_model)
    }

    # Validate requested trackers
    valid_trackers = []
    for tracker_name in trackers_list:
        tracker_name = tracker_name.lower()
        if tracker_name in OPENCV_OBJECT_TRACKERS:
            valid_trackers.append(tracker_name)
        else:
            print(f"Warning: Tracker '{tracker_name}' not found. Skipping.")

    if not valid_trackers:
        print("No valid trackers specified. Using CSRT as default.")
        valid_trackers = ["csrt"]

    print(f"Running comparison with these trackers: {valid_trackers}")

    # Load video and extract frames
    print(f"Loading video: {video_path}")
    vs = cv2.VideoCapture(video_path)

    if not vs.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Extract frames with compression
    print(f"Extracting frames (compression rate: {compress_rate})")
    frame_list = []
    frame_cnt = 0

    while vs.isOpened():
        ret, frame = vs.read()
        if ret and frame_cnt % compress_rate == 0:
            frame = imutils.resize(frame, width=500)
            frame_list.append(frame)
            frame_cnt = 1
        elif ret and frame_cnt % compress_rate != 0:
            frame_cnt += 1
        else:
            break

    frame_list = np.array(frame_list)
    vs.release()
    print(f"Extracted {len(frame_list)} frames")

    # Have user select ROI
    print("Select the object to track in the first frame")
    bbox = cv2.selectROI('Frame', frame_list[0], False)
    cv2.destroyAllWindows()
    (x, y, w, h) = [int(v) for v in bbox]

    # Run each tracker and collect metrics
    metrics_results = {}

    for tracker_name in valid_trackers:
        print(f"\nRunning tracker: {tracker_name.upper()}")

        # Initialize tracker and evaluator
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        evaluator = TrackingEvaluator()

        # Initialize with first frame
        tracker.init(frame_list[0], (x, y, w, h))
        evaluator.initialize(frame_list[0], (x, y, w, h))

        # Track through all frames
        bbox_list = np.zeros((len(frame_list), 4))
        bbox_list[0] = [x, y, w+x, h+y]  # Store in x1, y1, x2, y2 format

        tracking_time_start = cv2.getTickCount()

        failures = 0
        for i in range(1, len(frame_list)):
            success, bbox = tracker.update(frame_list[i])

            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                bbox_list[i] = [x, y, w+x, h+y]
                evaluator.update(frame_list[i], (x, y, w, h), True)
            else:
                failures += 1
                print(f"Frame {i}: Tracking failed")
                # Use last known position
                if i > 0:
                    bbox_list[i] = bbox_list[i-1]
                evaluator.update(frame_list[i], (0, 0, 0, 0), False)

        tracking_time_end = cv2.getTickCount()
        processing_time = (tracking_time_end - tracking_time_start) / cv2.getTickFrequency()
        fps = len(frame_list) / processing_time

        # Get metrics
        metrics = evaluator.get_overall_performance()
        metrics['failures'] = failures
        metrics['failure_rate'] = failures / (len(frame_list) - 1)
        metrics['fps'] = fps

        print(f"Tracker: {tracker_name.upper()}")
        print(f"Effective tracking: {metrics['effective_tracking']}/{len(frame_list)} frames ({metrics['effective_tracking_percentage']:.1f}%)")
        print(f"Drift detected: {metrics['drift_frames']} frames ({metrics['drift_percentage']:.1f}%)")
        print(f"Average appearance similarity: {metrics['avg_appearance_similarity']:.3f}")
        print(f"Processing speed: {fps:.1f} FPS")
        print(f"Composite score: {metrics['composite_score']:.3f}")

        metrics_results[tracker_name] = metrics

    # Generate comparative visualization
    generate_comparison_charts(metrics_results, video_path)

    return metrics_results

def generate_comparison_charts(metrics_results, video_path):
    """Generate visual comparison of tracker performance."""
    if not metrics_results:
        return

    # Create output directory for charts
    output_dir = "tracking_metrics"
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.basename(video_path).split('.')[0]
    output_base = os.path.join(output_dir, f"{video_name}_comparison")

    # Create bar charts for key metrics
    metrics_to_plot = {
        'effective_tracking_percentage': {'title': 'Effective Tracking (higher is better)', 'percentage': True},
        'drift_percentage': {'title': 'Background Drift (lower is better)', 'percentage': True},
        'avg_appearance_similarity': {'title': 'Appearance Similarity (higher is better)', 'percentage': True},
        'composite_score': {'title': 'Overall Score (higher is better)', 'percentage': False},
        'failure_rate': {'title': 'Failure Rate (%) — Lower is Better', 'percentage': True}
    }

    num_metrics = len(metrics_to_plot)
    cols = 2
    rows = (num_metrics + 1) // 2
    plt.figure(figsize=(6 * cols, 4.5 * rows))

    for i, (metric_name, config) in enumerate(metrics_to_plot.items(), 1):
        plt.subplot(rows, cols, i)

        trackers = list(metrics_results.keys())
        values = [metrics_results[t].get(metric_name, 0) for t in trackers]

        # Define color map for visual clarity
        if metric_name in ['avg_appearance_similarity']:
            colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
        elif metric_name in ['drift_percentage', 'failure_rate']:
            colors = ['green' if v < 10 else 'orange' if v < 30 else 'red' for v in values]
        elif metric_name == 'effective_tracking_percentage':
            colors = ['green' if v > 80 else 'orange' if v > 50 else 'red' for v in values]
        elif metric_name == 'composite_score':
            colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
        else:
            colors = 'skyblue'

        bars = plt.bar(trackers, values, color=colors)

        plt.title(config['title'])
        plt.xticks(rotation=45, ha='right')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            label = f"{height:.1f}%" if config['percentage'] else f"{height:.3f}"
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     label, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_base}_metrics.png")
    print(f"Comparison charts saved to {output_base}_metrics.png")

    # Save numerical results as CSV
    csv_path = f"{output_base}_metrics.csv"
    with open(csv_path, 'w') as f:
        # Write header
        metrics_header = list(next(iter(metrics_results.values())).keys())
        f.write("tracker," + ",".join(metrics_header) + "\n")

        # Write values
        for tracker, metrics in metrics_results.items():
            f.write(f"{tracker}")
            for metric in metrics_header:
                value = metrics.get(metric, "N/A")
                if isinstance(value, float):
                    f.write(f",{value:.4f}")
                else:
                    f.write(f",{value}")
            f.write("\n")

    print(f"Metrics data saved to {csv_path}")

    # Return to command line that metrics are ready
    print("\nTracking comparison completed. See results in the tracking_metrics directory.")

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
	help="OpenCV object tracker type")
ap.add_argument("-c", "--compress", type=int, default=5,
	help="compress rate(default set as 5, which read 1 frame among 5 frames from input video)")
ap.add_argument("-p", "--compute", type=str, default="cuda",
	help="platform for computing")
ap.add_argument("-d", "--deepsort_model", type=str, default="mars-small128.pb",
    help="DeepSORT feature extractor model")
#This is for performance comparision
ap.add_argument("--metrics", action="store_true",
    help="Run metrics comparison between trackers")
ap.add_argument("--trackers", nargs='+',
    help="Trackers to compare (space-separated list)")
args = vars(ap.parse_args())

if args["metrics"] and args["video"] and args["trackers"]:
    metrics_results = run_tracking_comparison(
        video_path=args["video"],
        trackers_list=args["trackers"],
        compress_rate=args["compress"],
        compute=args["compute"],
        deepsort_model=args["deepsort_model"]
    )
    exit()

vs = cv2.VideoCapture(args["video"])
compress_rate = args["compress"]
if args["compute"] == "cuda":
    torch.cuda.set_device(0) #You can edit the GPU number here
    device = "cuda"
else:
    device = args["compute"]

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create,
    "deepsort": lambda: DeepSORTWrapper(model_path=args["deepsort_model"])
}

tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# This part is the input preprocessing part, we separate the video into frames, resize the frames(for fast processing)
frame_list = []
frame_cnt = 0
while vs.isOpened():
    ret, frame = vs.read()
    if ret and frame_cnt % compress_rate == 0:
        frame = imutils.resize(frame, width=500)
        frame_list.append(frame)
        frame_cnt = 1
    elif ret and  frame_cnt % compress_rate != 0:
        frame_cnt += 1
    else:
        break
frame_list = np.array(frame_list)
vs.release()
cv2.destroyAllWindows()

# This is the Object Tracking part, with assigned tracker and bounding box drew by ROIselect, we worked out the bounding boxes for target object in the video.
bbox_list = np.zeros((len(frame_list), 4))
bbox = cv2.selectROI('Frame', frame_list[0], False)
(x, y, w, h) = [int(v) for v in bbox]
bbox_list[0] = [x, y, w+x, h+y]
tracker.init(frame_list[0], bbox)

for i in range(1, len(frame_list)):
    success, bbox = tracker.update(frame_list[i])
    if success:
        (x, y, w, h) = [int(v) for v in bbox]
        bbox_list[i] = [x, y, w+x, h+y]
    else:
        print('tracker failed.')
print("Bounding boxes finished!")

# This is the SAM segmentation part, with the frames and bounding box of each frame, we generated the mask of object in each frame and combine it with grayscaled background
sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
input_boxes = torch.tensor(bbox_list, device=predictor.device)

i = 0
output_frames = []
while i < len(frame_list):
    predictor.set_image(frame_list[i])
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes[i], frame_list[i].shape[:2])
    masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
    mask = masks[0][0].cpu().numpy().astype(np.uint8)
    # Working out the object's mask

    # Reverse the mask and cut-out the background, grayscale the background part
    mask_inv = cv2.bitwise_not(mask)
    gray_frame = cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bitwise_and(gray_frame,gray_frame,mask = mask_inv)
    gray_bg = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Draw the contour of object and cut-out the RGB version of object.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame_list[i], contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
    color_fg = cv2.bitwise_and(frame_list[i],frame_list[i],mask = mask)

    # Combine the grayscale background and RGB frontgraound together
    final_pic = cv2.add(color_fg,gray_bg)

    output_frames.append(final_pic)
    i += 1

# This is the last Output video part. After getting all the required frames, we write them into a new video and return.
height, width, _ = output_frames[0].shape
fps = 10
video = cv2.VideoWriter(args["video"]+'_output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
for image in output_frames:
    video.write(image)
video.release()
print("Object Tracking Video finished")
