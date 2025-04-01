import cv2
import torch
import numpy as np

from tracking.deepsort_wrapper import DeepSORTWrapper
from utils.video_utils import extract_video_frames
from utils.visualization import generate_comparison_charts

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
            print(f"[Warning] Drift detected at frame {self.frame_count} (similarity: {similarity:.3f})")
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
            'effective_tracking_percentage': 0.5,  # The higher the better
            'avg_appearance_similarity': 0.3,      # The higher the better
            'drift_percentage': -0.2,              # The lower the better (negative correlation)
            # 'avg_bbox_jitter': -0.1,             # Optional metric
            # 'failure_rate': -0.2                 # Optional metric
        }

        # 组合得分：每个指标先归一化（百分比/100），再按权重累加
        composite_score = 0
        for metric_name, weight in metric_weights.items():
            if metric_name == 'effective_tracking_percentage':
                value = effective_tracking_percent  # Already in [0, 1]
            elif metric_name == 'avg_appearance_similarity':
                value = avg_similarity              # Usually in [0, 1]
            elif metric_name == 'drift_percentage':
                value = drift_percentage / 100
            elif metric_name == 'failure_rate':
                value = failure_rate / 100
            elif metric_name == 'avg_bbox_jitter':
                value = avg_jitter / 100  # Normalize based on expected jitter range
            else:
                value = 0  # Skip unrecognized metrics
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
        print("[SETUP] Using CUDA for processing")
    else:
        device = "cpu"
        print("[SETUP] Using CPU for processing")

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
        print("Warning: No valid trackers specified. Using CSRT as default.")
        valid_trackers = ["csrt"]

    print(f"Running comparison with these trackers: {valid_trackers}")


    frame_list = extract_video_frames(video_path, compress_rate, resize_width=500)

    # Have user select ROI
    print("[INSTRUCTION] Select the object to track in the first frame")
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
                print(f"Warning: Frame {i} Tracking failed")
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