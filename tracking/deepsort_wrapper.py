import numpy as np

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

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
