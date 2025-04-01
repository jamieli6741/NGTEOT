import cv2
import torch
import argparse
import numpy as np

from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from tracking.deepsort_wrapper import DeepSORTWrapper
from utils.video_utils import extract_video_frames, write_video
from utils.evaluation import run_tracking_comparison

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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
                    help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="csrt",
                    help="OpenCV object tracker type")
    ap.add_argument("-c", "--compress", type=int, default=5,
                    help="compress rate(default set as 5, which read 1 frame among 5 frames from input video)")
    ap.add_argument("-p", "--compute", type=str, default="cpu",
                    help="platform for computing")
    ap.add_argument("-d", "--deepsort_model", type=str, default="mars-small128.pb",
                    help="DeepSORT feature extractor model")
    # This is for performance comparision
    ap.add_argument("--metrics", action="store_true",
                    help="Run metrics comparison between trackers")
    ap.add_argument("--trackers", nargs='+',
                    help="Trackers to compare (space-separated list)")

    return vars(ap.parse_args())


if __name__ == '__main__':
    args = parse_args()

    if args["metrics"] and args["video"] and args["trackers"]:
        metrics_results = run_tracking_comparison(
            video_path=args["video"],
            trackers_list=args["trackers"],
            compress_rate=args["compress"],
            compute=args["compute"],
            deepsort_model=args["deepsort_model"]
        )
        exit()

    if args["compute"] == "cuda":
        torch.cuda.set_device(0) # You can edit the GPU number here
        device = "cuda"
    else:
        device = args["compute"]

    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

    frame_list = extract_video_frames(args["video"], args["compress"])

    # This is the Object Tracking part, with assigned tracker and bounding box drew by ROIselect, we worked out the bounding boxes for target object in the video.
    bbox_list = np.zeros((len(frame_list), 4))
    bbox = cv2.selectROI('Frame', frame_list[0], False)
    (x, y, w, h) = [int(v) for v in bbox]
    bbox_list[0] = [x, y, w+x, h+y]
    tracker.init(frame_list[0], bbox)

    for i in tqdm(range(1, len(frame_list)), desc="Tracking"):
        success, bbox = tracker.update(frame_list[i])
        if success:
            (x, y, w, h) = [int(v) for v in bbox]
            bbox_list[i] = [x, y, w+x, h+y]
        else:
            print('Warning: tracker failed.')
    print("Tracking Finished.")

    # This is the SAM segmentation part, with the frames and bounding box of each frame, we generated the mask of object in each frame and combine it with grayscaled background
    print("Running Segment Everything on Tracking Results ...")
    sam_checkpoint = "sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    input_boxes = torch.tensor(bbox_list, device=predictor.device)

    i = 0
    output_frames = []
    for i in tqdm(range(len(frame_list)), desc="Segmenting frames"):
        frame = frame_list[i]
        predictor.set_image(frame)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes[i], frame.shape[:2])
        masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
        mask = masks[0][0].cpu().numpy().astype(np.uint8)
        # Working out the object's mask

        # Reverse the mask and cut-out the background, grayscale the background part
        mask_inv = cv2.bitwise_not(mask)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bitwise_and(gray_frame,gray_frame,mask = mask_inv)
        gray_bg = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # Draw the contour of object and cut-out the RGB version of object.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
        color_fg = cv2.bitwise_and(frame,frame,mask = mask)

        # Combine the grayscale background and RGB frontgraound together
        final_pic = cv2.add(color_fg,gray_bg)

        output_frames.append(final_pic)
        i += 1

    # Write the result into a new video.
    output_video_path = args["video"].split(".")[0]+'_output.mp4'
    write_video(output_video_path, output_frames, fps=10)

