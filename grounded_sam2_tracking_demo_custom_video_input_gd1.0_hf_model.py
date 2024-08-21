import os
import cv2
import torch
import numpy as np
import supervision as sv
from itertools import chain
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import glob

"""
Hyperparam for Ground and Tracking
"""
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
VIDEO_PATH = "./assets/mustard_bottle_real.mp4"
TEXT_PROMPT = "yellow mustard bottle." # NOTE: Need to add a dot ('.') at the end for this to work!
NEGATIVE_TEXT_PROMT = "robot." # Prompt of what not to mask. Can be `None`.
NUM_NEG_FRAMES = 3 # Number of frames to add negatives to (uniformly spaced). Increasing this might lead to OOM.
OUTPUT_VIDEO_PATH = "./mustard_real.mp4"
SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames"
SAVE_TRACKING_RESULTS_DIR = "./tracking_results"
SAVE_MASKS_DIR = "./masks"
PROMPT_TYPE_FOR_VIDEO = "point" # choose from ["point", "box", "mask"]
# The interval at which frames are sampled from the input image. Increase to sample coarser.
# Note that his might reduce quality and is thus not recommended. Try the offload options instead.
INPUT_VIDEO_STRIDE = 1
OFFLOAD_VIDEO_TO_CPU = True # Prevents OOM for large videos but is slower.
OFFLOAD_STATE_TO_CPU = False
STARTING_FRAME = 80 # Frame number to start at

"""
Step 1: Environment settings and model initialization for SAM 2
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs
    # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
# sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
# model_cfg = "sam2_hiera_t.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# build grounding dino from huggingface
model_id = MODEL_ID
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


"""
Custom video input directly using video files
"""
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
frame_generator = sv.get_video_frames_generator(
    VIDEO_PATH, stride=INPUT_VIDEO_STRIDE, start=STARTING_FRAME, end=None
)

# saving video to frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

frame_count = 0
with sv.ImageSink(
    target_dir_path=source_frames, 
    overwrite=True, 
    image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)
        frame_count +=1

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(
    video_path=SOURCE_VIDEO_FRAME_DIR,
    offload_video_to_cpu=OFFLOAD_VIDEO_TO_CPU,
    offload_state_to_cpu=OFFLOAD_STATE_TO_CPU,
)

ann_frame_idx = 0  # the frame index we interact with
"""
Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
"""

# prompt grounding dino to get the box coordinates on specific frame
def get_dino_boxes(text, frame_idx):
    img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx])
    image = Image.open(img_path)

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    input_boxes = results[0]["boxes"].cpu().numpy()
    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]
    return input_boxes, confidences, class_names

input_boxes, confidences, class_names = get_dino_boxes(TEXT_PROMPT, ann_frame_idx)

print(input_boxes)
assert len(input_boxes) > 0, "No results found for the text prompt. Make sure that the prompt ends with a dot '.'!"

# prompt SAM image predictor to get the mask for the object
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image = Image.open(img_path)
image_predictor.set_image(np.array(image.convert("RGB")))

# process the detection results
OBJECTS = class_names

print("pos objects", OBJECTS)

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)
# convert the mask shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

# If you are using point prompts, we uniformly sample positive points based on the mask
if PROMPT_TYPE_FOR_VIDEO == "point":
    # sample the positive points from mask for each objects
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

    for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
        # label one means positive (do mask), label zero means negative (don't mask)
        labels = np.ones((points.shape[0]), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
# Using box prompt
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
# Using mask prompt is a more straightforward way
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask
        )
else:
    raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

if NEGATIVE_TEXT_PROMT is not None:
    image_predictor.reset_predictor()
    
    neg_id_start_orig = neg_id_start = object_id+1
    for idx in tqdm(
        np.linspace(0, frame_count - 1, NUM_NEG_FRAMES, dtype=int), desc="Adding negative", leave=False
    ):
        neg_input_boxes, _, neg_class_names = get_dino_boxes(NEGATIVE_TEXT_PROMT, idx)
        if len(neg_input_boxes) == 0:
            continue
        
        img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[idx])
        image = Image.open(img_path)
        image_predictor.set_image(np.array(image.convert("RGB")))

        neg_masks, _, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=neg_input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if neg_masks.ndim == 4:
            neg_masks = neg_masks.squeeze(1)
            
        # sample the negative points from mask for each objects
        num_points = 1
        neg_all_sample_points = sample_points_from_masks(masks=neg_masks, num_points=num_points)

        for object_id, (label, points) in enumerate(
            zip(neg_class_names, neg_all_sample_points), start=neg_id_start
        ):
            # label one means positive (do mask), label zero means negative (don't mask)
            labels = np.zeros((points.shape[0]), dtype=np.int32)
            _, _, _ = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
        
        neg_id_start += num_points
        image_predictor.reset_predictor()
        
# Clear GPU memory.
image_predictor.model.cpu()
del image_predictor

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

"""
Step 5: Visualize the segment results across the video and save them
"""

if not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
    os.makedirs(SAVE_TRACKING_RESULTS_DIR)

SAVE_TRACKING_RESULTS_DIR_DEBUG = SAVE_TRACKING_RESULTS_DIR + "_debug"
if not os.path.exists(SAVE_TRACKING_RESULTS_DIR_DEBUG):
    os.makedirs(SAVE_TRACKING_RESULTS_DIR_DEBUG)
    
if not os.path.exists(SAVE_MASKS_DIR):
    os.makedirs(SAVE_MASKS_DIR)
    
for dir in [SAVE_TRACKING_RESULTS_DIR, SAVE_TRACKING_RESULTS_DIR_DEBUG, SAVE_MASKS_DIR]:
    png_files = glob.glob(os.path.join(dir, '*.png'))
    jpg_files = glob.glob(os.path.join(dir, '*.jpg'))

    # Loop through the list of .png files and delete them.
    for file in chain(png_files, jpg_files):
        try:
            os.remove(file)
        except Exception as e:
            print(f'Error deleting {file}: {e}')

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
    
    if NEGATIVE_TEXT_PROMT is not None:
        pos_segments = {k: v for k, v in segments.items() if k < neg_id_start_orig}
    else:
        pos_segments = segments
    
    object_ids = list(pos_segments.keys())
    masks = list(pos_segments.values())
    masks = np.concatenate(masks, axis=0)
    
    # Save masks.
    union_mask = np.any(masks, axis=0)
    union_mask_8bit = (union_mask.astype(np.uint8)) * 255
    cv2.imwrite(os.path.join(SAVE_MASKS_DIR, f"mask_{frame_idx:05d}.png"), union_mask_8bit)
    
    # Create masked images.
    mask_3d = np.stack([union_mask] * 3, axis=-1)  # Shape: (W, H, 3)
    masked_image = np.where(mask_3d, img.copy(), 0)
    cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"masked_frame_{frame_idx:05d}.jpg"), masked_image)
    
    # Create debug overlay images.
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks, # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR_DEBUG, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)


"""
Step 6: Convert the annotated frames to video
"""

create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)

OUTPUT_VIDEO_PATH_DEBUG = OUTPUT_VIDEO_PATH[:-4] + "_debug.mp4"
create_video_from_images(SAVE_TRACKING_RESULTS_DIR_DEBUG, OUTPUT_VIDEO_PATH_DEBUG)
