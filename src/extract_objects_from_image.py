import os
import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO



# ---------------------------
# Helper function: check if vertical intervals [y1, y2] overlap
# ---------------------------
def vertical_overlap(box1, box2):
    # box: (x1, y1, x2, y2) where y defines the vertical axis.
    overlap = min(box1[3], box2[3]) - max(box1[1], box2[1])
    return overlap > 0

# ---------------------------
# Detection processor class
# ---------------------------
class DetectionProcessor:
    TABLE_PICTURE_LABELS = {"table", "picture"}

    def __init__(self, model_path, conf_threshold=0.2, iou_threshold=0.8, high_quality=True):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.high_quality = high_quality  # New parameter for image quality

    def process_image(self, image_path):
        """
        Run inference on the given image and perform detection grouping.
        Returns the image array, list of grouped detections and remaining detections.
        """
        image = cv2.imread(str(image_path))
        # Run inference on the image
        results = self.model(str(image_path), conf=self.conf_threshold, iou=self.iou_threshold)
        result = results[0]  # Single-image inference

        # Collect detections
        detections = []
        for i, (box, cls_idx) in enumerate(zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist())):
            x1, y1, x2, y2 = map(int, box)
            label = result.names[int(cls_idx)] if result.names else f"class_{int(cls_idx)}"
            detections.append({
                "box": (x1, y1, x2, y2),
                "label": label,
                "index": i
            })

        # Sort detections by vertical (y1) then horizontal (x1) position.
        sorted_detections = sorted(detections, key=lambda d: (d["box"][1], d["box"][0]))

        # Group detections for Table/Picture items
        grouped_indices = set()   # Indices already grouped
        table_picture_groups = [] # List to store groups (Table/Picture + extra Caption/Text)
        others = []               # Remaining detections not grouped

        for i, det in enumerate(sorted_detections):
            if i in grouped_indices:
                continue

            label_lower = det["label"].lower()
            # Process only Table or Picture as primary detection
            if label_lower in self.TABLE_PICTURE_LABELS:
                group_items = [det]
                grouped_indices.add(i)
                # Look for additional groupable items (Caption/Text)
                for j, candidate in enumerate(sorted_detections):
                    if j == i or j in grouped_indices:
                        continue
                    candidate_label = candidate["label"].lower()
                    # For Caption: group if immediately before, immediately after, or if there is vertical overlap.
                    if candidate_label == "caption":
                        if j == i - 1 or j == i + 1 or vertical_overlap(det["box"], candidate["box"]):
                            group_items.append(candidate)
                            grouped_indices.add(j)
                    # For Text: group if it's immediately before OR if there is vertical overlap.
                    elif candidate_label == "text":
                        if j == i - 1 or vertical_overlap(det["box"], candidate["box"]):
                            group_items.append(candidate)
                            grouped_indices.add(j)
                # Compute the union bounding box for the group
                group_x1 = min(item["box"][0] for item in group_items)
                group_y1 = min(item["box"][1] for item in group_items)
                group_x2 = max(item["box"][2] for item in group_items)
                group_y2 = max(item["box"][3] for item in group_items)
                table_picture_groups.append({
                    "group_box": (group_x1, group_y1, group_x2, group_y2),
                    "label": det["label"],
                    "index": det["index"]
                })
            else:
                if i not in grouped_indices:
                    others.append(det)

        return image, table_picture_groups, others

    def save_detections(self, image, image_path, table_picture_groups, others, output_dir):
        """
        Save each Table/Picture group separately and composite all remaining detections into a "Text" image.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set JPEG quality based on the high_quality parameter.
        quality = 95 if self.high_quality else 75

        # Save each Table/Picture group (with associated Caption/Text)
        for count, group in enumerate(table_picture_groups, start=1):
            x1, y1, x2, y2 = group["group_box"]
            crop = image[y1:y2, x1:x2]
            out_filename = f"{image_path.stem}_{group['label']}_{count}.jpg"
            cv2.imwrite(str(output_dir / out_filename), crop, [cv2.IMWRITE_JPEG_QUALITY, quality])

        # Composite remaining detections (non-grouped) into a "Text" image
        h, w = image.shape[:2]
        composite = np.ones((h, w, 3), dtype=np.uint8) * 255  # Blank white canvas
        for det in others:
            x1, y1, x2, y2 = det["box"]
            crop = image[y1:y2, x1:x2]
            composite[y1:y2, x1:x2] = crop

        text_filename = f"{image_path.stem}_Text.jpg"
        cv2.imwrite(str(output_dir / text_filename), composite, [cv2.IMWRITE_JPEG_QUALITY, quality])

# ---------------------------
# Main function to process all images in a directory
# ---------------------------
def extract_objects_from_image(input_dir, output_dir, model_path, conf_threshold=0.2, iou_threshold=0.8, high_quality=True):
    processor = DetectionProcessor(model_path, conf_threshold, iou_threshold, high_quality=high_quality)
    input_dir = Path(input_dir)
    # Loop over all image files (filtering common image extensions)
    print(f"\nAnalyzing the images in the directory {input_dir} ...")
    for image_path in input_dir.glob("*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        print(f"\nProcessing {image_path.name}...")
        image, table_picture_groups, others = processor.process_image(image_path)
        processor.save_detections(image, image_path, table_picture_groups, others, output_dir)
        print(f"Objects were detected successfully and stored in the folder {output_dir}")

    print("\nAll images were processed successfully!")


