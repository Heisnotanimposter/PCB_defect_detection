#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCB Defect Detection & Automated Quality Sorting Engine (IPC-A-610 Standard)
-----------------------------------------------------------------------------
A modular computer vision & deep learning pipeline for printed circuit board (PCB) inspection.

Pipeline Architecture:
  Stage 1 (Alignment)     : ORB/SIFT Feature Matching & RANSAC Homography Registration.
  Stage 2 (Difference)    : Absolute Difference & SSIM Contour Extraction.
  Stage 3 (Classification): YOLO Deep Learning Model / OpenCV Heuristics (Open, Short, Mousebite, Spur, Pinhole, Solder Splatter).
  Stage 4 (IPC Sorting)   : IPC-A-610 Standards Compliance (Class 1, Class 2, Class 3) Quality Decision Engine.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np


class PCBAligner:
    """Stage 1: Aligns the test PCB image to the Golden Reference image using feature matching & homography."""

    def __init__(self, nfeatures: int = 5000):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align(self, ref_img: np.ndarray, test_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Aligns test_img to ref_img. Returns (aligned_image, alignment_mask, metrics)."""
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if len(test_img.shape) == 3 else test_img

        # Detect keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(ref_gray, None)
        kp2, des2 = self.orb.detectAndCompute(test_gray, None)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            # Fallback if feature matching fails
            return test_img.copy(), np.ones_like(ref_gray), {"matches": 0, "inliers": 0, "status": "fallback_raw"}

        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Retain top 25% matches
        num_good_matches = max(int(len(matches) * 0.25), 4)
        good_matches = matches[:num_good_matches]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Homography via RANSAC
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return test_img.copy(), np.ones_like(ref_gray), {"matches": len(matches), "inliers": 0, "status": "homography_failed"}

        h, w = ref_gray.shape
        aligned_img = cv2.warpPerspective(test_img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        inliers_count = int(np.sum(mask)) if mask is not None else 0

        metrics = {
            "total_keypoints_ref": len(kp1),
            "total_keypoints_test": len(kp2),
            "matches": len(matches),
            "inliers": inliers_count,
            "status": "success"
        }
        return aligned_img, mask, metrics


class PCBDifferenceDetector:
    """Stage 2: Subtracts aligned test image from Golden Reference image to locate difference ROIs."""

    def __init__(self, min_defect_area: int = 15, blur_kernel: int = 5):
        self.min_defect_area = min_defect_area
        self.blur_kernel = blur_kernel

    def detect_differences(self, ref_img: np.ndarray, aligned_img: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Calculates difference map and extracts candidate defect region bounding boxes."""
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
        aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY) if len(aligned_img.shape) == 3 else aligned_img

        # Gaussian smoothing to suppress high-frequency noise
        ref_blur = cv2.GaussianBlur(ref_gray, (self.blur_kernel, self.blur_kernel), 0)
        aligned_blur = cv2.GaussianBlur(aligned_gray, (self.blur_kernel, self.blur_kernel), 0)

        # Absolute Difference
        diff = cv2.absdiff(ref_blur, aligned_blur)

        # Otsu Adaptive Thresholding
        _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological Opening & Closing to clean small speckles and connect components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_diff = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned_diff = cv2.morphologyEx(cleaned_diff, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours of difference regions
        contours, _ = cv2.findContours(cleaned_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        defect_rois = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_defect_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi_ref = ref_img[y:y+h, x:x+w]
            roi_test = aligned_img[y:y+h, x:x+w]

            defect_rois.append({
                "id": i + 1,
                "bbox": (x, y, w, h),
                "area": float(area),
                "perimeter": float(cv2.arcLength(cnt, True)),
                "center": (x + w // 2, y + h // 2),
                "roi_ref": roi_ref,
                "roi_test": roi_test
            })

        return cleaned_diff, defect_rois


class YOLODefectClassifier:
    """Stage 3: Deep learning / Heuristic classifier for defect categorization."""

    DEFECT_CLASSES = {
        0: "open",             # Open Circuit
        1: "short",            # Short Circuit / Copper Bridge
        2: "mousebite",        # Mousebite / Edge Erosion
        3: "spur",             # Copper Spur / Spike
        4: "copper_excess",    # Excess Copper / Splatter
        5: "pinhole",          # Pinhole / Hole in Copper Pad
        6: "solder_ball",      # Solder Ball / Void
        7: "missing_component" # Missing Component
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"[+] Successfully loaded YOLO model weights from '{model_path}'")
            except Exception as e:
                print(f"[!] Could not load Ultralytics YOLO model ({e}). Using OpenCV heuristic classifier.")

    def classify(self, defect_rois: List[Dict[str, Any]], ref_img: np.ndarray, test_img: np.ndarray) -> List[Dict[str, Any]]:
        """Classifies each candidate ROI into a specific defect type."""
        classified_defects = []

        for roi in defect_rois:
            x, y, w, h = roi["bbox"]
            roi_ref = roi["roi_ref"]
            roi_test = roi["roi_test"]

            if self.model is not None:
                # Model inference on crop
                results = self.model(roi_test, verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    defect_type = self.DEFECT_CLASSES.get(cls_id, "unknown")
                else:
                    defect_type, conf = self._heuristic_classify(roi_ref, roi_test, roi)
            else:
                defect_type, conf = self._heuristic_classify(roi_ref, roi_test, roi)

            classified_defects.append({
                "id": roi["id"],
                "type": defect_type,
                "confidence": round(conf, 3),
                "bbox": [x, y, w, h],
                "area": roi["area"],
                "center": roi["center"]
            })

        return classified_defects

    def _heuristic_classify(self, roi_ref: np.ndarray, roi_test: np.ndarray, roi_info: Dict[str, Any]) -> Tuple[str, float]:
        """Computer vision heuristic classification based on color, intensity, and shape analysis."""
        if roi_ref.size == 0 or roi_test.size == 0:
            return "unknown", 0.5

        ref_gray = cv2.cvtColor(roi_ref, cv2.COLOR_BGR2GRAY) if len(roi_ref.shape) == 3 else roi_ref
        test_gray = cv2.cvtColor(roi_test, cv2.COLOR_BGR2GRAY) if len(roi_test.shape) == 3 else roi_test

        ref_mean = float(np.mean(ref_gray))
        test_mean = float(np.mean(test_gray))
        diff_mean = test_mean - ref_mean

        aspect_ratio = roi_info["bbox"][2] / max(roi_info["bbox"][3], 1)

        # 1. Darker in test -> Open Circuit or Mousebite or Pinhole
        if diff_mean < -15:
            if roi_info["area"] < 80 and 0.7 < aspect_ratio < 1.3:
                return "pinhole", 0.88
            elif aspect_ratio > 2.0 or aspect_ratio < 0.5:
                return "open", 0.92
            else:
                return "mousebite", 0.85

        # 2. Brighter in test -> Short, Spur, or Solder Ball
        elif diff_mean > 15:
            # Check for metallic solder silver vs copper
            if len(roi_test.shape) == 3:
                std_color = np.std(roi_test, axis=(0, 1))
                if np.max(std_color) < 15:  # Low color saturation -> Silver solder ball
                    return "solder_ball", 0.90
            
            if aspect_ratio > 1.8 or aspect_ratio < 0.55:
                return "short", 0.89
            else:
                return "spur", 0.82

        return "copper_excess", 0.75


class IPCSorter:
    """Stage 4: Evaluates board against IPC-A-610 inspection standards for Class 1, 2, or 3."""

    CLASS_SPECS = {
        1: {
            "name": "Class 1: General Electronic Products",
            "max_defects_allowed": 3,
            "allow_cosmetic_defects": True,
            "prohibited_defects": ["short"]
        },
        2: {
            "name": "Class 2: Dedicated Service Electronic Products",
            "max_defects_allowed": 1,
            "allow_cosmetic_defects": False,
            "prohibited_defects": ["open", "short", "missing_component"]
        },
        3: {
            "name": "Class 3: High Performance / Aerospace / Medical Electronics",
            "max_defects_allowed": 0,
            "allow_cosmetic_defects": False,
            "prohibited_defects": ["open", "short", "mousebite", "spur", "copper_excess", "pinhole", "solder_ball", "missing_component"]
        }
    }

    def evaluate(self, defects: List[Dict[str, Any]], ipc_class: int = 3) -> Dict[str, Any]:
        """Evaluates defect list against specified IPC-A-610 class."""
        spec = self.CLASS_SPECS.get(ipc_class, self.CLASS_SPECS[3])
        
        num_defects = len(defects)
        defect_types = [d["type"] for d in defects]

        # Violations check
        critical_violations = [d for d in defects if d["type"] in spec["prohibited_defects"]]
        
        is_pass = True
        fail_reasons = []

        if num_defects > spec["max_defects_allowed"]:
            is_pass = False
            fail_reasons.append(f"Total defect count ({num_defects}) exceeds IPC Class {ipc_class} limit ({spec['max_defects_allowed']}).")

        if len(critical_violations) > 0:
            is_pass = False
            violating_types = set([d["type"] for d in critical_violations])
            fail_reasons.append(f"Contains prohibited defect types for Class {ipc_class}: {', '.join(violating_types)}.")

        # Quality Score Calculation (100% minus deductions per defect)
        score = max(0.0, round(100.0 - (num_defects * 15.0 + len(critical_violations) * 20.0), 1))

        return {
            "ipc_class": ipc_class,
            "class_name": spec["name"],
            "status": "PASS" if is_pass else "FAIL",
            "quality_score": score,
            "total_defects": num_defects,
            "critical_violations": len(critical_violations),
            "fail_reasons": fail_reasons,
            "defect_summary": {t: defect_types.count(t) for t in set(defect_types)}
        }


class PCBInspector:
    """Main Orchestrator for 4-Stage PCB Inspection Pipeline."""

    def __init__(self, model_path: Optional[str] = None, min_defect_area: int = 15):
        self.aligner = PCBAligner()
        self.detector = PCBDifferenceDetector(min_defect_area=min_defect_area)
        self.classifier = YOLODefectClassifier(model_path=model_path)
        self.sorter = IPCSorter()

    def inspect(self, ref_path: str, test_path: str, ipc_class: int = 3, output_dir: str = "output") -> Dict[str, Any]:
        """Runs complete inspection on reference and test image pair."""
        os.makedirs(output_dir, exist_ok=True)

        ref_img = cv2.imread(ref_path)
        test_img = cv2.imread(test_path)

        if ref_img is None:
            raise FileNotFoundError(f"Could not load reference image from '{ref_path}'")
        if test_img is None:
            raise FileNotFoundError(f"Could not load test image from '{test_path}'")

        print("[Stage 1/4] Aligning Test Board to Golden Reference Board...")
        aligned_img, mask, align_metrics = self.aligner.align(ref_img, test_img)

        print("[Stage 2/4] Extracting Difference Regions & Candidate Defects...")
        diff_map, defect_rois = self.detector.detect_differences(ref_img, aligned_img)

        print("[Stage 3/4] Classifying Defects (YOLO / OpenCV Heuristics)...")
        classified_defects = self.classifier.classify(defect_rois, ref_img, aligned_img)

        print("[Stage 4/4] Evaluating IPC-A-610 Standard Quality Sorting...")
        eval_result = self.sorter.evaluate(classified_defects, ipc_class=ipc_class)

        # Generate Visual Collage & Save Annotations
        vis_collage = self._create_visual_collage(ref_img, test_img, aligned_img, diff_map, classified_defects, eval_result)
        
        vis_path = os.path.join(output_dir, "inspection_result.png")
        cv2.imwrite(vis_path, vis_collage)

        report = {
            "reference_image": ref_path,
            "test_image": test_path,
            "alignment": align_metrics,
            "defects": classified_defects,
            "evaluation": eval_result,
            "output_visualization": vis_path
        }

        report_path = os.path.join(output_dir, "inspection_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n[+] Inspection Complete!")
        print(f"    - IPC Status     : {eval_result['status']} ({eval_result['class_name']})")
        print(f"    - Quality Score  : {eval_result['quality_score']}%")
        print(f"    - Defect Count   : {eval_result['total_defects']}")
        print(f"    - Visualization  : {vis_path}")
        print(f"    - JSON Report    : {report_path}\n")

        return report

    def _create_visual_collage(self, ref_img: np.ndarray, raw_test: np.ndarray, aligned_test: np.ndarray,
                               diff_map: np.ndarray, defects: List[Dict[str, Any]], eval_result: Dict[str, Any]) -> np.ndarray:
        """Draws bounding boxes, labels, and creates a side-by-side inspection grid."""
        h, w = ref_img.shape[:2]
        
        # Color mapping for defect categories (BGR)
        color_map = {
            "open": (0, 0, 255),          # Bright Red
            "short": (0, 165, 255),       # Orange
            "mousebite": (255, 0, 255),   # Magenta
            "spur": (255, 255, 0),        # Cyan
            "copper_excess": (0, 255, 255),# Yellow
            "pinhole": (255, 0, 0),       # Blue
            "solder_ball": (128, 0, 128), # Purple
            "missing_component": (0, 0, 128) # Dark Red
        }

        # Panel 1: Pristine Golden Reference
        p1 = ref_img.copy()
        cv2.putText(p1, "1. Golden Reference", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Panel 2: Aligned Test Board with Bounding Boxes
        p2 = aligned_test.copy()
        cv2.putText(p2, "2. Aligned Test Board (Defects)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        for d in defects:
            x, y, bw, bh = d["bbox"]
            color = color_map.get(d["type"], (0, 255, 0))
            cv2.rectangle(p2, (x, y), (x + bw, y + bh), color, 2)
            label = f"#{d['id']} {d['type']} ({d['confidence']:.2f})"
            cv2.putText(p2, label, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Panel 3: Difference Heatmap
        diff_bgr = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
        cv2.putText(diff_bgr, "3. Difference Heatmap", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Panel 4: Inspection Summary Dashboard
        p4 = np.zeros_like(ref_img)
        p4[:] = (30, 30, 30)  # Dark gray background
        
        status = eval_result["status"]
        status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        
        cv2.putText(p4, "4. IPC-A-610 Sorting Report", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.rectangle(p4, (20, 70), (w - 20, 140), status_color, -1)
        cv2.putText(p4, f"STATUS: {status}", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        cv2.putText(p4, f"Target Grade : IPC Class {eval_result['ipc_class']}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(p4, f"Quality Score: {eval_result['quality_score']}%", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(p4, f"Total Defects: {eval_result['total_defects']}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        y_offset = 280
        cv2.putText(p4, "Defect Breakdown:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        y_offset += 25
        for dtype, count in eval_result["defect_summary"].items():
            cv2.putText(p4, f" - {dtype}: {count}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1)
            y_offset += 22

        if eval_result["fail_reasons"]:
            y_offset += 10
            cv2.putText(p4, "Failure Reasons:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            y_offset += 25
            for reason in eval_result["fail_reasons"]:
                cv2.putText(p4, f" ! {reason}", (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
                y_offset += 20

        # Assemble 2x2 Grid
        top_row = np.hstack((p1, p2))
        bottom_row = np.hstack((diff_bgr, p4))
        collage = np.vstack((top_row, bottom_row))

        return collage


def setup_dataset_directory():
    """Helper to download and organize DeepPCB and Roboflow dataset references."""
    print("==========================================================")
    print("PCB Defect Detection Dataset Setup Helper")
    print("==========================================================")
    print("Recommended Open-Source Datasets:")
    print("1. DeepPCB Dataset (1,500 image pairs - Open, Short, Mousebite, Spur, Copper, Pinhole)")
    print("   GitHub: https://github.com/tangsanli5201/DeepPCB")
    print("2. Roboflow PCB Defects Dataset (YOLOv8/v5 format)")
    print("   Roboflow: https://universe.roboflow.com/object-detection-dt-wzpc6/pcb-dataset-defect")
    print("\nTo set up DeepPCB, run:")
    print("  git clone https://github.com/tangsanli5201/DeepPCB.git dataset/DeepPCB")
    print("==========================================================")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="PCB Defect Detection & IPC Quality Sorting Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Run 4-stage inspection on reference and test PCB images")
    inspect_parser.add_argument("-r", "--reference", required=True, help="Path to Golden Reference PCB image")
    inspect_parser.add_argument("-t", "--test", required=True, help="Path to Test PCB image")
    inspect_parser.add_argument("-c", "--ipc-class", type=int, choices=[1, 2, 3], default=3, help="IPC-A-610 Class strictness (1, 2, or 3)")
    inspect_parser.add_argument("-m", "--model", default=None, help="Optional path to custom YOLO weights (.pt)")
    inspect_parser.add_argument("-o", "--output", default="output", help="Directory to save visual collage and JSON report")

    # Command: demo
    demo_parser = subparsers.add_parser("demo", help="Generate synthetic PCB samples and run inspection demo")
    demo_parser.add_argument("-c", "--ipc-class", type=int, choices=[1, 2, 3], default=3, help="IPC-A-610 Class strictness (1, 2, or 3)")
    demo_parser.add_argument("-o", "--output", default="output", help="Output directory")

    # Command: setup-dataset
    subparsers.add_parser("setup-dataset", help="Print dataset download and structure setup instructions")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        inspector = PCBInspector(model_path=args.model)
        inspector.inspect(args.reference, args.test, ipc_class=args.ipc_class, output_dir=args.output)

    elif args.command == "demo":
        from create_demo_samples import generate_samples
        ref_path, test_path = generate_samples()
        inspector = PCBInspector()
        inspector.inspect(ref_path, test_path, ipc_class=args.ipc_class, output_dir=args.output)

    elif args.command == "setup-dataset":
        setup_dataset_directory()

    else:
        # Default behavior if no sub-command passed
        parser.print_help()


if __name__ == "__main__":
    main()
