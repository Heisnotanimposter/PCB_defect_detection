#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic PCB Sample Generator for PCB Defect Detection & Inspection Testing.
Generates a pristine 'Golden Reference Board' image and a slightly warped/defective 'Test Board' image.
"""

import os
import cv2
import numpy as np


def draw_pcb_base(width=800, height=600):
    """Draws a base PCB board with substrate background, copper traces, pads, and chips."""
    # Dark green PCB substrate background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (20, 80, 35)

    # Grid gridlines (subtle silkscreen / texture)
    for x in range(0, width, 40):
        cv2.line(img, (x, 0), (x, height), (25, 95, 42), 1)
    for y in range(0, height, 40):
        cv2.line(img, (0, y), (width, y), (25, 95, 42), 1)

    # Corner mounting holes (gold/copper rim + black hole)
    holes = [(50, 50), (width - 50, 50), (50, height - 50), (width - 50, height - 50)]
    for h in holes:
        cv2.circle(img, h, 18, (30, 180, 220), -1)  # Copper/gold ring
        cv2.circle(img, h, 12, (10, 15, 10), -1)   # Hole interior

    # Gold/Copper Traces (Track 1: top horizontal, Track 2: middle angled, Track 3: bottom parallel)
    trace_color = (40, 200, 240)  # Bright copper/gold BGR
    
    # Trace 1
    cv2.line(img, (100, 150), (400, 150), trace_color, 8)
    cv2.line(img, (400, 150), (500, 250), trace_color, 8)
    cv2.line(img, (500, 250), (700, 250), trace_color, 8)

    # Trace 2
    cv2.line(img, (100, 200), (350, 200), trace_color, 8)
    cv2.line(img, (350, 200), (450, 300), trace_color, 8)
    cv2.line(img, (450, 300), (700, 300), trace_color, 8)

    # Trace 3
    cv2.line(img, (100, 450), (700, 450), trace_color, 8)

    # Solder pads & vias
    pads = [
        (100, 150), (700, 250),
        (100, 200), (700, 300),
        (100, 450), (400, 450), (700, 450)
    ]
    for p in pads:
        cv2.circle(img, p, 14, (180, 220, 240), -1)  # Silver/solder finish
        cv2.circle(img, p, 5, (10, 20, 10), -1)     # Via hole

    # Main Integrated Circuit (IC Chip) in center
    ic_rect = (280, 320, 240, 100)  # x, y, w, h
    x, y, w, h = ic_rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)  # Black IC body
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 2)   # Outline
    cv2.circle(img, (x + 20, y + 20), 5, (150, 150, 150), -1)     # Pin 1 mark
    
    # Silkscreen text
    cv2.putText(img, "MCU-CORE-v1", (x + 40, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # IC Pins (Silver metallic leads)
    num_pins = 6
    for i in range(num_pins):
        px = x + 30 + i * 35
        # Top pins
        cv2.rectangle(img, (px, y - 15), (px + 10, y), (200, 210, 220), -1)
        cv2.line(img, (px + 5, y - 15), (px + 5, y - 35), trace_color, 4)
        # Bottom pins
        cv2.rectangle(img, (px, y + h), (px + 10, y + h + 15), (200, 210, 220), -1)
        cv2.line(img, (px + 5, y + h + 15), (px + 5, y + h + 35), trace_color, 4)

    return img


def add_defects(reference_img):
    """Introduces synthetic defects onto a copy of the reference board."""
    defective = reference_img.copy()
    trace_color = (40, 200, 240)
    bg_color = (20, 80, 35)

    # 1. Defect: Open Circuit (Gap cut in Trace 1)
    cv2.rectangle(defective, (230, 142), (260, 158), bg_color, -1)

    # 2. Defect: Short Circuit (Copper bridge between Trace 1 and Trace 2)
    cv2.rectangle(defective, (300, 150), (308, 200), trace_color, -1)

    # 3. Defect: Mousebite (Notch bitten out of Trace 3)
    cv2.circle(defective, (320, 446), 7, bg_color, -1)

    # 4. Defect: Spur (Unintended copper spike branching off Trace 2)
    pts = np.array([[550, 300], [570, 340], [555, 300]], np.int32)
    cv2.fillPoly(defective, [pts], trace_color)

    # 5. Defect: Pinhole (Small hole in copper pad)
    cv2.circle(defective, (700, 250), 6, bg_color, -1)

    # 6. Defect: Copper Excess / Solder Splatter (Unintended solder ball near IC)
    cv2.circle(defective, (240, 370), 9, (200, 210, 220), -1)

    return defective


def apply_slight_transformation(img, angle_deg=1.5, tx=8, ty=-5):
    """Applies realistic perspective shift / rotation / translation to simulate camera misalignment."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Rotation & Scaling matrix
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty

    warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped


def generate_samples(output_dir="samples"):
    """Generates and saves pristine reference and defective test images."""
    os.makedirs(output_dir, exist_ok=True)

    print("[*] Generating Pristine Golden Reference PCB Image...")
    ref_img = draw_pcb_base()
    ref_path = os.path.join(output_dir, "golden_reference.png")
    cv2.imwrite(ref_path, ref_img)

    print("[*] Injecting PCB Defects (Open, Short, Mousebite, Spur, Pinhole, Solder Splatter)...")
    defective_img = add_defects(ref_img)

    print("[*] Simulating Camera Misalignment (Rotation & Translation)...")
    aligned_defective_img = apply_slight_transformation(defective_img, angle_deg=1.2, tx=6, ty=-4)
    test_path = os.path.join(output_dir, "test_defective.png")
    cv2.imwrite(test_path, aligned_defective_img)

    print(f"[+] Demo samples successfully generated in '{output_dir}/':")
    print(f"    - Reference Board : {ref_path}")
    print(f"    - Test Board      : {test_path}")

    return ref_path, test_path


if __name__ == "__main__":
    generate_samples()
