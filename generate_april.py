import os
import cv2
import numpy as np

def main():
    out_dir = "./assets/AprilTag36h11"
    os.makedirs(out_dir, exist_ok=True)

    # AprilTag family
    family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    n_markers = 20        # 0..19
    marker_px = 800       # size of tag image (pixels)
    margin_px = 50        # white border around tag (pixels)

    for marker_id in range(n_markers):
        tag = cv2.aruco.generateImageMarker(family, marker_id, marker_px)

        h = marker_px + 2 * margin_px
        w = marker_px + 2 * margin_px

        # create white canvas
        out = np.ones((h, w, 3), dtype=np.uint8) * 255

        tag_bgr = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)

        out[
            margin_px:margin_px + marker_px,
            margin_px:margin_px + marker_px
        ] = tag_bgr

        path = os.path.join(out_dir, f"tag36h11_id_{marker_id:02d}.png")
        cv2.imwrite(path, out)

    print(f"Saved {n_markers} AprilTags to: {out_dir}")


if __name__ == "__main__":
    main()