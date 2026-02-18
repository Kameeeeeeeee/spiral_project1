import os
import cv2

def main():
    out_dir = "./assets/Aruco4x4"
    os.makedirs(out_dir, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    n_markers = 20          # 0..19
    marker_px = 800         # size of marker image (pixels)
    margin_px = 50         # white border around marker (pixels)

    for marker_id in range(n_markers):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)

        canvas = 255 * (cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR) * 0 + 1)
        canvas = canvas.astype("uint8")

        h = marker_px + 2 * margin_px
        w = marker_px + 2 * margin_px
        img = 255 * (cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR) * 0 + 1)
        img = img.astype("uint8")
        img = 255 * (img * 0 + 1)

        out = 255 * (img * 0 + 1)
        out = out.astype("uint8")
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)

        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        out[margin_px:margin_px + marker_px, margin_px:margin_px + marker_px] = marker_bgr

        path = os.path.join(out_dir, f"aruco_4x4_id_{marker_id:02d}.png")
        cv2.imwrite(path, out)

    print(f"Saved {n_markers} markers to: {out_dir}")

if __name__ == "__main__":
    main()
