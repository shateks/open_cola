import cv2
import numpy as np
from matplotlib import pyplot as plt

HUE_BOTTOML_VAL = 25.0
HUE_UPPER_VAL = 35.0
CROP_ROI = 10
GRADIENT_THRESHOLD = 3.5


def imshow(title="Image", image=None, size=2):
    image = image.astype(np.uint8)
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def detect_qr_codes(image):
    """Detect QR codes in an image."""
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale")
    qr_code_detector = cv2.QRCodeDetector()
    decoded_text, info_string, points, _ = qr_code_detector.detectAndDecodeMulti(
        image)
    if hasattr(points, '__len__') and len(points) == 2:
        points_dict = {k: points[i].reshape(-1, 2)
                       for i, k in enumerate(info_string)}
        return points_dict
    return {}


def draw_qr_markers(image, contours):
    """Draw markers at the center of QR code contours."""
    centers = np.empty((0, 2), dtype=int)
    for vertices in contours:
        c_x, c_y = vertices.mean(axis=0)
        centers = np.vstack([centers, [int(c_x), int(c_y)]])
        cv2.drawMarker(image, (int(c_x), int(c_y)), (0, 255, 0),
                       markerSize=200, thickness=20)
    return centers


def Find_Level_n_ROI(image_path, hue_lower, hue_higher, debug=False):
    _d = debug
    image_org = cv2.imread(image_path)
    image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    _d and imshow('Bottle in gray', image)
    points = detect_qr_codes(image)
    contours = []
    if len(points) == 2:
        for point in points.values():
            pts = point.reshape(-1, 2).astype(int)
            contours.append(pts)
    else:
        _d and print('No QR codes found')
        return None
    _d and draw_qr_markers(image_org, contours)
    _d and imshow('Bottle in color with marked QR', image_org)

    # Calculate the distance between the each two points
    point_candidates = []
    for p1 in contours[0]:
        for p2 in contours[1]:
            point_candidates.append([p1, p2, np.linalg.norm(p1-p2)])

    # Take the two closest points
    point_candidates.sort(key=lambda x: x[2])
    pts = np.array([point_candidates[i][0:2] for i in range(2)]).reshape(-1, 2)
    # Calculate which point is top left, bottom right, top right and bottom left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    # Cropping to roi by percentage
    roi_width = abs(top_left[0] - bottom_right[0])
    roi_height = abs(top_left[1] - bottom_right[1])

    roi_margin_x = int(roi_width * (CROP_ROI/100))
    roi_margin_y = int(roi_height * (CROP_ROI/100))
    roi_top_left = top_left + (roi_margin_x, roi_margin_y)
    roi_bottom_right = bottom_right + (-roi_margin_x, -roi_margin_y)
    roi_top_right = top_right + (roi_margin_x, -roi_margin_y)
    roi_bottom_left = bottom_left + (-roi_margin_x, roi_margin_y)
    roi_tl_tr_br_bl = np.array(
        [roi_top_left, roi_top_right, roi_bottom_right, roi_bottom_left], dtype=np.int32)
    image_cropped_org = image_org[roi_top_left[1]                                  :roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
    image_cropped_blr = cv2.medianBlur(image_cropped_org, 77)
    # if _d:
    image_roi_hsv = cv2.cvtColor(image_cropped_blr, cv2.COLOR_BGR2HSV_FULL)
    roi_hue = image_roi_hsv[:, :, 0:1]
    roi_hue = np.squeeze(roi_hue)
    roi_sat = image_roi_hsv[:, :, 1:2]
    roi_sat = np.squeeze(roi_sat)
    roi_val = image_roi_hsv[:, :, 2:3]
    roi_val = np.squeeze(roi_val)
    roi_hue_mean = np.mean(roi_hue, axis=1, keepdims=True).astype(np.uint8)
    roi_val_mean = np.mean(roi_val, axis=1, keepdims=True).astype(np.uint8)
    roi_sat_mean = np.mean(roi_sat, axis=1, keepdims=True).astype(np.uint8)
    roi_val_mean_grad = np.abs(np.gradient(roi_val_mean, axis=0))
    if _d:
        cv2.rectangle(image_org, tuple(roi_tl_tr_br_bl[0]), tuple(
            roi_tl_tr_br_bl[2]), (0, 255, 0), thickness=10)
        imshow('Rectangle between QR', image_org)
        image_roi_gy = cv2.cvtColor(image_cropped_blr, cv2.COLOR_BGR2GRAY)
        imshow('Between QR gray, cropped by 10%', image_roi_gy)
        image_roi_gy_crop_maen = np.mean(
            image_roi_gy, axis=1, keepdims=True).astype(np.uint8)
        image_roi_gy_crop_maen = np.repeat(image_roi_gy_crop_maen, 50, axis=1)
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].plot(image_roi_gy_crop_maen[:, 0], label="Grey")
        axs[0].plot(roi_hue_mean[:, 0], label="Hue")
        axs[0].plot(roi_sat_mean[:, 0], label="Sat")
        axs[0].plot(roi_val_mean[:, 0], label="Val")
        axs[0].plot(roi_val_mean_grad[:, 0], label="Grad(Val)")
        axs[0].legend()
        axs[0].set_title("Combination of mean every row.")
        axs[1].imshow(cv2.cvtColor(image_cropped_org, cv2.COLOR_BGR2RGB))
        # axs[1].axis('off')
        axs[1].set_title("Cropped original image")
        plt.show()

    # If from bottom to top is found gradient of Val bigger than GRADIENT_THRESHOLD, get this coordinate.
    # Then check if 70% of Hue from bottom to gradient is in range hue_lower to hue_higher.
    # If yes this range can be considered as fill level.
    if np.max(roi_val_mean_grad[::-1]) > GRADIENT_THRESHOLD:
        idx = roi_val_mean_grad.shape[0] - \
            np.argmax(roi_val_mean_grad[::-1] > GRADIENT_THRESHOLD)
    else:
        idx = 0
    hue_bottom_to_grad = roi_hue_mean[idx:]
    level = 0.0
    try:
        percent_in_range = np.count_nonzero((hue_bottom_to_grad > hue_lower) & (
            hue_bottom_to_grad < hue_higher))/len(hue_bottom_to_grad)
        if percent_in_range > 0.7:
            level = len(roi_val_mean_grad[idx:]) / len(roi_val_mean_grad)
    except ZeroDivisionError:
        _d and print('ZeroDivisionError')
        return None
    image_cropped_org_m = np.copy(image_org)
    cv2.line(image_cropped_org_m, (roi_top_left[0], roi_top_left[1]+idx),
             (roi_bottom_right[0], roi_top_left[1]+idx), (0, 0, 255), thickness=10)
    cv2.putText(image_cropped_org_m, '{:2.1f}%'.format(
        level*100), (int(roi_top_left[0]), roi_top_left[1]+idx), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), thickness=5)
    _d and imshow('Space between', image_cropped_org_m)
    return level


if __name__ == "__main__":
    image_list = [['img\ColaQR001.jpg', 25.0, 35.0],
                  ['img\ColaQR002.jpg', 25.0, 35.0],
                  ['img\ColaQR011.jpg', 150.0, 165.0],
                  ]

    for im, lower, upper in image_list:
        print(im, Find_Level_n_ROI(im, hue_lower=lower,
              hue_higher=upper,  debug=False))
