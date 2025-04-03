import cv2
import numpy as np
from matplotlib import pyplot as plt


def imshow(title="Image", image=None, size=10):
    image = image.astype(np.uint8)
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def Find_Level_n_ROI(image_path, debug=False):
    _d = debug
    # Inicjalizacja detektora kodów QR
    qr_code_detector = cv2.QRCodeDetector()

    image_list = ['.\img\ColaQR001.jpg'
                  #             ,
                  #  '.\images\ColaQR002.jpg',
                  #  '.\images\ColaQR003.jpg',
                  #  '.\images\ColaQR004.jpg',
                  #  '.\images\ColaQR005.jpg',
                  #  '.\images\ColaQR006.jpg',
                  #  '.\images\ColaQR007.jpg',
                  #  '.\images\ColaQR008.jpg',
                  #  '.\images\ColaQR009.jpg',
                  #  '.\images\ColaQR010.jpg'
                  ]

    for i, im in enumerate(image_list):
        image_org = cv2.imread(im)
        # if i % 2 == 0:
        #     image_org = cv2.flip(image_org, 1)

        print(im)

    image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    image.shape
    _d and imshow('Bottle in gray', image)
    decoded_text, info_string, points, _ = qr_code_detector.detectAndDecodeMulti(
        image)
    points_dict = {k: points[i].reshape(-1, 2)
                   for i, k in enumerate(info_string)}

    contours = []
    # Jeśli znaleziono kody QR, rysujemy prostokąty
    if points is not None:
        for point in points:
            # point = point[0]
            # print(point)
            pts = point.reshape(-1, 2).astype(int)
            # image = cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            # cv2.drawContours(image, [pts], -1, (0,255,0), thickness = 5)
            contours.append(pts)
    # Wykrywanie kodów QR i rysowanie prostokątów

    centers = np.empty((0, 2), dtype=int)
    for vertices in contours:
        c_x, c_y = vertices.mean(axis=0)
        centers = np.vstack([centers, [int(c_x), int(c_y)]])
    centers[0]
    for c in centers:
        cv2.drawMarker(image_org, c, (0, 255, 0),
                       markerSize=200,	thickness=20)

    _d and imshow('Bottle in color with marked QR', image_org)

    np.linalg.norm(centers[0]-centers[1])

    point_candidates = []
    for p1 in contours[0]:
        for p2 in contours[1]:
            point_candidates.append([p1, p2, np.linalg.norm(p1-p2)])

    point_candidates.sort(key=lambda x: x[2])

    pts = np.array([point_candidates[i][0:2] for i in range(2)]).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    roi_width = abs(top_left[0] - bottom_right[0])
    roi_height = abs(top_left[1] - bottom_right[1])
    roi_margin_x = roi_width // 10
    roi_margin_y = roi_height // 10

    roi_top_left = top_left + (roi_margin_x, roi_margin_y)
    roi_bottom_right = bottom_right + (-roi_margin_x, -roi_margin_y)
    roi_top_right = top_right + (roi_margin_x, -roi_margin_y)
    roi_bottom_left = bottom_left + (-roi_margin_x, roi_margin_y)
    roi_tl_tr_br_bl = np.array(
        [roi_top_left, roi_top_right, roi_bottom_right, roi_bottom_left], dtype=np.int32)

    image_cropped_org = image_org[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    type(image_org)
    image_cropped_blr = cv2.medianBlur(image_cropped_org, 77)
    image_org.shape

    cv2.rectangle(image_org, tuple(roi_tl_tr_br_bl[0]), tuple(
        roi_tl_tr_br_bl[2]), (0, 255, 0), thickness=10)
    _d and imshow('Rectangle between QR', image_org)

    image_roi_gy = cv2.cvtColor(image_cropped_blr, cv2.COLOR_BGR2GRAY)

    image_roi_gy.shape

    _d and imshow('Between QR gray, cropped by 10%', image_roi_gy)
    image_roi_gy_crop_maen = np.mean(
        image_roi_gy, axis=1, keepdims=True).astype(np.uint8)
    image_roi_gy_crop_maen.shape
    image_roi_gy_crop_maen = np.repeat(image_roi_gy_crop_maen, 50, axis=1)
    threshold = image_roi_gy_crop_maen.mean()
    _, image_roi_gy_crop_maen_tsh = cv2.threshold(
        image_roi_gy_crop_maen, 110, 255, cv2.THRESH_BINARY)
    roi_bar = np.full((image_roi_gy_crop_maen.shape[0], 5), 255)
    roi_bar.shape
    image_roi_hsv = cv2.cvtColor(image_cropped_blr, cv2.COLOR_BGR2HSV_FULL)
    roi_hue = image_roi_hsv[:, :, 0:1]
    roi_hue = np.squeeze(roi_hue)

    roi_sat = image_roi_hsv[:, :, 1:2]
    roi_sat = np.squeeze(roi_sat)
    roi_val = image_roi_hsv[:, :, 2:3]
    roi_val = np.squeeze(roi_val)
    roi_component = np.hstack((roi_hue, roi_sat, roi_val))
    _d and imshow('Roi H S V', roi_component)
    # _d and imshow('Space between', roi_hue)
    roi_hue_mean = np.mean(roi_hue, axis=1, keepdims=True).astype(np.uint8)
    roi_hue_mean_50 = np.repeat(roi_hue_mean, 50, axis=1)
    # roi_hue_mean_tsh = cv2.adaptiveThreshold(roi_hue_mean_50, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, roi_hue_mean_tsh = cv2.threshold(
        roi_hue_mean_50, roi_hue_mean_50.mean(), 255, cv2.THRESH_BINARY)
    roi_hue_mean_tsh = roi_hue_mean_tsh.reshape(*roi_hue_mean_tsh.shape)
    roi_val_mean = np.mean(roi_val, axis=1, keepdims=True).astype(np.uint8)
    roi_val_mean_50 = np.repeat(roi_val_mean, 50, axis=1)
    _, roi_val_mean_tsh = cv2.threshold(
        roi_val_mean_50, roi_val_mean_50.mean()+15, 255, cv2.THRESH_BINARY)
    roi_val_mean_tsh = roi_val_mean_tsh.reshape(*roi_val_mean_tsh.shape)
    roi_val_mean_grad = np.abs(np.gradient(roi_val_mean, axis=0))
    roi_sat_mean = np.mean(roi_sat, axis=1, keepdims=True).astype(np.uint8)
    roi_sat_mean_50 = np.repeat(roi_sat_mean, 50, axis=1)
    _, roi_sat_mean_tsh = cv2.threshold(
        roi_sat_mean_50, roi_sat_mean_50.mean(), 255, cv2.THRESH_BINARY)
    roi_sat_mean_tsh = roi_sat_mean_tsh.reshape(*roi_sat_mean_tsh.shape)
    roi_bar = np.full((roi_hue.shape[0], 5), 128)
    # roi_hue.shape
    # roi_bar.shape
    # roi_hue_mean_50.shape
    # roi_hue_mean_50.shape
    # roi_hue_mean_tsh.shape
    # roi_sat.shape
    # roi_sat_mean_50.shape
    # roi_sat_mean_tsh.shape
    # roi_val.shape
    # roi_val_mean.shape
    # roi_val_mean_50.shape
    # roi_val_mean_tsh.shape

    roi_component_mean = np.hstack(
        (roi_hue, roi_bar, roi_hue_mean_50, roi_hue_mean_50, roi_hue_mean_tsh, roi_sat, roi_sat_mean_50, roi_sat_mean_tsh, roi_val, roi_val_mean_50, roi_val_mean_tsh))
    _d and imshow('H S V with bar next to, calculated as mean from every row',
                  roi_component_mean)
    image_roi_gy.shape
    roi_bar.shape
    image_roi_gy_crop_maen.shape
    image_roi_gy_crop_maen_tsh.shape
    image_roi_gy_crop_component = np.hstack(
        (image_roi_gy, roi_bar, image_roi_gy_crop_maen, roi_bar, image_roi_gy_crop_maen_tsh))
    if _d:
        imshow('Roi in gray, with mean and arbitrary threshold',
               image_roi_gy_crop_component)
        plt.plot(image_roi_gy_crop_maen[:, 0], label="Grey")
        plt.plot(roi_hue_mean[:, 0], label="Hue")
        plt.plot(roi_sat_mean[:, 0], label="Sat")
        plt.plot(roi_val_mean[:, 0], label="Val")
        plt.plot(roi_val_mean_grad[:, 0], label="Grad(Val)")
        plt.legend()
        plt.title("Combination of mean every row.")
        plt.show()

    # If from bottom to top is found gradient of Val bigger than 3.5, get this coordinate.
    # Then check if 70% of Hue from bottom to gradient is in range 27 to 32.
    # If yes this range can be considered as fill level.
    idx = np.argmax(roi_val_mean_grad)
    # idx = roi_val_mean_grad.shape[0] -np.argmax(roi_val_mean_grad[::-1] > 3.5)
    hue_bottom_to_grad = roi_hue_mean[idx:]
    percent_in_range = np.count_nonzero((hue_bottom_to_grad > 25.0) & (
        hue_bottom_to_grad < 35.0))/len(hue_bottom_to_grad)
    level = len(roi_val_mean_grad[idx:]) / len(roi_val_mean_grad)
    image_cropped_org_m = np.copy(image_org)
    cv2.line(image_cropped_org_m, (roi_top_left[0], roi_top_left[1]+idx),
             (roi_bottom_right[0], roi_top_left[1]+idx), (0, 0, 255), thickness=10)
    cv2.putText(image_cropped_org_m, '{:2.1f}%'.format(
        level*100), (int(roi_top_left[0]), roi_top_left[1]+idx), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), thickness=5)
    _d and imshow('Space between', image_cropped_org_m)
    return level


if __name__ == "__main__":
    image_list = ['images\ColaQR001.jpg'
                  #             ,
                  #  '.\images\ColaQR002.jpg',
                  #  '.\images\ColaQR003.jpg',
                  #  '.\images\ColaQR004.jpg',
                  #  '.\images\ColaQR005.jpg',
                  #  '.\images\ColaQR006.jpg',
                  #  '.\images\ColaQR007.jpg',
                  #  '.\images\ColaQR008.jpg',
                  #  '.\images\ColaQR009.jpg',
                  #  '.\images\ColaQR010.jpg'
                  ]

    for i, im in enumerate(image_list):
        image_path = cv2.imread(im)
        # if i % 2 == 0:
        #     image_org = cv2.flip(image_org, 1)

        print(image_path, Find_Level_n_ROI(image_path))
