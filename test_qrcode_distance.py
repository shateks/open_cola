from qrcode_distance import Find_Level_n_ROI


def test_Find_Level_n_ROI():
    image_set = [
        ('images/ColaQR001.jpg', 0.61, 0.65),
        # ('images/ColaQR002.jpg', 0.60, 0.62),
        # ('images/ColaQR003.jpg', 0.70, 0.72),
        # ('images/ColaQR004.jpg', 0.68, 0.70),
        # ('images/ColaQR005.jpg', 0.63, 0.65),
        # ('images/ColaQR006.jpg', 0.66, 0.68),
        # ('images/ColaQR007.jpg', 0.64, 0.66),
        # ('images/ColaQR008.jpg', 0.61, 0.63),
        # ('images/ColaQR009.jpg', 0.69, 0.71),
        # ('images/ColaQR010.jpg', 0.62, 0.64),
    ]

    for image_path, lower_bound, upper_bound in image_set:
        result = Find_Level_n_ROI(image_path, debug=False)
        assert result is not None, f"Test failed for {image_path}: No result returned."
        assert isinstance(result, float), (
            f"Test failed for {image_path}: Result is not a tuple."
        )
        assert lower_bound <= result <= upper_bound, (
            f"Test failed for {image_path}: Result {result} not in range {lower_bound} to {upper_bound}."
        )
