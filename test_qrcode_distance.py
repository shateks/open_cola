from qrcode_distance import Find_Level_n_ROI


def test_Find_Level_n_ROI():
    image_set = [
        ('img/ColaQR001.jpg', 0.61, 0.65, 25.0, 35.0),
        ('img/ColaQR002.jpg', 0.95, 1.00, 25.0, 35.0),
        ('img/ColaQR011.jpg', 0.45, 0.50, 150.0, 165.0)
    ]

    for image_path, lower_bound, upper_bound, hue_l, hue_h in image_set:
        result = Find_Level_n_ROI(
            image_path, hue_lower=hue_l, hue_higher=hue_h, debug=False)
        assert result is not None, f"Test failed for {image_path}: No result returned."
        assert isinstance(result, float), (
            f"Test failed for {image_path}: Result is not a tuple."
        )
        assert lower_bound <= result <= upper_bound, (
            f"Test failed for {image_path}: Result {result} not in range {lower_bound} to {upper_bound}."
        )
