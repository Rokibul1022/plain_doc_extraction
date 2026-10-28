"""Classical license-plate scan: 4 corners -> straighten -> clean -> plain page. OpenCV + NumPy only.

    from opencv import scan
    result = scan(cv2.imread("plate.jpg"))        # result.document is the plain black-on-white page
"""

from opencv.scan import CvScan, find_corners, scan

__all__ = ["scan", "find_corners", "CvScan"]
