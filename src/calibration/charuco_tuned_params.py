# charuco_tuned_params.py
import cv2

def tuned_charuco_params(resolution=(1920, 1080)):
    params = cv2.aruco.DetectorParameters()

    w, h = resolution
    short_side = min(w, h)

    # adaptive threshold windows
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshConstant = 7

    # marker perimeter relative to image size
    if short_side < 900:  # 720p
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
    else:  # 1080p+
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 5.0

    params.minCornerDistanceRate = 0.05
    params.minOtsuStdDev = 3.0
    params.polygonalApproxAccuracyRate = 0.03
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementMaxIterations = 60
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMinAccuracy = 0.1
    params.detectInvertedMarker = True  # robust for lighting reversals

    return params
