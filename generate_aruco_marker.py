import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

marker_id = 23
marker_pixels = 1000

marker_img = cv2.aruco.generateImageMarker(
    aruco_dict,
    marker_id,
    marker_pixels
)

cv2.imwrite("aruco_6x6_id23.png", marker_img)

print("已生成 aruco_6x6_id23.png")
print("打印时请设置 marker 黑色外边框实际边长为 96 mm。")
