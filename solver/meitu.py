import cv2

# 加载一张图片
img = cv2.imread('h.jpg')

# # 创建一个窗口
# cv2.namedWindow('image')
#
#
# # 定义函数：实时进行鼠标状态的监听
# def draw(event, x, y, flags, param):
#     # 判断鼠标的事件
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print('鼠标按下')
#     elif event == cv2.EVENT_MOUSEMOVE:
#         print('鼠标移动')
#     elif event == cv2.EVENT_LBUTTONUP:
#         print('鼠标弹起')
#
#
# # 鼠标事件的回调
# cv2.setMouseCallback('image', draw)
#
# # 展示窗口
# cv2.imshow('image', img)
#
# # 窗口等待
# cv2.waitKey(0)
#
# # 销毁窗口
# cv2.destroyAllWindows()


# 图像模糊效果
# 模糊程度，越大越模糊
# img_dist = cv2.blur(img, (15, 15))


img_dist = cv2.blur(img, (15, 15))

# 美白效果
# val = 120  # val值越大，美白程度越大
# img_dist = cv2.bilateralFilter(img, val, val * 2, val / 2)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img_dist)
print(cv2.imwrite('heyulong_dist.jpg', img_dist))
cv2.waitKey(0)
cv2.destroyAllWindows()


