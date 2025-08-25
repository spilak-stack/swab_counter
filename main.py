# 라이브러리 불러오기
import cv2
from utils import *
import time

# 이미지 열기
filename = "swab.jpg" # 입력 이미지 파일명을 적으세요.
img = cv2.imread(filename, cv2.IMREAD_COLOR)



start = time.time()
img = preprocess(img, margin_px=30)
mask_img = masking(img, radius_ratio=0.97, thresh=100)
hsv_correction_img = hsvCorrection(mask_img)
fine_img = removeEdge(hsv_correction_img, thres=50)
centroids1, stats, labels, background = centroidExtract(fine_img, kernel_size=19, margin=1.5)
skeleton = skeletonize(background, stats, labels)
centroids2, stats, labels, background2 = centroidExtract(skeleton, kernel_size=11, margin=5.5)
centroids1 = remove_duplicate_centroids(centroids1, centroids2, 25)
centroids = np.vstack((centroids1,centroids2))

img_vis=img.copy()
overlay = img_vis.copy()
alpha=0.3
for i in centroids.astype(np.int16):
    cv2.circle(overlay, (i[0], i[1]), 18, (60, 60, 200, int(alpha * 255)), -1)
cv2.addWeighted(overlay, alpha, img_vis, 1 - alpha, 0, img_vis)


for i in centroids.astype(np.int16):
    cv2.circle(img_vis, (i[0], i[1]), 18, (40, 40, 255), 2) 

print(centroids)

cv2.putText(img_vis,'COUNT: '+str(len(centroids)), (10,50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255,255,255))
cv2.putText(img_vis, filename, (10,img_vis.shape[0]-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255,255,255))
cv2.putText(img_vis, str(round(time.time() - start,3))+'sec', (10,img_vis.shape[0]-50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255,255,255))





cv2.imshow('visualization', img_vis) # 시각화
cv2.waitKey(0) 
cv2.destroyAllWindows()
