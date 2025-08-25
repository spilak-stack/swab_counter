import cv2
import numpy as np

def calPixelmean(img):
  black_mask = np.all(img == [0, 0, 0], axis=-1)
  white_mask=~black_mask
  a=np.sum(white_mask)
  if a!=0:                # 검은색 이미지가 아닌 경우에만 진행
    b=np.sum(img)
    x=b/a/3
  else: x=0
  return x

def hsvCorrection(img, num_divisions=6):
  """
  이미지를 25등분하여 각 부분을 저장하는 함수

  Args:
    img: 마스킹된 이미지
    num_divisions: 가로/세로 등분 수 (기본값 6)
  """

  height, width, _ = img.shape

  # 각 부분의 높이와 너비 계산
  part_height = height // num_divisions
  part_width = width // num_divisions
  merged_img = np.zeros((height, width), dtype=np.uint8)

  for i in range(num_divisions):
    for j in range(num_divisions):
      # 각 부분의 좌상단, 우하단 좌표 계산
      y1 = i * part_height
      y2 = (i + 1) * part_height
      x1 = j * part_width
      x2 = (j + 1) * part_width
      
      # 이미지 자르기
      part_img = img[y1:y2, x1:x2]
      part_img_lst=removePartEdge(part_img)
      if len(part_img_lst)==1:
        hsv_cor_part_img=hsvTF(part_img_lst[0])
      else:
        hsv_cor_part_img=cv2.add(hsvTF(part_img_lst[0]),hsvTF2(part_img_lst[1]))
      merged_img[y1:y2, x1:x2]=hsv_cor_part_img
  return merged_img


def hsvTF(mask_img):
  '''
  hsv 보정 이미지 추출
  '''
  x = calPixelmean(mask_img)
  if x!=0:                # 검은색 이미지가 아닌 경우에만 진행
    hsv_image = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
    v_high=1.11*x+40.25
    v_low=0.79*x+20.36
    s_low=0.5
    s_high=40

    lower_hsv = np.array([0, s_low, v_low])  # H / S / V
    upper_hsv = np.array([180, s_high, v_high])  # H / S / V
    correction_img = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
  else:
    correction_img=cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
  return correction_img

def hsvTF2(mask_img):
  '''
  hsv 보정 이미지 추출
  '''
  x = calPixelmean(mask_img)
  if x!=0:                # 검은색 이미지가 아닌 경우에만 진행
    hsv_image = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
    v_high=0.88*x+100.60
    v_low=0.78*x+66.80
    s_low=0.5
    s_high=35

    lower_hsv = np.array([0, s_low, v_low])  # H / S / V
    upper_hsv = np.array([180, s_high, v_high])  # H / S / V
    correction_img = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
  else:
      correction_img=cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
  return correction_img



def removePartEdge(part_img):
  black_mask = np.all(part_img == [0, 0, 0], axis=-1)
  white_mask=~black_mask
  if np.sum(white_mask)==0:
    return [part_img]
  else: 
    rate=np.sum(black_mask)/np.sum(white_mask)
  if rate>=0.05:
      gray_part_img=cv2.cvtColor(part_img,cv2.COLOR_BGR2GRAY)
      _, bin_part_img=cv2.threshold(gray_part_img,1,255,cv2.THRESH_BINARY)
      kernel=np.ones((3,3),np.uint8)
      bin_part_img=cv2.morphologyEx(bin_part_img,cv2.MORPH_OPEN, kernel, iterations=1)
      bin_part_img=cv2.morphologyEx(bin_part_img,cv2.MORPH_CLOSE, kernel, iterations=1)
      bin_part_img2=cv2.morphologyEx(bin_part_img,cv2.MORPH_ERODE, kernel, iterations=20)
      mask=cv2.subtract(bin_part_img,bin_part_img2)
      
      mask = mask > 0
      masked_img = np.zeros_like(part_img)
      masked_img[mask] = part_img[mask]
      bin_part_img2=bin_part_img2>0
      masked2_img = np.zeros_like(part_img)
      masked2_img[bin_part_img2] = part_img[bin_part_img2]

      x1 = calPixelmean(masked_img)
      x2 = calPixelmean(masked2_img)


      if x1<(x2*0.80):      # 1에 가까울수록 테두리 별도보정을 많이 하게 됨
          return [masked2_img, masked_img]    #  [면봉 부분, 필요없는 부분]
      else: return [part_img]
  else: return [part_img]



def removeEdge(bw_img, thres=50):
  '''
  엣지를 제거한 정제된 이진 이미지 추출
  '''
  _, bin_img=cv2.threshold(bw_img,thres,255,cv2.THRESH_BINARY)
  edges = cv2.Canny(bin_img, 150, 255)
  kernel=np.ones((3,3),np.uint8)
  edges=cv2.morphologyEx(edges,cv2.MORPH_DILATE, kernel, iterations=2)
  fine_img=cv2.subtract(bin_img,edges)
  fine_img=cv2.morphologyEx(fine_img,cv2.MORPH_CLOSE, kernel, iterations=2)
  return fine_img

def preprocess(img, margin_px=50):
  '''
  원검출 하기 쉬운 이미지로 사이즈를 조절
  '''
  bw_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 원본이미지에서 테두리 원검출
  blurred = cv2.GaussianBlur(bw_img, (5, 5), 0)
  _, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  largest_contour = max(contours, key=cv2.contourArea)
  (x, y), radius = cv2.minEnclosingCircle(largest_contour)

  adjusted_radius = radius * 1.03  
  mask = np.zeros(img.shape[:2], dtype="uint8")
  cv2.circle(mask, (int(x), int(y)), int(adjusted_radius), 255, -1)
  masked_image = cv2.bitwise_and(bw_img, bw_img, mask=~mask)

  # 검출된 테두리 원 바깥쪽의 최대 픽셀값 탐색
  adjusted_img=cv2.subtract(bw_img,int(np.max(masked_image)))
  adjusted_img=cv2.multiply(adjusted_img, int(255/np.max(adjusted_img)))
  _, bin_img=cv2.threshold(adjusted_img,30,255,cv2.THRESH_BINARY)


  # 이미지에서 흰색 픽셀의 좌표 찾기
  white_pixels = np.argwhere(bin_img == 255)
  x_coords, y_coords = white_pixels[:, 1], white_pixels[:, 0]
  width = x_coords.max() - x_coords.min() + 1
  height = y_coords.max() - y_coords.min() + 1


  if y_coords.min()-margin_px < 0 or x_coords.min()-margin_px < 0 or y_coords.min()+height+margin_px > img.shape[0] or x_coords.min()+width+margin_px > img.shape[1] :
    crop_size=(y_coords.min(), y_coords.min()+height, 
                x_coords.min(), x_coords.min()+width)
  else:
    crop_size=(y_coords.min()-margin_px, y_coords.min()+height+margin_px, 
              x_coords.min()-margin_px, x_coords.min()+width+margin_px)
  

  crop_image=img[crop_size[0] : crop_size[1],
                  crop_size[2] : crop_size[3]]

  ratio=width/height
  resize=(1000, int(1000*ratio))
  resized_img=cv2.resize(crop_image, resize)

  return resized_img

def masking(img, radius_ratio:float=0.98, thresh:int=100):
  '''
  배경을 없애는 마스킹 작업
  '''

  bw_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(bw_img, (5, 5), 0)
  _, threshold = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  largest_contour = max(contours, key=cv2.contourArea)
  (x, y), radius = cv2.minEnclosingCircle(largest_contour)


  adjusted_radius = radius * radius_ratio  

  # 마스크 초기화 (1000x1000 크기의 검은색 배열 생성)
  mask = np.zeros(img.shape[:2], dtype="uint8")

  # 마스크에 원 그리기 (줄인 반지름으로 원 그리기)
  cv2.circle(mask, (int(x), int(y)), int(adjusted_radius), 255, -1)

  # 원형 마스크 적용 (비트 연산 사용)
  masked_image = cv2.bitwise_and(img, img, mask=mask)

  return masked_image

def centroidExtract(fine_img, kernel_size=19, margin=0.5):
  '''
  가우시안 블러 기반 면봉 중심점 추출
  '''     
  blurred_img = cv2.GaussianBlur(fine_img.astype(np.float32), (kernel_size, kernel_size), 0)
  maximum_value=np.max(blurred_img)
  matching_pixels = np.where(blurred_img >= maximum_value-margin)

  # 좌표 목록 생성 (행, 열 순서)
  coor = np.row_stack(matching_pixels)

  background = np.zeros(shape=fine_img.shape, dtype=np.uint8) # type: ignore
  for x, y in zip(coor[1],coor[0]):
      background[y, x]=255

  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(background)
  unique_centroids=remove_close_points(centroids[1:], 25)
  return unique_centroids, stats, labels, background

def remove_close_points(points, threshold):
  """
  2차원 좌표 배열에서 서로 가까운 점들을 제거합니다.

  Args:
    points: 2차원 좌표 배열 (shape: (n, 2))
    threshold: 두 점 사이의 거리 임계값

  Returns:
    가까운 점들이 제거된 좌표 배열
  """

  keep_indices = np.ones(len(points), dtype=bool)
  for i in range(len(points)):
    if keep_indices[i]:
      for j in range(i + 1, len(points)):
        if keep_indices[j] and np.linalg.norm(points[i] - points[j]) < threshold:
          keep_indices[j] = False
  return points[keep_indices]




def skeletonize(background, stats, labels):
  sizes = stats[1:, cv2.CC_STAT_AREA]  # 첫 번째 객체는 배경이므로 제외
  mean_size = np.mean(sizes)

  background2 = np.zeros_like(background)
  for i in range(1, len(stats)):  # 배경(라벨 0) 제외
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= mean_size*1.5:
      background2[labels == i] = 255

  dst = cv2.distanceTransform(background2, cv2.DIST_L2, 5)
  # 거리 값을 0 ~ 255 범위로 정규화 ---②
  dst = (dst/(dst.max()-dst.min()) * 255).astype(np.uint8)
  # 거리 값에 쓰레시홀드로 완전한 뼈대 찾기 ---③
  skeleton = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -3)
  kernel=np.ones((3,3),np.uint8)

  dilate_skeleton=cv2.morphologyEx(skeleton,cv2.MORPH_DILATE, kernel, iterations=2)
  return dilate_skeleton


def remove_duplicate_centroids(centroids1, centroids2, threshold):
    """
    centroids1과 centroids2에서 중복된 좌표를 제거합니다.
    centroids1의 각 좌표와 centroids2의 각 좌표를 비교해 임계값 내에 있는 경우 centroids1에서 해당 좌표를 제거합니다.

    Args:
      centroids1: 첫 번째 좌표 배열 (shape: (n, 2))
      centroids2: 두 번째 좌표 배열 (shape: (m, 2))
      threshold: 두 좌표 간의 거리 임계값

    Returns:
      중복된 좌표가 제거된 centroids1 배열
    """
    # 유지할 centroids1의 인덱스를 추적하는 리스트
    keep_indices = np.ones(len(centroids1), dtype=bool)

    # centroids1과 centroids2의 각 좌표 비교
    for i in range(len(centroids1)):
        for j in range(len(centroids2)):
            distance = np.linalg.norm(centroids1[i] - centroids2[j])  # 두 좌표 간의 거리 계산
            if distance < threshold:  # 거리가 임계값보다 작으면 중복된 것으로 간주
                keep_indices[i] = False  # 해당 좌표를 제거하기 위해 False로 설정
                break  # 중복이 확인되면 더 이상 비교할 필요 없으므로 반복문 종료

    # 중복된 좌표가 제거된 centroids1 반환
    return centroids1[keep_indices]
# centroids3 = remove_duplicate_centroids(centroids1, centroids2, 25)
