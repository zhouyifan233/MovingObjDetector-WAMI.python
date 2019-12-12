import cv2
import numpy as np
import timeit

def CalcHomography(frame1, frame2, num_of_features=3000):
    starttime = timeit.default_timer()

    #surfdetector = cv2.xfeatures2d.SURF_create(hessianThreshold=3000)
    siftdetector = cv2.xfeatures2d.SIFT_create(num_of_features, edgeThreshold=5, contrastThreshold=0.05)
    sift_pnts1, sift_des1 = siftdetector.detectAndCompute(frame1, None)
    sift_pnts2, sift_des2 = siftdetector.detectAndCompute(frame2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(sift_des1, sift_des2)

    pts_src = []
    pts_dst = []
    cnt_matched_features = 0
    for mat in matches:
        dist = mat.distance
        if dist <= 200:
            pnt1 = sift_pnts1[mat.queryIdx].pt
            pnt2 = sift_pnts2[mat.trainIdx].pt
            pts_src.append(pnt1)
            pts_dst.append(pnt2)
            cnt_matched_features += 1

    if cnt_matched_features <= 1000:
        print("Less number of matched features(" + str(cnt_matched_features) + "), estimation might be unreliable...")
    H, status = cv2.findHomography(np.array(pts_src), np.array(pts_dst), method=cv2.RANSAC, maxIters=10000, confidence=0.998)
    match_ratio = np.sum(status) / len(status)

    endtime = timeit.default_timer()
    #print("Processing Time (Homography Estimation): " + str(endtime - starttime) + " s, score: " + str(match_ratio) + "...")
    return H, match_ratio

def ImageRegistration(srcImg, dstShape, H):
    starttime = timeit.default_timer()
    if dstShape is None:
        im_out = cv2.warpPerspective(srcImg, H, (srcImg.shape[1], srcImg.shape[0]), borderValue=255, flags=cv2.INTER_LINEAR)
    else:
        im_out = cv2.warpPerspective(srcImg, H, (dstShape[1], dstShape[0]), borderValue=255, flags=cv2.INTER_LINEAR)

    endtime = timeit.default_timer()
    #print("Processing Time (Image Registration): " + str(endtime - starttime) + " s... ")

    return im_out

