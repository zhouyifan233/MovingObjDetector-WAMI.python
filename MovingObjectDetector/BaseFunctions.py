import numpy as np
import skimage.draw as draw

def TimePropagate(detections, transformation_matrix):
    new_detections = []
    for thisdetection in detections:
        x = thisdetection[0,0]
        y = thisdetection[1,0]
        newxyz = transformation_matrix @ np.array([[x], [y], [1]])
        newx = newxyz[0] / newxyz[2]
        newy = newxyz[1] / newxyz[2]
        new_detections.append(np.array([newx, newy]).reshape(2,1))
    return new_detections


def TimePropagate_(detection, transformation_matrix):
    x = detection[0]
    y = detection[1]
    newxyz = transformation_matrix @ np.array([[x], [y], [1]])
    newx = newxyz[0,0] / newxyz[2,0]
    newy = newxyz[1,0] / newxyz[2,0]
    new_detection = [newx, newy]
    return new_detection


def draw_error_ellipse2d(image, mu, sigma, color="k"):
    if image.shape[2] != 3:
        print('The input image should has 3 channels')
    else:
        eigen_values, eigen_vecters = np.linalg.eigh(sigma)
        angle = np.arctan2(eigen_vecters[1,1], eigen_vecters[0,0])
        angle = angle/np.pi*180
        # 95%---5.991     90%---4.605     99%---9.21
        x_value = np.sqrt(eigen_values[0]*5.991)
        y_value = np.sqrt(eigen_values[1]*5.991)
        points = draw.ellipse_perimeter(int(mu[1]), int(mu[0]), int(y_value), int(x_value), angle)
        pointsr = points[0]
        pointsc = points[1]

        (sizerow, sizecol, _) = image.shape
        valid_points_r = []
        valid_points_c = []
        for i in range(len(pointsr)):
            if pointsr[i] > 0 and pointsc[i] > 0 and pointsr[i] < sizerow and pointsc[i] < sizecol:
                valid_points_r.append(pointsr[i])
                valid_points_c.append(pointsc[i])
        image[valid_points_r, valid_points_c, 1] = 255
    return image

