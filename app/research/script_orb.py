import cv2
import numpy as np

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


def meth(feature_detector, img, frame, grayframe, flann, number):
    print("ORB")
    # ORB Features Detector
    orb = cv2.ORB_create(nfeatures=15000)

    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # features of picture
    kp_image, desc_image = orb.detectAndCompute(img, None)

    # features of grayframe
    kp_grayframe, desc_grayframe = orb.detectAndCompute(grayframe, None)

    # Brute Force Matching
    # вместе с orb детектором обычно используется cv2.NORM_HAMMING
    # crossCheck=True - означает, что будет меньше совпадений, но более качественных
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_image, desc_grayframe)
    matches = sorted(matches, key=lambda x: x.distance)  # sort matches with distance

    # отбираем только хорошие совпадения
    good_points = []
    for m in matches:
        if m.distance < 30:
            good_points.append(m)

    # matches = sorted(matches, key=lambda x: x.distance)
    good_points = sorted(good_points, key=lambda x: x.distance)

    # создаём картинку, отображающую совпадения
    img_with_matching = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points[:30], grayframe)

    return img_with_matching


def get_rotated_image(image):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=20, scale=1)
    dst = cv2.warpAffine(image, M, (cols, rows))

    # cv2.namedWindow('Rotation', cv2.WINDOW_NORMAL)
    # cv2.imshow('Rotation', dst)
    return dst


def main():
    feature_detector = "ORB"

    # path_to_video = '../resources/beer_table.avi'
    # path_to_test_image = '../resources/images_initial/img1.jpg'
    # path_to_test_image = '../resources/images_initial/img2_2016-03-01 21.42.11.jpg'
    # path_to_test_image = '../resources/images_initial/img3_20160630_160547.jpg'
    path_to_test_image = '../resources/images_initial/img4_20160630_160548.jpg'

    # path_to_template1 = '../resources/images_initial/img1.jpg'
    # path_to_template1 = '../resources/images_initial/img2_2016-03-01 21.42.11.jpg'
    path_to_template1 = '../resources/images_initial/img3_20160630_160547.jpg'
    # path_to_template1 = '../resources/images_initial/img4_20160630_160548.jpg'
    img1 = cv2.imread(path_to_template1, cv2.IMREAD_GRAYSCALE)

    # FlannBasedMatcher - объект для матчинга с параметрами.
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    frame = cv2.imread(path_to_test_image)
    frame_rotated = get_rotated_image(frame)
    frame = frame_rotated

    # convert to gray
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_with_matching1 = meth(feature_detector, img1, frame, grayframe, flann, 1)

    cv2.namedWindow('img_with_matching1', cv2.WINDOW_NORMAL)
    cv2.imshow("img_with_matching1", img_with_matching1)

    PATH_TO_SAVE_IMAGE = '../output/test_img4_template_img3_orb_template_matching_30_best_rotated20.png'
    cv2.imwrite(PATH_TO_SAVE_IMAGE, img_with_matching1)

    # if 'Esc' (k==27) is pressed then break
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
