from django.shortcuts import render,get_object_or_404
from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
import cv2 as cv
import numpy as np
import io
from .models import *
from .serializers import *
import base64
from imageio import imread
import urllib.request


def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv.imdecode(image, cv.IMREAD_COLOR)
	# return the image
	return image


# Create your views here.

def filterOutSaltPepperNoise(edgeImg):
    # Get rid of salt & pepper noise.
    count = 0
    lastMedian = edgeImg
    median = cv.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        # get those pixels that gets zeroed out
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 50:
            break
        lastMedian = median
        median = cv.medianBlur(edgeImg, 3)

def findLargestContour(edgeImg):
    contours, hierarchy = cv.findContours(
        edgeImg,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    # From among them, find the contours with large surface area.
    contoursWithArea = []
    for contour in contours:
        area = cv.contourArea(contour)
        contoursWithArea.append([contour, area])
		
    contoursWithArea.sort(key=lambda tupl: tupl[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

def bg_removal(src):
    # src = cv.imread('coolgate.jpeg')
    blurred = cv.GaussianBlur(src, (5, 5), 0)

    blurred_float = blurred.astype(np.float32) / 255.0
    edgeDetector = cv.ximgproc.createStructuredEdgeDetection(r"D:/model.yml")
    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    # cv.imwrite('edge-raw.jpg', edges)
    edges_8u = np.asarray(edges, np.uint8)
    filterOutSaltPepperNoise(edges_8u)
    # cv.imwrite('edge.jpg', edges_8u)
    contour = findLargestContour(edges_8u)
    # Draw the contour on the original image
    contourImg = np.copy(src)
    cv.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv.LINE_AA, maxLevel=1)
    # cv.imwrite('contour.jpg', contourImg)

    mask = np.zeros_like(edges_8u)
    cv.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    # mark inital mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv.GC_BGD
    trimap[mask == 255] = cv.GC_PR_BGD
    trimap[mapFg == 255] = cv.GC_FGD

    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv.GC_FGD] = 255
    # cv.imwrite('trimap.png', trimap_print)

    # run grabcut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv.grabCut(src, trimap, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

    # create mask again
    mask2 = np.where((trimap == cv.GC_FGD) | (trimap == cv.GC_PR_FGD),255,0).astype('uint8')
    # cv.imwrite('mask2.jpg', mask2)

    contour2 = findLargestContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv.fillPoly(mask3, [contour2], 255)

    # blended alpha cut-out
    mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    mask4 = cv.GaussianBlur(mask3, (3, 3), 0)
    alpha = mask4.astype(float) * 1.1  # making blend stronger
    alpha[mask3 > 0] = 255.0
    alpha[alpha > 255] = 255.0

    foreground = np.copy(src).astype(float)
    foreground[mask4 == 0] = 0
    background = np.ones_like(foreground, dtype=float) * 255.0

    # cv.imwrite('foreground.png', foreground)
    # cv.imwrite('background.png', background)
    # cv.imwrite('alpha.png', alpha)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha / 255.0
    # Multiply the foreground with the alpha matte
    foreground = cv.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv.multiply(1.0 - alpha, background)
    # Add the masked foreground and background.
    cutout = cv.add(foreground, background)

    # cv.imwrite('cutout.jpg', cutout)
    return cutout

def image_cmp(image1, image1_ref):
    imageref8bit = cv.normalize(image1_ref, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    image8bit = cv.normalize(image1, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    img_ref = imageref8bit
    img = image8bit

#     img_ref = cv.resize(img_ref, (960, 540))
#     img = cv.resize(img, (960, 540))

    sift = cv.xfeatures2d.SIFT_create()

    kp_1, desc_1 = sift.detectAndCompute(img_ref, None)
    kp_2, desc_2 = sift.detectAndCompute(img, None)

    # print("Key points 1st image " + str(len(kp_1)))
    # print("Key points 2nd image " + str(len(kp_2)))

#     bf = cv.BFMatcher(
#         # cv.NORM_L2, 
#         # crossCheck=True
#         )

    # Match descriptors.
    # matches = bf.match(des1,des2)
    # matches=bf.knnMatch(desc_1, desc_2, k=2)

    # FLANN BASED MATCHING

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    # flann identifies nearest neighbours, knn srearch for k closet key points
    # It gives list with set of two points in it which contains feature vector of image1 and image2

    good_points = []

    for m, n in matches:
        if(m.distance < 0.8*n.distance):
            good_points.append(m)


    # print(len(matches))
    # print("Good matches " + str(len(good_points)))

    points = min(len(kp_1), len(kp_2))
    perc = len(good_points)/points * 100
    # print("How good 2nd image is ", perc)
    # print()
    #match_perc.append(perc)

    result = cv.drawMatches(img_ref, kp_1, img, kp_2, good_points, None)

    result = cv.resize(result, (960, 540))
    return perc
    # cv.imshow("result", result)
    # cv.imshow("Ref", img_ref)
    # cv.imshow("Img", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()



class ListCompany(APIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = ()

    def get(self,request,format=None):
        companies=Company.objects.all()
        print(companies)
        serializer = CompanySerializer(companies,many=True)
        print(serializer)
        
        return Response(data=serializer.data,status=status.HTTP_200_OK)

class UploadImage(APIView):
    permission_classes = (permissions.AllowAny,)
    authentication_classes = ()

    def post(self,request,format=None):
        companyID=request.data.get('companyID')
        productID=request.data.get('productID')
        print(companyID)
        print(productID)
        image=request.data.get('image')
        # print(image)
        # fd = image.read()
        print(type(image))
        print(type(image.file))
        image_b64 = base64.b64encode(image.read())
        img = imread(io.BytesIO(base64.b64decode(image_b64)))
        
        half=cv.resize(img,(960,540))
        # cv.imshow("Image",half)
        cutout=bg_removal(half)
        prodobj=get_object_or_404(Product,pk=productID)
        prodimgurl=prodobj.productImg
        prodimg=url_to_image(prodimgurl)
        prodimgnobg=bg_removal(prodimg)

        result=image_cmp(cutout,prodimgnobg)
        print(f"Result={result}")

        # cv.waitKey(0)
        # cv.destroyAllWindows()
        data={"percentage":result}
        return Response(data=data,status=status.HTTP_200_OK)


