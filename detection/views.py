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
import os

# Create your views here.

def url_to_image(url):
    # dl img,conv to numpy and read
	r1 = urllib.request.urlopen(url)
	image = np.asarray(bytearray(r1.read()), dtype="uint8")
	image = cv.imdecode(image, cv.IMREAD_COLOR)
	return image



def filterSPNoise(edgeImg):
    # remove salt pepper noise.
    ctr = 0
    lastMedian = edgeImg
    median = cv.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        # get those pixels that gets zeroed out
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        ctr = ctr + 1
        if ctr > 50:
            break
        lastMedian = median
        median = cv.medianBlur(edgeImg, 3)

def findLargestContour(edgeImg):
    contours, hierarchy = cv.findContours(edgeImg,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # get areas of contour,sort and get largest
    contoursArea = []
    for contour in contours:
        area = cv.contourArea(contour)
        contoursArea.append([contour, area])
		
    contoursArea.sort(key=lambda tupl: tupl[1], reverse=True)
    largestContour = contoursArea[0][0]
    return largestContour

def bg_removal(src):
    # src = cv.imread('coolgate.jpeg')
    blurred = cv.GaussianBlur(src, (5, 5), 0)

    blurredFloat = blurred.astype(np.float32) / 255.0
    module_dir = os.path.dirname(__file__)  # get current directory
    file_path = os.path.join(module_dir, 'model.yml')
    edgeDetector = cv.ximgproc.createStructuredEdgeDetection(file_path)
    edges = edgeDetector.detectEdges(blurredFloat) * 255.0
    edges_u8 = np.asarray(edges, np.uint8)
    filterSPNoise(edges_u8)
    contour = findLargestContour(edges_u8)
    # Draw the contour on the original image
    # contourImg = np.copy(src)
    # cv.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv.LINE_AA, maxLevel=1)
    mask = np.zeros_like(edges_u8)
    cv.fillPoly(mask, [contour], 255)

    # finding surefg using erosion
    fgmap = cv.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    # mask1 is probably bg,fgmap is sure fg
    trimap = np.copy(mask)
    trimap[mask == 0] = cv.GC_BGD
    trimap[mask == 255] = cv.GC_PR_BGD
    trimap[fgmap == 255] = cv.GC_FGD

    # do grabcut segmentations
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv.grabCut(src, trimap, rect, bgModel, fgModel, 5, cv.GC_INIT_WITH_MASK)

    # create mask again
    mask2 = np.where((trimap == cv.GC_FGD) | (trimap == cv.GC_PR_FGD),255,0).astype('uint8')

    contour2 = findLargestContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv.fillPoly(mask3, [contour2], 255)

    # blended alpha cut-out
    mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    mask4 = cv.GaussianBlur(mask3, (3, 3), 0)

    alpha = mask4.astype(float) * 1.1
    alpha[mask3 > 0] = 255.0
    alpha[alpha > 255] = 255.0

    foreground = np.copy(src).astype(float)
    foreground[mask4 == 0] = 0
    background = np.ones_like(foreground, dtype=float) * 255.0


    # make alpha mask a binary mask
    alpha = alpha / 255.0
    foreground = cv.multiply(alpha, foreground)
    background = cv.multiply(1.0 - alpha, background)
    cutout = cv.add(foreground, background)

    return cutout

def image_cmp(image1, image1_ref):
    imageref8bit = cv.normalize(image1_ref, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    # imageref8bit=cv.resize(imageref8bit,(960,540))
    image8bit = cv.normalize(image1, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    img_ref = imageref8bit
    img = image8bit

    # img_ref = cv.resize(img_ref, (960, 540))
    # img = cv.resize(img, (960, 540))

    sift = cv.xfeatures2d.SIFT_create()

    keypt1, descriptor1 = sift.detectAndCompute(img_ref, None)
    keypt2, descriptor2 = sift.detectAndCompute(img, None)

    # FLANN BASED MATCHING

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptor1, descriptor2, k=2)

    # flann identifies nearest neighbours, knn srearch for k closest key points
    # It gives list with set of two points in it which contains feature vector of image1 and image2

    good_points = []

    for m, n in matches:
        if(m.distance < 0.9*n.distance):
            good_points.append(m)

    points = min(len(keypt1), len(keypt2))
    perc = len(good_points)/points * 100
    perc=round(perc,2)
    return perc



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
        prodimg=cv.resize(prodimg,(960,540))
        prodimgnobg=bg_removal(prodimg)
        # cv.imshow("Image",cutout)
        result=image_cmp(cutout,prodimgnobg)
        print(f"Result={result}")

        # cv.waitKey(0)
        # cv.destroyAllWindows()
        data={"percentage":result}
        return Response(data=data,status=status.HTTP_200_OK)


