import cv2
import numpy as np

#cwd(),dir()
import os

#ocr
from PIL import Image
import pytesseract
import argparse

#small object removal
from skimage import morphology

#
import matplotlib.pyplot as plt
from matplotlib import interactive

def Show(img,title="image"):
    return
    cv2.imshow(title,img)
    cv2.waitKey()
    

    
def main():
    img1 = cv2.imread("troll_template.png", cv2.IMREAD_GRAYSCALE)
    #Show(img1)
    ##    img2 = cv2.imread("img2.png")
    ##    diff = cv2.absdiff(img1, img2)
        # threshold the diff image so that we get the foreground
    _,im_bin = cv2.threshold(img1, 25, 255, cv2.THRESH_BINARY)

    (trollmask,trollsliced)=segment(im_bin,img1)
    trollmoment=cv2.moments(trollmask)
    trollmoment = [i for i in trollmoment.values()]
    trollmoment=( np.log10(np.abs(trollmoment)), np.sign(trollmoment) )

    namelist = os.listdir('data')

    N=len(namelist)
    errors = np.zeros( N )
    errors_moment = np.zeros( N )
    nos = np.linspace(1,N,N)


    for i,fname in enumerate(namelist):
    ##    if (i<48) | (i>49):
    ##        continue
        print('---'*30)
        print('image '+str(i+1)+'/'+str(N)+' : '+fname)
        img=preprocIm(fname)
        
        img=segment(img)
        cv2.imwrite('gener/'+fname,img)

        merged = overlap(img, trollsliced)
        cv2.imwrite('merged/'+fname,merged)    

        
    ##    Show(img,'Sliced')
    ##        cv2.waitKey()
    ##        cv2.destroyAllWindows()

        errors[i] = computeErr(img, trollmask)
        errors_moment[i] = computeErrMoment(img, trollmoment)
        


    ##    fname = namelist[0]

    M = np.stack((nos,errors,errors_moment))
        #sort by 2nd row, reverse
    ##        a[:,a[1,:].argsort()[::-1] ]
    M = M[:,M[1,:].argsort()[::-1] ]
    interactive(True)
    fig,ax=plt.subplots()
    Zoom=20
    ax.plot(M[1,:Zoom])
    #labels=[str(int(i)) for i in M[0,:Zoom]]
    #ax.set_xticklabels(labels)
    labels=[int(i)-1 for i in M[0,:Zoom]]
    labels=[namelist[i].replace('.jpg','') for i in labels]
    ax.set_xticklabels(labels)
    ax.set_xticks(nos[:Zoom]-1)
    ax.set_title('pixel diff')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    M = M[:,M[2,:].argsort()]
    fig,ax=plt.subplots()
    Zoom=20
    ax.plot(M[2,:Zoom])

    labels=[int(i)-1 for i in M[0,:Zoom]]
    labels=[namelist[i].replace('.jpg','') for i in labels]
    ax.set_xticklabels(labels)
    ax.set_xticks(nos[:Zoom]-1)
    ax.set_title('rms hu moment diff')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def computeErr(img, mask):

    if img.ndim<2:
        print("failed")
        return -1
    hi,wi = img.shape[:2]
    hm,wm = mask.shape[:2]

    #fit mask to img dims
    if (hi != hm) & ( wi != wm):
        mask = cv2.resize(mask, (wi,hi), interpolation=cv2.INTER_CUBIC)
        _,mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

    #compute same part, subtract different
    net = cv2.bitwise_and(img,mask)
    sub = cv2.bitwise_and(img,cv2.bitwise_not(mask))
    SS = np.sum(net/255) - np.sum(sub/255)
    SS/=(hi*wi)
##    print(SS)
##    Show(mask,"mask")
    Show(net,"product")
    return SS


def computeErrMoment(img, moment):
    m1 = cv2.moments(img)
    m1=[i for i in m1.values()]
    #clean up zero moments (empty image)
    L1 = [np.log10(i) if i>0 else 10 for i in np.abs(m1)]
    S1 = [i if i!=0 else 1 for i in np.sign(m1)]
##    print(L1,S1)
    print('len: '+str(len(L1)))
    SS = np.multiply(L1,S1)-np.multiply(moment[0],moment[1])
    SS = np.linalg.norm(SS)/len(L1)
##    print(SS)
    return SS

def overlap(img1, img2):
    hi,wi = img1.shape[:2]
    hm,wm = img2.shape[:2]
        #fit img2 to img dims
    if (hi != hm) & ( wi != wm):
        img2 = cv2.resize(img2, (wi,hi), interpolation=cv2.INTER_CUBIC)

    print('hi:',hi,wi)
    #assume img1,img2 are bw
    M=cv2.bitwise_and(np.ones((hi,wi),dtype='uint8'), img1)
    #background black -> white
    Bg=cv2.bitwise_not(M)
##    print(type(Bg),type(Bg[0][0]),Bg[0][0])

    R=26*M+Bg
    G=240*M+Bg
    B=40*M+Bg
    #build color from mask
    img1 = cv2.merge((B,G,R))
    #restore 3channel from bw
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
    
    out = cv2.addWeighted(img1, 0.6, img2, 0.4, gamma=0)
##    Show(img1,'colored')
##    Show(out,'mixed')
    return out


def segment(image, colored=[]):


    # Copy the thresholded image.
    im_floodfill = cv2.bitwise_not(image.copy())
    im_bin_inv = cv2.bitwise_not(image)

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (1,1)
    cv2.floodFill(im_floodfill, mask, (1,1), 255);

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
##    out = cv2.bitwise_not(im_bin_inv | im_floodfill_inv)
    out = im_bin_inv | im_floodfill_inv

    # Display images.
##    cv2.imshow("OG Image", image)
##    cv2.imshow("Floodfilled Image", im_floodfill)
##    cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
##    cv2.imshow("Inv thres", im_bin_inv)
##    cv2.imshow("Inv thres + inv flood", out)
    

    #remove small islands
##    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8) )
##    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2) )
##    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, se1)
##    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, se2)
    #remove with morph
    lumpSize=np.round( np.min([w, h])/6 )
    processed = morphology.remove_small_objects(out.astype(bool), min_size=lumpSize, connectivity=1).astype(int)

# black out pixels
    mask_x, mask_y = np.where(processed == 0)
    out[mask_x, mask_y] = 0
    Show(out,"Eroded")


        # get the contours in the thresholded image
##    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##    x,y,w,h = cv2.boundingRect(contours[0])
##    cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),2)
##    cv2.imwrite("troll_mask.png",out)

    # get the outline for proper scaling
    pos = np.where(out==255)
    py,px = pos

    if len(px)==0:
        #empty outline
        out = np.zeros((300,300),dtype='uint8')
    else:
        xmin = np.min(px)
        xmax = np.max(px)

        ymin = np.min(py)
        ymax = np.max(py)
        #slice ROI
        out = out[ymin:ymax,xmin:xmax]
        if len(colored) !=0:
            return (out, colored[ymin:ymax,xmin:xmax])
##    cv2.imwrite("troll_mask_sliced.png",out)

    return out

def preprocIm(fname):
    img = cv2.imread('data/'+fname)
    Show(img,'before')


    #threshold
    clow = np.array([0,0,0])
    chigh = np.array([150,150,150])

    mask = cv2.inRange(img, clow, chigh)
    
    #grain remov
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2) )
    mask = cv2.dilate(mask, ker, iterations=1)
    Show(mask,'morphed')
        
    h, w = mask.shape

    letters = pytesseract.image_to_boxes(mask) #"img" older
    letters = letters.split('\n')
    letters = [letter.split() for letter in letters]
    if not letters[0]:
        #print("No letters detected")
        Show(mask,'No letters')

    else:
        Show(mask,'Det '+str(len(letters))+' letters')
##        print(letters)
        for letter in letters:
            if letter[0].isalpha():
                letter[1:] = list(map(int,letter[1:]))

            #fill found letters with black
                cv2.rectangle(mask, (letter[1], h - letter[2]),
                              (letter[3], h - letter[4]), (0,0,0), -1)
    #Show(mask,'mask')
    img = cv2.bitwise_not(mask)
    Show(img,'final preproc')
    
##    mask=np.stack([mask, mask, mask],axis=2)
##    img = img & mask
##    Show(img)
    
    return img




if __name__ == '__main__':
    main()
