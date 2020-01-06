# coding: utf-8
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import neighbors

class PlotScaleDetermine(object):
    """
    detect image plot scale

    usage:
        >>> psd = PlotScaleDetermine(dataset_dir='data/dataset_digits.npy')
        >>> length_per_pixel, length_of_scale_bar = psd.determine(file_path, verbose=0)
    """
    def __init__(self, dataset_dir='dataset_digits.npy'):
        self.dataset_digits = np.load(dataset_dir)
        self.dataset_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'u', 'm', '*', 'n', '.']
        self.clf = neighbors.KNeighborsClassifier(3, weights='distance')
        self.clf.fit(self.dataset_digits.reshape(len(self.dataset_digits), -1), [i for i in range(15)]*3)
    
    def digit_predict(self, img):
        if img.shape != (15, 10):
            img = cv.resize(img, (10,15), cv.INTER_NEAREST)
        res = self.clf.predict(img.reshape(1,-1))[0]
        return self.dataset_labels[res]
    
    def s4800_determine(self, img, verbose=0, threshold=230):   
        if verbose == 2:
            plt.subplot(3, 1, 1), plt.imshow(img[445:474,350:], cmap='gray')
            plt.xticks([]),plt.yticks([]);
            
        _, scale_bar = cv.threshold(img[445:458,350:], threshold, 255, 0)
        pixels_of_scale_bar = self.pixels_of_scale_bar_determine(scale_bar)
        
        _, length_scale = cv.threshold(img[457:474,550:], threshold, 255, 0)
        length_of_scale_bar = self.length_of_scale_bar_determine(length_scale, verbose)
            
        length_per_pixel = length_of_scale_bar / pixels_of_scale_bar
        
        if verbose == 2:
            print('pixel of scale bar: {} pixel'.format(pixels_of_scale_bar))
            print('length_of_scale_bar: {} nm'.format(length_of_scale_bar))
            print('length per pixel: {} pixel/nm'.format(length_per_pixel))
            plt.subplot(3, 1, 2), plt.imshow(scale_bar, cmap='gray')
            plt.xticks([]),plt.yticks([]);
            plt.title('pixel of scale bar: {} pixel\nlength_of_scale_bar: {} nm'.format(pixels_of_scale_bar, length_of_scale_bar))
            plt.subplot(3, 1, 3), plt.imshow(length_scale, cmap='gray')
            plt.xticks([]),plt.yticks([]);
            
        return length_per_pixel, length_of_scale_bar
        
    def zeiss_determine(self, img, verbose=0, threshold=70):
        # scale info from zeiss SEM is in black color, we reverse the color
        # before we process it with length_of_scale_bar_determine
        img_original = img.copy()
        img = 255 - img
        if verbose == 2:
            plt.subplot(4, 1, 1), plt.imshow(img_original[700:750,20:150], cmap='gray')
            plt.xticks([]),plt.yticks([]);
            
        _, scale_bar = cv.threshold(img[720:750,20:150], threshold, 255, 0)
        pixels_of_scale_bar = self.pixels_of_scale_bar_determine(scale_bar)
        
        _, length_scale = cv.threshold(img[700:720,20:150], threshold, 255, 0)
        length_of_scale_bar = self.length_of_scale_bar_determine(length_scale, verbose)
        
        length_per_pixel = length_of_scale_bar / pixels_of_scale_bar
        
        if verbose == 2:
            print('pixel of scale bar: {} pixel'.format(pixels_of_scale_bar))
            print('length_of_scale_bar: {} nm'.format(length_of_scale_bar))
            print('length per pixel: {} pixel/nm'.format(length_per_pixel))
            plt.subplot(4, 1, 3), plt.imshow(scale_bar, cmap='gray')
            plt.xticks([]),plt.yticks([]);
            plt.title('pixel of scale bar: {} pixel\nlength_of_scale_bar: {} nm'.format(pixels_of_scale_bar, length_of_scale_bar))
            plt.subplot(4, 1, 4), plt.imshow(length_scale, cmap='gray')         
            plt.xticks([]),plt.yticks([]);
        
        return length_per_pixel, length_of_scale_bar
        
    def pixels_of_scale_bar_determine(self, img):
        left = None
        right = None
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j]:
                    if left == None:
                        left = j
                    elif left > j:
                        left = j
                    if right == None:
                        right = j
                    else:
                        right = max(j, right)
        return right - left + 1
    
    def length_of_scale_bar_determine(self, img, verbose=0):
        _, contours, _ = cv.findContours(img.copy(), 0, 1)
        
        orc_res = []
        # sort contours by x coordinate to keep the content in correct order
        contours = sorted(contours, key=lambda x:x[0][0][0])
        for i in range(len(contours)):
            x,y,w,h = cv.boundingRect(contours[i])
            orc_res.append(self.digit_predict(img[y:y+h,x:x+w]))
            if verbose == 3:
                plt.subplot(len(contours), 1, i+1), plt.imshow(img[y:y+h,x:x+w], cmap='gray')
                plt.xticks([]),plt.yticks([]);
        
        if verbose == 3:
            print(orc_res)
        factor = 1
        res = ''
        # if the unit is um, then we convert it into nm by multiply 1000
        for i in orc_res:
            if i.isdigit() or i == '.':
                res += i
            elif i == 'u':
                factor = 1000
            else:
                pass

        res = float(res) * factor
        
        return res
        
    def determine(self, file_path, verbose=0):
        img = cv.imread(file_path, 0)
        if img.shape == (768, 1024):
            length_per_pixel, length_of_scale_bar = self.zeiss_determine(img, verbose)
        else:
            length_per_pixel, length_of_scale_bar = self.s4800_determine(img, verbose)

        if verbose:
            if img.shape == (768, 1024):
                device = 'zeiss'
            else:
                device = 's4800'
            print('{} | device {} | scale bar {}nm | length per pixel {}'.format(file_path, device, length_of_scale_bar, length_per_pixel))
        
        if length_of_scale_bar % 100 != 0:
            raise ValueError('This one may be wrong:', file_path)
            
        return length_per_pixel, length_of_scale_bar

if __name__ == '__main__':
 
    model = PlotScaleDetermine()

    filename2 = 'sem_original/20171005-sem/35-1-BPDC_q007.jpg'
    img2 = cv.imread(filename2, 0)
    model.s4800_determine(img2, 2, 230)

    filename = 'opencv_measure/6.jpg'
    img = cv.imread(filename, 0)
    model.zeiss_determine(img, 2)
