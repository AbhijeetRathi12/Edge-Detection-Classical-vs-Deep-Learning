#!/usr/bin/env python3

import numpy as np
import cv2
from scipy.signal import convolve
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import imutils

def visualize_filters(filters, columns, title):
    no_of_rows = int((len(filters) / columns) + ((len(filters) % columns) != 0))
    plt.figure(figsize=(15, 15))
    
    for index, filt in enumerate(filters, start=1):
        plt.subplot(no_of_rows, columns, index)
        plt.axis('off')
        plt.imshow(filt, cmap='gray')
    plt.savefig(title)
    
def Gaussian_kernel(sigma, kernel_size):
    
    if (kernel_size % 2) == 0:
        index = kernel_size / 2
    else:
        index = (kernel_size - 1) / 2

    x, y = np.meshgrid(np.linspace(-index, index, kernel_size), np.linspace(-index, index, kernel_size))
    sigma_x = sigma
    sigma_y = sigma
    kernel_value = np.exp((-1/2) * (np.square(x) / np.square(sigma_x) + np.square(y) / np.square(sigma_y))) * 1 /np.sqrt (2 * np.pi * sigma_x * sigma_y)
    kernel_2D = np.reshape(kernel_value, (kernel_size, kernel_size))
    # kernel_2D = kernel_value / np.sum(kernel_value)
    return kernel_2D

def DoGFilter(scales, orientation, kernel_size):
    filter_bank = []

    # Sobel Kernels in X and Y direction respectively
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    orientations = np.linspace(0, 360, orientation)
    
    for i in scales:
        Gaussian = Gaussian_kernel(i, kernel_size)
        Gaussian_x = convolve(Gaussian, Sx, mode='same')
        Gaussian_y = convolve(Gaussian, Sy, mode='same')
        
        for angle in orientations:
            cos_theta = np.cos(np.deg2rad(angle))
            sin_theta = np.sin(np.deg2rad(angle))
            
            filters = (Gaussian_x * cos_theta) + (Gaussian_y * sin_theta)
            filter_bank.append(filters)
    
    return filter_bank

def Gaussian_Kernel_LM(sigma, kernel_size):	

	sigma_x, sigma_y = sigma
	Gaussian = np.zeros([kernel_size, kernel_size])
   
	if (kernel_size%2) == 0:
		index = kernel_size/2
	else:
		index = (kernel_size - 1)/2

	x, y = np.meshgrid(np.linspace(-index, index, kernel_size), np.linspace(-index, index, kernel_size))
	Gaussian = (1 /np.sqrt (2 * np.pi * sigma_x * sigma_y)) * np.exp(-((np.square(x)/np.square(sigma_x)) + (np.square(y)/np.square(sigma_y))/2))
	return Gaussian

def LMFilter(scales, orientations, filter_size):
    scales_first_set = scales[0:3]
    scales_gaussian = scales
    scales_laplacian = scales + [i * 3 for i in scales]

    filter_bank = []
    gaussian_1d_filters = []
    gaussian_2d_filters = []
    gaussian_filters = []
    laplacian_filters = []

    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for scale in scales_first_set:
        sigma_values = [3 * scale, scale]
        gaussian_kernel = Gaussian_Kernel_LM(sigma_values, filter_size)

        gaussian_1d = cv2.filter2D(gaussian_kernel, -1, Sx) + cv2.filter2D(gaussian_kernel, -1, Sy)
        gaussian_2d = cv2.filter2D(gaussian_1d, -1, Sx) + cv2.filter2D(gaussian_1d, -1, Sy)

        for orientation in range(orientations):
            orientation_degrees = orientation * 180 / orientations

            rotated_gaussian_1d = imutils.rotate(gaussian_1d, orientation_degrees)
            gaussian_1d_filters.append(rotated_gaussian_1d)

            rotated_gaussian_2d = imutils.rotate(gaussian_2d, orientation_degrees)
            gaussian_2d_filters.append(rotated_gaussian_2d)

    for scale in scales_laplacian:
        sigma_values = [scale, scale]
        gaussian_kernel = Gaussian_Kernel_LM(sigma_values, filter_size)
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        laplacian_filters.append(cv2.filter2D(gaussian_kernel, -1, laplacian_kernel))

    for scale in scales_gaussian:
        sigma_values = [scale, scale]
        gaussian_filters.append(Gaussian_Kernel_LM(sigma_values, filter_size))

    filter_bank = gaussian_1d_filters + gaussian_2d_filters + laplacian_filters + gaussian_filters
    return filter_bank

def gabor(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

def sin_func(kernel_size, freq, angle):
    if (kernel_size % 2) == 0:
        index = kernel_size / 2
    else:
        index = (kernel_size - 1) / 2

    (x, y) = np.meshgrid(np.linspace(-index, index, kernel_size), np.linspace(-index, index, kernel_size))
    theta_x = x * np.cos(angle) + y * np.sin(angle)
    theta_y = -x * np.sin(angle) + y * np.cos(angle)

    theta = theta_x + theta_y
    sine_wave = np.sin(theta * 2 * np.pi * freq / kernel_size)

    return sine_wave

def GaborFilter(scales_list, orientations, freq_list, kernel_size):
    filter_bank = []
    for scale in scales_list:
        sigma = [scale, scale]
        gaussian_kernel = Gaussian_Kernel_LM(sigma, kernel_size)
        for frequency in freq_list:
            for orientation in range(orientations):
                angle = orientation * np.pi / orientations
                sin_wave_2d = sin_func(kernel_size, frequency, angle)
                gabor_filter = gaussian_kernel * sin_wave_2d
                filter_bank.append(gabor_filter)
    return filter_bank

def Half_Disk(radius, angle):
    size = 2 * radius + 1
    center = radius
    half_disk = np.zeros([size, size])
    
    for i in range(radius):
        for j in range(size):
            distance = (i - center) ** 2 + (j - center) ** 2
            if distance <= radius ** 2:
                half_disk[i, j] = 1
    
    half_disk = np.rot90(half_disk, k=int(angle / 90))
    half_disk = np.where(half_disk <= 0.5, 0, 1)
    return half_disk

def HalfDiskFilters(radii_list, orientations):
    filter_bank = []
    for radius in radii_list:
        temp = []
        for orientation in range(orientations):
            angle = orientation * 360 / orientations
            half_disk_filter = Half_Disk(radius, angle)
            temp.append(half_disk_filter)
        
        filter_bank.extend(temp[:orientations // 2])
        filter_bank.extend(temp[orientations // 2:])
    
    return filter_bank

def TextronMap(img, DoG, LM, Gabor):
    maps = np.array(img)
    for i in range(len(DoG)):
        conv = cv2.filter2D(img,-1, DoG[i])
        maps = np.dstack((maps,conv))
    for i in range(len(LM)):
        conv = cv2.filter2D(img,-1, LM[i])
        maps = np.dstack((maps,conv))
    for i in range(len(Gabor)):
        conv = cv2.filter2D(img,-1, Gabor[i])
        maps = np.dstack((maps,conv))
    maps = maps[:,:,1:]
    return maps

def Texton(img, num):
    a,b,c = img.shape
    img = np.reshape(img,((a*b),c))
    kmeans = KMeans(n_clusters = num, random_state = 4)
    kmeans.fit(img)
    labels = kmeans.predict(img)
    map = np.reshape(labels,(a,b))
    return map

def ChiSquareDist(img, bins, filter1, filter2):
    Img = np.float32(img)
    Chi_Square_Distance = Img.copy()
    tmp = np.zeros(Img.shape)
    for i in range(bins):
        tmp[Img == i] = 1.0
        tmp[Img != i] = 0.0
        g_i = cv2.filter2D(tmp, -1, filter1)
        h_i = cv2.filter2D(tmp, -1, filter2)
        Chi_Square_Distance = (Chi_Square_Distance + ((g_i - h_i)**2 /(g_i + h_i + np.exp(-10))))/2.0
    return Chi_Square_Distance

def Gradient(img, bins, half_disk):
    for i in range(int(len(half_disk)/2)):
        left = half_disk[i]
        right = half_disk[i+1]
        dist = ChiSquareDist(img, bins, left, right)
        gradient = np.dstack((img, dist))
        Gradient = gradient[:,:,1:]
    return Gradient

def BrightnessMap(img, num):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a,b= img.shape
    img = np.reshape(img,((a*b),1))
    kmeans = KMeans(n_clusters = num, random_state = 4)
    kmeans.fit(img)
    labels = kmeans.predict(img)
    map = np.reshape(labels,(a,b))
    return map

def ColorMap(img, num):
    a,b,c= img.shape
    img = np.reshape(img,((a*b),c))
    kmeans = KMeans(n_clusters = num, random_state = 4)
    kmeans.fit(img)
    labels = kmeans.predict(img)
    map = np.reshape(labels,(a,b))
    return map

def main():
    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    scale = 2
    orientations = 16
    kernel_size = 41
    
    DOG_filter_bank = DoGFilter([scale], orientations, kernel_size)
    visualize_filters(DOG_filter_bank, columns=4, title="DoG")
    img = cv2.imread("DoG.png")
    cv2.imwrite("Phase1\Code\Results\DoG.png", img)
    os.remove("DoG.png")
    img = cv2.imread("Phase1\Code\Results\DoG.png")
    cv2.imshow("Difference of Gaussian Filters", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """
    scales_set_1 = [1, np.sqrt(2), 2, 2 * np.sqrt(2)]
    scales_set_2 = [np.sqrt(2), 2, 2 * np.sqrt(2), 4]
    orientations = 6

    filter_bank_set_S = LMFilter(scales_set_1, orientations, kernel_size)
    visualize_filters(filter_bank_set_S, 12, "LMS Filters")
    filter_bank_set_L = LMFilter(scales_set_2, orientations, kernel_size)
    visualize_filters(filter_bank_set_L, 12, "LML Filters")
    LM_filter_bank = filter_bank_set_S + filter_bank_set_L
    visualize_filters(LM_filter_bank, 12, "LM Filters")
    img = cv2.imread("LMS Filters.png")
    cv2.imwrite("Phase1\Code\Results\LMS.png", img)
    os.remove("LMS Filters.png")
    img = cv2.imread("Phase1\Code\Results\LMS.png")
    cv2.imshow("LMS Filters", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    img = cv2.imread("LML Filters.png")
    cv2.imwrite("Phase1\Code\Results\LML.png", img)
    os.remove("LML Filters.png")
    img = cv2.imread("Phase1\Code\Results\LML.png")
    cv2.imshow("LML Filters", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    img = cv2.imread("LM Filters.png")
    cv2.imwrite("Phase1\Code\Results\LM.png", img)
    os.remove("LM Filters.png")
    img = cv2.imread("Phase1\Code\Results\LM.png")
    cv2.imshow("LM Filters", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    Gabor_filter_bank = GaborFilter([3, 6, 12], 8, [4, 8, 16], kernel_size)
    visualize_filters(Gabor_filter_bank, columns=5, title='Gabor Filters')
    img = cv2.imread("Gabor Filters.png")
    cv2.imwrite("Phase1\Code\Results\Gabor.png", img)
    os.remove("Gabor Filters.png")
    img = cv2.imread("Phase1\Code\Results\Gabor.png")
    cv2.imshow("Gabor Filters", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    radii_values = [2, 4, 8, 16, 32]
    orientations = 18
    half_disk_filter_bank = HalfDiskFilters(radii_values, orientations)
    visualize_filters(half_disk_filter_bank, columns=8, title='Half Disk Filters')
    img = cv2.imread("Half Disk Filters.png")
    cv2.imwrite("Phase1\Code\Results\HDMasks.png", img)
    os.remove("Half Disk Filters.png")
    img = cv2.imread("Phase1\Code\Results\HDMasks.png")
    cv2.imshow("Half Disk Filters", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for i in range(10):
        n = i + 1
        img = cv2.imread("Phase1\BSDS500\Images/"+ str(n)+ ".jpg")
        """
        Generate Texton Map
        Filter image using oriented gaussian filter bank
        """
        Texton_map = TextronMap(img, DOG_filter_bank, LM_filter_bank, Gabor_filter_bank)
        """
        Generate texture ID's using K-means clustering
        Display texton map and save image as TextonMap_ImageName.png,
        use command "cv2.imwrite('...)"
        """
        texton = Texton(Texton_map,64)
        plt.imsave("Phase1\Code\Results\Tp_"+ str(n)+ ".png", texton)
        img1 = cv2.imread("Phase1\Code\Results\Tp_"+ str(n)+ ".png")
        cv2.imwrite("Phase1\Code\Results\T_"+ str(n)+ ".png", img1)
        os.remove("Phase1\Code\Results\Tp_"+ str(n)+ ".png")
        
        """
        Generate Texton Gradient (Tg)
        Perform Chi-square calculation on Texton Map
        Display Tg and save image as Tg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Texton_Gradient = Gradient(texton, 64, half_disk_filter_bank)
        Texton_Gradient =Texton_Gradient[:,:,0]
        plt.imsave("Phase1\Code\Results\Tg_"+ str(n)+ ".png", Texton_Gradient)
        # img1 = cv2.imread("Phase1\Code\Results\TG_"+ str(n)+ ".png")
        # cv2.imwrite("Phase1\Code\Results\Tg_"+ str(n)+ ".png", img1)
        # os.remove("Phase1\Code\Results\TG_"+ str(n)+ ".png")

        """
        Generate Brightness Map
        Perform brightness binning 
        """
        Brightness_Map =BrightnessMap(img, 16)
        plt.imsave("Phase1\Code\Results\Bp_"+ str(n)+ ".png", Brightness_Map)
        img1 = cv2.imread("Phase1\Code\Results\Bp_"+ str(n)+ ".png")
        cv2.imwrite("Phase1\Code\Results\B_"+ str(n)+ ".png", img1)
        os.remove("Phase1\Code\Results\Bp_"+ str(n)+ ".png")

        """
        Generate Brightness Gradient (Bg)
        Perform Chi-square calculation on Brightness Map
        Display Bg and save image as Bg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Brightness_Gradient = Gradient(Brightness_Map, 16, half_disk_filter_bank)
        Brightness_Gradient = Brightness_Gradient[:,:,0]
        plt.imsave("Phase1\Code\Results\Bg_"+ str(n)+ ".png", Brightness_Gradient)
        # img1 = cv2.imread("Phase1\Code\Results\BG_"+ str(n)+ ".png")
        # cv2.imwrite("Phase1\Code\Results\Bg_"+ str(n)+ ".png", img1)
        # os.remove("Phase1\Code\Results\BG_"+ str(n)+ ".png")

        """
        Generate Color Map
        Perform color binning or clustering
        """
        Color_Map = ColorMap(img, 16)
        plt.imsave("Phase1\Code\Results\Cp_"+ str(n)+ ".png", Color_Map)
        img1 = cv2.imread("Phase1\Code\Results\Cp_"+ str(n)+ ".png")
        cv2.imwrite("Phase1\Code\Results\C_"+ str(n)+ ".png", img1)
        os.remove("Phase1\Code\Results\Cp_"+ str(n)+ ".png")

        """
        Generate Color Gradient (Cg)
        Perform Chi-square calculation on Color Map
        Display Cg and save image as Cg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        Color_Gradient = Gradient(Color_Map, 16, half_disk_filter_bank)
        Color_Gradient = Color_Gradient[:,:,0]
        plt.imsave("Phase1\Code\Results\Cg_"+ str(n)+ ".png", Color_Gradient)
        # img1 = cv2.imread("Phase1\Code\Results\CG_"+ str(n)+ ".png")
        # cv2.imwrite("Phase1\Code\Results\Cg_"+ str(n)+ ".png", img1)
        # os.remove("Phase1\Code\Results\CG_"+ str(n)+ ".png")

        """
        Read Sobel Baseline
        use command "cv2.imread(...)"
        """
        Sobel_Baseline = cv2.imread("Phase1/BSDS500/SobelBaseline/" +str(n)+ ".png")

        """
        Read Canny Baseline
        use command "cv2.imread(...)"
        """
        Canny_Baseline = cv2.imread("Phase1/BSDS500/CannyBaseline/" +str(n)+ ".png")

        """
        Combine responses to get pb-lite output
        Display PbLite and save image as PbLite_ImageName.png
        use command "cv2.imwrite(...)"
        """
        Average_of_gradient = (Texton_Gradient + Brightness_Gradient + Color_Gradient)/3.0
        Wght = (0.5 * Canny_Baseline + 0.5 * Sobel_Baseline)
        Wght = Wght[:,:,0]
        Pb_Lite_Coloured = np.multiply(Average_of_gradient, Wght)
        plt.imsave("Phase1\Code\Results\PbLiteCol_"+ str(n)+ ".png", Pb_Lite_Coloured)
        img1 = cv2.imread("Phase1\Code\Results\PbLiteCol_"+ str(n)+ ".png")
        Pb_Lite = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("Phase1\Code\Results\PbLite_"+ str(n)+ ".png", Pb_Lite)
        os.remove("Phase1\Code\Results\PbLiteCol_"+ str(n)+ ".png")

if __name__ == '__main__':
    main()