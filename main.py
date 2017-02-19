
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# camera calibration routine, return calibration matrix, and dist coefficients
def camera_cal(calimg_path):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    gray = None
    
    # Make a list of calibration images
    images = glob.glob(calimg_path + '/calibration*.jpg')
    
    for i, fn in enumerate(images):
        img  = cv2.imread(fn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    # Calibrate camer given object and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return mtx, dist


# Apply absolute thresh on sobel operation, routine from carnd-term1 lesson
def abs_sobel_thresh(img, orient='x', sobel_kernel = 3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.abs(sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    #    is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return sxbinary


# For a given sobel kernel size and threshold values return the magnitude of the
# gradient for the image, routine from carnd-term1
def mag_thresh(img, sobel_kernel = 3, mag_thresh = (0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # return the binary image
    return binary_output


# threshold an image for a given range of direction threshold and a sobel kernel
# routine from carnd-term1
def dir_threshold(img, sobel_kernel = 3, thresh = (0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # return the binary image
    return binary_output


# image processing pipeline, threshold image by x gradient, gradient direction
# and l and s-channels
def hsv_pipeline(img, s_thresh=(170, 255), l_thresh=(30, 255), sx_thresh=(20, 150)):
    
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold dir gradient
    dir_binary = dir_threshold(img, 3, (0.7, 1.2))
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Threshold l channel
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (l_binary == 1)) | ((sxbinary == 1) & (dir_binary == 1))] = 1
    
    return combined_binary


# image processing pipe line
def thresh_pipeline(img):
    # sobel kernel size
    ksize = 3
    
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel = ksize, thresh = (20, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel = ksize, thresh = (20, 100))
    
    mag_binary = mag_thresh(img, ksize, (30, 120))

    dir_binary = dir_threshold(img, ksize, (0.7, 1.2))
    
    # combine threahold images together
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined


# perspective transformed image for lane detection
class Perspective_xform:
    def __init__(self):
        # source points
        #src = np.float32([(img_shape[1]*0.15, img_shape[0]),
        #                  (img_shape[1]*0.85, img_shape[0]),
        #                  (img_shape[1]*0.465, img_shape[0]*0.62),
        #                  (img_shape[1]*0.535, img_shape[0]*0.62)])
        
        #dst = np.float32([(img_shape[1]*0.20, img_shape[0]),
        #                  (img_shape[1]*0.75, img_shape[0]),
        #                  (img_shape[1]*0.20, 0.0),
        #                  (img_shape[1]*0.75, 0.0)])
        
        src1 = np.float32([(255, 690), (1060, 690), (585, 455), (700, 455)])
        dst1 = np.float32([(305, 690), (1010, 690), (305, 0), (1010, 0)])
        
        self.M = cv2.getPerspectiveTransform(src1, dst1)
        self.Minv = cv2.getPerspectiveTransform(dst1, src1)
    
    def transform(self, img):
        # return warped image
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]),  flags=cv2.INTER_LINEAR)
    
    def inv_transform(self, img):
        # return unwarped image
        return cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]),  flags=cv2.INTER_LINEAR)


# line class for lane detections and processing
class Line():
    def __init__(self, n = 10):  # n is the number of most recent fits used to smooth output
        
        # was the line detected in the last iteration?
        self.detected  = False
        
        self.left_fit  = None
        self.right_fit = None
        
        self.mtx, self.dist  = camera_cal('camera_cal')
        self.ps_xform  = Perspective_xform()
        
        self.radius_of_curvature = None # last saved radio of curvature
        #self.max_left_curv_change = 0
        #self.max_right_curv_change = 0
        #self.max_dd = 0
        
        self.buf_pt  = 0  # roud robin buffer pointer
        self.counter = 0  # counting frames we have used
        
        self.fitx_n = n
        self.recent_leftx  = np.zeros((n, 720))
        self.recent_rightx = np.zeros((n, 720))
    
    # searching lane lines by window search, routine from carnd-term1        
    def histogram_line_detect(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        
        # Create an output image to draw on and  visualize the result
        #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base  = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Set the width of the windows +/- margin
        margin = 80
        # Set minimum number of pixels found to recenter window
        minpix = 35
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds  = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low  = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            win_xleft_low   = leftx_current - margin
            win_xleft_high  = leftx_current + margin
            win_xright_low  = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img, (win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0, 255, 0), 2) 
            #cv2.rectangle(out_img, (win_xright_low,win_y_low),(win_xright_high,win_y_high),(0, 255, 0), 2) 
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) 
                              & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low)
                               & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds  = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx  = nonzerox[left_lane_inds]
        lefty  = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to left/right lines
        self.left_fit  = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
        self.detected  = True
        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        
        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        #plt.imshow(out_img)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        
        return left_fitx, right_fitx

        
    def detect_lanes(self, binary_warped):
        if self.detected == False:
            return self.histogram_line_detect(binary_warped)
        
        # we have previous detected lanes as our base
        # It's now much easier to find line pixels!
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        margin = 100
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                                       self.left_fit[2] - margin)) & 
                          (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                                       self.left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                                        self.right_fit[2] - margin)) & 
                           (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                                        self.right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        if ((not leftx.all()) or (not lefty.all()) or (not rightx.all()) or (not righty.all())):
            return self.histogram_line_detect(binary_warped)
        
        # Fit a second order polynomial to each lines
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx  = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        
        return left_fitx, right_fitx
    
    def calculate_curvature(self, img_shape, leftx, rightx):
        y_eval = img_shape[0]
                        
        # the lane width in pixels
        left_intercpt  = self.left_fit[0]*y_eval**2 + self.left_fit[1]*y_eval + self.left_fit[2]
        right_intercpt = self.right_fit[0]*y_eval**2 + self.right_fit[1]*y_eval + self.right_fit[2]
        
        lane_width_in_pixels = right_intercpt - left_intercpt
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / img_shape[0]
        xm_per_pix = 3.7 / lane_width_in_pixels
        
        # fit new polynomials to x, y in world space
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0] )
        
        left_fit_cr  = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / \
        		  np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / \
                          np.absolute(2*right_fit_cr[0])
        
        offset = ((left_intercpt + right_intercpt)/2.0 - img_shape[1]/2.0)*xm_per_pix
                
        #print(left_curverad, 'm', right_curverad, 'm')
        return left_curverad, right_curverad, offset
    
    # utility method to ploy the fitted poly line
    def draw_poly_fit(self, binary_img):
        # generate y space
        ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
        
        left_fitx  = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        
        out_img = np.dstack((binary_img, binary_img, binary_img))*255
        
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, binary_img.shape[1])
        plt.ylim(binary_img.shape[0], 0)
    
    # paint the lane to image
    def fill_lane(self, binary_img, left_fitx, right_fitx):
        # generate y pixel coordinates
        ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
        
        warp_zero = np.zeros_like(binary_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        return color_warp
    
    def process_image(self, img):
        # save a color image copy
        img_copy = np.copy(img)
        #img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # first undistort the image
        img = cv2.undistort(img, self.mtx, self.dist)
        # lane lines finding pipeline
        lane_img = hsv_pipeline(img)
        
        # perspective transform
        dst = self.ps_xform.transform(lane_img)
        
        # detect the left/right lane lines
        leftx, rightx = self.detect_lanes(dst)
        
        # calculate curvature
        leftcurv, rightcurv, c_offset = self.calculate_curvature(img.shape, leftx, rightx)
        
        # calculate curvature change
        d_leftcurv  = 0
        d_rightcurv = 0
        if self.radius_of_curvature == None:
            self.radius_of_curvature = [leftcurv, rightcurv]
        else:
            # check the change of curvatures, we expect small change
            d_leftcurv = abs(leftcurv - self.radius_of_curvature[0])/self.radius_of_curvature[0]
            d_rightcurv = abs(rightcurv - self.radius_of_curvature[1])/self.radius_of_curvature[1]
            # save the last detected curvature
            self.radius_of_curvature = [leftcurv, rightcurv]
        
        # check if the two lane line are parallel by checking the tangent of
        # mid-points of the two lines
        y2e = img.shape[1]*0.8
        left_line_tangent  = abs(2*self.left_fit[0]*y2e + self.left_fit[1])
        right_line_tangent = abs(2*self.right_fit[0]*y2e + self.right_fit[1])
        d = abs(left_line_tangent - right_line_tangent)/((left_line_tangent + right_line_tangent)/2.0)
        
        if (d_leftcurv > 0.1 or d_rightcurv > 0.1 or d > 0.1):
            self.detected = False
            leftx, rightx = self.detect_lanes(dst)
            leftcurv, rightcurv, c_offset = self.calculate_curvature(img.shape, leftx, rightx)
            self.radius_of_curvature = [leftcurv, rightcurv]
        
        # save current fitx to recent buffer
        self.recent_leftx[self.buf_pt] = leftx
        self.recent_rightx[self.buf_pt] = rightx
        
        # move the round robin buffer pointer
        self.buf_pt = (self.buf_pt + 1) % self.fitx_n
        
        if self.counter < self.fitx_n:
            self.counter += 1
        
        best_leftx = np.sum(self.recent_leftx, axis = 0) / self.counter
        best_rightx = np.sum(self.recent_rightx, axis = 0) / self.counter            
            
        # add curvature text
        curvlabel = 'Radius of Curvature: left {:.2f} m  right {:.2f} m'.format(leftcurv, rightcurv)
        cv2.putText(img_copy, curvlabel, (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 25, 120), 2)
        # add car deviation text
        offset_text = 'Deviation from Lane Center: {:.3f} m'.format(c_offset)
        cv2.putText(img_copy, offset_text, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 25, 120), 2)

        # fill lane polygon
        color_lane = self.fill_lane(dst, best_leftx, best_rightx)
        # unwarped image by inverse perspective transform
        unwarp_lane = self.ps_xform.inv_transform(color_lane)
        
        # paint lane to image and return result
        proj_img = cv2.addWeighted(img_copy, 1, unwarp_lane, 0.3, 0)
        #proj_img = cv2.cvtColor(proj_img, cv2.COLOR_BGR2RGB)
        return proj_img



if __name__ == "__main__":
    # calculate camera and distortion coefficients first
    from moviepy.editor import VideoFileClip
        
    lane = Line()
    
    input_file = 'project_video.mp4'
    clip = VideoFileClip(input_file)
    
    output_file = 'out_project_video.mp4'
    out_clip = clip.fl_image(lane.process_image)
    out_clip.write_videofile(output_file, audio=False)



