import numpy as np
from pylibdmtx.pylibdmtx import decode
import cv2




class VcFrame:


    def __init__(self,frame):
        self.frame=frame
        self.codes_and_contours=[]
        self.analyze_frame()


    def auto_canny(self,img, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(img)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(img, lower, upper)
        # return the edged image
        return edged

    def crop_rotated(self,img, cnt):
        """gets an image and a contour,
        calculates the min rect of the contour,then crops, warps, and
        returns the min rect area out of the image"""

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[width, height],
                            [0, height],
                            [0, 0],
                            [width, 0]
                            ], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(img, M, (width, height))

        return warped

    def resize_contour(self,contour, scalefactor):
        cnt = np.float32(contour)
        # compute centroid based of moments
        M = cv2.moments(cnt)
        cx = (M['m10'] / M['m00'])
        cy = (M['m01'] / M['m00'])
        centroid = np.array((cx, cy), dtype=np.float32)
        # compute vectors from centroid to vertices
        vectors = cnt - centroid
        # resizing vectors
        vectors = scalefactor * vectors
        result = vectors + centroid
        result = result.astype(np.int32)

        return result

    def cont_to_minareacont(self,cnt):
        rect = cv2.minAreaRect(cnt)

        # calc aspect ratio
        width = int(rect[1][0])
        height = int(rect[1][1])

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        try:
            aspect_ratio = float(width / height)
        except ZeroDivisionError:
            return box, 0

        return box, aspect_ratio

    def blockshaped(self,arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    def unblockshaped(self,arr, h, w):
        """
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols = arr.shape
        return (arr.reshape(h // nrows, -1, nrows, ncols)
                .swapaxes(1, 2)
                .reshape(h, w))

    def get_2dgaussian(self,size, sigma):
        xdir_gauss = cv2.getGaussianKernel(size, sigma)
        kernel = np.multiply(xdir_gauss.T, xdir_gauss)
        return kernel

    def get_2dcorrelation(self,mtx1, mtx2):
        return np.dot(mtx1.ravel(), mtx2.ravel())

    def finecut_dmtx2(self,img, margin=0.15, thresh_whole=0.7, thresh_finder=0.45):
        """gets an image of already cut out data matrix (data matrix should occupy
            most of the width and height of the image, and should not be rotated. the return
            value of the function is a FINE-CUT of the data matrix suitable for mesh-and-reconstruction.

            margin:the approximate portion of width and height which is not occupied by dmtx.defaults to 0.15.

            threshold: if the amount of white pixels (non-zero) for each row or column (in the margin) exceeds
            threshold percentage (out of total height or width), it is assumed that that is part of the cutable!!!
            """

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(11, 11))
        excl = clahe.apply(img)
        ret, clahe_otsu = cv2.threshold(excl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        image_binary = clahe_otsu

        xmargin = int(margin * img.shape[1])
        ymargin = int(margin * img.shape[0])

        height, width = img.shape[0:2]

        top = np.count_nonzero(image_binary[0:ymargin, :], axis=1)
        bottom = np.flip(np.count_nonzero(image_binary[height - ymargin:height, :], axis=1))
        left = np.count_nonzero(image_binary[:, 0:xmargin], axis=0)
        right = np.flip(np.count_nonzero(image_binary[:, width - xmargin:width], axis=0))

        cut_pixels = []
        for (direction, dim) in [(top, height), (bottom, height), (left, width), (right, width)]:
            for i in range(direction.shape[0]):
                if i < direction.shape[0] - 1 and (
                        direction[i] / dim > thresh_whole or direction[i + 1] / dim > thresh_whole):
                    continue
                else:
                    cut_pixels.append(i);
                    break

        # this is to further trim the data matrix(smaller threshold) on the sides of finder pattern ONLY!
        cut_pixels_finder = []
        for (direction, dim) in [(top, height), (bottom, height), (left, width), (right, width)]:
            for i in range(direction.shape[0]):
                if i < direction.shape[0] - 1 and (
                        direction[i] / dim > thresh_finder or direction[i + 1] / dim > thresh_finder):
                    continue
                else:
                    cut_pixels_finder.append(i);
                    break

        top_low = np.min((top[cut_pixels[0]:cut_pixels[0] + 3]))
        bottom_low = np.min((bottom[cut_pixels[1]:cut_pixels[1] + 3]))
        left_low = np.min((left[cut_pixels[2]:cut_pixels[2] + 3]))
        right_low = np.min((right[cut_pixels[3]:cut_pixels[3] + 3]))

        # finding the minimum of the 3 first numbers after the cut pixels, for every direction
        # of the four directions, the two for which the aforementioned number is lower, are considered
        # directions of the finder pattern
        finding_patterns = sorted([(top_low, "top", (
            np.where(((top[cut_pixels[0] + 1:]) - ((top + (top_low * 2))[cut_pixels[0]:])[:-1]) > 0)[0][:1])),
                                   (bottom_low, "bottom", (np.where(
                                       ((bottom[cut_pixels[0] + 1:]) - ((bottom + (bottom_low * 2))[cut_pixels[0]:])[
                                                                       :-1]) > 0)[
                                                               0][:1])),
                                   (left_low, "left", (
                                       np.where(
                                           ((left[cut_pixels[0] + 1:]) - ((left + (left_low * 2))[cut_pixels[0]:])[
                                                                         :-1]) > 0)[
                                           0][:1])),
                                   (right_low, "right", (np.where(
                                       ((right[cut_pixels[0] + 1:]) - ((right + (right_low * 2))[cut_pixels[0]:])[
                                                                      :-1]) > 0)[0][
                                                         :1]))], key=lambda x: x[0])[:2]

        # average_thickness is the average of found finding patterns
        # if the shape is messed up and it cannot find a value for average thickness, returns None
        try:
            average_thickness = int((finding_patterns[0][2] + finding_patterns[1][2]) / 2)
        except TypeError:
            average_thickness = None

        img_cut = img[cut_pixels[0]:(height - cut_pixels[1]), cut_pixels[2]:(width - cut_pixels[3])]
        cut_width = img_cut.shape[1]
        cut_height = img_cut.shape[0]

        directions = (finding_patterns[0][1], finding_patterns[1][1])
        directions = sorted(directions)

        if directions == ['right', 'top'] or directions == ['top', 'right']:
            img_cut = img_cut[abs(cut_pixels_finder[0] - cut_pixels[0]):,
                      :cut_width - abs(cut_pixels_finder[3] - cut_pixels[3])]
            img_cut = cv2.rotate(img_cut, cv2.ROTATE_180)
        elif directions == ['left', 'top'] or directions == ['top', 'left']:
            img_cut = img_cut[abs(cut_pixels_finder[0] - cut_pixels[0]):, abs(cut_pixels_finder[2] - cut_pixels[2]):]
            img_cut = cv2.rotate(img_cut, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif directions == ['bottom', 'right'] or directions == ['right', 'bottom']:
            img_cut = img_cut[:cut_height - abs(cut_pixels_finder[1] - cut_pixels[1]),
                      :cut_width - abs(cut_pixels_finder[3] - cut_pixels[3])]
            img_cut = cv2.rotate(img_cut, cv2.cv2.ROTATE_90_CLOCKWISE)
        else:
            img_cut = img_cut[:cut_height - abs(cut_pixels_finder[1] - cut_pixels[1]),
                      abs(cut_pixels_finder[2] - cut_pixels[2]):]

        return img_cut

    def reconstruct_dmtx_fast(self,img, timing_count):

        """give it a fine_cut dmtx, and tell it the dimensions. it will rebuild it for you.
        remember that you need to specify the dimension"""

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(11, 11))
        img = clahe.apply(img)

        # ret value shall be used to threshold pixels after cross correlation
        ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # making a 2d gaussian kernel,used to perform cross correlation with each tile
        kernel = self.get_2dgaussian(10, 0.5)

        # of various resizing methods,this one produces optimum results
        img = cv2.resize(img, (timing_count * 10, timing_count * 10), interpolation=cv2.INTER_CUBIC)

        # could have used blockshaped function instead of spelling it out here
        blockshaped = img.reshape(timing_count, 10, -1, 10).swapaxes(1, 2).reshape(-1, 10, 10)
        raveled = blockshaped.ravel()
        reshaped = raveled.reshape((timing_count ** 2), 100)
        reduced = np.dot(reshaped, kernel.ravel())
        result = reduced.reshape(timing_count, timing_count)
        final = np.asarray(result, dtype=np.uint8)
        _, dmtx = cv2.threshold(final, (ret *0.7), 255, cv2.THRESH_BINARY)

        # manually setting finding and timing pattern for 26x26 datamatrix
        clock = np.array([0, 255], dtype=np.uint8)
        clock = np.resize(clock, timing_count)

        dmtx[:, 0:1] = 0;
        dmtx[-1:, :, ] = 0
        dmtx[0:1, :] = clock.reshape((1, timing_count))
        dmtx[:, -1:] = np.flip(clock).reshape((timing_count, 1))

        # adding border and making the barcode larger in size so to be readable by most libraries
        dmtx = cv2.copyMakeBorder(dmtx, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255)
        dmtx = cv2.resize(dmtx, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        return dmtx

    def extract_dmtx(self,img, contour_resize_factor=1):
        """gets the image(already cut) and extracts the area with datamatrix
        then returns that part of the image which contains the data matrix"""
        # note to self:applying clahe before extracting dmtx might result in selection
        # of wrong area as dmtx
        # print(np.average(img))
        if np.average(img) > 6:
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            ret, ots = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # creating a canny edge image from the original image
            edges = self.auto_canny(ots)

            # replacing the original canny edge image with the dilated version of it
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # calculating contours on dilated canny image and calculating the 10 with largest area
            contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            sortedContours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            sortedContours = sortedContours[0:15]

            # calculating minrect for 10 largest contours and returning them as drawable contours
            sortedMinRect = [self.cont_to_minareacont(cnt) for cnt in sortedContours]

            # checking the squareness of the minAreaRect
            squareMinRect = [cnt[0] for cnt in sortedMinRect if
                             0.95 <= cnt[1] <= 1.05]

            if len(squareMinRect) > 0:
                finalContour = sorted(squareMinRect, key=lambda x: cv2.contourArea(x), reverse=True)[0:5]
                output = [(self.crop_rotated(img, self.resize_contour(fcontour, contour_resize_factor)),fcontour) for fcontour in
                          finalContour]
                # to Omit mostly absolute black but also very dark images/not a real value for now
                output = [res_img for res_img in output if np.average(res_img[0]) > 3 and (min((res_img[0]).shape)>40)]
                return output
            else:
                return []
                # return [cv2.putText(np.zeros((200, 200), dtype=np.uint8), "NOT\nYET", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                #                     1, 255, 2, cv2.LINE_AA)]
        else:
            return []

    def analyze_frame(self):
        likely_dmtx_areas=self.extract_dmtx(self.frame.copy(),1.05)

        if len(likely_dmtx_areas)>0:

            for item in likely_dmtx_areas:
                reconstructed=self.reconstruct_dmtx_fast(self.finecut_dmtx2(item[0]),22)
                decoded = decode(reconstructed, max_count=1, shrink=1, deviation=10, timeout=2, shape=6)

                if len(decoded)>0:
                    try:
                        decoded_str=(decoded[0].data).decode("ASCII")
                    except UnicodeDecodeError:pass
                    if len(decoded_str)>47 and len(decoded_str)<80:
                            if str(decoded_str[:2])=="01" and str(decoded_str[16:18])=="21"and \
                               str(decoded_str[38:40]) == "17" and str(decoded_str[46:48]) == "10":
                               self.codes_and_contours.append((decoded_str,item[1]))





