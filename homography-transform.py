import cv2
import numpy as np
import imutils

class EditWindow():

    def __init__(self, points) -> None:
    
        self.radius = 15
        self.selected_point = None
        self.points = points

        # Create window and set mouse callback
        cv2.namedWindow("Homography Transform")
        cv2.setMouseCallback("Homography Transform", self.mouse_callback)

    def draw_points(self, img, points):
        """Draw circles around each point and display instruction text."""
        for point in points:

            cv2.circle(img, point[0], self.radius, (0, 255, 0), 2)
            cv2.circle(img, point[0], 1, (0, 0, 255), -1)
        
        # Display instructions on the image
        cv2.putText(img, "Press 'q' to extract", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, "Drag the circles to correct", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def find_closest_point(self, x, y):
        """Return index of the closest point within the radius, or None if no point is close."""
        for i, (px, py) in enumerate([p[0] for p in self.points]):
            if (x - px)**2 + (y - py)**2 < self.radius**2:
                return i
        return None

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if a point is clicked and select it for dragging
            self.selected_point = self.find_closest_point(x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            # Move the selected point if any
            if self.selected_point is not None:
                self.points[self.selected_point] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # Release the selected point
            self.selected_point = None

    def run(self, image):

        while True:
            # Draw points on a copy of the image
            img_copy = image.copy()
            self.draw_points(img_copy, self.points)

            # Display the image
            cv2.imshow("Homography Transform", img_copy)

            # Exit on 'q' key press
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()



def shadow_remove( img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((17,17), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=40, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov


def preprocess(image):

    img_copy = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    kernel = np.ones((2,2), np.uint8)
    img_erosion = cv2.erode(img_copy, kernel, iterations=3)
    dilated = cv2.dilate(img_erosion, kernel, iterations=3)
    
    blur = cv2.GaussianBlur(dilated, (15,15), sigmaX=33, sigmaY=33)
    divided = cv2.divide(dilated, blur, scale=255)

    clahe = cv2.createCLAHE(clipLimit = 1)
    final_img = clahe.apply(divided) + 30

    final_img = cv2.bitwise_not(final_img) 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, kernel)

    (thresh, im_bw) = cv2.threshold(opening, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    final_img = cv2.bitwise_not(im_bw)

    return final_img

def order_points(pts):
    
        rect = np.zeros((4, 2), dtype = "float32")
       
        s = pts.sum(axis = 1) 
        rect[0] = pts[np.argmin(s)] 
        rect[2] = pts[np.argmax(s)] 

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)] 
        rect[3] = pts[np.argmax(diff)] 
       
        return rect

def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect 

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)) 
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


if __name__ == "__main__":

    img = cv2.imread("img.jpg")

    ratio = img.shape[0] / 500.0
    orig = img.copy()
    image = imutils.resize(shadow_remove(img), height = 500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 100, 255)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


    for c in cnts:
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            editWindow = EditWindow(approx)
            editWindow.run(image)
            screenCnt = approx
            break

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    cv2.imshow(mat=warped,winname="result")
    cv2.waitKey(0)
    cv2.destroyAllWindows()