import Image, tempfile, math

from os.path import abspath, dirname, join

from opencv.highgui import cvCreateFileCapture, cvQueryFrame
from opencv.cv import cvClearMemStorage, cvCopy, cvCreateImage, cvCvtColor, \
                      cvEqualizeHist, cvHaarDetectObjects, \
                      cvLoadHaarClassifierCascade, cvResize, cvRound, cvSize, \
                      CV_HAAR_DO_CANNY_PRUNING, CV_INTER_LINEAR, CV_BGR2GRAY, \
                      IPL_DEPTH_8U, cvCreateMemStorage


CASCADES_DIR   = dirname(abspath(__file__))
CASCADES_NAMES = ['haarcascade_frontalface_alt_tree.xml']
CASCADES = [ join(CASCADES_DIR, name) for name in CASCADES_NAMES ]

MIN_SIZE             = cvSize(100, 100)
IMAGE_SCALE          = 1.3
MIN_NEIGHBORS        = 3
HAAR_SCALE           = 1.3
HAAR_FLAGS           = CV_HAAR_DO_CANNY_PRUNING
DEFAULT_ROTATE_ANGLE = 45
COPY_DEPTH           = 8
COPY_CHANNELS        = 1
STORAGE              = cvCreateMemStorage(0)

 
def _detect(image):
    """ Detects faces on `image`
    Parameters:
        @image: image file path

    Returns:
        [((x1, y1), (x2, y2)), ...] List of coordenates for top-left
                                    and bottom-right corner
    """
    # the OpenCV API says this function is obsolete, but we can't
    # cast the output of cvLoad to a HaarClassifierCascade, so use
    # this anyways the size parameter is ignored
    capture = cvCreateFileCapture(image) 

    if not capture:
        return []

    frame = cvQueryFrame(capture)
    if not frame:
        return []

    img = cvCreateImage(cvSize(frame.width, frame.height),
                        IPL_DEPTH_8U, frame.nChannels)
    cvCopy(frame, img)

    # allocate temporary images
    gray          = cvCreateImage((img.width, img.height),
                                  COPY_DEPTH, COPY_CHANNELS)
    width, height = (cvRound(img.width / IMAGE_SCALE),
                     cvRound(img.height / IMAGE_SCALE))
    small_img     = cvCreateImage((width, height), COPY_DEPTH, COPY_CHANNELS)

    # convert color input image to grayscale
    cvCvtColor(img, gray, CV_BGR2GRAY)

    # scale input image for faster processing
    cvResize(gray, small_img, CV_INTER_LINEAR)
    cvEqualizeHist(small_img, small_img)
    cvClearMemStorage(STORAGE)

    coords = []
    for haar_file in CASCADES:
        cascade = cvLoadHaarClassifierCascade(haar_file, cvSize(1, 1))
        if cascade:
            faces = cvHaarDetectObjects(small_img, cascade, STORAGE, HAAR_SCALE,
                                        MIN_NEIGHBORS, HAAR_FLAGS, MIN_SIZE) or []
            for face_rect in faces:
                # the input to cvHaarDetectObjects was resized, so scale the 
                # bounding box of each face and convert it to two CvPoints
                x, y = face_rect.x, face_rect.y
                pt1 = (int(x * IMAGE_SCALE), int(y * IMAGE_SCALE))
                pt2 = (int((x + face_rect.width) * IMAGE_SCALE),
                       int((y + face_rect.height) * IMAGE_SCALE))
                coords.append((pt1, pt2))
    return coords


def detect(image):
    """ Aims to detect faces in `image`. If no faces are detected,
    the image is rotated DEFAULT_ROTATE_ANGLE left and right trying to
    get potentially inclined faces.

    Parameters:
        @image: image path

    Returns:
        [((x1, y1), (x2, y2)), ...] List of coordenates for top-left
                                    and bottom-right corner

    Rotated coordinates are rotated back to original image position.
    """
    coords = _detect(image) or []

    if not coords:
        img = Image.open(image)

        for degree in (-DEFAULT_ROTATE_ANGLE, DEFAULT_ROTATE_ANGLE):
            tmp = tempfile.NamedTemporaryFile(suffix='.' + image.split('.')[-1])
            img.rotate(degree).save(tmp)
            coords += [rotate(-1 * degree, box) for box in _detect(tmp.name)]
            tmp.close()
    return coords


def rotate(degree, box):
    """ Rotates `degree` the coords on `box` using the `box`
    center as the rotate axis.

    Parameters:
        @degree: Degrees to rotate back the box
        @box: Tuple with top-left and bottom-right coordinates

    Returns:
        [(x1, y1), (x2, y2)]: Rotated top-left and bottom-right
                              coordinats
    """
    (x1, y1), (x2, y2) = box
    orig_x, orig_y = int((x2 - x1)/2), int((y2 - y1)/2)

    x0 = x1 - orig_x
    y0 = y1 - orig_y

    rad = math.radians(degree) # conver to radians
    cos_rad = math.cos(rad)
    sin_rad = math.sin(rad)

    new_x = int((x0 * cos_rad) - (y0 * sin_rad))
    new_y = int((x0 * sin_rad) + (y0 * cos_rad))

    return [(new_x + orig_x, new_y + orig_y),
            (new_x + orig_x + x2 - x1, new_y + orig_y + y2 - y1)]
