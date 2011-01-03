import sys, optparse, Image

from eigenfaces import find_matching_image, parse_folder
from utils import merge_images


FACES     = None
THRESHOLD = 0.5


class PyFaces(object):
    """
    Detect if the face in `image` is similar to any face at `directory`,
    applies EigenFaces method with `threshold` as minum distance. Images
    on `directory` should be of the same size as `image`.
    """
    def __init__(self, image, directory, faces=FACES, threshold=THRESHOLD,
                 resize=True):
        """ Init method
        Parameters:
            @image: Image with a face
            @directory: Directory containing probe images
            @faces: Number of faces to compare (defaults to all images
                    in derectory)
            @threshold: Minimun distance between two images to accept them
                        as similar
            @resize: Resize image if it's bigger or smaller than comparision
                     images
        """
        self.image = image
        self.directory = directory
        self.faces = faces
        self.threshold = threshold
        self.resize = resize

    def match(self):
        """ Returns the match image and the distance between them, if any. """
        return find_matching_image(self.image, self.directory, self.threshold,
                                   self.faces, self.resize)

    def show(self):
        """ Shows the matching images joined """
        dist, match = self.match()
        if match is not None:
            merge_images([self.image, match]).show()
        return dist, match


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-i', '--image', dest='image')
    parser.add_option('-d', '--directory', dest='directory')
    parser.add_option('-f', '--faces', type='int', default=FACES, dest='faces')
    parser.add_option('-t', '--threshold', type='float', default=THRESHOLD,
                      dest='threshold')
    parser.add_option('-s', '--show', default=False, action='store_true',
                      dest='show')
    parser.add_option('-r', '--resize', default=True, action='store_true',
                      dest='resize')
    (options, args) = parser.parse_args()

    if not options.image or not options.directory:
        parser.print_help()
        sys.exit(2)

    pyfaces = PyFaces(options.image, options.directory,
                      options.faces, options.threshold,
                      options.resize)
    if options.show:
        dist, match = pyfaces.show()
    else:
        dist, match = pyfaces.match()

    if match is not None:
        print 'The image "%s" matches "%s" with a distance of "%s"' % \
                    (options.image, match, dist)
