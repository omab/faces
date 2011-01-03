import pickle, math, shutil, tempfile, Image
from os import listdir, mkdir
from os.path import exists, isdir, isfile, join, normpath, basename

from numpy import max, zeros, average, dot, asfarray, sort, trace, argmin
from numpy.linalg import eigh, svd


CACHE_FILE_NAME   = 'saveddata.cache'
AVERAGE_FILE_NAME = 'average.png'
RECON_DIRNAME     = 'reconfaces'
EIGENFACES_DIR    = 'eigenfaces'
EIGENFACE_IMG_FMT = 'eigenface-%s.png'
RECONPHI_FMT      = 'reconphi-%s.png'
RECONX_FMT        = 'reconx-%s.png'
USE_EIGH          = True
MAKE_AVERAGE      = False


class FaceBundle(object):
    """ Faces Bundle representation """
    def __init__(self, directory, images_list, width, height, adjfaces,
                 eigenfaces, avg, evals):
        self.directory = directory
        self.images_list = map(basename, images_list)
        self.width = width
        self.height = height
        self.adjfaces = adjfaces
        self.eigenfaces = eigenfaces
        self.average = avg
        self.evals = evals

    def as_dict(self):
        return { 'directory': self.directory, 'images_list': self.images_list,
                 'width': self.width, 'height': self.height,
                 'adjfaces': self.adjfaces, 'eigenfaces': self.eigenfaces,
                 'avg': self.average, 'evals': self.evals }


def find_matching_image(image, directory, threshold, egfnum=None, resize=True):
    """ Finds an image at `directory` that matches `image` between
    a `threshold` distance

    Parameters:
        @image: image to match
        @directory: images to compare against
        @threshold: max distance allowed between candidate and image
        @egfnum: max eigenfaces to compare with
        @resize: resize the candidate image if it's smaller or bigger than
                 faces to compare
    Returns:
        (mindist, image)
        @mindist: distance between image and best coincidence
        @image: conicidence image
    """
    extension = image.split('.')[-1]
    images_list = parse_folder(directory,
                               lambda name: name.lower().endswith(extension))

    numimgs = len(images_list)
    if not egfnum or egfnum >= numimgs:
        egfnum = numimgs / 2

    bundle = get_bundle(directory, images_list)

    # calculate weights
    weights = dot(bundle.eigenfaces[:egfnum,:],
                  bundle.adjfaces.transpose()).transpose()

    img = Image.open(image).convert('L')
    if img.size != (bundle.width, bundle.height):
        if resize:
            tmp = tempfile.NamedTemporaryFile(suffix='.' + extension)
            img.resize((bundle.width, bundle.height)).save(tmp)
            img = Image.open(tmp.name)
        else:
            raise IOError, 'Select image of correct size.'

    pixels = asfarray(img.getdata())
    face = (pixels / max(pixels)) - bundle.average

    input_weight = dot(bundle.eigenfaces[:egfnum,:], face.transpose())
    dist = ((weights - input_weight.transpose()) ** 2).sum(axis=1)
    idx = argmin(dist)
    mindist = math.sqrt(dist[idx])
    #reconstruct_faces(bundle, egfnum, weights)

    if mindist < threshold:
        return (mindist, join(directory, bundle.images_list[idx]))
    else:
        return (None, None)


def create_face_bundle(directory, images_list):
    """ Creates FaceBundle """
    images = validate_directory(images_list)

    img = images[0]
    width, height = img.size
    pixels = width * height
    numimgs = len(images)

    # Create a 2d array, each row holds pixvalues of a single image
    facet_matrix = zeros((numimgs, pixels))
    for i in xrange(numimgs):
        pixels = asfarray(images[i].getdata())
        facet_matrix[i] = pixels / max(pixels)

    # Create average values, one for each column (ie pixel)
    avg = average(facet_matrix, axis=0)
    if MAKE_AVERAGE:
        # Create average image in current directory just for fun of viewing
        make_image(avg, AVERAGE_FILE_NAME, (width, height))

    # Substract avg val from each orig val to get adjusted faces (phi of T&P)
    adjfaces = facet_matrix - avg
    adjfaces_transpose = adjfaces.transpose()
    L = dot(adjfaces, adjfaces_transpose)

    if USE_EIGH:
        evals1, evects1 = eigh(L)
    else:
        evects1, evals1, vt = svd(L, 0)
    reversed_evalue_order = evals1.argsort()[::-1]
    evects = evects1[:,reversed_evalue_order]
    evals = sort(evals1)[::-1]

    # Rows in eigen_space are eigenfaces
    eigen_space = dot(adjfaces_transpose, evects).transpose()
    # Normalize rows of eigen_space
    for i in xrange(numimgs):
        ui = eigen_space[i]
        ui.shape = (height, width)
        eigen_space[i] = eigen_space[i] / trace(dot(ui.transpose(), ui))

    bundle = FaceBundle(directory, images_list, width, height, adjfaces,
                        eigen_space, avg, evals)
    #create_eigenimages(bundle, eigen_space) # create eigenface images
    return bundle


def get_bundle(directory, images_list):
    """ Builds or retrives bundle for images at `directory` directory.
    Checks cache file on `directory` which is a pickled dictionary from
    FaceBundle values instance with stats about the directory, like
    images list to detect if it has changed (images were added or removed,
    etc). The cache is created if it doesn't exists or the directory changed.
    """
    bundle, build_cache = None, True
    cache_file = join(directory, CACHE_FILE_NAME)
    if exists(cache_file):
        cache = open(join(directory, CACHE_FILE_NAME))
        bundle = FaceBundle(**pickle.load(cache))
        cache.close()
        build_cache = images_list != bundle.images_list

    if build_cache: # Cache doesn't exists or `directory` changed
        bundle = create_face_bundle(directory, images_list)
        cache = open(cache_file, 'w')
        pickle.dump(bundle.as_dict(), cache)
        cache.close()
    return bundle


def parse_folder(directory, filter_rule=None):
    """ Returns a list of files in `directory` that complies `filter_rule`.
    Raises IOError if `directory` is not a directory. Returns all files
    in `directory` if no filter rule is passed"""
    if not isdir(directory):
        raise IOError, '%s is not a directory' % directory
    return sorted(filter(filter_rule,
                         (normpath(join(directory, name))
                            for name in listdir(directory))))


def validate_directory(images_list):
    """ Validates images directory, all should be images
    of the same size """
    if not images_list:
        raise IOError, 'Folder empty'

    file_list, sizes = [], set()
    for name in images_list:
        img = Image.open(name).convert('L')
        file_list.append(img)
        sizes.add(img.size)

    if len(sizes) != 1:
        raise IOError, 'Select folder with all images of equal dimensions'
    return file_list


# def reconstruct_faces(bundle, egfnum, weights):
#     """ Reconstructs probe faces """
#     evals_sub = bundle.evals[:egfnum]
#     new_weights = zeros(weights.shape)
# 
#     x_coords, y_coords = xrange(len(weights)), xrange(len(evals_sub))
#     for x, y in ((x, y) for x in x_coords for y in y_coords):
#         new_weights[x][y] = weights[x][y] * evals_sub[y]
# 
#     try:
#         if isdir(RECON_DIRNAME):
#             shutil.rmtree(RECON_DIRNAME, True)
#         mkdir(RECON_DIRNAME)
#     except Exception, e:
#         raise IOError, 'Some problem removing directory: "%s"', e.message
#     else:
#         phinew = dot(new_weights, bundle.eigenfaces[:egfnum,:])
#         xnew = phinew + bundle.average
#         for x in xrange(len(bundle.images_list)):
#             make_image(phinew[x], join(RECON_DIRNAME, RECONPHI_FMT % x),
#                        (bundle.width, bundle.height), True)
#             make_image(xnew[x], join(RECON_DIRNAME, RECONX_FMT % x),
#                        (bundle.width, bundle.height), True)
# 
# 
# def create_eigenimages(bundle, eigen_space):
#     """ Creates eigenfaces images from `eigen_space` at EIGENFACES_DIR """
#     if isdir(EIGENFACES_DIR):
#         shutil.rmtree(EIGENFACES_DIR, True)
#     mkdir(EIGENFACES_DIR)
# 
#     for idx in xrange(len(bundle.images_list)):
#         make_image(eigen_space[idx],
#                    join(EIGENFACES_DIR, EIGENFACE_IMG_FMT % idx),
#                    (bundle.width, bundle.height))
# 
# 
# def make_image(v, filename, size, scaled=True):
#     """ Builds an image named `filename` of `size` dimensions
#     that might be scaled """
#     v.shape = (-1,) #change to 1 dim array
#     im = Image.new('L', size)
#     if scaled:
#         a, b = v.min(), v.max()
#         v = ((v - a) * 255 / (b - a))
#     im.putdata(v)
#     im.save(filename)
