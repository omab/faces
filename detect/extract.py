#!/bin/sh
import optparse, Image

from os import mkdir
from os.path import join, split, exists, isdir, isfile 

from facedetect import detect


DEFAULT_EXTENSION  = 'jpg'
OUTPUT_NAME_FORMAT = '%s_%sx%s.' + DEFAULT_EXTENSION


def extract(image, directory):
    def crop(box):
        if box:
            (x1, y1), (x2, y2) = box
            name = OUTPUT_NAME_FORMAT % \
                        ('.'.join(split(image)[-1].split('.')[:-1]),
                         x2 - x1, y2 - y1)
            image_name = join(directory, name)
            result = open(image_name, 'w')
            Image.open(image).crop((x1, y1, x2, y2)).save(result)
            result.close()
            return image_name

    if not exists(directory):
        mkdir(directory)
    elif not isdir(directory):
        raise IOError, '"%s" is not a directory' % directory

    if not exists(image) or not isfile(image):
        raise IOError, '"%s" does not exists or is not an image' % image

    return filter(None, map(crop, detect(image)))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-i', '--image', dest='image')
    parser.add_option('-d', '--directory', dest='directory')
    parser.add_option('-l', '--list', default=True,
                      action='store_true', dest='list')
    (options, args) = parser.parse_args()

    if not options.image or not options.directory:
        parser.print_help()
        sys.exit(2)

    image, directory = options.image, options.directory
    faces = extract(image, directory)
    if faces:
        if options.list:
            print 'Faces found for "%s": %s' % (image, ', '.join(faces))
    else:
        print 'No faces found'
