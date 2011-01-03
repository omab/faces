#!/usr/bin/python
"""
Who is an integration between OpenCV face detection/extraction
and pyfaces face recognition (eigenfaces method).
"""

import sys
import optparse
from os import listdir
from os.path import abspath, dirname, join, isdir, basename

from detect.extract import extract
from pyfaces.pyfaces import PyFaces, THRESHOLD
from pyfaces.eigenfaces import parse_folder
from pyfaces.utils import merge_images


PEOPLE           = 'people'
PEOPLE_DIRECTORY = join(dirname(abspath(__file__)), PEOPLE)


def reduce_result(matches):
    """ Reduces the matches at `matches` removing the
    repetitions keeping the values with small distance """ 
    results = {}
    for name, match, dist in matches:
        if name not in results or results[name][-1] > dist:
            results[name] = (name, match, dist)
    values = results.values()
    values.sort(key=lambda x: x[-1])
    return values
            

def find_people(image, faces, people, threshold):
    """ Finds people in `image` using faces from `faces` that
    souits the `threshold` limit
    """
    matches = []
    for face in extract(image, faces):
        for person in parse_folder(people, lambda name: isdir(name)):
            dist, match = PyFaces(face, person, threshold=threshold).match()
            if match:
                matches.append((basename(person), match, dist))
    return reduce_result(matches)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-i', '--image', dest='image')
    parser.add_option('-f', '--faces', dest='faces')
    parser.add_option('-t', '--threshold', default=THRESHOLD, dest='threshold')
    parser.add_option('-e', '--extract', default=False, action='store_true',
                      dest='extract')
    parser.add_option('-d', '--extract-directory', dest='extract_src')
    parser.add_option('-s', '--show', default=False, action='store_true',
                      dest='show')
    parser.add_option('-p', '--people', default=PEOPLE_DIRECTORY,
                      dest='people')

    (options, args) = parser.parse_args()


    if options.extract:
        if not options.faces or not options.extract_src:
            parser.print_help()
            sys.exit(2)
        src = options.extract_src
        images = listdir(src)
        if images:
            for image in images:
                faces = extract(join(src, image), options.faces)
                print 'Faces found for "%s": %s' % (image, ', '.join(faces))
        else:
            print 'No faces recognised on the photo'
    else:
        if not options.image or not options.faces:
            parser.print_help()
            sys.exit(2)
        people = find_people(options.image, options.faces, options.people,
                             options.threshold)
        if people:
            print 'People found on "%s": %s' % \
                    (options.image,
                     ', '.join(('\n\t%s (%s %s)' % person for person in people)))

            if options.show:
                merge_images([options.image] + [person[1] for person in people]).show()
        else:
            print 'No body was recognised on the photo'
