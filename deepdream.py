# encoding:utf-8
# deepdream.py
#
# Deep Dream is an algorithm that makes an pattern detection algorithm
# over-interpret patterns. The Deep Dream algorithm is a modified neural network.
# Instead of identifying objects in an input image, it changes the image into
# the direction of its training data set, which produces impressive surrealistic,
# dream-like images
#
# The code is modified from
# https://github.com/google/deepdream/blob/master/dream.ipynb
#
# How to Use:
# Use `-i` to specify your input content image. It will deep dream at a random layer.
#   python deepdream.py -i {your_image}.jpg
#
# If you want to start Deep Dream at a layer depth, type and octave manually:
#   python deepdream.py -d 1 -t 1 -o 6 -i Style_StarryNight.jpg
#

import os, sys

from random import randint

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
import argparse, datetime, shutil
from google.protobuf import text_format

# disable logging before net creation
os.environ["GLOG_minloglevel"] = "2"
import caffe

def create_net(model_file):
    """ Create the neural network tmp.prototxt.
    """

    net_fn = os.path.join(os.path.split(model_file)[0], 'deploy.prototxt')

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True

    open('tmp.prototxt', 'w').write(str(model))


def load_net(model_file):
    """ Load the neural network tmp.prototxt.
    """

    net = caffe.Classifier('tmp.prototxt', model_file,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    return net


# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


# regular, non-guided objective
def objective_L2(destination):
    destination.diff[:] = destination.data


class DeepDream(object):
    def __init__(self, net):
        self.net = net

    def iterated_dream(self, source_path, end, octaves):
        frame = np.float32(PIL.Image.open(source_path))

        frame = self.deepdream(frame, end=end, octave_n=octaves)
        extension = os.path.splitext(source_path)[1]
        dream_path = source_path.replace(extension, '_dream' + extension)
        PIL.Image.fromarray(np.uint8(frame)).save(dream_path)

    def deepdream(self, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
                              end='inception_4c/output'):

        # prepare base images for all octaves
        octaves = [preprocess(self.net, base_img)]
        for i in xrange(octave_n-1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

        source = self.net.blobs['data']  # original image
        detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details

        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]  # octave size
            if octave > 0:
                # upscale details from previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1, 1.0*w/w1), order=1)

            source.reshape(1, 3, h, w) # resize the network's input image size
            source.data[0] = octave_base + detail

            for i in xrange(iter_n):
                self.make_step(end=end)

            # extract details produced on the current octave
            detail = source.data[0] - octave_base

        return deprocess(self.net, source.data[0])  # return final image

    def make_step(self, step_size=1.5, end='inception_4c/output', jitter=32):
        """ Basic gradient ascent step.
        """

        source = self.net.blobs['data'] # input image is stored in Net's 'data' blob
        destination = self.net.blobs[end]

        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        source.data[0] = np.roll(np.roll(source.data[0], ox, -1), oy, -2) # apply jitter shift

        self.net.forward(end=end)  # step in direction of the target layer
        objective_L2(destination)  # specify the optimization objective
        self.net.backward(start=end) # step in direction of the input layer

        # apply normalized ascent step to the input image
        gradient = source.diff[0]
        source.data[:] += step_size/np.abs(gradient).mean() * gradient

        source.data[0] = np.roll(np.roll(source.data[0], -ox, -1), -oy, -2) # unshift image

        bias = self.net.transformer.mean['data']
        source.data[:] = np.clip(source.data, -bias, 255-bias)


def start_dream(args):
    """ Gather all parameters (source image, layer descriptor and octave),
    create a net and start to dream.
    """

    source_path = get_source_image(args)
    layer = get_layer_descriptor(args)
    octave = (args.octaves if args.octaves else randint(1, 9))

    model_file = 'bvlc_googlenet/bvlc_googlenet.caffemodel'

    if args.network:
        create_net(model_file)
    net = load_net(model_file)

    deepdream = DeepDream(net=net)
    deepdream.iterated_dream(source_path=source_path, end=layer, octaves=octave)


def parse_arguments(sysargs):
    """ Setup the command line options.
    """

    description = '''DeepDream is an implementation of the Google DeepDream algorithm.
        See the original Googleresearch blog post
        http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html
        '''

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--depth', nargs='?', metavar='int', type=int,
                                    choices=xrange(1, 10),
                                    help='Depth of the dream as an value between 1 and 10')
    parser.add_argument('-t', '--type', nargs='?', metavar='int', type=int,
                                    choices=xrange(1, 6),
                                    help='Layer type as an value between 1 and 6')
    parser.add_argument('-o', '--octaves', nargs='?', metavar='int', type=int,
                                         choices=xrange(1, 12),
                                         help='The number of scales the neural network is applied for')
    parser.add_argument('-n', '--network', action='store_true',
                                         help='Create a new neural network model file')
    parser.add_argument('-i', '--input', nargs='?', metavar='path', type=str,
                                    help='Use the path passed behind -i as source for the dream')

    return parser.parse_args(sysargs)


def get_layer_descriptor(args):
    """ Process input arguments into a layer descriptor and return it.
    """

    layer_depths = ['3a', '3b', '4a', '4b', '4c', '4d', '4e', '5a', '5b']
    layer_types = ['1x1', '3x3', '5x5', 'output', '5x5_reduce', '3x3_reduce']

    # if given, take the input parameter; use random value elseway
    l_depth = (args.depth - 1 if args.depth else randint(0, len(layer_depths)-1))
    l_type = (args.type - 1 if args.type else randint(0, len(layer_types)-1))

    layer = 'inception_' + layer_depths[l_depth] + '/' + layer_types[l_type]

    print(''.join(['\nLayer: ', layer, '\n']))

    return layer


def get_source_image(args):
    """ Input processing: if a source image is supplied, make a time-stamped
    duplicate;  if no image is supplied, make a snapshot."""

    if args.input:
        layer_name = get_layer_descriptor(args).replace('/', '_')
        source_path = args.input.replace('.jpg', '_' + layer_name + '.jpg')
        shutil.copyfile(args.input, source_path)

    print(''.join(['\nBase image for the DeepDream: ', source_path, '\n']))

    return source_path


if __name__ == "__main__":
    try:
        args = parse_arguments(sys.argv[1:])
        while True:
            start_dream(args)
            break

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print('Quitting DeepDream')
