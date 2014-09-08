import collections
import glob
import numpy as np
import random
from scipy import misc

np.random.seed(12345)
random.seed(12345)

class Datapoint(collections.namedtuple("Datapoint", ("data", "label", "fn"))):
    def __repr__(self):
        return "Datapoint(data=array%s, label=%r, fn=%r)" % (self.data.shape, self.label, self.fn)

def loadTrainingSet():
    images = []
    if not glob.glob("training_set/*"):
        R, C = 8, 8
        funcs = [
                (lambda x, y: x % 4 in (0, 1), 'v'),
                (lambda x, y: x % 4 in (1, 2), 'v'),
                (lambda x, y: x % 4 in (2, 3), 'v'),
                (lambda x, y: x % 4 in (3, 0), 'v'),
                (lambda x, y: y % 4 in (0, 1), 'h'),
                (lambda x, y: y % 4 in (1, 2), 'h'),
                (lambda x, y: y % 4 in (2, 3), 'h'),
                (lambda x, y: y % 4 in (3, 0), 'h'),
                ]
        for i, (f, l) in enumerate(funcs):
            im = np.zeros((R, C))

            for x in xrange(R):
                for y in xrange(C):
                    if f(x, y):
                        im[x,y] = 1.0

            fn = "training_set/test%d.bmp" % i
            # misc.imsave(fn, im)
            images.append(Datapoint(im, l, fn))
    for fn in glob.glob("training_set/*"):
        print fn
        1/0
    return images

def loadTestSet():
    images = []
    if not glob.glob("test_set/*"):
        R, C = 8, 8
        funcs = [
                (lambda x, y: x % 4 in (2, 3), 'v'),
                (lambda x, y: y % 4 in (2, 3), 'h'),
                ]
        for i, (f, l) in enumerate(funcs):
            im = np.zeros((R, C))

            for x in xrange(R):
                for y in xrange(C):
                    if f(x, y):
                        im[x,y] = 1.0

            fn = "test_set/test%d.bmp" % i
            # misc.imsave(fn, im)
            images.append(Datapoint(im, l, fn))
    for fn in glob.glob("test_set/*"):
        print fn
        1/0
    return images

class CropLayer(object):
    def __init__(self, R, C):
        self.R = R
        self.C = C

    def fprop(self, data):
        return data[:self.R,:self.C]

    def bprop(self, data, grad, scale, label):
        return None, 0.0

class SimpleLinear(object):
    def __init__(self, R, C):
        self.R = R
        self.C = C

        self.weights = np.random.random((R, C)) - 0.5
        self.bias = np.random.random() - 0.5

    def fprop(self, data):
        return (self.weights * data).sum() + self.bias

    def bprop(self, data, grad, scale, label):
        assert isinstance(grad, float)

        u = 0.1
        self.weights -= scale * (data * grad + 2 * u * self.weights)
        self.bias -= scale * grad
        return self.weights * grad, (u * self.weights ** 2).sum()

class SimpleClassPicker(object):
    def fprop(self, data):
        return data
        assert isinstance(data, float), data
        if -0.5 < data < 0.5:
            return "unknown"
        if data > 0:
            return 'h'
        return 'v'

    def bprop(self, data, grad, scale, label):
        if label == 'h':
            return 2 * (data - 1), (data - 1) ** 2
        return 2 * (data + 1), (data + 1) ** 2

class Classifier(object):
    def __init__(self):
        self.layers = []

        self.layers.append(CropLayer(4, 4))
        self.layers.append(SimpleLinear(4, 4))
        self.layers.append(SimpleClassPicker())

    def _update(self, t):
        data = t.data

        datas = [data]
        for l in self.layers:
            data = l.fprop(data)
            datas.append(data)

        assert isinstance(data, float)
        total_err = 0.0
        # print err

        grad = 1.0
        for i, l in reversed(list(enumerate(self.layers))):
            grad, err = l.bprop(datas[i], grad, 0.001, t.label)
            total_err += err

        return total_err

    def train(self, training_set):
        training_set = list(training_set)

        last = []

        for _i in xrange(2000000):
            random.shuffle(training_set)

            total_err = 0.0

            for t in training_set:
                # print t.label
                total_err += self._update(t)

            total_err /= len(training_set)
            print total_err

            if len(last) >= 10:
                if total_err > max(last):
                    print "Stopped getting better after %d iterations" % _i
                    break
                last.pop(0)
            last.append(total_err)

    def classify(self, im):
        layers = self.layers + [SimpleClassPicker()]

        data = im
        for l in layers:
            data = l.fprop(data)
        return data

def main():
    training_set = loadTrainingSet()
    classifier = Classifier()

    classifier.train(training_set)

    print classifier.layers[1].weights

    test_set = loadTestSet()
    # for t in test_set:
    for t in training_set:
        label = classifier.classify(t.data)
        print t, label
    print
    for t in test_set:
        label = classifier.classify(t.data)
        print t, label

if __name__ == "__main__":
    main()
