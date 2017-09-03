# %pylab inline
import warnings
warnings.filterwarnings("ignore")
import nolearn
from nolearn.lasagne import NeuralNet
from progress_bar import ProgressBar
import createdata 
import lasagne
from lasagne import layers
from sklearn import metrics
import detectobjects as det

opts = {'img_dir': '../data/intestinalparasites_Images/',
        'models_dir': '../models/',
        'annotation_dir': '../data/intestinalparasites_annotation/',
        'train-dir': 'train_dir/',
        'test-dir': 'test_dir/',
        'val-dir': 'val_dir/',
        'patches_dir': 'patches_dir/',
        'augment-training-data': False,
        'model': '2C-1FC-O',
        'threshold': 0.9, 
        'overlapThreshold': 0.3, 
        'lim': 0, 
        'gauss': 1,
        'prob': det.non_maximum_suppression, 
        'pos': det.non_maximum_suppression, 
        'probs_area': 90,
        'input_scale': None,
        'raw_scale': 255,
        'image_dims': (600,600),
        'image_downsample' : 10,
        'channel_swap': None,
        'probs_area': 40,
        'detection-step': 10,
        'patch-creation-step': 40,
        'object-class': 'hookworm',
        'negative-training-discard-rate': .9
       }
opts['patch_stride_training'] = int(opts['image_dims'][0]*.5)

reload(createdata)
trainfiles, valfiles, testfiles = createdata.create_sets(opts['img_dir'], train_set_proportion=.6, 
                                                  test_set_proportion=.39,
                                                  val_set_proportion=.01)

train_y, train_X = createdata.create_patches(trainfiles, opts['annotation_dir'], opts['img_dir'],
opts['image_dims'][0], opts['patch_stride_training'], grayscale=False, progressbar=True, downsample=opts['image_downsample'], 
objectclass=opts['object-class'], negative_discard_rate=opts['negative-training-discard-rate'])

test_y, test_X = createdata.create_patches(testfiles,  opts['annotation_dir'], opts['img_dir'], 
opts['image_dims'][0], opts['patch_stride_training'], grayscale=False, progressbar=True, downsample=opts['image_downsample'], 
objectclass=opts['object-class'], negative_discard_rate=opts['negative-training-discard-rate'])

val_y, val_X = createdata.create_patches(valfiles, opts['annotation_dir'], opts['img_dir'], 
opts['image_dims'][0], opts['patch_stride_training'], grayscale=False, progressbar=True, downsample=opts['image_downsample'], 
objectclass=opts['object-class'], negative_discard_rate=opts['negative-training-discard-rate'])

# For training/validation, cut down on disproportionately large numbers of negative patches
train_X, train_y = createdata.balance(train_X, train_y, mult_neg=100)
val_X, val_y = createdata.balance(val_X, val_y, mult_neg=100)

# Create rotated and flipped versions of the positive patches
train_X, train_y = createdata.augment_positives(train_X, train_y)
val_X, val_y = createdata.augment_positives(val_X, val_y)
test_X, test_y = createdata.augment_positives(test_X, test_y)
