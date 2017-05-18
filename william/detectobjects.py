
import numpy as np
import  skimage.filters as filters
from scipy import misc
    
def detect(imfile, cnn, opts):
    step = opts['detection-step']
    downsample = opts['image_downsample']
    size = opts['image_dims'][0]
    p = predict(cnn, imfile, step, size, downsample)

    boxes = get_boxes(imfile, p, step, size, gauss=opts['gauss'], threshold=opts['threshold'] )
    found = non_maximum_suppression(boxes, overlapThreshold = 15 )
    return found
     


def predict(classifier, img_filename, step, size, downsample=1):
            img = misc.imread(img_filename)
            height, width,channels = img.shape

            probs = np.zeros((img.shape[0]*1.0/step,img.shape[1]*1.0/step))
            patches = []

            y=0
            while y+(size) < height:
                     #rows
                     x = 0
                     predictions=[]
                     while (x+(size) < width):
                         left = x
                         right = x+(size)
                         top = y
                         bottom = y+(size)
                         patches.append(img[top:bottom:downsample, left:right:downsample,:])
                         x += step
                     y += step

            p = np.array(patches)
            p = np.swapaxes(p,1,3)
            p = np.swapaxes(p,2,3)
            predictions = classifier.predict_proba(p)
    
            i=0
            y=0
            while y+(size) < height:
                  x = 0
                  while (x+(size) < width):
                      left = x
                      right = x+(size)
                      top = y
                      bottom = y+(size)
                      probs[y/step,x/step]=predictions[i,1]
                      i+=1
                      x += step
                  y += step

            return probs
     



def get_boxes(img_filename, probs, step, size, gauss=0,threshold=0.5):

    if gauss != 0:
        probs = filters.gaussian_filter(probs, gauss)
        
    img = misc.imread(img_filename)
    height, width,channels = img.shape

    boxes=[]

    i=0
    y=0
    while y+(size) < height:
                x = 0
                while (x+(size) < width):
                    left = int(x)
                    right = int(x+(size))
                    top = int(y)
                    bottom = int(y+(size))
                    if probs[y/step,x/step] > threshold:
                        boxes.append([left,top,right,bottom,probs[y/step,x/step]])
                    i+=1
                    x += step
                y += step

    if len(boxes) == 0:
        return np.array([])

    boxes =  np.vstack(boxes)
    return boxes


# Malisiewicz et al.
# Python port by Adrian Rosebrock
def non_maximum_suppression(boxes, overlapThreshold = 0.5):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []

  # if the bounding boxes integers, convert them to floats --
  # this is important since we'll be doing a bunch of divisions
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")

  # initialize the list of picked indexes 
  pick = []

  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
  scores = boxes[:,4]
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the score/probability of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(scores)[::-1]

  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]

    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > overlapThreshold)[0])))

  # return only the bounding boxes that were picked using the
  # integer data type
  return boxes[pick].astype("int")
