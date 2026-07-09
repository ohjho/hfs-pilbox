###############################################################################
# Home of all Boudning Box Utility Functions
###############################################################################
import os, sys, json
import numpy as np

def get_bbox_dict(x, y, width, height, im_wh = None)-> dict:
    '''given top-left (x,y), return a bbox dict with absolute coordinates x0, y0, x1, y1
    Args:
        im_width: if provided, treats x,y,w,h as relative coordinates
        im_height: if provided, treats x,y,w,h as relative coordinates
    '''
    is_relative = type(im_wh) != type(None)
    if is_relative:
        w,h = im_wh
    return {'x0': int(x * w) if is_relative else int(x),
        'y0': int(y * h) if is_relative else int(y),
        'x1': int((x+width) * w) if is_relative else int(x + width),
        'y1': int((y+height) * h) if is_relative else int(y + height)}

def bbox_rebase_xy(x,y,w,h, to_yolo_format = False):
    '''convert bbox format frorm yolo (centroid) to standard (top-left) or vice versa
    '''
    x = x+w/2 if to_yolo_format else x - w/2
    y = y+h/2 if to_yolo_format else y - h/2
    return {'x': x, 'y': y, 'w': w, 'h': h}

def bbox_get_xywh(x0,y0,x1,y1):
    '''returns dict of x,y,w,h
    '''
    return {'x': x0, 'y': y0, 'w': x1-x0 ,'h': y1-y0}

def bbox_convert(x0, y0, x1, y1, width, height):
    '''convert bounding box from relative to absolute and vice versa
    Args:
        width: reference image's width
        height: reference image's height
    '''
    if all([i <= 1 for i in [x0,y0,x1,y1]]):
        # relative to absolute
        return {
            'x0': int(x0 * width), 'x1': int(x1 * width),
            'y0': int(y0 * height), 'y1': int(y1 * height)
        }
    else:
        # absolute to relative
        if x0 > width or x1> width:
            raise ValueError(f'{x0} or {x1} is greater than width: {width}')
        if y0 > height or y1> height:
            raise ValueError(f'{y0} or {y1} is greater than height: {height}')

        return {
            'x0': x0 / width, 'x1': x1 / width,
            'y0': y0 / height, 'y1': y1 / height
        }

def bboxes_to_im_mask(l_bboxes, im_wh):
    ''' return a binary mask given a list of bounding boxes
    '''
    mask = np.zeros(shape = (im_wh[::-1]), dtype = np.uint8)
    for bbox in l_bboxes:
        mask[bbox['y0']: bbox['y1'], bbox['x0']: bbox['x1']] = 1
    return mask

def bbox_intersects(bbox_a, bbox_b):
    if bbox_b['x0'] >= bbox_a['x0'] and bbox_b['x0'] <= bbox_a['x1'] and \
        bbox_b['y0'] >= bbox_a['y0'] and bbox_b['y0'] <= bbox_a['y1']:
        # top-left of b within a
        return True
    elif bbox_b['x1'] >= bbox_a['x0'] and bbox_b['x1'] <= bbox_a['x1'] and \
        bbox_b['y1'] >= bbox_a['y0'] and bbox_b['y1'] <= bbox_a['y1']:
        # bottom-right of b within a
        return True
    elif bbox_a['x0'] >= bbox_b['x0'] and bbox_a['x0'] <= bbox_b['x1'] and \
        bbox_a['y0'] >= bbox_b['y0'] and bbox_a['y0'] <= bbox_b['y1']:
        # top-left of a within b
        return True
    elif bbox_a['x1'] >= bbox_b['x0'] and bbox_a['x1'] <= bbox_b['x1'] and \
        bbox_a['y1'] >= bbox_b['y0'] and bbox_a['y1'] <= bbox_b['y1']:
        # bottom-right of a within b
        return True
    return False

def bbox_area(x0, y0, x1, y1):
    return (x1-x0+1) * (y1-y0+1)

def get_bbox_iou(bbox_a, bbox_b):
    if bbox_intersects(bbox_a, bbox_b):
        x_left = max(bbox_a['x0'], bbox_b['x0'])
        x_right = min(bbox_a['x1'], bbox_b['x1'])
        y_top = max(bbox_a['y0'], bbox_b['y0'])
        y_bottom = min(bbox_a['y1'], bbox_b['y1'])

        inter_area = bbox_area(x0 = x_left, x1 = x_right, y0 = y_top , y1 = y_bottom)
        bbox_a_area = bbox_area(**bbox_a)
        bbox_b_area = bbox_area(**bbox_b)

        return inter_area / float(bbox_a_area + bbox_b_area - inter_area)
    else:
        return 0

def boxer(lsXY, pctBuffer = 0.3, lXBounds = None, lYBounds = None):
	'''
	Create a minimum Bounding Box given a list of x,y coordinates
	and a buffer given in percentages of the output image size
	if pctBuffer is given as a tuple:
		pctBuffer[0] will be the x buffer
		pctBuffer[1] will be the y buffer
	Optional: Provide XBounds and YBounds to ensure the returned values
				fits within range.
	'''
	minX = minY = float("inf")
	maxX = maxY = float("-inf")

	for x, y in lsXY:
		# set min coords
		if x < minX:
			minX = x
		if y < minY:
			minY = y

		# set max coords
		if x > maxX:
			maxX = x
		if y > maxY:
			maxY = y

	width = maxX - minX
	height = maxY - minY

	if type(pctBuffer) == tuple:
		xBuffer = pctBuffer[0]
		yBuffer = pctBuffer[1]
	else:
		xBuffer = yBuffer = pctBuffer

	coordsDict ={
		'x1': int(minX - xBuffer * width),
		'x2': int(maxX + xBuffer * width),
		'y1': int(minY - yBuffer * height),
		'y2': int(maxY + yBuffer * height)
	}

	if lXBounds:
		coordsDict['x1'] = max( lXBounds[0], coordsDict['x1'])
		coordsDict['x2'] = min( lXBounds[1], coordsDict['x2'])
	if lYBounds:
		coordsDict['y1'] = max( lYBounds[0], coordsDict['y1'])
		coordsDict['y2'] = min( lYBounds[1], coordsDict['y2'])

	return coordsDict
