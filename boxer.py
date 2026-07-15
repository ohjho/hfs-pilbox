###############################################################################
# Home of all Boudning Box Utility Functions
###############################################################################
import numpy as np


# Supported bounding-box conventions, all converted to pascal_voc {x0,y0,x1,y1}.
BBOX_FORMATS = ("pascal_voc", "albumentations", "coco", "coco_normalized")


def to_pascal_voc(coords, fmt: str, im_w: int, im_h: int) -> dict:
    """Convert a bounding box in any supported format to pascal_voc.

    All formats collapse to the pascal_voc convention — a dict of absolute
    top-left/bottom-right pixels ``{x0, y0, x1, y1}`` (matching :func:`get_bbox_dict`
    and ``pilbox``). Given four ordered values ``(c0, c1, c2, c3)``:

    - ``pascal_voc``: ``(x0, y0, x1, y1)`` absolute pixels.
    - ``albumentations``: ``(x0, y0, x1, y1)`` normalized to ``[0, 1]``.
    - ``coco``: ``(x0, y0, w, h)`` absolute pixels (top-left + size).
    - ``coco_normalized``: ``(x0, y0, w, h)`` normalized to ``[0, 1]``.

    Args:
        coords: The four box values ``(c0, c1, c2, c3)`` in the given ``fmt``.
        fmt: One of :data:`BBOX_FORMATS`.
        im_w: Reference image width in pixels (for the normalized formats).
        im_h: Reference image height in pixels (for the normalized formats).

    Returns:
        ``{"x0", "y0", "x1", "y1"}`` of absolute integer pixels.

    Raises:
        ValueError: If ``fmt`` is not one of :data:`BBOX_FORMATS`.

    >>> to_pascal_voc((10, 20, 40, 60), "pascal_voc", 100, 100)
    {'x0': 10, 'y0': 20, 'x1': 40, 'y1': 60}
    >>> to_pascal_voc((0.1, 0.2, 0.4, 0.6), "albumentations", 100, 200)
    {'x0': 10, 'y0': 40, 'x1': 40, 'y1': 120}
    >>> to_pascal_voc((10, 20, 30, 40), "coco", 100, 100)
    {'x0': 10, 'y0': 20, 'x1': 40, 'y1': 60}
    >>> to_pascal_voc((0.1, 0.2, 0.3, 0.4), "coco_normalized", 100, 200)
    {'x0': 10, 'y0': 40, 'x1': 40, 'y1': 120}
    """
    c0, c1, c2, c3 = coords
    if fmt == "pascal_voc":
        return {"x0": int(c0), "y0": int(c1), "x1": int(c2), "y1": int(c3)}
    if fmt == "albumentations":
        return {
            "x0": int(c0 * im_w),
            "y0": int(c1 * im_h),
            "x1": int(c2 * im_w),
            "y1": int(c3 * im_h),
        }
    if fmt == "coco":
        return get_bbox_dict(c0, c1, c2, c3)
    if fmt == "coco_normalized":
        return get_bbox_dict(c0, c1, c2, c3, im_wh=(im_w, im_h))
    raise ValueError(f"unknown bbox format {fmt!r}; expected one of {list(BBOX_FORMATS)}")


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
    '''return True if two pascal_voc boxes overlap.

    Uses the standard axis-aligned overlap test rather than corner-in-rect
    checks: the latter miss "cross" overlaps where the boxes intersect but no
    corner of either lies inside the other (e.g. a tall thin box crossing a
    short wide one).

    >>> a = {'x0': 0, 'y0': 0, 'x1': 10, 'y1': 10}
    >>> bbox_intersects(a, {'x0': 5, 'y0': 5, 'x1': 15, 'y1': 15})
    True
    >>> bbox_intersects(a, {'x0': 20, 'y0': 20, 'x1': 30, 'y1': 30})
    False
    >>> # cross overlap: no corner of either box is inside the other
    >>> bbox_intersects({'x0': 4, 'y0': 0, 'x1': 6, 'y1': 10},
    ...                 {'x0': 0, 'y0': 4, 'x1': 10, 'y1': 6})
    True
    '''
    return (
        bbox_a['x0'] <= bbox_b['x1'] and bbox_a['x1'] >= bbox_b['x0']
        and bbox_a['y0'] <= bbox_b['y1'] and bbox_a['y1'] >= bbox_b['y0']
    )

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
