import numpy as np
from python.utils import transformations

def test_gt_state():
    # sigma1 = s1
    # sigma2 = s1, s2
    # sigma3 = s1, s2, s3
    gt = [1,3,7]
    assert(transformations.gt_state(0, gt) == 0)
    assert(transformations.gt_state(4, gt) == 4)
    assert(transformations.gt_state(5, gt) == 3)
    assert(transformations.gt_state(7, gt) == 5)

def test_gt_model():
    # s1 = sigma1
    # s2 = sigma1, sigma2
    # s3 = sigma2, sigma3
    gt = [1,3,6]
    assert(np.all(transformations.gt_model([1,2,3,4,5,6,7], gt) == [1,3,2,6,7,5,4]))
