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

def test_gt_state_discrete():
    # sigma1 = s1 * s2
    # sigma2 = s1 * s2^2
    gt = [4,5]
    assert(np.all(transformations.gt_state_discrete([0,0], gt, q=3) == [0,0]))
    assert(np.all(transformations.gt_state_discrete([0,1], gt, q=3) == [1,2]))
    assert(np.all(transformations.gt_state_discrete([0,2], gt, q=3) == [2,1]))
    assert(np.all(transformations.gt_state_discrete([1,0], gt, q=3) == [1,1]))
    assert(np.all(transformations.gt_state_discrete([1,1], gt, q=3) == [2,0]))
    assert(np.all(transformations.gt_state_discrete([1,2], gt, q=3) == [0,2]))
    assert(np.all(transformations.gt_state_discrete([2,0], gt, q=3) == [2,2]))
    assert(np.all(transformations.gt_state_discrete([2,1], gt, q=3) == [0,1]))
    assert(np.all(transformations.gt_state_discrete([2,2], gt, q=3) == [1,0]))

def test_gt_model():
    # s1 = sigma1
    # s2 = sigma1, sigma2
    # s3 = sigma2, sigma3
    gt = [1,3,6]
    assert(np.all(transformations.gt_model([1,2,3,4,5,6,7], gt) == [1,3,2,6,7,5,4]))
