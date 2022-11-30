from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.model_ops.learn_rate import PiecewiseConstLearnRate

def learn_rate_schedule_generator(learn_rate_config):

    """
     Generator function for learn rate schedule object
    """

    learn_rate_type = None if (learn_rate_config is None) else learn_rate_config['type']
    if learn_rate_type == 'piecewise_const':
        learn_rate_obj = PiecewiseConstLearnRate(learn_rate_config[learn_rate_type])
    else:
        learn_rate_obj = None

    return learn_rate_obj
