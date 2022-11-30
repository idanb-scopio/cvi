from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

class PiecewiseConstLearnRate(object):

    def __init__(self, config):

        self.config = copy.deepcopy(config)
        self.step_size = int(self.config.get('epochs_step_size', 8))
        self.steps_factors = self.config.get('epochs_steps_factors', [1.])
        self.base_learn_rate = float(self.config.get('base_learn_rate', 1e-4))

    def get_scheduler_fn(self):

        def schedule_fn(epoch):

            learn_rate = self.base_learn_rate
            step_idx = epoch // self.step_size
            if step_idx >= len(self.steps_factors):
                factor = self.steps_factors[-1]
            else:
                factor = self.steps_factors[step_idx]
            learn_rate *= float(factor)

            tf.summary.scalar('learning rate', data=learn_rate, step=epoch)

            return learn_rate

        return schedule_fn

