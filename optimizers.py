import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.backend as K


class AdamAccumulation(Optimizer):
    def __init__(
        self,
        accumulation_steps,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        bias_correction=True,
        **kwargs
    ):
        kwargs['name'] = kwargs.get('name') or 'Adam'
        super(AdamAccumulation, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('accumulation_steps', accumulation_steps)
        self.epsilon = epsilon or K.epislon()
        self.bias_correction = bias_correction

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            self.add_slot(var, 'accum_grad')

    def _resource_apply(self, grad, var, indices=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        accum_grad = self.get_slot(var, 'accum_grad')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        accumulation_steps = self._get_hyper('accumulation_steps', var_dtype)
        epsilon_t = K.cast(self.epsilon, var_dtype)
        local_step = K.cast(self.iterations + 1, var_dtype)
        complete_step = K.cast((local_step - 1) // accumulation_steps, var_dtype) + 1

        update_switch = K.equal(local_step // accumulation_steps * accumulation_steps, local_step)
        update_switch = K.cast(update_switch, K.floatx())
        
        beta_1_t_power = K.pow(beta_1_t, complete_step)
        beta_2_t_power = K.pow(beta_2_t, complete_step)

        
        if indices is None:
            accum_grad_t = accum_grad + grad
            avg_grad_t = accum_grad_t / accumulation_steps
            with tf.control_dependencies([avg_grad_t]):
                m_t = beta_1_t * m + (1 - beta_1_t) * avg_grad_t
                v_t = beta_2_t * v + (1 - beta_2_t) * K.square(avg_grad_t)
        else:
            accum_grad_t = self._resource_scatter_add(accum_grad, indices, (1 - beta_1_t) * grad)
            avg_grad_t = accum_grad_t / accumulation_steps
            with tf.control_dependencies([avg_grad_t]):
                m_t = beta_1_t * m + (1 - beta_1_t) * avg_grad_t
                v_t = beta_2_t * v + (1 - beta_2_t) * K.square(avg_grad_t)

        if self.bias_correction:
            lr_t = lr_t * K.sqrt(1.0 - beta_2_t_power) / (1.0 - beta_1_t_power)
        with tf.control_dependencies([lr_t, update_switch]):
            var_t = var - lr_t * m_t / (K.sqrt(v_t) + epsilon_t)
            # only update m and v when update switch is on
            _ = K.update(m, (1 - update_switch) * m + update_switch * m_t)
            _ = K.update(v, (1 - update_switch) * v + update_switch * v_t)
            # turn accum_grad to zero when update switch is on
            _ = K.update(accum_grad, (1 - update_switch) * accum_grad_t)
            return K.update(var, (1 - update_switch) * var + update_switch * var_t)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'bias_correction': self.bias_correction,
            'accumulation_steps': self.accumulation_steps,
        }
        base_config = super(AdamAccumulation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))