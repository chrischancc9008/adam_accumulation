import tensorflow as tf
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
            self.add_slot(var, 'avg_grad', initializer=tf.keras.initializers.Zeros())

    def _resource_apply(self, grad, var, indices=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        avg_grad = self.get_slot(var, 'avg_grad')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        accumulation_steps = self._get_hyper('accumulation_steps', var_dtype)
        epsilon_t = K.cast(self.epsilon, var_dtype)
        local_step = K.cast(self.iterations + 1, var_dtype)
        iteration = K.cast(self.iterations, var_dtype)

        update_switch_t = K.equal(local_step // accumulation_steps * accumulation_steps, local_step)
        grad_update_switch = K.equal(iteration - iteration // accumulation_steps * accumulation_steps, 0)
        complete_step = K.cast((local_step - 1) // accumulation_steps, var_dtype) + 1

        beta_1_t_power = K.pow(beta_1_t, complete_step)
        beta_2_t_power = K.pow(beta_2_t, complete_step)

        avg_grad_t1 = K.update(avg_grad, tf.where(grad_update_switch,
                                                  avg_grad * 0,
                                                  avg_grad,
                                                  ))

        with tf.control_dependencies([avg_grad_t1]):
            if indices is None:
                avg_grad_t2 = K.update(avg_grad, avg_grad_t1 + grad / accumulation_steps)
            else:
                avg_grad_t2 = self._resource_scatter_add(avg_grad, indices, grad / accumulation_steps)

        with tf.control_dependencies([avg_grad_t1, avg_grad_t2, update_switch_t]):
            m_t = beta_1_t * m + (1 - beta_1_t) * avg_grad_t2
            v_t = beta_2_t * v + (1 - beta_2_t) * K.square(avg_grad_t2)
            m_t = K.update(m, tf.where(update_switch_t, m_t, m))
            v_t = K.update(v, tf.where(update_switch_t, v_t, v))

        if self.bias_correction:
            with tf.control_dependencies([beta_1_t_power, beta_2_t_power]):
                lr_t = lr_t * K.sqrt(1.0 - beta_2_t_power) / (1.0 - beta_1_t_power)

        with tf.control_dependencies([lr_t, update_switch_t, m_t, v_t, avg_grad_t1, avg_grad_t2, ]):
            var_t = var - lr_t * m_t / (K.sqrt(v_t) + epsilon_t)
            return K.update(var, tf.where(update_switch_t, var_t, var))

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
        base_config.update(config)
        return config
