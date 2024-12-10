import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class DistillationModel(tf.keras.Model):
    def __init__(self, teacher_model, student_model, alpha, temperature, **kwargs):
        super(DistillationModel, self).__init__(**kwargs)
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.alpha = alpha
        self.temperature = temperature

    def compile(self, optimizer, metrics):
        super(DistillationModel, self).compile()
        self.optimizer = optimizer
        self.train_metrics = metrics

    def train_step(self, data):
        x, y = data
        teacher_preds = self.teacher_model(x, training=False)

        with tf.GradientTape() as tape:
            student_preds = self.student_model(x, training=True)
            ce_loss = tf.keras.losses.BinaryCrossentropy()(y, student_preds)
            kl_loss = tf.keras.losses.KLDivergence()(
                tf.nn.softmax(teacher_preds / self.temperature),
                tf.nn.softmax(student_preds / self.temperature)
            )
            loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss

        grads = tape.gradient(loss, self.student_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student_model.trainable_weights))

        for metric in self.train_metrics:
            metric.update_state(y, student_preds)

        return {"loss": loss, **{m.name: m.result() for m in self.train_metrics}}

    def test_step(self, data):
        x, y = data
        student_preds = self.student_model(x, training=False)
        loss = tf.keras.losses.BinaryCrossentropy()(y, student_preds)

        for metric in self.train_metrics:
            metric.update_state(y, student_preds)

        return {"loss": loss, **{m.name: m.result() for m in self.train_metrics}}

    def call(self, inputs, training=False):
        return self.student_model(inputs, training=training)

    def get_config(self):
        """Salva os parâmetros necessários para recriar o modelo."""
        config = super(DistillationModel, self).get_config()
        config.update({
            "teacher_model": tf.keras.models.serialize(self.teacher_model),
            "student_model": tf.keras.models.serialize(self.student_model),
            "alpha": self.alpha,
            "temperature": self.temperature,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Recria o modelo a partir do dicionário de configuração."""
        teacher_model_config = config.pop("teacher_model")
        student_model_config = config.pop("student_model")

        # Desserializar os modelos do professor e do aluno
        teacher_model = tf.keras.models.model_from_config(teacher_model_config['config'])
        student_model = tf.keras.models.model_from_config(student_model_config['config'])

        return cls(
            teacher_model=teacher_model,
            student_model=student_model,
            **config
        )


