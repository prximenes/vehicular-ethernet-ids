# distillation_model.py
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class DistillationModel(tf.keras.Model):
    def __init__(self, teacher_model, student_model, alpha, temperature):
        super(DistillationModel, self).__init__()
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

        # Atualizar métricas
        for metric in self.train_metrics:
            metric.update_state(y, student_preds)

        return {"loss": loss, **{m.name: m.result() for m in self.train_metrics}}

    def test_step(self, data):
        x, y = data
        student_preds = self.student_model(x, training=False)
        loss = tf.keras.losses.BinaryCrossentropy()(y, student_preds)

        # Atualizar métricas
        for metric in self.train
