#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from keras_tuner import Hyperband

class PromoterClassifier:
    def __init__(self, sequence_length=600, n_features=5, model_path=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = load_model(model_path) if model_path else None

    def build_model(self, hp):
        """Builds a CNN model for promoter classification with hyperparameter tuning."""
        model = models.Sequential()
        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))
        model.add(layers.Conv1D(filters=hp.Int('conv1_filters', 32, 256, step=32),
                                kernel_size=hp.Int('conv1_kernel', 3, 20, step=2),
                                activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=hp.Int('pool1_size', 2, 4)))

        if hp.Boolean('conv2'):
            model.add(layers.Conv1D(filters=hp.Int('conv2_filters', 32, 256, step=32),
                                    kernel_size=hp.Int('conv2_kernel', 3, 20, step=2),
                                    activation='relu'))
            model.add(layers.MaxPooling1D(pool_size=hp.Int('pool2_size', 2, 4)))

        model.add(layers.Flatten())
        model.add(layers.Dense(units=hp.Int('dense1_units', 64, 512, step=64), activation='relu'))
        model.add(layers.Dropout(hp.Float('dropout1', 0.25, 0.5, step=0.1)))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-6, 1e-3, sampling='log')),
                      loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    def train_model(self, X_train, y_train, X_val, y_val):
        """Runs hyperparameter tuning and trains the model."""
        tuner = Hyperband(self.build_model,
                          objective='val_accuracy',
                          max_epochs=50,
                          directory='gc_model_dir',
                          project_name='promoter_classification')

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        tuner.search(X_train, y_train, validation_data=(X_val, y_val),
                     callbacks=[early_stopping], epochs=30, batch_size=32)

        best_hps = tuner.get_best_hyperparameters(1)[0]
        self.model = tuner.hypermodel.build(best_hps)

        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                 epochs=50, batch_size=32, callbacks=[early_stopping])
        return history, best_hps

    def predict(self, X):
        """Runs predictions using the trained model."""
        prediction = self.model.predict(X)
        return (prediction >= 0.5).astype(int), prediction
