"""
Multi-input Keras model:
 - Image branch: EfficientNetB0 (ImageNet weights) as feature extractor
 - Metadata branch: small MLP that accepts encoded metadata vectors
 - Concatenate -> Dense heads -> Binary output (sigmoid)
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

def build_multiinput_model(input_image_shape=(224,224,3),
                           metadata_vector_size=10,
                           base_trainable=False,
                           dropout_rate=0.3,
                           learning_rate=1e-4):
    # === Image branch ===
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_image_shape)
    base.trainable = base_trainable  # freeze by default

    x = base.output
    x = layers.GlobalAveragePooling2D(name="img_gap")(x)
    x = layers.Dropout(dropout_rate, name="img_dropout")(x)
    x = layers.Dense(256, activation='relu', name="img_fc1")(x)
    x = layers.BatchNormalization(name="img_bn")(x)
    image_features = layers.Dense(64, activation='relu', name="img_feats")(x)  # final image feature vector

    # === Metadata branch ===
    meta_input = layers.Input(shape=(metadata_vector_size,), name="metadata_input")
    m = layers.Dense(64, activation='relu')(meta_input)
    m = layers.BatchNormalization()(m)
    m = layers.Dropout(dropout_rate)(m)
    m = layers.Dense(32, activation='relu')(m)
    metadata_features = layers.Dense(16, activation='relu')(m)

    # === Concatenate ===
    combined = layers.Concatenate(name="concat")([image_features, metadata_features])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(dropout_rate)(combined)
    combined = layers.Dense(64, activation='relu')(combined)
    combined = layers.Dense(1, activation='sigmoid', name="output")(combined)

    # Full model (image input from base, metadata input defined above)
    image_input = base.input
    model = Model(inputs=[image_input, meta_input], outputs=combined, name="image_meta_classifier")

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model
