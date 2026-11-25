"""MobileNetV2 transfer-learning model factory."""
from backend.config import MODEL_INPUT_SIZE
import tensorflow as tf

def build_model(class_count:int,pretrained=True,dropout=.3):
    if class_count<2: raise ValueError('At least two breed classes are required for training.')
    try:
        base=tf.keras.applications.MobileNetV2(input_shape=(*MODEL_INPUT_SIZE,3),include_top=False,weights='imagenet' if pretrained else None)
    except Exception as exc:
        if pretrained: raise RuntimeError('Could not load ImageNet weights. Check network access or disable pretrained weights.') from exc
        raise
    base.trainable=False
    inputs=tf.keras.Input((*MODEL_INPUT_SIZE,3),name='rgb_normalized')
    x=tf.keras.applications.mobilenet_v2.preprocess_input(inputs*255.0)
    x=base(x,training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(dropout)(x)
    outputs=tf.keras.layers.Dense(class_count,activation='softmax',name='breed_probabilities')(x)
    model=tf.keras.Model(inputs,outputs,name='BreedAugment_MobileNetV2')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
