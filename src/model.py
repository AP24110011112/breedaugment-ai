"""MobileNetV2 classifier factory."""
import tensorflow as tf

def build_model(class_count,pretrained=True,dropout=.3):
    if class_count<2: raise ValueError('At least two breed classes are required.')
    base=tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet' if pretrained else None)
    base.trainable=False
    inputs=tf.keras.Input((224,224,3),name='rgb_0_1')
    x=tf.keras.applications.mobilenet_v2.preprocess_input(inputs*255.)
    x=base(x,training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(dropout)(x)
    outputs=tf.keras.layers.Dense(class_count,activation='softmax')(x)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
