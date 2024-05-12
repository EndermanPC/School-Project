import os
import tensorflow as tf

# Cấu hình TensorFlow để sử dụng GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
from tensorflow.keras.models import Model # type: ignore

Dataset = 'Dataset/Train'

# Khởi tạo ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Tải dữ liệu từ Dataset
train_data = datagen.flow_from_directory(
    Dataset,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

classQuantity = len([name for name in os.listdir(Dataset) if os.path.isdir(os.path.join(Dataset, name))])

# Load pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, classQuantity))

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(classQuantity, activation='softmax')(x)

# Create a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Số lượng epoch bạn muốn huấn luyện
epochs = 150

# Huấn luyện mô hình với dữ liệu đã tải
model.fit(train_data, epochs=epochs)

# Lưu mô hình
model.save('Model.h5')