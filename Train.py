import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

# Đường dẫn đến thư mục chứa dữ liệu huấn luyện
TRAINING_DATA_DIR = 'Dataset/Train'

# Cấu hình TensorFlow để sử dụng GPU
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

# Tạo ImageDataGenerator với Data Augmentation
def create_datagen():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

# Tải dữ liệu từ thư mục
def load_data(datagen, directory, target_size=(224, 224), batch_size=32, class_mode='categorical'):
    return datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode
    )

# Tạo mô hình dựa trên VGG16
def create_model(input_shape, num_classes):
    # Load pre-trained VGG16 model without the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Thêm lớp Dropout
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create a new model
    return Model(inputs=base_model.input, outputs=predictions)

# Biên dịch mô hình
def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
def train_model(model, train_data, epochs):
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
    model.fit(train_data, epochs=epochs, callbacks=[early_stop, checkpoint])

# Lưu mô hình
def save_model(model, filename):
    model.save(filename)

# Chạy chương trình
def main():
    configure_gpu()
    datagen = create_datagen()
    train_data = load_data(datagen, TRAINING_DATA_DIR)
    num_classes = len(os.listdir(TRAINING_DATA_DIR))
    model = create_model((224, 224, 3), num_classes)
    compile_model(model)
    train_model(model, train_data, epochs=1000)
    save_model(model, 'Model.keras')

if __name__ == "__main__":
    main()
