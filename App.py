import os
import time
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

def Get_Information(predicted_name):
    # Đường dẫn đến thư mục 'Dataset/Train'
    train_dir = 'Dataset/Train'

    # Duyệt qua mỗi lớp trong thư mục 'Dataset/Train'
    for class_name in class_names:
        # Nếu class_name không phải là predicted_name, tiếp tục vòng lặp
        if class_name != predicted_name:
            continue

        # Đường dẫn đến tệp 'information.text' trong thư mục của lớp
        info_file_path = os.path.join(train_dir, class_name, 'information.text')

        # Kiểm tra xem tệp 'information.text' có tồn tại không
        if os.path.exists(info_file_path):
            # Đọc nội dung của tệp
            with open(info_file_path, 'r', encoding='utf-8') as f:
                info_text = f.read()
            
            return info_text
        else:
            return f"Không tìm thấy tệp 'information.text' trong thư mục của lớp '{class_name}'."
        
def Get_Notes(predicted_name):
    # Đường dẫn đến thư mục 'Dataset/Train'
    train_dir = 'Dataset/Train'

    # Duyệt qua mỗi lớp trong thư mục 'Dataset/Train'
    for class_name in class_names:
        # Nếu class_name không phải là predicted_name, tiếp tục vòng lặp
        if class_name != predicted_name:
            continue

        # Đường dẫn đến tệp 'notes.info' trong thư mục của lớp
        info_file_path = os.path.join(train_dir, class_name, 'notes.info')

        # Kiểm tra xem tệp 'notes.info' có tồn tại không
        if os.path.exists(info_file_path):
            # Đọc nội dung của tệp
            with open(info_file_path, 'r', encoding='utf-8') as f:
                info_text = f.read()
            
            return info_text
        else:
            return f"Không tìm thấy tệp 'notes.info' trong thư mục của lớp '{class_name}'."

# Tải mô hình
model = load_model('Model.keras')

# Khởi tạo ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Tải dữ liệu từ thư mục 'Dataset/Train'
train_data = datagen.flow_from_directory(
    'Dataset/Train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Get the dictionary of class indices
class_indices = train_data.class_indices

# Get the list of class names
class_names = list(class_indices.keys())

st.title("Nhận diện và tra cứu thông tin động vật quý hiếm thông qua học sâu")

uploaded_file = st.sidebar.file_uploader("Hoặc chọn một hình ảnh từ máy tính của bạn", type=['jpg', 'png'])
photo = st.camera_input("Chụp một bức ảnh")

if uploaded_file is not None or photo is not None:
    if photo is not None:
        img = Image.open(photo)
    else:
        img = Image.open(uploaded_file)
        st.image(img, caption='Hình ảnh đã tải lên.', use_column_width=True)
    
    img = img.resize((224, 224))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Add a third dimension (for batch size), and rescale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    with st.spinner('Đang xử lý...'):
        # Use the model to predict the image's label
        predictions = model.predict(img_array)
        time.sleep(3)

    # The output of the model is a 2D array, with shape (1, num_classes). 
    # To get the predicted label, we get the index of the maximum value in the array.
    predicted_label = np.argmax(predictions)

    # Use the predicted label to get the class name
    predicted_class_name = class_names[predicted_label]

    st.write("Độ Chính Xác")
    st.bar_chart(predictions)

    st.write("`ID: " + str(predicted_label) + "`")
    st.write("## Tên: " + predicted_class_name)
    st.markdown(Get_Notes(predicted_class_name))
    with st.expander("Xem thông tin chi tiết"):
        st.markdown(Get_Information(predicted_class_name))
