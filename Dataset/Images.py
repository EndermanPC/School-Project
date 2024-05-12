import os
import time
from PIL import Image, ImageTk
import tkinter as tk

def Images(name):
    # Định nghĩa thư mục cần lọc
    dir_path = 'Dataset/Train/' + name

    # Lọc file
    for filename in os.listdir(dir_path):
        if not (filename.endswith('.jpg') or filename.endswith('.text') or filename.endswith('.info') or filename.endswith('.links')):
            os.remove(os.path.join(dir_path, filename))

    # Tạo cửa sổ Tkinter
    root = tk.Tk()

    # Hiển thị và xóa hình ảnh
    for filename in os.listdir(dir_path):
        if filename.endswith('.jpg'):
            # Tạo cửa sổ mới cho mỗi hình ảnh
            new_window = tk.Toplevel(root)
            img = Image.open(os.path.join(dir_path, filename))
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(new_window, image=photo)
            label.pack()

            # Hỏi người dùng có muốn giữ hình ảnh hay không
            user_input = tk.StringVar()
            entry = tk.Entry(new_window, textvariable=user_input)
            entry.pack()
            button = tk.Button(new_window, text='Submit', command=new_window.destroy)
            button.pack()

            root.wait_window(new_window)

            if user_input.get().lower() == 'n':
                os.remove(os.path.join(dir_path, filename))

    # Đặt tên lại hình ảnh
    for filename in os.listdir(dir_path):
        if filename.endswith('.jpg'):
            timestamp = time.strftime('%Y/%m/%d/%H/%M/%S', time.gmtime())
            new_filename = timestamp + '.jpg'
            os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, new_filename))

    root.mainloop()
