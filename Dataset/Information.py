import wikipedia

def replace_and_trim(s):
    # Thay thế tất cả các dấu '=' bằng '#'
    s = s.replace('=', '#')
    # Loại bỏ tất cả các dấu '#' ở cuối chuỗi
    s = s.rstrip('#')
    return s

def process_text(text):
    # Chia đoạn văn bản thành từng chuỗi dựa trên dấu xuống dòng
    lines = text.split('\n')
    # Áp dụng hàm replace_and_trim cho từng chuỗi
    processed_lines = [replace_and_trim(line) for line in lines]
    # Tạo một danh sách mới để lưu các dòng đã xử lý
    new_lines = []
    for i in range(len(processed_lines)):
        # Kiểm tra nếu dòng bắt đầu bằng '#' hoặc là dòng trống, hoặc dòng tiếp theo trống, hoặc là dòng cuối cùng
        if processed_lines[i].startswith('#') or processed_lines[i].strip() == '' or (i < len(processed_lines) - 1 and processed_lines[i+1].strip() == '') or i == len(processed_lines) - 1:
            new_lines.append(processed_lines[i])
        else:
            new_lines.append(processed_lines[i] + '\\')
    # Nối các chuỗi đã xử lý lại thành một đoạn văn bản
    return '\n'.join(new_lines)

def Wikipedia_Information(name):
    wikipedia.set_lang("vi")

    page_py = wikipedia.page('"' + name + '"').content

    page_py = process_text(page_py)
    
    file_path = f'Dataset/Train/{name}/information.text'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(page_py)
