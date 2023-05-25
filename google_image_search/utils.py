import re 
import os
import base64
import hashlib
from functools import wraps
    

def download_img_via_base64(string, filename):
        """当图片以base64形式存储在网页中时, 可使用该方式下载该图片"""
        pattern = re.compile("data:image/(.*?);base64,(.*?$)", re.I | re.M)
        data = re.findall(pattern, string)[0]
        if not data[1].endswith("=="):
            data_ = data[1] + '=='
        else:
            data_ = data[1]
        img_data = base64.b64decode(data_)
        img_name = str(filename) + "." + data[0]
        with open(img_name, "wb") as file:
            file.write(img_data)

def process_filename(filename):
    """将文件名进行处理，尽量避免出现同名的文件"""
    if os.path.exists(filename):
        md = hashlib.md5()
        md.update("就用这个加密吧".encode("utf-8"))
        hexdigest = md.hexdigest()
        basename, extention = os.path.splitext(filename)
        filename = basename + "_" + hexdigest[-3:] + extention
        return filename
    else:
        return filename

def is_img(img_path, extention):
        """判断该文件是否是需要的图片类型"""
        ext = os.path.splitext(img_path)[1]
        if ext in extention:
            return True
        else:
            return False


# 装饰器，用来进行日志记录
def log(separate: bool, content: str, file):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if separate:
                file.write(content + "\n")
            return func(*args, **kwargs)
        return wrapper   
    return decorate


if __name__ == "__main__":

    f = open("data.log", mode='a', encoding='utf-8')

    @log(True, "hello world", f)
    def hello():
        print(hello)
    
    hello()
