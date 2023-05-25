# coding=utf-8
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions

import os
import time
from colorama import Fore
import urllib

from utils import is_img
from settings import load_settings

class GoogleSearcher:
    def __init__(self):
        super().__init__()
        settings = load_settings()
        self._upload = settings['upload']  # 上传的图片所在目录
        self._download = settings['download']  # 下载的文件
        self.sleep_time = settings["sleep_time"]  # 下载网页源代码所等待的时间
        self.separate = settings["separate"]  # 是否分割开下载的数据文件和当前的图片
        self.extention = settings["extention"]
        self.mirror = settings["mirror"]  # 是否使用镜像网站
        
        brower = settings["brower"]
        profile = settings["profile_path"]
        # profile 自定义可以参考 https://blog.csdn.net/weixin_44676081/article/details/106322068
        try:
            self.driver = webdriver.Chrome(executable_path=settings["webdriver_path"]) 
        except Exception as e:
            print(e)
            
    def upload_img_get_url(self, file):
        """上传图片, 并获取对应的html源码"""
        print(f"{Fore.GREEN}Uploading image {os.path.split(file)[1]} {Fore.RESET}")
        self.driver.get("https://www.google.com/searchbyimage/upload")

        # 等待输入框右边的相机图片出现
        condition_1 = expected_conditions.visibility_of_element_located(
            (By.CLASS_NAME, "nDcEnd"))
        WebDriverWait(self.driver, timeout=20,
                      poll_frequency=0.5).until(condition_1)
        # 出现之后点击该按钮
        image_button = self.driver.find_element(By.CLASS_NAME,"nDcEnd")
        image_button.send_keys(Keys.ENTER)

        # 等待界面上出现upload an image
        condition_2 = expected_conditions.visibility_of_element_located(
            (By.CLASS_NAME, "DV7the"))
        WebDriverWait(self.driver, timeout=20, poll_frequency=0.5).until(
            condition_2)

        # 上传文件，此处由于图片的控件是个input，可以直接使用send_keys
        self.driver.find_element(By.XPATH,"//input[@type='file']").send_keys(file)
        print(f"{Fore.GREEN}image completely upload{Fore.RESET}")

        # 当转到另外一个页面的时候
        condition_4 = expected_conditions.visibility_of_element_located(
            (By.ID, "yDmH0d"))
        WebDriverWait(self.driver, timeout=20,
                      poll_frequency=0.5).until(condition_4)
        # driver.implicitly_wait(20)

        time.sleep(self.sleep_time)  # 网络好一点的话可以调小一点

        print(f"{Fore.GREEN}got page url{Fore.RESET}")
        print(self.driver.current_url)
        return self.driver.current_url



    # def analyse(self, url, img_dir, data_text_name):
        """
        下载网页中的图片和文本
        """
        self.driver.get(url)  
        # 紀錄下載過的圖片網址，避免重複下載 
        img_url_dic = {}  
        # 模擬滾動視窗瀏覽更多圖片
        pos = 0 
        m = 0 # 圖片編號 
        for i in range(100):
            pos += i*500 # 每次下滾500  
            js = "document.documentElement.scrollTop=%d" % pos  
            self.driver.execute_script(js)  
            time.sleep(1)
            for j in range(60):
                for k in range(60):
                    xpath = '//*[@id="islrg"]/div[1]/div['+str(j)+']/div['+str(k)+']/a[1]/div[1]/img'
                    for element in self.driver.find_elements(By.XPATH,xpath):
                        try:
                            img_url = element.get_attribute('src')
                            # 保存圖片到指定路徑
                            if img_url != None and not img_url in img_url_dic:
                                img_url_dic[img_url] = ''  
                                m += 1
                                filename = 'image' + str(m) +'.jpg'
                                print('download' +filename)
                                # 保存圖片
                                urllib.request.urlretrieve(img_url, os.path.join(img_dir , filename))
                        except OSError:
                            print('發生OSError!')
                            print(pos)
                            break
            self.driver.close()

    def simple_file_run(self, img, download_path):
        """对单独的一个文件进行搜索"""
        if os.path.isfile(img): # 这里的img是一个完成的路径
            img_name = os.path.splitext(os.path.split(img)[1])[0]  # 所要上传图片的名字
            
            print("--> processing image:  {}  ".format(img_name))
            if is_img(img, self.extention):
                # 在对应的目录下创建新的目录来储存对应获取的内容
                # this_download_dir = os.path.join(download_path, img_name + "_search_data_folder")

                # if not os.path.exists(this_download_dir):
                #     os.mkdir(this_download_dir)
                url_source = self.upload_img_get_url(img)# 获取上传图片之后获取的html source
                
                # self.analyse(url_source, this_download_dir,
                #              this_download_dir + "/" + img_name)  # 解析网页，下载图片，写入网页文本

                print("{}Image url{}process completely\n{}".format(Fore.GREEN, img_name, Fore.RESET))
                print("url: "+str(url_source))
                path = os.path.join(download_path, img_name+ "_search_url.txt")
                txtfile = open(path, 'w')
                txtfile.write(str(url_source))
            else:
                print(f"{Fore.RED}This document {img_name} is not an image{Fore.RESET}")

    def run(self):
        cwd = os.getcwd()
        upload_path = os.path.join(cwd, self._upload)
        for i in os.walk(upload_path):
            current_upload_directory = i[0]
            
            if self.separate:
                related_upload = current_upload_directory.split(self._upload)[1].lstrip("\\").lstrip("/")
                related_download = os.path.join(self._download, related_upload)
                download = os.path.join(cwd, related_download)
            else:
                download = current_upload_directory
            
            if not os.path.exists(download):
                os.mkdir(download)

            if i[-1]:  # 同一目录下的文件列表
                for j in i[-1]:  # 每一个文件
                    img_path = os.path.join(current_upload_directory, j)
                    self.simple_file_run(img_path, download)

    def __del__(self):
        self.driver.close()


if __name__ == "__main__":
    start_time = time.time()
    test = GoogleSearcher()

    test.run()
    end_time = time.time()

    print(f"{Fore.GREEN}Cost: {end_time - start_time} {Fore.RESET}")
