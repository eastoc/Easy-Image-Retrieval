import pymysql
import cv2
import pandas as pd
import urllib.request
import uuid
import os
import numpy as np
from PIL import Image
from io import BytesIO

 # bash
 # mysql -h 139.224.75.52 -P 6013 -u root -p baixin_datatrain
#定义一个函数，用来创建连接数据库

def generate_image_uid():
    return str(uuid.uuid4())
class mysql():
    def __init__(self) -> None:
        self.con = pymysql.connect(
        host='139.224.75.52',
        port=6013,
        database='baixin_datatrain',
        charset='utf8',
        user='root',
        password='lilishop'
    )
    

    def get_data(self):
        with self.con.cursor() as cursor:
            # 二、插入
            #1-插入一条数据
            sql = 'select * from image where status in (1,2,3,4);'
            #keys = ('id', 'file_path')
            #placeholders = ', '.join(['%s'] * len(keys))
            #sql = "select * from image where status in '{keys}'"
            #执行SQL语句
            cursor.execute(sql)
            #执行玩SQL语句要提交
            self.con.commit()
        
            results = cursor.fetchall()
            return results
            

def mysql_db():
    con = pymysql.connect(
        host='139.224.75.52',
        port=6013,
        database='baixin_datatrain',
        charset='utf8',
        user='root',
        password='lilishop'
    )
    try:
        with con.cursor() as cursor:
            # 二、插入
            #1-插入一条数据
            sql = 'select * from image where status in (1,2,3,4);'
            #keys = ('id', 'file_path')
            #placeholders = ', '.join(['%s'] * len(keys))
            #sql = "select * from image where status in '{keys}'"
            #执行SQL语句
            cursor.execute(sql)
            #执行玩SQL语句要提交
            con.commit()
        
            results = cursor.fetchall()
            return results
            for ℹ, row in enumerate(results):
                id = row[0]
                url = row[2]
                img_dir = row[3]
                
                #cv2.imwrite('%d.jpg'%(i), img)
                print(id, url, img_dir)
                url_response, = urllib.request.urlopen
                img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, -1)
            #print("获取的数据：\n",datas[])
            print("提交成功")
 
    except Exception as e:
        # 如果执行失败要回滚
        con.rollback()
        print("数据库异常：\n",e)
    finally:
        #不管成功还是失败，都要关闭数据库连接
        con.close()
 
if __name__ == '__main__':
    database = mysql()
    results = database.get_data()
    #url = 'https://img.zcool.cn/community/01exee0ulgvlgoihkzeuzr3333.jpg'
    #url_response, = urllib.request.urlopen(url)
    title = 'https://minio-api.chengma-ai.com/'
    for ℹ, row in enumerate(results):
        id = row[0]
        img_dir = row[3]        
        img_path = title+img_dir
        print()
        print(id, img_path)
        url_response, = urllib.request.urlopen(img_path)
        #print(img_path)
        #image = Image.open(BytesIO(url_response.read()))
        #img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        #img = cv2.imdecode(img_array, -1)
        #print("获取的数据：\n",datas[])
        print("提交成功")
  