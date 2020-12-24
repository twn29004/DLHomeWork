#!/usr/bin/python

import os
import random
import struct

from PIL import Image

path = 'F:\TODO\深度学习课程\作业\大作业\dataset'
writerStart = 1241
writerEnd = 1301


# 需要先把图片按字分类，得出每个字有几个，然后选出最频繁的几个
def wordNum():
    dictionary = dict()
    cnt = 0
    for writer in range(writerStart, writerEnd):
        ff = path + '\\HWDB1.1tst_gnt\\' + str(writer) + '-c.gnt'
        f = open(ff, 'rb')
        while f.read(1) != "":
            cnt += 1
            # -1 表示回到起点，4表示跳过数据的长度
            f.seek(-1, 1)
            # <表示的是小端存储，I表示的unsignedint
            # 使用[0]的原因是返回值是一个元组
            try:
                length_bytes = struct.unpack('<I', f.read(4))[0]
            except struct.error:
                break
            tag_code = f.read(2)
            tag_code = int.from_bytes(tag_code, byteorder="big")
            if tag_code in dictionary.keys():
                dictionary[tag_code] += 1
            else:
                dictionary[tag_code] = 1
            # 需要重新设置读指针
            f.seek(-6 + length_bytes, 1)
        f.close()

    tmp = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)[0:140]
    # 随机的选取30个
    length = len(tmp) if len(tmp) < 140 else 140
    indexs = random.sample(range(0, length), 30)
    wordbag = set()
    for index in indexs:
        wordbag.add(tmp[index][0])
    return wordbag


def createImage():
    # 创建存储图片的目录
    if not os.path.exists(path + "\\image"):
        os.mkdir(path + "\\image")
        print("新建文件夹：", path + "\\image")

    # 把这30个tag作为字典的key,然后按照这里面的值来初始化,count作为文件名
    wordCount = dict()
    wordBag = wordNum()
    for item in wordBag:
        wordCount[item] = 0

    for z in range(writerStart, writerEnd):
        ff = path + '\\HWDB1.1tst_gnt\\' + str(z) + '-c.gnt'
        f = open(ff, 'rb')
        print("start process {}".format(z))
        while f.read(1) != "":
            # 1表示当前位置，-1表示往前移
            f.seek(-1, 1)
            try:
                length_bytes = struct.unpack('<I', f.read(4))[0]
            except struct.error:
                break
            tag_code = f.read(2)
            tag_code = int.from_bytes(tag_code, byteorder="big")
            if tag_code not in wordBag:
                # 重新设置读指针，跳到下一个模式串
                f.seek(-6 + length_bytes, 1)
                # 读指针直到下一个模式串，继续处理下一个
                continue
            else:
                count = wordCount[tag_code]
                wordCount[tag_code] += 1
            # H表示的unsignedshort
            width = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]

            # 由于Bitmap是按行存储的，所以使用一个循环把他读出来，然后再放到图片里面。
            im = Image.new('L', (width, height))
            img_array = im.load()
            for x in range(0, height):
                for y in range(0, width):
                    pixel = struct.unpack('<B', f.read(1))[0]
                    img_array[y, x] = pixel

            filename = str(count) + '.png'
            tag_code = str(tag_code)
            filename = path + "\\image\\" + tag_code + '\\' + filename
            if os.path.exists(path + "\\image\\" + tag_code):
                im = im.resize((32, 32))
                im.save(filename)
            else:
                os.makedirs(path + "\\image\\" + tag_code)
                im = im.resize((32, 32))
                im.save(filename)
        f.close()
        print("writer {} has processed".format(z))


if __name__ == "__main__":
    createImage()
