__author__ = 'MiaFeng'
import os, sys
import imageio
from PIL import Image

def makeGif(figDir,figName,method=1):
    __imageio = 1
    __PIL = 2

    images = []

    filenames = sorted(fn for fn in os.listdir(figDir) if fn.endswith('.png'))
    if method==__imageio:
        for filename in filenames:
            images.append(imageio.imread(figDir +'/'+ filename))
            print('start')
            imageio.mimsave(figDir+'/'+figName, images, duration=0.5, loop=1)  # duration 每帧间隔时间，loop 循环次数
            print('Done')

        # elif method ==__PIL:  # PIL 方法
        #     imN = 1
        #     for filename in filenames:
        #         if imN == 1:  # 执行一次 im的open操作，PIL在保存gif之前，必须先打开一个生成的帧，默认第一个frame的大小、调色
        #             im = Image.open(figDir +'/'+ filename)
        #             imN = 2
        #
        #         images.append(Image.open(figDir +'/'+ filename))
        #     print('start')
        #     im.save(figDir+'/'+figName, save_all=True, append_images=images, loop=1, duration=500,
        #             comment=b"this is my weather.gif")
        #     print('Done')