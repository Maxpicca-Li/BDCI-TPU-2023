import cv2
from PIL import Image
from npuengine import EngineOV
from fix import *
import os
import glob
import argparse
import math
import json
import time
import warnings
from metrics.niqe import calculate_niqe


class UpscaleModel:

    def __init__(self, model=None, model_size=(200, 200), upscale_rate=4, tile_size=(196, 196), padding=4, device_id=0):
        self.tile_size = tile_size
        self.padding = padding
        self.upscale_rate = upscale_rate
        if model is None:
            print("use default upscaler model")
            model = "./models/other/resrgan4x.bmodel"
        # 导入bmodel
        self.model = EngineOV(model, device_id=device_id)
        self.model_size = model_size

    def calc_tile_position(self, width, height, col, row):
        # generate mask
        # (left, top) = A(x,y)
        # (right, bottom) = D(x, y)
        # tile 之间可能有重合的
        tile_left = col * self.tile_size[0]
        tile_top = row * self.tile_size[1]
        tile_right = (col + 1) * self.tile_size[0] + self.padding
        tile_bottom = (row + 1) * self.tile_size[1] + self.padding
        if tile_right > height:
            tile_right = height
            tile_left = height - self.tile_size[0] - self.padding * 1
        if tile_bottom > width:
            tile_bottom = width
            tile_top = width - self.tile_size[1] - self.padding * 1
        print("calc_tile_position", tile_top, tile_left, tile_bottom, tile_right, self.tile_size, self.padding)
        return tile_top, tile_left, tile_bottom, tile_right

    def calc_upscale_tile_position(self, tile_left, tile_top, tile_right, tile_bottom):
        return int(tile_left * self.upscale_rate), int(tile_top * self.upscale_rate), int(
            tile_right * self.upscale_rate), int(tile_bottom * self.upscale_rate)

    def modelprocess(self, images):
        for i in range(len(images)):
            images[i] = images[i].resize(self.model_size)
            images[i] = np.array(images[i]).astype(np.float32)
        
        ntile = np.array(images)
        ntile = ntile / 255
        ntile = np.transpose(ntile, (0, 3, 1, 2))

        resAll = self.model([ntile])[0]
        res_list = []
        for i in range(resAll.shape[0]):
            res = resAll[i]
            res = np.transpose(res, (1, 2, 0))
            res = res * 255
            res[res > 255] = 255
            res[res < 0] = 0
            res = res.astype(np.uint8)
            res_list.append(res)
        return res_list

    def extract_and_enhance_tiles(self, images, upscale_ratio=4.0):
        for i in range(len(images)):
            if images[i].mode != "RGB":
                images[i] = images[i].convert("RGB")        
        self.upscale_rate = upscale_ratio
        res_list = self.modelprocess(images.copy())

        res_imgs = []
        for i in range(len(res_list)):
            res = Image.fromarray(res_list[i])
            width, height = images[i].size
            target_tile_size = (int(width * upscale_ratio), int(height * upscale_ratio))
            res = res.resize(target_tile_size)
            # res = np.array(res).astype(np.float32)
            # res = Image.fromarray(res.astype(np.uint8))
            res_imgs.append(res)
        return res_imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, default="./models/resrgan4x.bmodel",
                        help="Model names")
    parser.add_argument("-i", "--input", type=str, default="./dataset/test",
                        help="Input image or folder")
    parser.add_argument("-o", "--output", type=str, default="./results/test_fix",
                        help="Output image folder")
    parser.add_argument("-r", "--report", type=str, default="./results/test.json",
                             help="report model runtime to json file")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, "*")))

    # parameters
    model_size = (400, 400)
    tile_size = model_size
    padding = 0

    # set models
    model = args.model_path
    upmodel = UpscaleModel(model=model, model_size=model_size, upscale_rate=4, tile_size=tile_size, padding=padding)

    start_all = time.time()
    batch = 4
    images = []
    img_names = []
    result, runtime, niqe = [], [], []
    
    for idx, path in enumerate(paths):
        img_name, extension = os.path.splitext(os.path.basename(path))
        img = Image.open(path)
        images.append(img)
        img_names.append(img_name)
        
        if (idx+1) % batch == 0 or idx==len(paths)-1:
            real_len = len(images)
            if idx==len(paths)-1 and ((idx+1) % batch) != 0:
                tmp = Image.new("RGB", model_size, color=(0, 0, 0))  # 创建一个白色背景的图像
                now_len = len(images)
                images.extend([tmp]*(batch-now_len))
            
            start = time.time()
            # 推理优化点：时间也只是算这里的时间
            res = upmodel.extract_and_enhance_tiles(images, upscale_ratio=4.0)
            end = time.time() - start
            singleTime = end/batch
            runtime.extend([singleTime]*batch)
            
            for k in range(0,real_len):
                output_path = os.path.join(args.output, img_names[k] + extension)
                res[k].save(output_path)

                # 计算niqe
                output = cv2.imread(output_path)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    niqe_output = calculate_niqe(output, 0, input_order='HWC', convert_to='y')
                niqe_output = format(niqe_output, '.4f')
                niqe.append(niqe_output)

                result.append({"img_name": img_names[k], "runtime": format(singleTime,".4f"), "niqe": niqe_output})
            images = []
            img_names = []

    model_size = os.path.getsize(model)
    runtime_avg = np.mean(np.array(runtime, dtype=float))
    niqe_avg = np.mean(np.array(niqe, dtype=float))

    end_all = time.time()
    time_all = end_all - start_all
    #print('time_all:', time_all)
    params = {"A": [{"model_size": model_size, "time_all": time_all, "runtime_avg": format(runtime_avg, '.4f'),
                     "niqe_avg": format(niqe_avg, '.4f'), "images": result}]}
    #print("params: ", params)

    output_fn = f'{args.report}'
    with open(output_fn, 'w') as f:
        json.dump(params, f, indent=4)

if __name__ == "__main__":
    main()