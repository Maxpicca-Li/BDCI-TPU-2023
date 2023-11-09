import cv2
from PIL import Image
from fix import *
import os
import glob
import argparse
import math
import json
import time
import warnings
import signal
import sys
import sophon.sail as sail
from metrics.niqe import calculate_niqe
from concurrent.futures import ThreadPoolExecutor

device_id = 0

class UpscaleModel:

    def __init__(self, model=None, model_size=(200, 200), upscale_rate=4, tile_size=(196, 196), padding=4, device_id=0):
        self.tile_size = tile_size
        self.padding = padding
        self.upscale_rate = upscale_rate
        if model is None:
            print("use default upscaler model")
            model = "./models/other/resrgan4x.bmodel"
        self.engine = sail.Engine(device_id)
        self.engine.load(model)
        self.image = None
        self.model_size = model_size

    def calc_tile_position(self, width, height, col, row):
        # generate mask
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

        return tile_top, tile_left, tile_bottom, tile_right

    def calc_upscale_tile_position(self, tile_left, tile_top, tile_right, tile_bottom):
        return int(tile_left * self.upscale_rate), int(tile_top * self.upscale_rate), int(
            tile_right * self.upscale_rate), int(tile_bottom * self.upscale_rate)

    # NOTE lyq: 这个时候已经是 tile 了，一个线程是合理的
    # def modelprocess(self, tile):
    #     ntile = tile.resize(self.model_size)
    #     # preprocess
    #     ntile = np.array(ntile).astype(np.float32)
    #     ntile = ntile / 255
    #     ntile = np.transpose(ntile, (2, 0, 1))
    #     ntile = ntile[np.newaxis, :, :, :]

    #     res = self.model([ntile])[0]
    #     # extract padding
    #     res = res[0]
    #     res = np.transpose(res, (1, 2, 0))
    #     res = res * 255
    #     res[res > 255] = 255
    #     res[res < 0] = 0
    #     res = res.astype(np.uint8)
    #     res = Image.fromarray(res)
    #     res = res.resize(self.target_tile_size)
    #     return res
    
    def thread_infer(self, engine, tile):
        # get model info
        # only one model loaded for this engine
        # only one input tensor and only one output tensor in this graph
        graph_name = engine.get_graph_names()[0]
        input_name = engine.get_input_names(graph_name)[0]
        input_shape = engine.get_input_shape(graph_name, input_name)
        output_name = engine.get_output_names(graph_name)[0]
        output_shape = engine.get_output_shape(graph_name, output_name)
        in_dtype = engine.get_input_dtype(graph_name, input_name)
        out_dtype = engine.get_output_dtype(graph_name, output_name)
        # get handle to create input and output tensors
        handle = engine.get_handle()
        input = sail.Tensor(handle, input_shape, in_dtype, True, True)
        output = sail.Tensor(handle, output_shape, out_dtype, True, True)
        input_tensors = {input_name:input}
        output_tensors = {output_name:output}
        # set io_mode
        engine.set_io_mode(graph_name, sail.SYSIO)
        # FIXME lyq: comment it
        sail.set_print_flag(True)

        
        # 使用超分辨率模型放大瓦片
        ntile = tile.resize(self.model_size)
        # preprocess
        ntile = np.array(ntile).astype(np.float32)
        ntile = ntile / 255
        ntile = np.transpose(ntile, (2, 0, 1))
        ntile = ntile[np.newaxis, :, :, :]

        # res = self.model([ntile])[0] # old train
        input_tensors[input_name].update_data(ntile)
        engine.process(graph_name, input_tensors, output_tensors)
        res = output.asnumpy()
        # extract padding
        res = res[0]
        res = np.transpose(res, (1, 2, 0))
        res = res * 255
        res[res > 255] = 255
        res[res < 0] = 0
        res = res.astype(np.uint8)
        res = Image.fromarray(res)
        res = res.resize(self.target_tile_size)

        # 将放大后的瓦片粘贴到输出图像上
        # overlap
        ntile = np.array(res).astype(np.float32)
        ntile = np.transpose(ntile, (2, 0, 1))
        return ntile

    def extract_and_enhance_tiles(self, image, upscale_ratio=2.0):
        if image.mode != "RGB":
            image = image.convert("RGB")
        # 获取图像的宽度和高度
        width, height = image.size
        self.image = image
        self.upscale_rate = upscale_ratio
        self.target_tile_size = (int((self.tile_size[0] + self.padding * 1) * upscale_ratio),
                                 int((self.tile_size[1] + self.padding * 1) * upscale_ratio))
        target_width, target_height = int(width * upscale_ratio), int(height * upscale_ratio)
        # 计算瓦片的列数和行数
        num_cols = math.ceil((width - self.padding) / self.tile_size[0])
        num_rows = math.ceil((height - self.padding) / self.tile_size[1])

        # 遍历每个瓦片的行和列索引
        # TODO lyq: 如果推理优化也算的话，这里也可以改
        # 如果要模型进行多线程推理，那这里应该就是最好的切入点，因为modelprocess内的数据单位太小了，不太好又分
        engine_list = [self.engine] * (num_rows*num_cols)
        tile_list = []
        for row in range(num_rows):
            for col in range(num_cols):
                tile_top, tile_left, tile_bottom, tile_right = self.calc_tile_position(width, height, col, row)
                tile = image.crop((tile_left, tile_top, tile_right, tile_bottom))
                tile_list.append(tile)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = pool.map(self.thread_infer, engine_list, tile_list)
        img_tiles = []
        img_h_tiles = []
        for i, tile in enumerate(results):
            img_h_tiles.append(tile)
            if (i+1)%num_cols==0:
                img_tiles.append(img_h_tiles)
                img_h_tiles = []
        # 二维的 2*3=6
        res = imgFusion(img_list=img_tiles, overlap=int(self.padding * upscale_ratio), res_w=target_width,
                        res_h=target_height)
        res = Image.fromarray(np.transpose(res, (1, 2, 0)).astype(np.uint8))
        return res

def signal_handle():
  print("INT*****")
  sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handle)

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

    # set models 利用model_path设置模型 -> 目前是单线程逻辑
    # TODO lyq: 多线程怎么加的问题
    model = args.model_path
    upmodel = UpscaleModel(model=model, model_size=(200, 200), upscale_rate=4, tile_size=(196, 196), padding=20)

    # 模型推理开始
    start_all = time.time()
    result, runtime, niqe = [], [], []
    for idx, path in enumerate(paths):
        img_name, extension = os.path.splitext(os.path.basename(path))
        img = Image.open(path)
        print("Testing", idx, img_name)

        start = time.time()
        # 模型执行
        res = upmodel.extract_and_enhance_tiles(img, upscale_ratio=4.0)
        end = format((time.time() - start), '.4f')
        runtime.append(end)

        output_path = os.path.join(args.output, img_name + extension)
        res.save(output_path)

        # 计算niqe
        output = cv2.imread(output_path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_output = calculate_niqe(output, 0, input_order='HWC', convert_to='y')
        niqe_output = format(niqe_output, '.4f')
        niqe.append(niqe_output)

        result.append({"img_name": img_name, "runtime": end, "niqe": niqe_output})

    model_size = os.path.getsize(model)
    runtime_avg = np.mean(np.array(runtime, dtype=float))
    niqe_avg = np.mean(np.array(niqe, dtype=float))

    end_all = time.time()
    time_all = end_all - start_all
    print('time_all:', time_all)
    params = {"A": [{"model_size": model_size, "time_all": time_all, "runtime_avg": format(runtime_avg, '.4f'),
                     "niqe_avg": format(niqe_avg, '.4f'), "images": result}]}
    print("params: ", params)

    output_fn = f'{args.report}'
    with open(output_fn, 'w') as f:
        json.dump(params, f, indent=4)

if __name__ == "__main__":
    main()