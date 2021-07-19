#!/usr/bin/env python3
"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from argparse import ArgumentParser, SUPPRESS
import logging
import sys
import time
import glob
import os
import pyaudio
import cv2

import numpy as np
from openvino.inference_engine import IECore


def type_overlap(arg):
    if arg.endswith('%'):
        res = float(arg[:-1]) / 100
    else:
        res = int(arg)
    return res

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')

    args.add_argument('-m', "--model", type=str, required=True,
                      help="Required. Path to an .xml file with a trained model.")
    args.add_argument("-l", "--cpu_extension", type=str, default=None,
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is"
                           " acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU")
    args.add_argument('--labels', type=str, default=None,
                      help="Optional. Labels mapping file")
    args.add_argument('--illustration_dir', type=str, default=None,
                      help="Optional. Directory of label illusration")

    return parser.parse_args()

def main():
    args = build_argparser()

    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
    log = logging.getLogger()

    log.info("Creating Inference Engine")
    ie = IECore()

    if args.device == "CPU" and args.cpu_extension:
        ie.add_extension(args.cpu_extension, 'CPU')

    log.info("Loading model {}".format(args.model))
    net = ie.read_network(args.model, args.model[:-4] + ".bin")

    if len(net.input_info) != 1:
        log.error("Demo supports only models with 1 input layer")
        sys.exit(1)
    input_blob = next(iter(net.input_info))
    input_shape = net.input_info[input_blob].input_data.shape
    if len(net.outputs) != 1:
        log.error("Demo supports only models with 1 output layer")
        sys.exit(1)
    output_blob = next(iter(net.outputs))

    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    log.info("Preparing input")

    labels = []
    if args.labels:
        with open(args.labels, "r") as file:
            labels = [line.rstrip() for line in file.readlines()]

    illustrations = {}
    if args.illustration_dir:
        image_files = glob.glob(os.path.join(args.illustration_dir, '*.png'))
        for image_file in image_files:
            p, f = os.path.split(image_file)
            base, ext = os.path.splitext(f)
            illustrations[base] = cv2.imread(image_file)

    batch_size, channels, one, length = input_shape
    if one != 1:
        raise RuntimeError("Wrong third dimension size of model input shape - {} (expected 1)".format(one))

    input_size = input_shape[-1]
    sample_rate = 16000
    audio = pyaudio.PyAudio()
    record_stream   = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input =True, frames_per_buffer=input_size)

    log.info("Starting inference")
    infer_time = 0
 
    idx = 0
    chunk = np.zeros(input_shape, dtype=np.float32)
    hop = int(input_size * 0.8)
    overlap = int(input_size - hop)

    log.info('Press ESC key to terminate.')
    key = -1
    while key != 27:
        input_audio = np.frombuffer(record_stream.read(num_frames=hop), dtype=np.int16).astype(np.float32)
        scale = np.std(input_audio)
        scale = 4000 if scale < 4000 else scale     # Limit scale value
        input_audio = (input_audio - np.mean(input_audio)) / scale
        input_audio = np.reshape(input_audio, ( 1, 1, 1, hop))

        chunk[:, :, :, :overlap] = chunk[:, :, :, -overlap:]
        chunk[:, :, :, overlap:] = input_audio
        
        infer_start_time = time.perf_counter()
        output = exec_net.infer(inputs={input_blob: chunk})
        infer_time += time.perf_counter() - infer_start_time
        output = output[output_blob]
        for batch, data in enumerate(output):
            start_time =  (idx*batch_size + batch)   *hop  / sample_rate
            end_time   = ((idx*batch_size + batch +1)*hop) / sample_rate
            label = np.argmax(data)
            if data[label]<0.7:
                label_txt = '????'
                illust_idx = 99
            else:
                label_txt = labels[label]
                illust_idx = label
            log.info("[{:.2f}-{:.2f}] - {:6.2%} {:s}".format(start_time, end_time, data[label], label_txt))
            illustration = illustrations[str(illust_idx)]
            illustration = cv2.resize(illustration, (600, 600))
            cv2.imshow('result', illustration)
        idx += 1
        key = cv2.waitKey(1)

    cv2.destroyAllWindows()
    record_stream.stop_stream()
    record_stream.close()
    audio.terminate()
    log.info('Terminated.')
 
if __name__ == '__main__':
    main()
