#!/usr/bin/env python3

import sys
sys.path.append('../')
import os.path
from os import path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
import sys
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.utils import long_to_int
from common.FPS import GETFPS
import pyds
import requests
from datetime import datetime
import cv2 
import numpy as np
import yaml
import argparse
import time

failed_to_post_max_retries = 2
failed_to_post_image_max_retries = 2
failed_to_post_reached = False
failed_to_post_image_reached = False
fps_streams={}
MAX_DISPLAY_LEN=64
MAX_TIME_STAMP_LEN=32


# input_file = None
cfg = None
no_display = False
current_time = None


# getFps = GETFPS("/dev/video0")

pgie_classes_str=["whiteRect"]

def draw_bounding_boxes(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[obj_meta.class_id]
    image = cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255, 0), 2, cv2.LINE_4)
    color = (0, 0, 255, 0)
    w_percents = int(width * 0.05) if width > 100 else int(width * 0.1)
    h_percents = int(height * 0.05) if height > 100 else int(height * 0.1)
    linetop_c1 = (left + w_percents, top)
    linetop_c2 = (left + width - w_percents, top)
    image = cv2.line(image, linetop_c1, linetop_c2, color, 6)
    linebot_c1 = (left + w_percents, top + height)
    linebot_c2 = (left + width - w_percents, top + height)
    image = cv2.line(image, linebot_c1, linebot_c2, color, 6)
    lineleft_c1 = (left, top + h_percents)
    lineleft_c2 = (left, top + height - h_percents)
    image = cv2.line(image, lineleft_c1, lineleft_c2, color, 6)
    lineright_c1 = (left + width, top + h_percents)
    lineright_c2 = (left + width, top + height - h_percents)
    image = cv2.line(image, lineright_c1, lineright_c2, color, 6)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image = cv2.putText(image, obj_name + ',C=' + str(confidence), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255, 0), 2)
    return image

# osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.
# IMPORTANT NOTE:
# a) probe() callbacks are synchronous and thus holds the buffer
#    (info.get_buffer()) from traversing the pipeline until user return.
# b) loops inside probe() callback could be costly in python.
#    So users shall optimize according to their use-case.
def osd_sink_pad_buffer_probe(pad,info,u_data):
    global failed_to_post_max_retries
    global failed_to_post_image_max_retries
    global failed_to_post_reached
    global failed_to_post_image_reached
    global cfg, current_time
    # frame_number=0
    # source = cfg['source']
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

    obj_id = 0
    #Intiallizing object counter with 0.
    obj_counter = {
        "whiteRect": 0
    }
    box_data = []

    data_to_send = {"timeStamp":dt_string ,"data": box_data}
    is_first_object=True
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return


    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    # latency_meta = pyds.measure_buffer_latency(hash(gst_buffer))
    # print(latency_meta)
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            continue
        is_first_object = True

        '''
        print("Frame Number is ", frame_meta.frame_num)
        print("Source id is ", frame_meta.source_id)
        print("Batch id is ", frame_meta.batch_id)
        print("Source Frame Width ", frame_meta.source_frame_width)
        print("Source Frame Height ", frame_meta.source_frame_height)
        print("Num object meta ", frame_meta.num_obj_meta)
        '''
        frame_number=frame_meta.frame_num
        sensor_ID = frame_meta.source_id
        # print(frame_meta.misc_frame_info)
        obj_counter["frame"] = frame_number
        obj_counter["sensor_ID"]= sensor_ID

        data_to_send["frame"] = frame_number
        data_to_send["sensor_ID"]= sensor_ID
        # print("sensor_ID", sensor_ID, "frame", frame_number)
        l_obj=frame_meta.obj_meta_list
        save_image = False
        while l_obj is not None:
            try:
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                continue
            # lat=pyds.NvDsMeta.NvDsFrameLatencyInfo
            # latencyF=lat.latency
            # print(latencyF)
            # Update the object text display
            
            txt_params=obj_meta.text_params

            # Set display_text. Any existing display_text string will be
            # freed by the bindings module.
            txt_params.display_text = pgie_classes_str[obj_meta.class_id]

            obj_counter[pgie_classes_str[obj_meta.class_id]] += 1
            # Font , font-color and font-size
            txt_params.font_params.font_name = "Serif"
            txt_params.font_params.font_size = 10
            # set(red, green, blue, alpha); set to White
            txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            # Text background color
            txt_params.set_bg_clr = 1
            # set(red, green, blue, alpha); set to Black
            txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

            # Ideally NVDS_EVENT_MSG_META should be attached to buffer by the
            # component implementing detection / recognition logic.
            # Here it demonstrates how to use / attach that meta data.

            if not frame_number % dataSend:
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                n_frame = draw_bounding_boxes(n_frame, obj_meta, obj_meta.confidence)
                # convert python array into numpy array format in the copy mode.
                frame_copy = np.array(n_frame, copy=True, order='C')
                # convert the array into cv2 default color format
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                save_image = True   

            bbox = { "top": obj_meta.rect_params.top, "left":  obj_meta.rect_params.left, "width":  obj_meta.rect_params.width, "height":obj_meta.rect_params.height}
            obj = { 'label':pgie_classes_str[obj_meta.class_id],
                    'bbox':bbox,
                    'confidence':'{0:.2f}'.format(obj_meta.confidence) 
                    }
            box_data.append(obj)
            # print(obj) 

            # if(is_first_object and not (frame_number%30)):
            #     # Frequency of messages to be send will be based on use case.
            #     # Here message is being sent for first object every 30 frames.
            try:
                l_obj=l_obj.next
            except StopIteration:
                break
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
        # if not frame_number % 4:
            # print(data_to_send)
        if not save_image and not frame_number % dataSend:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            # convert python array into numpy array format in the copy mode.
            frame_copy = np.array(n_frame, copy=True, order='C')
            # convert the array into cv2 default color format
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
            save_image = True    

        if save_image:
            img_name = "{}/stream_{}_time_{}.jpg".format(folder_name, frame_meta.pad_index, dt_string)
            cv2.imwrite(img_name, frame_copy)
            # print(img_path)

    if failed_to_post_max_retries > 0 :
        try:
            if not frame_number % dataSend:
                # print("!!!!!!!")
                requests.post(conn_http, json = data_to_send, timeout=1)
                # print("sent data")
        except:
            print(f"failed to post {failed_to_post_max_retries} to {conn_http}")
            failed_to_post_max_retries -= 1
    else:
        if not failed_to_post_reached:
            print("max retries reached, not sending")
            failed_to_post_reached = True

    if failed_to_post_image_max_retries > 0 :
        try:
            if not frame_number % picSend:
                files = {'media': open(img_name, 'rb')}
                requests.post(conn_http, files=files, timeout=1)
        except:
            print(f"failed to post image {failed_to_post_image_max_retries} to {conn_http}")
            failed_to_post_image_max_retries -= 1
    else:
        if not failed_to_post_image_reached:
            print("max retries reached, not picture sending")
            failed_to_post_image_reached = True
            
    for filename in os.listdir(image_dir_path):
        file_path = os.path.join(image_dir_path, filename)
        filestamp = os.stat(file_path).st_mtime
        filecompare = time.time() - (picDeleteDays * 86400 + picDeleteSec)
        if filestamp < filecompare:
            # print(f"removing {file_path}")
            os.remove(file_path)
    # getFps.get_fps()
    # print(obj_counter["frame"])
    # print(obj_counter)                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    return Gst.PadProbeReturn.OK


def make_camera_source(camera):
    device = camera["device"]
    extraControls = camera.get("extra-controls", None)
    target_fps = camera.get("target_fps", None)
    pipe = f"v4l2src device={device} "
    if extraControls != None:
        pipe += f"extra-controls=\"{extraControls}\" " 
    caps = camera.get("caps", None)
    if caps:
        pipe += f" ! capsfilter caps=\"{caps}\" "
    if not caps or caps.startswith("image/jpeg"):
        pipe += " ! jpegdec ! "
    else: 
        pipe += " ! "
    if target_fps != None:
        pipe += f"videorate ! video/x-raw, framerate={target_fps} ! " 
    pipe += "nvvideoconvert ! capsfilter caps=\"video/x-raw(memory:NVMM)\""
    print(pipe)
    source = Gst.parse_bin_from_description(pipe, True)
    return source

def link_to_tee(tee, dest):
    sink = dest.get_static_pad("sink")
    src = tee.get_request_pad('src_%u')
    src.link(sink)
    return src

def make_display_bin():
    return Gst.parse_bin_from_description(f"queue ! nvegltransform ! nveglglessink sync=false", True)

def make_empty_bin():
    return Gst.parse_bin_from_description(f"queue ! fakesink sync=false", True)

def main(args):

    global current_time

    MUXER_OUTPUT_WIDTH = width
    MUXER_OUTPUT_HEIGHT = height
    MUXER_BATCH_TIMEOUT_USEC=4000
    PGIE_CONFIG_FILE = config_file_path

    GObject.threads_init()
    Gst.init(None)
    global folder_name
    folder_name = image_dir_path
    checkFolder = os.path.isdir(folder_name)                        
    if not checkFolder:
        try:                                                                                                                                                                                                                                                                                                                                    
            os.makedirs(folder_name)
            print("created Folder:", folder_name)
        except:
            print(folder_name + " already exists.")

    print("Frames will be saved in ", folder_name)
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)
    
    print("Creating Pipeline \n ")

    pipeline = Gst.Pipeline()                                              
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)

    nvmultistreamtiler = Gst.ElementFactory.make("nvmultistreamtiler", None)
    if not nvmultistreamtiler:
        sys.stderr.write(" Unable to create nvmultistreamtiler ")

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", None)
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    tee=Gst.ElementFactory.make("tee", "nvsink-tee")
    if not tee:
        sys.stderr.write(" Unable to create tee \n")

    sink = make_empty_bin() if no_display else make_display_bin()
    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    nvmultistreamtiler.set_property('columns', 2)
    nvmultistreamtiler.set_property('rows', 2)
    nvmultistreamtiler.set_property('width', MUXER_OUTPUT_WIDTH)
    nvmultistreamtiler.set_property('height', MUXER_OUTPUT_HEIGHT)
    pgie.set_property('config-file-path', PGIE_CONFIG_FILE)

    print("Adding elements to Pipeline \n")
    pipeline.add(streammux)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvmultistreamtiler)
    pipeline.add(pgie)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(tee)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(filter1)
    filter1.link(nvmultistreamtiler)
    nvmultistreamtiler.link(nvvidconv1)
    nvvidconv1.link(nvosd)
    nvosd.link(tee)

    source_index = 0
    def newDevice(camera_source):
        nonlocal source_index
        device = camera_source.get("device")
        camera = { 
            "device": "/dev/video"+device, 
            "extra-controls": "c,backlight_compensation=2", 
            "target_fps": target_fps,
            "caps": camera_source.get("caps", None)
        }
        source = make_camera_source(camera)
        pipeline.add(source)
        source_src_pad = source.get_static_pad("src")
        if not source_src_pad:
            sys.stderr.write(f"Unable to get source pad of source {source_index} \n")
        streammux_sink = streammux.get_request_pad(f"sink_{source_index}")
        if not streammux_sink:
            sys.stderr.write(" Unable to get sink pad of nvstreammux \n")
        source_src_pad.link(streammux_sink)
        source_index += 1
    if camera_source and camera_source.get("enable", True):
        newDevice(camera_source)
    if camera_source1 and camera_source1.get("enable", True):
        newDevice(camera_source1)
    if camera_source2 and camera_source2.get("enable", True):
        newDevice(camera_source2)
    if camera_source3 and camera_source3.get("enable", True):
        newDevice(camera_source3)

    link_to_tee(tee, sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    tilerpad = nvmultistreamtiler.get_static_pad("sink")
    if not tilerpad:
        sys.stderr.write(" Unable to get src pad of nvmultistreamtiler \n")


    tilerpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    print("Starting pipeline \n")

    current_time = time.time()
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pyds.unset_callback_funcs()
    pipeline.set_state(Gst.State.NULL)

# Parse and validate input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Tag Tracker')
    parser.add_argument('-c', '--config-path', default=None, type=str, help="Path to a configuration .yaml file")
    args = parser.parse_args()
    assert args.config_path is not None, "Invalid config_path"
    with open(args.config_path, 'r') as cfg_f:
        cfg = yaml.load(cfg_f, Loader=yaml.FullLoader)
    global conn_http
    global camera_source
    global camera_source1
    global camera_source2
    global camera_source3
    global width
    global height
    global target_fps
    global image_dir_path
    global config_file_path
    global dataSend
    global picDeleteDays
    global picDeleteSec
    global picSend
    picSend = cfg["picSend"]
    camera_source = cfg.get("camera_source")
    camera_source1 = cfg.get("camera_source1")
    camera_source2 = cfg.get("camera_source2")
    camera_source3 = cfg.get("camera_source3")
    picDeleteSec = cfg["picDeleteSec"]
    picDeleteDays = cfg["picDeleteDays"]
    dataSend = cfg["dataSend"]
    config_file_path = cfg["config_file_path"]
    conn_http = cfg["conn_http"]
    width = cfg["width"]
    height = cfg["height"]
    target_fps = cfg.get("target_fps", None)
    image_dir_path = cfg["image_dir_path"]

    return 0

if __name__ == '__main__':
    ret = parse_args()
    #If argumer parsing fail, return failure (non-zero)
    if ret == 1:
        sys.exit(1)
    sys.exit(main(sys.argv))

