from enum import IntEnum

import depthai as dai
import cv2
import contextlib
import math
import time
from pathlib import Path
import signal
import threading
from waiting import wait

# DepthAI Record library
from depthai_sdk import Record, EncodingQuality
from depthai_sdk.managers.arg_manager import ArgsManager
import argparse

cam = []
_save_choices = ("color", "left", "right", "disparity", "depth", "pointcloud")  # TODO: IMU/ToF...
_quality_choices = tuple(str(q).split('.')[1] for q in EncodingQuality)
trig = False  # use to activate trig function
device_infos = dai.Device.getAllAvailableDevices()


def trigger():
    pipeline = dai.Pipeline()
    count = 0

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setIspScale(2, 3)
    camRgb.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
    camRgb.initialControl.setExternalTrigger(15, 5)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("color")
    camRgb.isp.link(xoutRgb.input)

    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoLeft.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
    monoLeft.initialControl.setExternalTrigger(15, 5)

    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutLeft.setStreamName("left")
    monoLeft.out.link(xoutLeft.input)

    monoRight = pipeline.createMonoCamera()
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoRight.initialControl.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
    monoRight.initialControl.setExternalTrigger(15, 5)

    xoutRight = pipeline.create(dai.node.XLinkOut)
    xoutRight.setStreamName("right")
    monoRight.out.link(xoutRight.input)

    # Connect to device with pipeline
    # print(pipeline)
    count = (count + 1)
    print(count)

    if len(device_infos) == 0:
        raise RuntimeError("No devices found!")
    else:
        print("Found", len(device_infos), "devices")
        print("     Wait for connection...")

    with dai.Device(pipeline) as device:
        arr = ['color']
        queues = {}
        frames = {}

        for name in arr:
            queues[name] = device.getOutputQueue(name)

        print("Conneceted...")
        print(queues)

        while True:
            wait(lambda: queues[name].has() == True)
            print("IMU recording begun...")
            break
        return (True)


def checkQuality(value: str):
    if value.upper() in _quality_choices:
        return value
    elif value.isdigit():
        num = int(value)
        if 0 <= num <= 100:
            return num
    raise argparse.ArgumentTypeError(
        f"{value} is not a valid quality. Either use number 0-100 or {'/'.join(_quality_choices)}.")


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-p', '--path', default="recordings", type=str, help="Path where to store the captured data")
parser.add_argument('-save', '--save', default=["color", "left", "right"], nargs="+", choices=_save_choices,
                    help="Choose which streams to save. Default: %(default)s")
# parser.add_argument('-f', '--fps', type=float, default=30,
# help='Number of frames to record per second')
parser.add_argument('-q', '--quality', default="MEDIUM", type=checkQuality,
                    help='Selects the quality of the recording. Default: %(default)s')
parser.add_argument('-fc', '--frame_cnt', type=int, default=-1,
                    help='Number of frames to record. Record until stopped by default.')
parser.add_argument('-tl', '--timelapse', type=int, default=-1,
                    help='Number of seconds between frames for timelapse recording. Default: timelapse disabled')
parser.add_argument('-mcap', '--mcap', action="store_true", help="MCAP file format")
args = ArgsManager.parseArgs(parser)

# TODO: make camera resolutions configrable
if 'color' in args.save and 1 < len(args.save) and args.monoFps != args.rgbFps:
    raise argparse.ArgumentTypeError('Recording app requires Mono and Color camera FPS to be the same!')
args.fps = args.rgbFps if 'color' in args.save else args.monoFps

save_path = Path.cwd() / args.path


# Host side sequence number syncing
def checkSync(queues, sequenceNum: int):
    matching_frames = []
    for q in queues:
        for i, msg in enumerate(q['msgs']):
            if msg.getSequenceNum() == sequenceNum:
                matching_frames.append(i)
                break

    if len(matching_frames) == len(queues):
        # We have all frames synced. Remove the excess ones
        for i, q in enumerate(queues):
            q['msgs'] = q['msgs'][matching_frames[i]:]
        return True
    else:
        return False


def run():
    global trig
    while trig == False:
        print("running false")
        trig = trigger()
        # print ("while false: "+ str(trigger()))

    while trig == True:
        print("running true")
        start_time = time.time()
        with contextlib.ExitStack() as stack:
            # Record from all available devices

            devices = []
            print(devices)
            # TODO: allow users to specify which available devices should record
            for device_info in device_infos:
                openvino_version = dai.OpenVINO.Version.VERSION_2021_4
                device = stack.enter_context(dai.Device(openvino_version, device_info, usb2Mode=False))
                if device.getUsbSpeed() == dai.UsbSpeed.HIGH and args.quality == "LOW":
                    print(
                        "USB2 speeds detected! Recorded video stream(s) might be look 'glitchy'. To avoid this, don't use LOW quality.")
                # Create recording object for this device
                recording = Record(save_path, device, args)
                # Set recording configuration
                recording.setTimelapse(args.timelapse)
                recording.setRecordStreams(args.save)
                recording.setQuality(args.quality)
                recording.setMcap(args.mcap)

                devices.append(recording)

            for recording in devices:
                recording.start()  # Start recording
                cam.append(time.time())

            timelapse = 0

            def roundUp(value, divisibleBy: float):
                return int(divisibleBy * math.ceil(value / divisibleBy))

            # If H265, we want to start recording with the keyframe (default keyframe freq is 30 frames)
            SKIP_FRAMES = roundUp(1.5 * args.fps, 30 if args.quality == "LOW" else 1)
            args.frame_cnt += SKIP_FRAMES

            # Terminate app handler
            quitEvent = threading.Event()
            signal.signal(signal.SIGTERM, lambda *_args: quitEvent.set())
            end_time = time.time()
            print("\nRecording started. Press 'Ctrl+C' to stop.")
            elapsed_time = end_time - start_time
            print("Elapsed time: ", elapsed_time)

            while not quitEvent.is_set():
                try:
                    allQueues = []
                    for recording in devices:
                        allQueues.append(recording.queues)
                        # checkSync(recording.queues, new_msg.getSequenceNum())
                        print(allQueues)
                        #print(recording.queues)
                        print(time.time())
                        if 0 < args.timelapse and time.time() - timelapse < args.timelapse:
                            continue
                        allQueues.clear()
                        print("after clear", allQueues)
                        # Loop through device streams
                        for q in recording.queues:
                            new_msg = q['q'].tryGet()
                            if new_msg is not None:
                                q['msgs'].append(new_msg)
                                if checkSync(recording.queues, new_msg.getSequenceNum()):
                                    # Wait for Auto focus/exposure/white-balance
                                    recording.frameCntr += 1
                                    if recording.frameCntr <= SKIP_FRAMES:  # 1.5 sec
                                        continue
                                    # Timelapse
                                    if 0 < args.timelapse: timelapse = time.time()
                                    if args.frame_cnt == recording.frameCntr:
                                        quitEvent.set()

                                    frames = dict()
                                    for stream in recording.queues:
                                        frames[stream['name']] = stream['msgs'].pop(0)
                                    recording.frame_q.put(frames)

                    time.sleep(0.001)  # 1ms, avoid lazy looping
                except KeyboardInterrupt:
                    break

            print('')  # For new line in terminal
            for recording in devices:
                recording.frame_q.put(None)
                recording.process.join()  # Terminate the process
            print("All recordings have stopped successfuly. Exiting the app.")
            print(cam)
            print(start_time)
            break
        break


if __name__ == '__main__':
    run()
