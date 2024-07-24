import cv2
import glob

before_path = r'\\10.10.10.8\determined\dengxiongshi\yolov5\runs\detect\test_result_before'
after_path = r'\\10.10.10.8\determined\dengxiongshi\yolov5\runs\detect\test_result'

before_images = glob.glob(before_path + '/*')
after_images = glob.glob(after_path + '/*')
