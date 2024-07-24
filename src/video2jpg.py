import subprocess
import imageio_ffmpeg
import os
import glob
from tqdm import tqdm


def convert_video_to_images(input_video, output_directory, output_format='jpg', frame_rate=1, names='', nums=None):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_path = os.path.join(output_directory, names + f'{nums}_' + f'%06d.{output_format}')

    ffmpeg_command = [
        'ffmpeg',
        '-i', input_video,
        '-vf', f'fps={frame_rate}',
        output_path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        print(f'视频转换成功，图片保存在 {output_directory}')
    except subprocess.CalledProcessError as e:
        print(f'视频转换失败: {e}')


# ffmpegPath = r"E:\soft\ffmpeg\bin\ffmpeg"
src_video = r"E:\downloads\compress\datasets\天气\video\thunder"
save_dir = r"E:\downloads\compress\datasets\天气\video\thunder\images"
videoFormat = '.mp4'
# out_path = r"E:\downloads\program\video_download\hand_hold_camera\logo\huawei mobile"
# pic = out_path
cmd = ' -r 0.5 -qscale:v 2 '
imgFormat = "jpg"

frame_rate = 1
video_list = glob.glob(src_video + '/*.mp4')

pbar = tqdm(video_list, desc=f"Converting {src_video}")

time = 'thunder20240614'

i = 0
for p in pbar:

    # p = "/data/video/fixed_camera/no_logo/mixkit_firefighters_putting_out_a_big_fire_5280.mp4"
    video = os.path.basename(p)

    video_name = os.path.splitext(video)[0]

    save_name = os.path.join(save_dir, video_name)
    if not os.path.exists(save_name):  # 如果图片文件夹不存在
        os.makedirs(save_name)

    convert_video_to_images(p, save_name, frame_rate=frame_rate, names=time,  nums=i)
    i += 1

    # command1 = "{} -i {} {} {}\%06d.{} -y".format(imageio_ffmpeg.get_ffmpeg_exe(), p, cmd, save_dir, imgFormat)
    # command1 = "{} -i {} {} {}/%06d.{}".format('ffmpeg', p, cmd, save_dir, imgFormat)
    # # command = imageio_ffmpeg.get_ffmpeg_exe() + ' -i ' + '"' + p + '"' + ' ' + cmd + ' '+ '"' + save_dir + '/' + video_name + '_%06d.' + imgFormat + '"'
    # f = subprocess.Popen(command1)


