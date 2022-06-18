import os
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision import transforms
from X3D import create_x3d
from I3D import I3Res50
import time
import numpy as np
import statistics
from config import *

import warnings
warnings.filterwarnings("ignore")


def measure_time(model_name, frames_dir, frequency, batch_size, sample_mode, crop_size):
    chunk_size = CHUNK_SIZE

    def forward_batch(batch_data):
        batch_data = batch_data.transpose([0, 4, 1, 2, 3]) # (16, 3, 16, 256, 340)
        batch_data = torch.from_numpy(batch_data)
        with torch.no_grad():
            batch_data = Variable(batch_data.cuda()).float()
            
            if model_name == "X3D":
                features = model(batch_data)
            else:
                inp = {"frames": batch_data}
                features = model(inp)

        return features.cpu().numpy() # torch.Size([16, 2048, 1, 1, 1])
    
    rgb_files = [i for i in os.listdir(frames_dir)]
    rgb_files.sort()
    frames_cnt = len(rgb_files) # 304
    
    clipped_length = ((frames_cnt - chunk_size) // frequency) * frequency # 288
    frames_indices = []

    for i in range(clipped_length // frequency + 1): # 19
        frames_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])

    if chunk_size != 16:
        final_frames = (clipped_length // frequency + 1) * chunk_size
        num_remain_frames = frames_cnt - final_frames
        
        frames_indices.append([j for j in range(final_frames, frames_cnt)])
        last_frame = frames_cnt - 1
        frames_indices[-1].extend([last_frame for _ in range(chunk_size - num_remain_frames)])

    frames_indices = np.array(frames_indices) # (19, 16)
    chunk_num = frames_indices.shape[0] # 19 (clips) # number of clips
    batch_num = int(np.ceil(chunk_num / batch_size)) # 19 (How many batches)
    frames_indices = np.array_split(frames_indices, batch_num, axis = 0) # (19, 1, 16) (batch_num, batch_size, num frames)
    
    if sample_mode == "OVERSAMPLE":
        full_features = [[] for i in range(10)]
    else:
        full_features = [[]]

    total_time_preprocessing = 0
    total_time_sample = 0
    for batch_id in range(batch_num):
        scale = crop_size/224

        batch_data, time_preprocessing = load_rgb_batch(frames_dir, rgb_files, frames_indices[batch_id], scale)
        total_time_preprocessing += time_preprocessing

        start_time = time.time()

        if sample_mode == "OVERSAMPLE":
            batch_data_ten_crop = oversample_data(batch_data, scale)
            for i in range(10):
                assert (batch_data_ten_crop[i].shape[-2] == crop_size)
                assert (batch_data_ten_crop[i].shape[-3] == crop_size)
                temp = forward_batch(batch_data_ten_crop[i])
                full_features[i].append(temp)

        elif sample_mode == "CENTER_CROP":
            batch_data = batch_data[:, :, int(16*scale):int(240*scale), int(58*scale):int(282*scale), :]
            assert (batch_data.shape[-2] == crop_size)
            assert (batch_data.shape[-3] == crop_size)
            temp = forward_batch(batch_data)
            full_features[0].append(temp)

        total_time_sample += time.time() - start_time

    start_time = time.time()
    full_features = [np.concatenate(feature, axis = 0) for feature in full_features]
    full_features = [np.expand_dims(feature, axis = 0) for feature in full_features]
    
    full_features = np.concatenate(full_features, axis = 0)
    full_features = full_features[:, :, :, 0, 0, 0]
    full_features = np.array(full_features).transpose([1, 0, 2])

    total_time = total_time_preprocessing + total_time_sample + (time.time() - start_time)

    avg_time = total_time / chunk_num
    return avg_time

def load_rgb_batch(frames_dir, rgb_files, frames_indices, scale):
    batch_data = np.zeros(frames_indices.shape + (int(256 * scale), int(340 * scale), 3)) # (height, width)
    
    total_time_preprocessing = 0
    
    for i in range(frames_indices.shape[0]):
        for j in range(frames_indices.shape[1]):
            data, time_preprocessing = load_frame(os.path.join(frames_dir, rgb_files[frames_indices[i][j]]), scale)
            batch_data[i, j, :, :, :] = data
            total_time_preprocessing += time_preprocessing
    
    return batch_data, total_time_preprocessing

def load_frame(frame_file, scale):
    data = Image.open(frame_file)

    start_time = time.time()

    data = data.resize((int(340 * scale), int(256 * scale)), Image.ANTIALIAS) # (width, height)
    data = np.array(data)
    data = data.astype(float)
    
    if MODEL_NAME == "X3D":
        data = data / 255
    else:
        pass

    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    data = transform_norm(data)
    
    data = np.array(torch.permute(data, (1, 2, 0)))

    time_preprocessing = time.time() - start_time
    return data, time_preprocessing

def oversample_data(data, scale):
    data_flip = np.array(data[:,:,:,::-1,:])

    data_1 = np.array(data[:, :, :int(224 * scale), :int(224 * scale), :])
    data_2 = np.array(data[:, :, :int(224 * scale), int(-224 * scale):, :])
    data_3 = np.array(data[:, :, int(16 * scale):int(240 * scale), int(58 * scale):int(282 * scale), :])
    data_4 = np.array(data[:, :, int(-224 * scale):, :int(224 * scale), :])
    data_5 = np.array(data[:, :, int(-224 * scale):, int(-224 * scale):, :])

    data_f_1 = np.array(data_flip[:, :, :int(224 * scale), :int(224 * scale), :])
    data_f_2 = np.array(data_flip[:, :, :int(224 * scale), -int(224 * scale):, :])
    data_f_3 = np.array(data_flip[:, :, int(16 * scale):int(240 * scale), int(58 * scale):int(282 * scale), :])
    data_f_4 = np.array(data_flip[:, :, -int(224 * scale):, :int(224 * scale), :])
    data_f_5 = np.array(data_flip[:, :, int(-224 * scale):, int(-224 * scale):, :])

    return [data_1, data_2, data_3, data_4, data_5,
            data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]

def run():
    list_avg_times = []
    for _ in range(10):
        avg_time = measure_time(MODEL_NAME, FRAMES_DIR, FREQUENCY, BATCH_SIZE, SAMPLE_MODE, CROP_SIZE)
        print("Process time: {}".format(avg_time))
        list_avg_times.append(avg_time)
    
    return list_avg_times

if __name__ == "__main__":
    model = create_x3d(input_clip_length = INPUT_CLIP_LENGTH, input_crop_size = CROP_SIZE, depth_factor = 2.2)
    # model = I3Res50(num_classes = 400, use_nl = False)
    
    model.load_state_dict(torch.load(PRETRAINED_PATH))
    print("Load pretrained weight successfully!!!")

    model = model.cuda().eval()

    list_avg_times = run()

    mean_time = statistics.mean(list_avg_times)
    std = statistics.stdev(list_avg_times)

    print("Mean time: {}".format(mean_time))
    print("Std: {}".format(std))