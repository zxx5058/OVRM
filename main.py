import cv2
import warnings
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy import signal
import matplotlib.pyplot as plt
import os
import math
from scipy.signal import find_peaks, savgol_filter
from detecta import detect_peaks
from scipy.fftpack import fft
from scipy.stats import multivariate_normal
from fastdtw import fastdtw
from scipy.integrate import simps
import pywt
import operator
from scipy.interpolate import interp1d
from PyEMD import EMD, EEMD

plt.interactive(False)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all='ignore')

def ImproveCrossingPoint(data, fs, shfit_distance, QualityLevel):
    N = len(data)
    data_shift = np.zeros(data.shape) - 1
    data_shift[shfit_distance:] = data[:-shfit_distance]
    cross_curve = data - data_shift

    zero_number = 0
    zero_index = []
    for i in range(len(cross_curve) - 1):
        if cross_curve[i] == 0:
            zero_number += 1
            zero_index.append(i)
        else:
            if cross_curve[i] * cross_curve[i + 1] < 0:
                zero_number += 1
                zero_index.append(i)

    cw = zero_number
    N = N
    fs = fs
    RR1 = ((cw / 2) / (N / fs)) * 60

    if (len(zero_index) <= 1 ) :
            RR2 = RR1
    else:
        time_span = 60 / RR1 / 2 * fs * QualityLevel
        zero_span = []
        for i in range(len(zero_index) - 1) :
            zero_span.append(zero_index[i + 1] - zero_index[i])

        while(min(zero_span) < time_span ) :
            doubt_point = np.argmin(zero_span)
            zero_index.pop(doubt_point)
            zero_index.pop(doubt_point)
            if len(zero_index) <= 1:
                break
            zero_span = []
            for i in range(len(zero_index) - 1):
                zero_span.append(zero_index[i + 1] - zero_index[i])

        zero_number = len(zero_index)
        cw = zero_number
        RR2 = ((cw / 2) / (N / fs)) * 60

    return RR2

def band_filter(orisignal, fs, bandpass_parameter = ( 3, 2/60, 40/60 )):
    filter_order = bandpass_parameter[0]
    LowPass = bandpass_parameter[1]
    HighPass = bandpass_parameter[2]
    b, a = signal.butter(filter_order, [2 * LowPass / fs, 2 * HighPass / fs], 'bandpass')
    filtedData = signal.filtfilt(b, a, orisignal)
    return filtedData

def normalizationfuc(wave):
    max_val = np.max(wave)
    min_val = np.min(wave)
    mind_val = max_val - min_val
    norm_wave = (wave - min_val) / mind_val
    return norm_wave

def peak_difference(signal1, signal2, framerate, peak_prominence=0.08):
    peaks1, _ = find_peaks(signal1, prominence=peak_prominence)
    peaks2, _ = find_peaks(signal2, prominence=peak_prominence)
    peak_values = np.diff(peaks1)
    average_amplitude = np.abs(np.mean(np.diff(peak_values)))

    areas = []
    for i in range(len(peaks1) - 1):
        start_index = peaks1[i]
        end_index = peaks1[i + 1]
        area = np.sum(signal1[start_index:end_index]) / framerate
        areas.append(area)
    mean_area = np.sum(areas)

    size1 = peaks1.shape
    size2 = peaks2.shape
    minsize = np.min((size1, size2))
    return minsize, average_amplitude, mean_area

def crosspoint(data, shfit_distance):
    data_shift = np.zeros(data.shape) - 1
    data_shift[shfit_distance:] = data[:-shfit_distance]
    cross_curve = data - data_shift
    zero_number = 0
    zero_index = []
    for i in range(len(cross_curve) - 1):
        if cross_curve[i] == 0:
            zero_number += 1
            zero_index.append(i)
        else:
            if cross_curve[i] * cross_curve[i + 1] < 0:
                zero_number += 1
                zero_index.append(i)
    return zero_number

def count_crossing_points(signal1, signal2):
    crossing_points = 0
    for i in range(1, len(signal1)):
        if (signal1[i-1] > signal2[i-1] and signal1[i] < signal2[i]) or (signal1[i-1] < signal2[i-1] and signal1[i] > signal2[i]):
            crossing_points += 1
    return crossing_points

def signal_energy(signal):
    squared_signal = np.square(signal)
    energy = np.sum(squared_signal)
    return energy

def feature_extraction(videoFile, QualityLevel = 0.7):
    cap = cv2.VideoCapture(videoFile)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = frame_width // 2
    center_y = frame_height // 2

    feature_params = dict(maxCorners=100,
                          qualityLevel=QualityLevel,
                          minDistance=7)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    while (p0 is None):
        QualityLevel = QualityLevel - 0.1
        feature_params = dict(maxCorners=100,
                              qualityLevel=QualityLevel,
                              minDistance=7)

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    lk_params = dict(winSize=(15, 15), maxLevel=2)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_num = 1
    u = np.zeros((int(total_frame), p0.shape[0], 2))
    u[0, :, 0] = p0[:, 0, 0].T
    u[0, :, 1] = p0[:, 0, 1].T

    while (frame_num < total_frame):
        frame_num += 1
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pl, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = pl

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        u[frame_num - 1, :, 0] = p0[:, 0, 0].T
        u[frame_num - 1, :, 1] = p0[:, 0, 1].T

    R = np.arctan(u[:, :, 0] / u[:, :, 1])
    y = u[:, :, 1]
    yk = np.sum(y, 1) / p0.shape[0]
    Rk = np.sum(R, 1) / p0.shape[0]
    M = np.sqrt(u[:, :, 0] ** 2 + u[:, :, 1] ** 2)
    Sk = np.sum(M, 1) / p0.shape[0]

    return R, Rk, yk, Sk

def moving_average(signal, window_size):
    # 使用np.convolve来进行移动平均滤波
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='same')
    return smoothed_signal

def sine_interpolation(start_value, end_value, interp_start_frame, interp_end_frame, direction):
    x = np.linspace(0, np.pi/2, interp_end_frame - interp_start_frame)
    if direction == 'up':
        interpolated_values = start_value + (end_value - start_value) * np.sin(x)
    elif direction == 'down':
        x_down = np.linspace(np.pi / 2, np.pi, interp_end_frame - interp_start_frame)
        interpolated_values = end_value + (start_value - end_value) * np.sin(x_down)
    return interpolated_values

def motion_artifact_remove(signal, peak_prominence=0.1, threshold=1.2):
    peaks, _ = find_peaks(signal, prominence=peak_prominence)
    valleys, _ = find_peaks(-signal, prominence=peak_prominence)

    combined_values = np.concatenate((peaks, valleys))
    sorted_combined = np.sort(combined_values)
    average_amplitude = np.mean(np.abs(signal[peaks]))
    average_amplitude_valley = np.mean(np.abs(signal[valleys]))

    abs_diff = np.abs(np.diff(signal[sorted_combined]))
    aver_diff = np.mean(abs_diff)
    normalized_diff = abs_diff / aver_diff
    # threshold = (average_amplitude + average_amplitude_valley) / 1.1
    motion_artifacts = []
    count_artifacts = []
    normal_diff = []
    abnormal_diff = []
    for i in range(0, len(sorted_combined) - 1):
        if normalized_diff[i] > threshold:
            motion_artifacts.append((sorted_combined[i], sorted_combined[i+1]))
            count_artifacts.extend((sorted_combined[i], sorted_combined[i+1]))
            abnormal_diff.append(abs_diff[i])
        else:
            normal_diff.append(abs_diff[i])
    duplicates = [item for item in count_artifacts if count_artifacts.count(item) == 2]
    duplicates = list(set(duplicates))

    filtered_peaks = [p for p in peaks if p not in count_artifacts]
    filtered_peak_average_amplitude = np.mean(np.abs(signal[filtered_peaks]))
    final_peak_amp = (filtered_peak_average_amplitude + average_amplitude) / 2
    filtered_valleys = [p for p in valleys if p not in count_artifacts]
    filtered_valley_average_amplitude = np.mean(np.abs(signal[filtered_valleys]))
    final_valley_amp = (filtered_valley_average_amplitude + average_amplitude_valley) / 2


    if abnormal_diff:
        for motion_range in motion_artifacts:
            start_frame, end_frame = motion_range
            motion_values = signal[start_frame:end_frame + 1]
            if start_frame in duplicates:
                peak_frame = start_frame
                if start_frame in peaks:
                    signal[peak_frame] = final_peak_amp
                    if end_frame in duplicates:
                        signal[end_frame] = final_valley_amp
                        interpolated_values = sine_interpolation(signal[peak_frame], signal[end_frame],
                                                                 start_frame,
                                                                 end_frame, direction='down')
                    else:
                        interpolated_values = sine_interpolation(signal[peak_frame], signal[end_frame],
                                                                 start_frame,
                                                                 end_frame, direction='down')
                else:
                    signal[peak_frame] = final_valley_amp
                    if end_frame in duplicates:
                        signal[end_frame] = final_peak_amp
                        interpolated_values = sine_interpolation(signal[peak_frame], signal[end_frame], start_frame,
                                                                 end_frame, direction='up')
                    else:
                        interpolated_values = sine_interpolation(signal[peak_frame], signal[end_frame], start_frame, end_frame, direction='up')
            elif end_frame in duplicates:
                peak_frame = end_frame
                if end_frame in peaks:
                    signal[peak_frame] = final_peak_amp
                    interpolated_values = sine_interpolation(signal[start_frame], signal[peak_frame], start_frame,
                                                             end_frame, direction='up')
                else:
                    signal[peak_frame] = final_valley_amp
                    interpolated_values = sine_interpolation(signal[start_frame], signal[peak_frame], start_frame,
                                                             end_frame, direction='down')
            else:
                if motion_values[0] < motion_values[-1]:  # 上升
                    signal[start_frame] = final_valley_amp
                    signal[end_frame] = final_peak_amp
                    interpolated_values = sine_interpolation(signal[start_frame], signal[end_frame],
                                                             start_frame, end_frame, direction='up')

                else:  # 下降
                    signal[start_frame] = final_peak_amp
                    signal[end_frame] = final_valley_amp
                    interpolated_values = sine_interpolation(signal[start_frame], signal[end_frame],
                                                             start_frame,
                                                             end_frame, direction='down')
            signal[start_frame:end_frame] = interpolated_values
    return signal

def video_split(video_file, num_blocks, output_folder, shfit_distance = 10, windowsize = 30):
    video_reader = cv2.VideoCapture(video_file)
    frame_rate = int(video_reader.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    block_width = frame_width // num_blocks
    block_height = frame_height // num_blocks
    os.makedirs(output_folder, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writers = [[cv2.VideoWriter(os.path.join(output_folder, f"block_{row}_{col}.avi"), fourcc, frame_rate,
                                      (block_width, block_height), isColor=True) for col in range(num_blocks)] for row
                     in range(num_blocks)]

    frame_index = 0
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break

        for row in range(num_blocks):
            for col in range(num_blocks):
                start_x = col * block_width
                start_y = row * block_height
                end_x = start_x + block_width
                end_y = start_y + block_height
                block_data = frame[start_y:end_y, start_x:end_x]

                video_writers[row][col].write(block_data)

        frame_index += 1
    for row in range(num_blocks):
        for col in range(num_blocks):
            video_writers[row][col].release()

    # 释放视频对象
    video_reader.release()

    R_total = np.zeros((num_blocks, num_blocks, total_frame))
    y_original = np.zeros((num_blocks, num_blocks, total_frame))
    y_total = np.zeros((num_blocks, num_blocks, total_frame))
    Sk_total = np.zeros((num_blocks, num_blocks, total_frame))
    Zero_number = np.zeros((num_blocks, num_blocks))
    IMF_number = np.zeros((num_blocks, num_blocks))
    total_cross = np.zeros((num_blocks, num_blocks))
    energy_total = np.zeros((num_blocks, num_blocks))
    similar_total = np.zeros((num_blocks, num_blocks))
    peak_num_total = np.zeros((num_blocks, num_blocks))
    peak_amp = np.zeros((num_blocks, num_blocks))
    gradient_total = np.zeros((num_blocks, num_blocks))
    area_total = np.zeros((num_blocks, num_blocks))
    std_total = np.zeros((num_blocks, num_blocks))
    for row in range(num_blocks):
        for col in range(num_blocks):
            block_video_path = os.path.join(output_folder, f"block_{row}_{col}.avi")
            R, Rk, y, Sk = feature_extraction(block_video_path)
            y_original[row, col, :] = y
            R_total[row, col, :] = savgol_filter(normalizationfuc(Rk), window_length=31, polyorder=2)
            y_total[row, col, :] = savgol_filter(y, window_length=35, polyorder=2)
            Sk_total[row, col, :] = savgol_filter(normalizationfuc(Sk), window_length=31, polyorder=2)
            data = y_total[row, col, :]
            zero_number = crosspoint(data, shfit_distance)
            Zero_number[row, col] = zero_number
            os.remove(block_video_path)

    mask = (Zero_number >= 6) & (Zero_number <= 32)
    for row in range(num_blocks):
        for col in range(num_blocks):
            if mask[row, col]:
                y_norm = normalizationfuc(motion_artifact_remove(y_total[row, col, :]))
                y_total[row, col, :] = savgol_filter(y_norm, window_length=27, polyorder=2)
                data = y_total[row, col, :]
                emd = EMD()
                IMF = emd.emd(data)
                IMF_number[row, col] = len(IMF)

                sym_data = np.flip(data)
                y_sym_data = 1 - data
                cross_number = count_crossing_points(data, sym_data)
                y_cross_number = count_crossing_points(data, y_sym_data)
                ener = signal_energy(data)
                energy_total[row, col] = ener
                total_cross[row, col] = cross_number + y_cross_number

                gradient = np.diff(y_total[row, col])
                gradient2 = np.diff(gradient)
                zero_number, _ = find_peaks(np.abs(gradient2))
                gradient_total[row, col] = np.sum(zero_number)

                diff_signal = np.diff(data)
                sign_changes = np.where(np.diff(np.sign(diff_signal)))[0]
                std_total[row, col] = len(sign_changes)

                peak_num, average_amplitude, mean_area = peak_difference(data, sym_data, frame_rate)
                time_span_minutes = len(data) / frame_rate / 60
                peak_frequency_per_minute = peak_num / time_span_minutes
                peak_num_total[row, col] = peak_frequency_per_minute
                peak_amp[row, col] = average_amplitude
                area_total[row, col] = mean_area

                similarity_measure = np.divide(np.minimum(np.abs(data), np.abs(sym_data)),
                                               np.maximum(np.abs(data), np.abs(sym_data)), out=np.zeros_like(data),
                                               where=(np.abs(data) != 0) & (np.abs(sym_data) != 0))
                if peak_num != 0:
                    # similar_total[row, col] = np.nansum(similarity_measure) / peak_num * cross_number
                    similar_total[row, col] = np.nansum(similarity_measure) / len(sign_changes) * cross_number
                else:
                    similar_total[row, col] = 0

    ave_peak = np.floor(np.mean(peak_num_total[mask]))
    mask_ori = mask & (peak_num_total >= 6) & (peak_num_total <= 34)
    peak_amp_no_nan = np.nan_to_num(peak_amp, nan=0)
    ave_peak_amp = np.mean(peak_amp_no_nan[mask_ori])
    ave_IMF = np.floor(np.mean(IMF_number[mask_ori]))
    ave_gradient = np.floor(np.mean(std_total[mask_ori]))
    ave_similar = np.mean(similar_total[mask_ori])
    ave_energy = np.mean(energy_total[mask_ori])
    row_indices, col_indices = np.where(mask_ori)
    total_score = np.zeros((num_blocks, num_blocks))
    for row, col in zip(row_indices, col_indices):
        score = 0
        if peak_amp_no_nan[row, col] <= ave_peak_amp:
            score += 1
        if IMF_number[row, col] <= ave_IMF:
            score += 1
        if std_total[row, col] <= ave_gradient:
            score += 1
        if similar_total[row, col] >= ave_similar:
            score += 1
        if energy_total[row, col] >= ave_energy:
            score += 1
        total_score[row, col] = score
    non_zero_mask = total_score != 0
    non_zero_scores = total_score[non_zero_mask]
    non_zero_mean = np.mean(non_zero_scores)
    new_mask = mask_ori & (total_score >= non_zero_mean)
    # mean_similar = np.mean(std_total[new_mask])
    # satisfying_max_mask = new_mask & (std_total <= mean_similar)
    # mean_similar = np.ceil(np.mean(std_total[new_mask]))
    # satisfying_max_mask = new_mask & (std_total <= mean_similar)
    # min_peak_amp = np.mean(similar_total[satisfying_max_mask])
    # final_satisfying_max_mask = satisfying_max_mask & (similar_total >= min_peak_amp)
    mean_similar = np.ceil(np.mean(std_total[new_mask]))
    satisfying_max_mask = new_mask & (std_total <= mean_similar)
    min_peak_amp = np.mean(similar_total[satisfying_max_mask])
    final_satisfying_max_mask = satisfying_max_mask & (similar_total >= min_peak_amp)

    row_indices_new, col_indices_new = np.where(final_satisfying_max_mask)
    average_signal = np.zeros(total_frame)
    for frame in range(total_frame):
        frame_values = y_total[row_indices_new, col_indices_new, frame]
        average_value = np.mean(frame_values)
        average_signal[frame] = average_value

    fil_average_signal = band_filter(average_signal, frame_rate)
    # plt.plot(y_total[3, 3, :])
    # plt.show()
    RR = ImproveCrossingPoint(fil_average_signal, frame_rate, shfit_distance=10, QualityLevel=0.6)


    return frame_rate, fil_average_signal, RR

if __name__ == '__main__':
    video_folder = 'E:/Users/wyl/Desktop/breath_codeepy/dataset_original'
    serialdata_folder = 'E:/Users/wyl/Desktop/breath_codeepy/SerialData_original'
    output_folder = 'E:/Users/wyl/Desktop/breath_codeepy/split_video'
    image_folder = 'E:/Users/wyl/Desktop/breath_codeepy/Motion_chest_signal_v3'

    video_files = [f for f in os.listdir(video_folder) if f.endswith('.avi')]
    total_MAE = 0
    total_MSE = 0
    num_samples = len(video_files)

    for video_file in video_files:
        # 构建视频文件的完整路径
        video_path = os.path.join(video_folder, video_file)
        print(video_path)
        numBlocks = 5
        frameRate, predict_signal, RR_predict = video_split(video_path, numBlocks, output_folder)

        # 生成真实呼吸信号
        fs = 50
        serialdata_file = os.path.splitext(video_file)[0] + '.npy'
        serialdata_path = os.path.join(serialdata_folder, serialdata_file)
        depthmap = np.load(serialdata_path)
        generate_signal = normalizationfuc(depthmap)
        lowfil_generate_signal = band_filter(generate_signal, fs)
        RR_real = ImproveCrossingPoint(lowfil_generate_signal, fs, shfit_distance=10, QualityLevel=0.6)

        video_name, _ = os.path.splitext(video_file)
        image_path = os.path.join(image_folder, video_name + '.png')
        t = np.linspace(1, len(predict_signal) / 30, len(predict_signal))
        t2 = np.linspace(1, len(depthmap) / 50, len(depthmap))
        plt.plot(t, predict_signal, color='red', label='Predict signal')
        plt.plot(t2, lowfil_generate_signal, color='blue', label='Real signal')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.savefig(image_path)
        plt.close()

        # plt.show()
        # print("RR_predict:", RR_predict)
        # print("RR_real:", RR_real)

        MAE = abs(RR_predict - RR_real)
        MSE = (RR_predict - RR_real) ** 2
        total_MAE += MAE
        total_MSE += MSE
        print("RR_predict", RR_predict)
        print("RR_real", RR_real)

    average_MAE = total_MAE / num_samples
    average_MSE = total_MSE / num_samples
    print("MAE:", average_MAE)
    print("MSE:", average_MSE)
