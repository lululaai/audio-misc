import io

import numpy as np
from pydub import AudioSegment
from pydub.utils import get_array_type
import pyroomacoustics as pra


class Audio:
    def __init__(self, stream: bytes, sample_width: int = 2, frame_rate: int = 24000, ratio: float = 1, channels=1):
        self.sample_width = sample_width
        self.frame_rate = frame_rate
        self.audio = Audio.bytes2audio(stream, sample_width=sample_width, frame_rate=frame_rate)
        self.channels = channels
        if not (0 <= ratio <= 1):
            raise RuntimeError("ratio illegal")
        self.ratio = ratio
        self.dtype = np.dtype(get_array_type(self.audio.sample_width * 8))

    @classmethod
    def bytes2audio(cls, stream: bytes, sample_width, frame_rate) -> AudioSegment:
        """
        将byte转换为 AudioSegment 对象
        :param stream: 原始流
        :param sample_width: 音频采样宽度, pcm格式音频采样宽度为2
        :param frame_rate: 采样率
        :return:
        """
        if stream.startswith(b'RIFF') and b'WAVE' in stream[:12]:
            return AudioSegment.from_file(io.BytesIO(stream), format="wav")
        else:
            return AudioSegment.from_raw(io.BytesIO(stream), sample_width=sample_width, frame_rate=frame_rate,
                                         channels=1)

    def scaled(self) -> np.ndarray:
        """
        将音频按照指定比例放大或缩小音量
        :return:
        """
        audio_array = np.array(self.audio.get_array_of_samples()).astype(np.float32)
        audio_array *= self.ratio
        audio_array = np.clip(audio_array, np.iinfo(self.dtype).min, np.iinfo(self.dtype).max)
        return audio_array.astype(self.dtype)

    def scaled2audio(self) -> AudioSegment:
        audio_array = self.scaled()
        return AudioSegment(audio_array.tobytes(), frame_rate=self.frame_rate, sample_width=self.sample_width,
                            channels=self.channels)

    def mono2stereo(self):
        """
        单声道音频转换为双声道
        :return:
        """
        self.audio = self.audio.set_channels(2)
        self.channels = 2

    def _norm_to_room_coords(self, norm_coords, room_dim):
        """
        Convert normalized coordinates (in [-1,1]) to actual room coordinates [0, dim] in meters.
        Each coordinate is mapped by: coord_phys = (coord_norm + 1) * (dim/2).
        """
        return [(coord + 1) * (dim / 2) for coord, dim in zip(norm_coords, room_dim)]

    def simulate_moving_source(self, p1, p2, mic_dist=0.2, room_dim=None, block_len=None) -> np.ndarray:
        """
        Simulate a moving sound source from p1 to p2 for the given mono signal.
        Args:
            p1 (list or array of 3 floats): Starting 3D position [x,y,z] in meters.
            p2 (list or array of 3 floats): Ending 3D position [x,y,z] in meters.
            mic_dist: distance from ear in meters.
            room_dim (list or array of 3 floats): Room dimensions [Lx, Ly, Lz] in meters.
            block_len (int, optional): Number of samples per interpolation block.
                If None, defaults to 50ms.
        Returns:
            stereo_out (2D numpy array): Simulated stereo signal of shape (2, N_out),
                where N_out >= len(mono_signal) (due to RIR tail).
        """

        if block_len is None:
            block_len = self.audio.frame_rate // 1000 * 50
        if room_dim is None:
            room_dim = [6.0, 6.0, 3.0]
        center_x = room_dim[0] / 2.0
        center_y = room_dim[1] / 2.0
        ear_height = 1.5
        mic_l = [center_x - mic_dist / 2.0, center_y, ear_height]
        mic_r = [center_x + mic_dist / 2.0, center_y, ear_height]

        # 麦克风位置
        mic_positions = np.array([mic_l, mic_r]).T  # shape (3, 2)
        # 归一化数据
        audio_array = np.array(self.audio.get_array_of_samples()).astype(np.float32) / (2 ** 15)
        # Calculate number of blocks (segments) for the motion
        total_len = len(audio_array)
        n_blocks = (total_len + block_len - 1) // (block_len // 2)
        # hanning window
        fade = np.hanning(block_len * 2)
        fade_in = fade[:block_len]
        out_len = (n_blocks + 1) * (block_len // 2)
        out = np.zeros((2, out_len))
        norm = np.zeros(out_len)

        for i in range(n_blocks):
            start = i * block_len // 2
            end = start + block_len
            audio_block = audio_array[start:end]
            if len(audio_block) < block_len:
                audio_block = np.pad(audio_block, (0, block_len - len(audio_block)))
            t = i / max(n_blocks - 1, 1)

            # Compute interpolation factor (0 at A, 1 at B)
            current_p_norm = [a + (b - a) * t for a, b in zip(p1, p2)]
            current_p = self._norm_to_room_coords(current_p_norm, room_dim)

            # init room
            room = pra.ShoeBox(room_dim, fs=self.audio.frame_rate, max_order=2, air_absorption=True,
                               materials=pra.Material(energy_absorption=0.3))
            room.add_microphone_array(mic_positions)
            room.add_source(current_p)
            room.compute_rir()
            rir_left = room.rir[0][0]
            rir_right = room.rir[1][0]

            # Convolve block
            block_left = np.convolve(audio_block, rir_left)[:block_len]
            block_right = np.convolve(audio_block, rir_right)[:block_len]

            # Apply fade in window
            block_left *= fade_in
            block_right *= fade_in

            # Overlap-add with normalization
            out[0, start:end] += block_left
            out[1, start:end] += block_right
            norm[start:end] += fade_in

        out[0] /= np.maximum(norm, 1e-6)
        out[1] /= np.maximum(norm, 1e-6)
        out = np.clip(out * (2 ** 15 - 1), np.iinfo(self.dtype).min, np.iinfo(self.dtype).max)
        return out.astype(self.dtype)


class Synthesis:
    def __init__(self, main_audio: Audio, merge_audio: Audio):
        # 如果传入了文件头
        if main_audio.sample_width != merge_audio.sample_width or main_audio.frame_rate != merge_audio.frame_rate:
            raise RuntimeError("audio format not same")
        self.main_audio = main_audio
        self.merge_audio = merge_audio

    def overlay(self, main_stream_timestamp=0) -> AudioSegment:
        """
        将音频叠加
        :param main_stream_timestamp: 主音频的毫秒级时间戳, 从这个时间开始叠加
        :param merge_audio_ratio: 辅助音频的音量比例
        :return:
        """
        if not (0 <= main_stream_timestamp <= len(self.main_audio.audio)):
            raise RuntimeError("audio merge timestamp illegal")

        main_scaled = self.main_audio.scaled()

        # 归一化
        audio_max = np.iinfo(main_scaled.dtype).max
        main_scaled = main_scaled.astype(np.float32) / audio_max

        # 这里计算1ms几个采样点
        point_1_ms = self.main_audio.frame_rate * self.main_audio.channels / 1000
        # 分割的数组长度
        main_split_idx = int(point_1_ms * main_stream_timestamp)
        main_before = main_scaled[:main_split_idx]
        main_after = main_scaled[main_split_idx:]

        merge_scaled = self.merge_audio.scaled()
        merge_scaled = merge_scaled.astype(np.float32) / audio_max

        if len(merge_scaled) < len(main_after):
            pad_len = len(main_after) - len(merge_scaled)
            merge_scaled = np.pad(merge_scaled, (0, pad_len))
        else:
            merge_scaled = merge_scaled[0:len(main_after)]

        # 混合
        mixed_part = np.add(main_after, merge_scaled)
        mixed_part = np.clip(mixed_part, -1.0, 1.0)
        mixed_part = np.append(main_before, mixed_part)
        mixed_part *= audio_max
        # 返回数据
        dtype = np.dtype(get_array_type(self.main_audio.sample_width * 8))
        mixed_part = mixed_part.astype(dtype)
        mixed_segment = AudioSegment(mixed_part.tobytes(), frame_rate=self.main_audio.frame_rate,
                                     sample_width=self.main_audio.sample_width, channels=self.main_audio.channels)

        # 拼接混音结果
        return mixed_segment


if __name__ == '__main__':
    v1 = AudioSegment.from_file("main.wav")
    v2 = AudioSegment.from_file("merge1.wav")
    main = Audio(v1.raw_data, v1.sample_width, v1.frame_rate)
    test_stereo = Audio(AudioSegment.from_file("test_stereo.wav").raw_data, 2, 24000, channels=2)
    merge = Audio(v2.raw_data, v2.sample_width, v2.frame_rate)

    s = Synthesis(main, merge)
    a = s.overlay()
    a.export('result3.wav', format='wav')
