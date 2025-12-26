/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
 * 
 * pybind11 wrapper for FunASR Offline VAD C API
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <map>
#include "funasrruntime.h"

namespace py = pybind11;

/**
 * Python wrapper class for FunASR Offline VAD
 */
class FsmnVad {
public:
    FsmnVad(const std::map<std::string, std::string>& model_path, int thread_num = 1) {
        std::map<std::string, std::string> path_map = model_path;
        vad_handle_ = FsmnVadInit(path_map, thread_num);
        if (!vad_handle_) {
            throw std::runtime_error("Failed to initialize VAD model");
        }
    }

    ~FsmnVad() {
        if (vad_handle_) {
            FsmnVadUninit(vad_handle_);
        }
    }

    // 禁止拷贝构造和赋值
    FsmnVad(const FsmnVad&) = delete;
    FsmnVad& operator=(const FsmnVad&) = delete;

    /**
     * 推理音频缓冲区
     * @param audio_data: numpy array of int16 audio samples or bytes
     * @param sampling_rate: 采样率，默认16000
     * @param wav_format: 音频格式，"pcm" 或 "wav"，默认"pcm"
     * @return: 检测到的语音段列表，格式为 [[start1, end1], [start2, end2], ...]
     */
    std::vector<std::vector<int>> infer_buffer(
        py::array_t<int16_t> audio_data,
        int sampling_rate = 16000,
        const std::string& wav_format = "pcm"
    ) {
        if (!vad_handle_) {
            throw std::runtime_error("VAD handle is not initialized");
        }

        // 获取音频数据指针和长度
        // py::array_t 自动确保数组是 C 连续的
        if (audio_data.ndim() != 1) {
            throw std::runtime_error("Audio data must be 1-dimensional");
        }

        const char* audio_ptr = reinterpret_cast<const char*>(audio_data.data());
        int audio_len = static_cast<int>(audio_data.size() * sizeof(int16_t));

        // 调用 C API，离线处理 input_finished 始终为 true
        FUNASR_RESULT result = FsmnVadInferBuffer(
            vad_handle_,
            audio_ptr,
            audio_len,
            nullptr,  // callback
            true,     // input_finished is always true for offline processing
            sampling_rate,
            wav_format
        );

        std::vector<std::vector<int>> segments;
        if (result) {
            std::vector<std::vector<int>>* vad_segments = FsmnVadGetResult(result, 0);
            if (vad_segments) {
                segments = *vad_segments;
            }
            FsmnVadFreeResult(result);
        }

        return segments;
    }

    /**
     * 推理音频缓冲区（接受 bytes 类型）
     */
    std::vector<std::vector<int>> infer_buffer_bytes(
        const std::string& audio_bytes,
        int sampling_rate = 16000,
        const std::string& wav_format = "pcm"
    ) {
        if (!vad_handle_) {
            throw std::runtime_error("VAD handle is not initialized");
        }

        const char* audio_ptr = audio_bytes.data();
        int audio_len = static_cast<int>(audio_bytes.size());

        FUNASR_RESULT result = FsmnVadInferBuffer(
            vad_handle_,
            audio_ptr,
            audio_len,
            nullptr,
            true,     // input_finished is always true for offline processing
            sampling_rate,
            wav_format
        );

        std::vector<std::vector<int>> segments;
        if (result) {
            std::vector<std::vector<int>>* vad_segments = FsmnVadGetResult(result, 0);
            if (vad_segments) {
                segments = *vad_segments;
            }
            FsmnVadFreeResult(result);
        }

        return segments;
    }

    /**
     * 推理音频文件
     * @param wav_path: 音频文件路径
     * @param sampling_rate: 采样率，默认16000
     * @return: 检测到的语音段列表
     */
    std::vector<std::vector<int>> infer_file(
        const std::string& wav_path,
        int sampling_rate = 16000
    ) {
        if (!vad_handle_) {
            throw std::runtime_error("VAD handle is not initialized");
        }

        FUNASR_RESULT result = FsmnVadInfer(
            vad_handle_,
            wav_path.c_str(),
            nullptr,  // callback
            sampling_rate
        );

        std::vector<std::vector<int>> segments;
        if (result) {
            std::vector<std::vector<int>>* vad_segments = FsmnVadGetResult(result, 0);
            if (vad_segments) {
                segments = *vad_segments;
            }
            FsmnVadFreeResult(result);
        }

        return segments;
    }

    /**
     * 获取结果中的音频时长
     * 注意：这个方法需要在 infer_buffer 之后立即调用，因为结果会被释放
     * 建议直接使用 infer_buffer 的返回值，这个方法主要用于调试
     */
    float get_snippet_time(FUNASR_RESULT result) {
        if (!result) {
            return 0.0f;
        }
        return FsmnVadGetRetSnippetTime(result);
    }

    /**
     * 重置 VAD 模型状态
     */
    void reset() {
        // 离线 VAD 通常不需要重置，但为了接口一致性提供此方法
        // 如果需要，可以在这里添加重置逻辑
    }

private:
    FUNASR_HANDLE vad_handle_ = nullptr;
};

PYBIND11_MODULE(funasr_vad, m) {
    m.doc() = "FunASR Offline VAD Python Binding";

    py::class_<FsmnVad>(m, "FsmnVad")
        .def(py::init<const std::map<std::string, std::string>&, int>(),
             py::arg("model_path"),
             py::arg("thread_num") = 1,
             "Initialize FunASR Offline VAD model\n"
             "Args:\n"
             "    model_path: Dictionary containing model configuration:\n"
             "        - 'model-dir': Path to model directory\n"
             "        - 'quantize': 'true' or 'false' (optional)\n"
             "    thread_num: Number of threads (default: 1)")
        
        .def("infer_buffer",
             &FsmnVad::infer_buffer,
             py::arg("audio_data"),
             py::arg("sampling_rate") = 16000,
             py::arg("wav_format") = "pcm",
             "Infer VAD on audio buffer (numpy array)\n"
             "Args:\n"
             "    audio_data: numpy array of int16 audio samples\n"
             "    sampling_rate: Audio sample rate (default: 16000)\n"
             "    wav_format: Audio format, 'pcm' or 'wav' (default: 'pcm')\n"
             "Returns:\n"
             "    List of segments: [[start1, end1], [start2, end2], ...]")
        
        .def("infer_buffer_bytes",
             &FsmnVad::infer_buffer_bytes,
             py::arg("audio_bytes"),
             py::arg("sampling_rate") = 16000,
             py::arg("wav_format") = "pcm",
             "Infer VAD on audio buffer (bytes)\n"
             "Args:\n"
             "    audio_bytes: bytes object containing audio data\n"
             "    sampling_rate: Audio sample rate (default: 16000)\n"
             "    wav_format: Audio format, 'pcm' or 'wav' (default: 'pcm')\n"
             "Returns:\n"
             "    List of segments: [[start1, end1], [start2, end2], ...]")
        
        .def("infer_file",
             &FsmnVad::infer_file,
             py::arg("wav_path"),
             py::arg("sampling_rate") = 16000,
             "Infer VAD on audio file\n"
             "Args:\n"
             "    wav_path: Path to audio file (wav or pcm)\n"
             "    sampling_rate: Audio sample rate (default: 16000)\n"
             "Returns:\n"
             "    List of segments: [[start1, end1], [start2, end2], ...]")
        
        .def("reset",
             &FsmnVad::reset,
             "Reset VAD model state (for offline VAD, this is a no-op)");
}

