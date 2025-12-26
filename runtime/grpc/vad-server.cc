/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* Modified for VAD-only service */

#include "vad-server.h"
#include "tclap/CmdLine.h"
#include "com-define.h"

VadEngine::VadEngine(
  grpc::ServerReaderWriter<Response, Request>* stream,
  std::shared_ptr<FUNASR_HANDLE> vad_handler)
  : stream_(std::move(stream)),
    vad_handler_(std::move(vad_handler)) {

  request_ = std::make_shared<Request>();
}

void VadEngine::VadThreadFunc() {
  int step = (sampling_rate_ * step_duration_ms_ / 1000) * 2; // int16 = 2bytes;
  bool is_final = false;
  std::vector<std::vector<int>> all_segments;

  LOG(INFO) << "VAD engine init, start processing loop";

  while (true) {
    if (audio_buffer_.length() > step || is_end_) {
      if (audio_buffer_.length() <= step && is_end_) {
        is_final = true;
        step = audio_buffer_.length();
      }

      // 调用VAD推理
      FUNASR_RESULT result = FsmnVadInfer(*vad_handler_,
                                         audio_buffer_.c_str(),
                                         nullptr,
                                         sampling_rate_);
      
      p_mutex_->lock();
      audio_buffer_ = audio_buffer_.substr(step);
      p_mutex_->unlock();

      if (result) {
        // 获取VAD结果
        std::vector<std::vector<int>>* vad_segments = FsmnVadGetResult(result, 0);
        float snippet_time = FsmnVadGetRetSnippetTime(result);
        
        if (vad_segments && !vad_segments->empty()) {
          // 合并所有VAD片段
          for (const auto& segment : *vad_segments) {
            all_segments.push_back(segment);
          }
          
          // 发送VAD结果
          std::string vad_result = "VAD segments: ";
          for (size_t i = 0; i < vad_segments->size(); i++) {
            const auto& seg = (*vad_segments)[i];
            vad_result += "[" + std::to_string(seg[0]) + "," + std::to_string(seg[1]) + "]";
            if (i < vad_segments->size() - 1) vad_result += ", ";
          }
          
          Response response;
          response.set_text(vad_result);
          response.set_is_final(is_final);
          stream_->Write(response);
          LOG(INFO) << "send VAD results: " << vad_result;
        }
        
        FsmnVadFreeResult(result);
      }

      if (is_final) {
        // 发送最终汇总结果
        std::string final_result = "Total VAD segments: " + std::to_string(all_segments.size());
        Response response;
        response.set_text(final_result);
        response.set_is_final(true);
        stream_->Write(response);
        LOG(INFO) << "VAD processing completed: " << final_result;
        break;
      }
    }
    sleep(0.001);
  }
}

void VadEngine::OnSpeechStart() {
  if (request_->sampling_rate() != 0) {
    sampling_rate_ = request_->sampling_rate();
  }
  LOG(INFO) << "sampling_rate is " << sampling_rate_;

  switch(request_->wav_format()) {
    case WavFormat::pcm: encoding_ = "pcm";
  }
  LOG(INFO) << "encoding is " << encoding_;
  
  vad_thread_ = std::make_shared<std::thread>(&VadEngine::VadThreadFunc, this);
  is_start_ = true;
}

void VadEngine::OnSpeechData() {
  p_mutex_->lock();
  audio_buffer_ += request_->audio_data();
  p_mutex_->unlock();
}

void VadEngine::OnSpeechEnd() {
  is_end_ = true;
  LOG(INFO) << "Read all pcm data, wait for VAD processing";
  if (vad_thread_ != nullptr) {
    vad_thread_->join();
  }
}

void VadEngine::operator()() {
  try {
    LOG(INFO) << "start VAD engine main loop";
    while (stream_->Read(request_.get())) {
      LOG(INFO) << "receive data";
      if (!is_start_) {
        OnSpeechStart();
      }
      OnSpeechData();
      if (request_->is_final()) {
        break;
      }
    }
    OnSpeechEnd();
    LOG(INFO) << "VAD connection finish";
  } catch (std::exception const& e) {
    LOG(ERROR) << e.what();
  }
}

VadService::VadService(std::map<std::string, std::string>& config, int onnx_thread)
  : config_(config) {

  // 只初始化VAD模型
  vad_handler_ = std::make_shared<FUNASR_HANDLE>(std::move(FsmnVadInit(config_, onnx_thread)));
  LOG(INFO) << "VadService VAD model loaded";

  // 模型预热
  int sampling_rate = 16000;
  int buffer_len = sampling_rate * 1;
  std::string tmp_data(buffer_len, '0');
  FUNASR_RESULT result = FsmnVadInfer(*vad_handler_, tmp_data.c_str(), nullptr, sampling_rate);
  if (result) {
      FsmnVadFreeResult(result);
  }
  LOG(INFO) << "VadService VAD model warmup completed";
}

grpc::Status VadService::Recognize(
  grpc::ServerContext* context,
  grpc::ServerReaderWriter<Response, Request>* stream) {
  LOG(INFO) << "Get VAD Recognize request";
  VadEngine engine(
    stream,
    vad_handler_
  );

  std::thread t(std::move(engine));
  t.join();
  return grpc::Status::OK;
}

void GetValue(TCLAP::ValueArg<std::string>& value_arg, std::string key, std::map<std::string, std::string>& config) {
  if (value_arg.isSet()) {
    config.insert({key, value_arg.getValue()});
    LOG(INFO) << key << " : " << value_arg.getValue();
  }
}

int main(int argc, char* argv[]) {
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);

  TCLAP::CmdLine cmd("funasr-vad-server", ' ', "1.0");
  TCLAP::ValueArg<std::string>  vad_dir("", VAD_DIR, "the vad model path, which contains model.onnx, vad.yaml, vad.mvn", true, "", "string");
  TCLAP::ValueArg<std::string>  vad_quant("", VAD_QUANT, "true (Default), load the model of model_quant.onnx in vad_dir. If set false, load the model of model.onnx in vad_dir", false, "true", "string");
  TCLAP::ValueArg<std::int32_t>  onnx_thread("", "onnx-inter-thread", "onnxruntime SetIntraOpNumThreads", false, 1, "int32_t");
  TCLAP::ValueArg<std::string> port_id("", PORT_ID, "port id", true, "", "string");

  cmd.add(vad_dir);
  cmd.add(vad_quant);
  cmd.add(onnx_thread);
  cmd.add(port_id);
  cmd.parse(argc, argv);

  std::map<std::string, std::string> config;
  GetValue(vad_dir, VAD_DIR, config);
  GetValue(vad_quant, VAD_QUANT, config);
  GetValue(port_id, PORT_ID, config);

  std::string port;
  try {
    port = config.at(PORT_ID);
  } catch(std::exception const &e) {
    LOG(INFO) << ("Error when read port.");
    exit(0);
  }
  std::string server_address;
  server_address = "0.0.0.0:" + port;
  VadService service(config, onnx_thread.getValue());

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "VAD Server listening on " << server_address;
  server->Wait();

  return 0;
}