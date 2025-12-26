/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* Modified for VAD-only service */

#include <string>
#include <thread>
#include <mutex>
#include <unistd.h>

#include "grpcpp/server_builder.h"
#include "paraformer.grpc.pb.h"
#include "funasrruntime.h"
#include "tclap/CmdLine.h"
#include "com-define.h"
#include "glog/logging.h"

using paraformer::WavFormat;
using paraformer::Request;
using paraformer::Response;
using paraformer::ASR;

// VAD结果结构
typedef struct {
  std::vector<std::vector<int>> segments;  // [[start1, end1], [start2, end2], ...]
  float snippet_time;
} VAD_RESULT;

class VadEngine {
 public:
  VadEngine(grpc::ServerReaderWriter<Response, Request>* stream, std::shared_ptr<FUNASR_HANDLE> vad_handler);
  void operator()();

 private:
  void VadThreadFunc();
  void OnSpeechStart();
  void OnSpeechData();
  void OnSpeechEnd();

  grpc::ServerReaderWriter<Response, Request>* stream_;
  std::shared_ptr<Request> request_;
  std::shared_ptr<Response> response_;
  std::shared_ptr<FUNASR_HANDLE> vad_handler_;
  std::string audio_buffer_;
  std::shared_ptr<std::thread> vad_thread_ = nullptr;
  bool is_start_ = false;
  bool is_end_ = false;

  int sampling_rate_ = 16000;
  std::string encoding_;
  int step_duration_ms_ = 100;

  std::unique_ptr<std::mutex> p_mutex_ = std::make_unique<std::mutex>();
};

class VadService final : public ASR::Service {
  public:
    VadService(std::map<std::string, std::string>& config, int num_thread);
    grpc::Status Recognize(grpc::ServerContext* context, grpc::ServerReaderWriter<Response, Request>* stream);

  private:
    std::map<std::string, std::string> config_;
    std::shared_ptr<FUNASR_HANDLE> vad_handler_;
};