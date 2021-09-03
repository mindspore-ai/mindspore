/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/benchmark_train/net_train.h"
#define __STDC_FORMAT_MACROS
#include <cinttypes>
#undef __STDC_FORMAT_MACROS
#include <algorithm>
#include <cstring>
#include <utility>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "src/common/common.h"
#include "include/ms_tensor.h"
#include "include/context.h"
#include "include/version.h"
#include "include/model.h"
#include "include/train/train_cfg.h"
#include "include/train/train_session.h"

namespace mindspore {
namespace lite {
static const char *DELIM_SLASH = "/";
constexpr const char *DELIM_COLON = ":";
constexpr const char *DELIM_COMMA = ",";
constexpr int RET_TOO_BIG = -9;
constexpr int kField0 = 0;
constexpr int kField1 = 1;
constexpr int kField2 = 2;
constexpr int kField3 = 3;
constexpr int kField4 = 4;
constexpr int kFieldsToPrint = 5;
constexpr int kPrintOffset = 4;
constexpr int kCPUBindFlag2 = 2;
constexpr int kCPUBindFlag1 = 1;
static const int kTHOUSAND = 1000;

namespace {
float *ReadFileBuf(const char *file, size_t *size) {
  if (file == nullptr) {
    MS_LOG(ERROR) << "file is nullptr";
    return nullptr;
  }
  MS_ASSERT(size != nullptr);
  std::string real_path = RealPath(file);
  std::ifstream ifs(real_path);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << " open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<float[]> buf = std::make_unique<float[]>(*size / sizeof(float) + 1);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << real_path;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buf.get()), *size);
  ifs.close();

  return buf.release();
}
}  // namespace

int NetTrain::GenerateRandomData(size_t size, void *data) {
  MS_ASSERT(data != nullptr);
  char *casted_data = static_cast<char *>(data);
  for (size_t i = 0; i < size; i++) {
    casted_data[i] = static_cast<char>(i);
  }
  return RET_OK;
}

int NetTrain::GenerateInputData(std::vector<mindspore::tensor::MSTensor *> *ms_inputs) {
  for (auto tensor : *ms_inputs) {
    MS_ASSERT(tensor != nullptr);
    auto input_data = tensor->MutableData();
    if (input_data == nullptr) {
      MS_LOG(ERROR) << "MallocData for inTensor failed";
      return RET_ERROR;
    }
    auto tensor_byte_size = tensor->Size();
    auto status = GenerateRandomData(tensor_byte_size, input_data);
    if (status != RET_OK) {
      std::cerr << "GenerateRandomData for inTensor failed: " << status << std::endl;
      MS_LOG(ERROR) << "GenerateRandomData for inTensor failed: " << status;
      return status;
    }
  }
  return RET_OK;
}

int NetTrain::LoadInput(std::vector<mindspore::tensor::MSTensor *> *ms_inputs) {
  if (flags_->in_data_file_.empty()) {
    auto status = GenerateInputData(ms_inputs);
    if (status != RET_OK) {
      std::cerr << "Generate input data error " << status << std::endl;
      MS_LOG(ERROR) << "Generate input data error " << status;
      return status;
    }
  } else {
    auto status = ReadInputFile(ms_inputs);
    if (status != RET_OK) {
      std::cerr << "Read Input File error, " << status << std::endl;
      MS_LOG(ERROR) << "Read Input File error, " << status;
      return status;
    }
  }
  return RET_OK;
}

int NetTrain::ReadInputFile(std::vector<mindspore::tensor::MSTensor *> *ms_inputs) {
  if (ms_inputs->empty()) {
    return RET_OK;
  }

  if (this->flags_->in_data_type_ == kImage) {
    MS_LOG(ERROR) << "Not supported image input";
    return RET_ERROR;
  } else {
    for (size_t i = 0; i < ms_inputs->size(); i++) {
      auto cur_tensor = ms_inputs->at(i);
      MS_ASSERT(cur_tensor != nullptr);
      size_t size;
      std::string file_name = flags_->in_data_file_ + std::to_string(i + 1) + ".bin";
      auto bin_buf = ReadFile(file_name.c_str(), &size);
      if (bin_buf == nullptr) {
        MS_LOG(ERROR) << "ReadFile return nullptr";
        return RET_ERROR;
      }
      auto tensor_data_size = cur_tensor->Size();
      if (size != tensor_data_size) {
        std::cerr << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size
                  << std::endl;
        MS_LOG(ERROR) << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size;
        delete bin_buf;
        return RET_ERROR;
      }
      auto input_data = cur_tensor->MutableData();
      memcpy(input_data, bin_buf, tensor_data_size);
      delete[](bin_buf);
    }
  }
  return RET_OK;
}

int NetTrain::CompareOutput(const session::LiteSession &lite_session) {
  std::cout << "================ Comparing Forward Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  bool has_error = false;
  auto tensors_list = lite_session.GetOutputs();
  if (tensors_list.empty()) {
    MS_LOG(ERROR) << "Cannot find output tensors, get model output failed";
    return RET_ERROR;
  }
  mindspore::tensor::MSTensor *tensor = nullptr;
  int i = 1;
  for (auto it = tensors_list.begin(); it != tensors_list.end(); ++it) {
    tensor = lite_session.GetOutputByTensorName(it->first);
    std::cout << "output is tensor " << it->first << "\n";
    auto outputs = tensor->data();
    size_t size;
    std::string output_file = flags_->data_file_ + std::to_string(i) + ".bin";
    auto bin_buf = std::unique_ptr<float[]>(ReadFileBuf(output_file.c_str(), &size));
    if (bin_buf == nullptr) {
      MS_LOG(ERROR) << "ReadFile return nullptr";
      std::cout << "ReadFile return nullptr" << std::endl;
      return RET_ERROR;
    }
    if (size != tensor->Size()) {
      MS_LOG(ERROR) << "Output buffer and output file differ by size. Tensor size: " << tensor->Size()
                    << ", read size: " << size;
      std::cout << "Output buffer and output file differ by size. Tensor size: " << tensor->Size()
                << ", read size: " << size << std::endl;
      return RET_ERROR;
    }
    float bias = CompareData<float>(bin_buf.get(), tensor->ElementsNum(), reinterpret_cast<float *>(outputs));
    if (bias >= 0) {
      total_bias += bias;
      total_size++;
    } else {
      has_error = true;
      break;
    }
    i++;
  }

  if (!has_error) {
    float mean_bias;
    if (total_size != 0) {
      mean_bias = total_bias / total_size * 100;
    } else {
      mean_bias = 0;
    }

    std::cout << "Mean bias of all nodes/tensors: " << mean_bias << "%"
              << " threshold is:" << this->flags_->accuracy_threshold_ << std::endl;
    std::cout << "=======================================================" << std::endl << std::endl;

    if (mean_bias > this->flags_->accuracy_threshold_) {
      MS_LOG(INFO) << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%";
      std::cout << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%" << std::endl;
      return RET_TOO_BIG;
    } else {
      return RET_OK;
    }
  } else {
    MS_LOG(ERROR) << "Error in CompareData";
    std::cerr << "Error in CompareData" << std::endl;
    std::cout << "=======================================================" << std::endl << std::endl;
    return RET_ERROR;
  }
}

int NetTrain::MarkPerformance(const std::unique_ptr<session::LiteSession> &session) {
  MS_LOG(INFO) << "Running train loops...";
  std::cout << "Running train loops..." << std::endl;
  uint64_t time_min = 0xFFFFFFFFFFFFFFFF;
  uint64_t time_max = 0;
  uint64_t time_avg = 0;

  for (int i = 0; i < flags_->epochs_; i++) {
    session->BindThread(true);
    auto start = GetTimeUs();
    auto status =
      flags_->time_profiling_ ? session->RunGraph(before_call_back_, after_call_back_) : session->RunGraph();
    if (status != 0) {
      MS_LOG(ERROR) << "Inference error " << status;
      std::cerr << "Inference error " << status;
      return status;
    }

    auto end = GetTimeUs();
    auto time = end - start;
    time_min = std::min(time_min, time);
    time_max = std::max(time_max, time);
    time_avg += time;
    session->BindThread(false);
  }

  if (flags_->time_profiling_) {
    const std::vector<std::string> per_op_name = {"opName", "avg(ms)", "percent", "calledTimes", "opTotalTime"};
    const std::vector<std::string> per_op_type = {"opType", "avg(ms)", "percent", "calledTimes", "opTotalTime"};
    PrintResult(per_op_name, op_times_by_name_);
    PrintResult(per_op_type, op_times_by_type_);
  }

  if (flags_->epochs_ > 0) {
    time_avg /= flags_->epochs_;
    MS_LOG(INFO) << "Model = " << flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
                 << ", NumThreads = " << flags_->num_threads_ << ", MinRunTime = " << time_min / 1000.0f
                 << ", MaxRuntime = " << time_max / 1000.0f << ", AvgRunTime = " << time_avg / 1000.0f;
    printf("Model = %s, NumThreads = %d, MinRunTime = %f ms, MaxRuntime = %f ms, AvgRunTime = %f ms\n",
           flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1).c_str(), flags_->num_threads_,
           time_min / 1000.0f, time_max / 1000.0f, time_avg / 1000.0f);
  }
  return RET_OK;
}

int NetTrain::MarkAccuracy(const std::unique_ptr<session::LiteSession> &session, bool enforce_accuracy) {
  MS_LOG(INFO) << "MarkAccuracy";
  for (auto &msInput : session->GetInputs()) {
    switch (msInput->data_type()) {
      case TypeId::kNumberTypeFloat:
        PrintInputData<float>(msInput);
        break;
      case TypeId::kNumberTypeFloat32:
        PrintInputData<float>(msInput);
        break;
      case TypeId::kNumberTypeInt32:
        PrintInputData<int>(msInput);
        break;
      default:
        MS_LOG(ERROR) << "Datatype " << msInput->data_type() << " is not supported.";
        return RET_ERROR;
    }
  }
  auto status = session->RunGraph();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Inference error " << status;
    std::cerr << "Inference error " << status << std::endl;
    return status;
  }

  status = CompareOutput(*session);
  if (status == RET_TOO_BIG && !enforce_accuracy) {
    MS_LOG(INFO) << "Accuracy Error is big but not enforced";
    std::cout << "Accuracy Error is big but not enforced" << std::endl;
    return RET_OK;
  }

  if (status != RET_OK) {
    MS_LOG(ERROR) << "Compare output error " << status;
    std::cerr << "Compare output error " << status << std::endl;
    return status;
  }
  return RET_OK;
}

static CpuBindMode FlagToBindMode(int flag) {
  if (flag == kCPUBindFlag2) {
    return MID_CPU;
  }
  if (flag == kCPUBindFlag1) {
    return HIGHER_CPU;
  }
  return NO_BIND;
}

std::unique_ptr<session::LiteSession> NetTrain::CreateAndRunNetworkForTrain(const std::string &filename,
                                                                            const std::string &bb_filename,
                                                                            const Context &context,
                                                                            const TrainCfg &train_cfg, int epochs) {
  std::unique_ptr<session::LiteSession> session = nullptr;
  std::string model_name = filename.substr(filename.find_last_of(DELIM_SLASH) + 1);
  if (!bb_filename.empty()) {
    MS_LOG(INFO) << "CreateTransferSession from models files" << filename << " and " << bb_filename;
    std::cout << "CreateTranferSession from model file " << filename << " and " << bb_filename << std::endl;
    session = std::unique_ptr<session::LiteSession>(
      session::TrainSession::CreateTransferSession(bb_filename, filename, &context, true, &train_cfg));
    if (session == nullptr) {
      MS_LOG(ERROR) << "RunNetTrain CreateTranferSession failed while running " << model_name.c_str();
      std::cout << "RunNetTrain CreateTranferSession failed while running " << model_name.c_str() << std::endl;
      return nullptr;
    }
  } else {
    MS_LOG(INFO) << "CreateTrainSession from model file" << filename.c_str();
    std::cout << "CreateTrainSession from model file " << filename.c_str() << std::endl;
    std::cout << "Is raw mix precision model: " << train_cfg.mix_precision_cfg_.is_raw_mix_precision_ << std::endl;
    session = std::unique_ptr<session::LiteSession>(
      session::TrainSession::CreateTrainSession(filename, &context, true, &train_cfg));
    if (session == nullptr) {
      MS_LOG(ERROR) << "RunNetTrain CreateTrainSession failed while running " << model_name.c_str();
      std::cout << "RunNetTrain CreateTrainSession failed while running " << model_name.c_str() << std::endl;
      return nullptr;
    }
  }
  if (epochs > 0) {
    if (flags_->virtual_batch_) {
      session->SetupVirtualBatch(epochs);
    }
    session->Train();
  }
  return session;
}

std::unique_ptr<session::LiteSession> NetTrain::CreateAndRunNetworkForInference(const std::string &filename,
                                                                                const Context &context) {
  std::unique_ptr<session::LiteSession> session = nullptr;
  std::string model_name = filename.substr(filename.find_last_of(DELIM_SLASH) + 1);
  std::string filenamems = filename;
  if (filenamems.substr(filenamems.find_last_of(".") + 1) != "ms") {
    filenamems = filenamems + ".ms";
  }

  MS_LOG(INFO) << "start reading model file " << filenamems.c_str();
  std::cout << "start reading model file " << filenamems.c_str() << std::endl;
  auto *model = mindspore::lite::Model::Import(filenamems.c_str());
  if (model == nullptr) {
    MS_LOG(ERROR) << "create model for train session failed";
    return nullptr;
  }
  session = std::unique_ptr<session::LiteSession>(session::LiteSession::CreateSession(&context));
  if (session == nullptr) {
    MS_LOG(ERROR) << "ExportedFile CreateSession failed while running " << model_name.c_str();
    std::cout << "CreateSession failed while running " << model_name.c_str() << std::endl;
    delete model;
    return nullptr;
  }
  if (session->CompileGraph(model) != RET_OK) {
    MS_LOG(ERROR) << "Cannot compile model";
    delete model;
    return nullptr;
  }
  delete model;
  return session;
}

int NetTrain::CreateAndRunNetwork(const std::string &filename, const std::string &bb_filename, int train_session,
                                  int epochs, bool check_accuracy) {
  auto start_prepare_time = GetTimeUs();
  Context context;
  context.device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = FlagToBindMode(flags_->cpu_bind_mode_);
  context.device_list_[0].device_info_.cpu_device_info_.enable_float16_ = flags_->enable_fp16_;
  context.device_list_[0].device_type_ = mindspore::lite::DT_CPU;
  context.thread_num_ = flags_->num_threads_;

  TrainCfg train_cfg;
  if (flags_->loss_name_ != "") {
    train_cfg.loss_name_ = flags_->loss_name_;
  }
  train_cfg.mix_precision_cfg_.is_raw_mix_precision_ = flags_->is_raw_mix_precision_;
  std::unique_ptr<session::LiteSession> session;
  if (train_session) {
    session = CreateAndRunNetworkForTrain(filename, bb_filename, context, train_cfg, epochs);
    if (session == nullptr) {
      MS_LOG(ERROR) << "CreateAndRunNetworkForInference failed.";
      return RET_ERROR;
    }
  } else {
    session = CreateAndRunNetworkForInference(filename, context);
    if (session == nullptr) {
      MS_LOG(ERROR) << "CreateAndRunNetworkForInference failed.";
      return RET_ERROR;
    }
  }

  if (!flags_->resize_dims_.empty()) {
    auto ret = session->Resize(session->GetInputs(), flags_->resize_dims_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Input tensor resize failed.";
      std::cout << "Input tensor resize failed.";
      return ret;
    }
  }

  auto end_prepare_time = GetTimeUs();
  MS_LOG(INFO) << "PrepareTime = " << ((end_prepare_time - start_prepare_time) / kTHOUSAND) << " ms";
  std::cout << "PrepareTime = " << ((end_prepare_time - start_prepare_time) / kTHOUSAND) << " ms" << std::endl;
  // Load input
  MS_LOG(INFO) << "Load input data";
  auto ms_inputs = session->GetInputs();
  auto status = LoadInput(&ms_inputs);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Load input data error";
    return status;
  }

  if ((epochs > 0) && train_session) {
    status = MarkPerformance(session);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run MarkPerformance error: " << status;
      std::cout << "Run MarkPerformance error: " << status << std::endl;
      return status;
    }
    SaveModels(session);  // save file if flags are on
  }
  if (!flags_->data_file_.empty()) {
    session->Eval();

    status = MarkAccuracy(session, check_accuracy);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run MarkAccuracy error: " << status;
      std::cout << "Run MarkAccuracy error: " << status << std::endl;
      return status;
    }
  }
  return RET_OK;
}

int NetTrain::RunNetTrain() {
  auto status = CreateAndRunNetwork(flags_->model_file_, flags_->bb_model_file_, true, flags_->epochs_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CreateAndRunNetwork failed for model " << flags_->model_file_ << ". Status is " << status;
    std::cout << "CreateAndRunNetwork failed for model " << flags_->model_file_ << ". Status is " << status
              << std::endl;
    return status;
  }

  status = CheckExecutionOfSavedModels();  // re-initialize sessions according to flags
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run CheckExecute error: " << status;
    std::cout << "Run CheckExecute error: " << status << std::endl;
    return status;
  }
  return RET_OK;
}

int NetTrain::SaveModels(const std::unique_ptr<session::LiteSession> &session) {
  if (!flags_->export_file_.empty()) {
    if (flags_->bb_model_file_.empty()) {
      auto status = session->Export(flags_->export_file_ + "_qt", lite::MT_TRAIN, lite::QT_WEIGHT);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Export quantized model error " << flags_->export_file_ + "_qt";
        std::cout << "Export quantized model error " << flags_->export_file_ + "_qt" << std::endl;
        return RET_ERROR;
      }
    }
    auto status = session->Export(flags_->export_file_, lite::MT_TRAIN, lite::QT_NONE);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Export non quantized model error " << flags_->export_file_;
      std::cout << "Export non quantized model error " << flags_->export_file_ << std::endl;
      return RET_ERROR;
    }
  }
  if (!flags_->inference_file_.empty()) {
    auto status = session->Export(flags_->inference_file_ + "_qt", lite::MT_INFERENCE, lite::QT_WEIGHT);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Export quantized inference model error " << flags_->inference_file_ + "_qt";
      std::cout << "Export quantized inference model error " << flags_->inference_file_ + "_qt" << std::endl;
      return RET_ERROR;
    }

    auto tick = GetTimeUs();
    status = session->Export(flags_->inference_file_, lite::MT_INFERENCE, lite::QT_NONE);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Export non quantized inference model error " << flags_->inference_file_ + "_qt";
      std::cout << "Export non quantized inference model error " << flags_->inference_file_ + "_qt" << std::endl;
      return status;
    }
    std::cout << "ExportInference() execution time is " << GetTimeUs() - tick << "us\n";
  }
  return RET_OK;
}

int NetTrain::CheckExecutionOfSavedModels() {
  int status = RET_OK;
  if (!flags_->export_file_.empty()) {
    status = NetTrain::CreateAndRunNetwork(flags_->export_file_, flags_->bb_model_file_, true, 0);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run Exported model " << flags_->export_file_ << " error: " << status;
      std::cout << "Run Exported model " << flags_->export_file_ << " error: " << status << std::endl;
      return status;
    }
    if (flags_->bb_model_file_.empty()) {
      status = NetTrain::CreateAndRunNetwork(flags_->export_file_ + "_qt", "", true, 0, false);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Run Exported model " << flags_->export_file_ << "_qt.ms error: " << status;
        std::cout << "Run Exported model " << flags_->export_file_ << "_qt.ms error: " << status << std::endl;
        return status;
      }
    }
  }
  if (!flags_->inference_file_.empty()) {
    status = NetTrain::CreateAndRunNetwork(flags_->inference_file_, "", false, 0);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Running saved model " << flags_->inference_file_ << ".ms error: " << status;
      std::cout << "Running saved model " << flags_->inference_file_ << ".ms error: " << status << std::endl;
      return status;
    }
    status = NetTrain::CreateAndRunNetwork(flags_->inference_file_ + "_qt", "", false, 0, false);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Running saved model " << flags_->inference_file_ << "_qt.ms error: " << status;
      std::cout << "Running saved model " << flags_->inference_file_ << "_qt.ms error: " << status << std::endl;
      return status;
    }
  }
  return status;
}

void NetTrain::CheckSum(mindspore::tensor::MSTensor *tensor, std::string node_type, int id, std::string in_out) {
  int tensor_size = tensor->ElementsNum();
  void *data = tensor->MutableData();
  TypeId type = tensor->data_type();
  std::cout << node_type << " " << in_out << id << " shape=" << tensor->shape() << " sum=";
  switch (type) {
    case kNumberTypeFloat32:
      TensorNan(reinterpret_cast<float *>(data), tensor_size);
      std::cout << TensorSum<float>(data, tensor_size) << std::endl;
      break;
    case kNumberTypeInt32:
      std::cout << TensorSum<int>(data, tensor_size) << std::endl;
      break;
#ifdef ENABLE_FP16
    case kNumberTypeFloat16:
      std::cout << TensorSum<float16_t>(data, tensor_size) << std::endl;
      break;
#endif
    default:
      std::cout << "unsupported type:" << type << std::endl;
      break;
  }
}

int NetTrain::InitCallbackParameter() {
  // before callback
  before_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                          const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                          const mindspore::CallBackParam &callParam) {
    if (before_inputs.empty()) {
      MS_LOG(INFO) << "The num of beforeInputs is empty";
    }
    if (before_outputs.empty()) {
      MS_LOG(INFO) << "The num of beforeOutputs is empty";
    }
    if (op_times_by_type_.find(callParam.node_type) == op_times_by_type_.end()) {
      op_times_by_type_.insert(std::make_pair(callParam.node_type, std::make_pair(0, 0.0f)));
    }
    if (op_times_by_name_.find(callParam.node_name) == op_times_by_name_.end()) {
      op_times_by_name_.insert(std::make_pair(callParam.node_name, std::make_pair(0, 0.0f)));
    }
    op_call_times_total_++;
    op_begin_ = GetTimeUs();
    if ((callParam.node_type == "Adam") || (callParam.node_type == "Assign") || callParam.node_type == "SGD") {
      for (auto tensor : before_outputs) {
        std::fill(reinterpret_cast<int8_t *>(tensor->MutableData()),
                  reinterpret_cast<int8_t *>(tensor->MutableData()) + tensor->Size(), 0);
      }
    }
    return true;
  };

  // after callback
  after_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                         const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                         const mindspore::CallBackParam &call_param) {
    uint64_t opEnd = GetTimeUs();
    if (after_inputs.empty()) {
      MS_LOG(INFO) << "The num of after inputs is empty";
    }
    if (after_outputs.empty()) {
      MS_LOG(INFO) << "The num of after outputs is empty";
    }
    float cost = static_cast<float>(opEnd - op_begin_) / 1000.0f;
    op_cost_total_ += cost;
    op_times_by_type_[call_param.node_type].first++;
    op_times_by_type_[call_param.node_type].second += cost;
    op_times_by_name_[call_param.node_name].first++;
    op_times_by_name_[call_param.node_name].second += cost;
    if (flags_->layer_checksum_) {
      for (size_t i = 0; i < after_inputs.size(); i++) {
        CheckSum(after_inputs.at(i), call_param.node_type, i, "in");
      }
      for (size_t i = 0; i < after_outputs.size(); i++) {
        CheckSum(after_outputs.at(i), call_param.node_type, i, "out");
      }
      std::cout << std::endl;
    }
    return true;
  };
  return RET_OK;
}

void NetTrainFlags::InitResizeDimsList() {
  std::string content = this->resize_dims_in_;
  std::vector<int> shape;
  auto shape_strs = StrSplit(content, std::string(DELIM_COLON));
  for (const auto &shape_str : shape_strs) {
    shape.clear();
    auto dim_strs = StrSplit(shape_str, std::string(DELIM_COMMA));
    std::cout << "Resize Dims: ";
    for (const auto &dim_str : dim_strs) {
      std::cout << dim_str << " ";
      shape.emplace_back(static_cast<int>(std::stoi(dim_str)));
    }
    std::cout << std::endl;
    this->resize_dims_.emplace_back(shape);
  }
}

int NetTrain::Init() {
  if (this->flags_ == nullptr) {
    return 1;
  }
  MS_LOG(INFO) << "ModelPath = " << this->flags_->model_file_;
  MS_LOG(INFO) << "InDataPath = " << this->flags_->in_data_file_;
  MS_LOG(INFO) << "InDataType = " << this->flags_->in_data_type_in_;
  MS_LOG(INFO) << "Epochs = " << this->flags_->epochs_;
  MS_LOG(INFO) << "AccuracyThreshold = " << this->flags_->accuracy_threshold_;
  MS_LOG(INFO) << "WarmUpLoopCount = " << this->flags_->warm_up_loop_count_;
  MS_LOG(INFO) << "NumThreads = " << this->flags_->num_threads_;
  MS_LOG(INFO) << "expectedDataFile = " << this->flags_->data_file_;
  MS_LOG(INFO) << "exportDataFile = " << this->flags_->export_file_;
  MS_LOG(INFO) << "enableFp16 = " << this->flags_->enable_fp16_;
  MS_LOG(INFO) << "virtualBatch = " << this->flags_->virtual_batch_;

  if (this->flags_->epochs_ < 0) {
    MS_LOG(ERROR) << "epochs:" << this->flags_->epochs_ << " must be equal/greater than 0";
    std::cerr << "epochs:" << this->flags_->epochs_ << " must be equal/greater than 0" << std::endl;
    return RET_ERROR;
  }

  if (this->flags_->num_threads_ < 1) {
    MS_LOG(ERROR) << "numThreads:" << this->flags_->num_threads_ << " must be greater than 0";
    std::cerr << "numThreads:" << this->flags_->num_threads_ << " must be greater than 0" << std::endl;
    return RET_ERROR;
  }

  this->flags_->in_data_type_ = this->flags_->in_data_type_in_ == "img" ? kImage : kBinary;

  if (flags_->in_data_file_.empty() && !flags_->data_file_.empty()) {
    MS_LOG(ERROR) << "expectedDataFile not supported in case that inDataFile is not provided";
    std::cerr << "expectedDataFile is not supported in case that inDataFile is not provided" << std::endl;
    return RET_ERROR;
  }

  if (flags_->in_data_file_.empty() && !flags_->export_file_.empty()) {
    MS_LOG(ERROR) << "exportDataFile not supported in case that inDataFile is not provided";
    std::cerr << "exportDataFile is not supported in case that inDataFile is not provided" << std::endl;
    return RET_ERROR;
  }

  if (flags_->model_file_.empty()) {
    MS_LOG(ERROR) << "modelPath is required";
    std::cerr << "modelPath is required" << std::endl;
    return 1;
  }

  if (flags_->time_profiling_) {
    auto status = InitCallbackParameter();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Init callback Parameter failed.";
      std::cerr << "Init callback Parameter failed." << std::endl;
      return RET_ERROR;
    }
  }
  flags_->InitResizeDimsList();
  if (!flags_->resize_dims_.empty() && !flags_->input_data_list_.empty() &&
      flags_->resize_dims_.size() != flags_->input_data_list_.size()) {
    MS_LOG(ERROR) << "Size of input resizeDims should be equal to size of input inDataPath";
    std::cerr << "Size of input resizeDims should be equal to size of input inDataPath" << std::endl;
    return RET_ERROR;
  }
  return RET_OK;
}

namespace {
constexpr int kNumToPrint = 5;
}

int NetTrain::PrintResult(const std::vector<std::string> &title,
                          const std::map<std::string, std::pair<int, float>> &result) {
  std::vector<size_t> columnLenMax(kFieldsToPrint);
  std::vector<std::vector<std::string>> rows;

  for (auto &iter : result) {
    std::string stringBuf[kFieldsToPrint];
    std::vector<std::string> columns;
    size_t len;

    len = iter.first.size();
    if (len > columnLenMax.at(kField0)) {
      columnLenMax.at(kField0) = len + kPrintOffset;
    }
    columns.push_back(iter.first);

    stringBuf[kField1] = to_string(iter.second.second / flags_->epochs_);
    len = stringBuf[kField1].length();
    if (len > columnLenMax.at(kField1)) {
      columnLenMax.at(kField1) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[kField1]);

    stringBuf[kField2] = to_string(iter.second.second / op_cost_total_);
    len = stringBuf[kField2].length();
    if (len > columnLenMax.at(kField2)) {
      columnLenMax.at(kField2) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[kField2]);

    stringBuf[kField3] = to_string(iter.second.first);
    len = stringBuf[kField3].length();
    if (len > columnLenMax.at(kField3)) {
      columnLenMax.at(kField3) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[kField3]);

    stringBuf[kField4] = to_string(iter.second.second);
    len = stringBuf[kField4].length();
    if (len > columnLenMax.at(kField4)) {
      columnLenMax.at(kField4) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[kField4]);

    rows.push_back(columns);
  }

  printf("-------------------------------------------------------------------------\n");
  for (int i = 0; i < kNumToPrint; i++) {
    auto printBuf = title[i];
    if (printBuf.size() > columnLenMax.at(i)) {
      columnLenMax.at(i) = printBuf.size();
    }
    printBuf.resize(columnLenMax.at(i), ' ');
    printf("%s\t", printBuf.c_str());
  }
  printf("\n");
  for (size_t i = 0; i < rows.size(); i++) {
    for (int j = 0; j < kNumToPrint; j++) {
      auto printBuf = rows[i][j];
      printBuf.resize(columnLenMax.at(j), ' ');
      printf("%s\t", printBuf.c_str());
    }
    printf("\n");
  }
  return RET_OK;
}

int RunNetTrain(int argc, const char **argv) {
  NetTrainFlags flags;
  Option<std::string> err = flags.ParseFlags(argc, argv);

  if (err.IsSome()) {
    std::cerr << err.Get() << std::endl;
    std::cerr << flags.Usage() << std::endl;
    return RET_ERROR;
  }

  if (flags.help) {
    std::cerr << flags.Usage() << std::endl;
    return RET_OK;
  }

  NetTrain net_trainer(&flags);
  auto status = net_trainer.Init();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "NetTrain init Error : " << status;
    std::cerr << "NetTrain init Error : " << status << std::endl;
    return RET_ERROR;
  }

  status = net_trainer.RunNetTrain();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run NetTrain "
                  << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
                  << " Failed : " << status;
    std::cerr << "Run NetTrain " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
              << " Failed : " << status << std::endl;
    return RET_ERROR;
  }

  MS_LOG(INFO) << "Run NetTrain " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
               << " Success.";
  std::cout << "Run NetTrain " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
            << " Success." << std::endl;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
