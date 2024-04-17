/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "include/api/serialization.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace lite {
constexpr int kField0 = 0;
constexpr int kField1 = 1;
constexpr int kField2 = 2;
constexpr int kField3 = 3;
constexpr int kField4 = 4;

int NetTrain::GenerateInputData() {
  for (auto tensor : ms_inputs_for_api_) {
    auto tensor_byte_size = tensor.DataSize();
    MS_ASSERT(tensor_byte_size != 0);
    auto data_ptr = new (std::nothrow) char[tensor_byte_size];
    if (data_ptr == nullptr) {
      MS_LOG(ERROR) << "Malloc input data buffer failed, data_size:  " << tensor_byte_size;
      return RET_ERROR;
    }
    inputs_buf_.emplace_back(data_ptr);
    inputs_size_.emplace_back(tensor_byte_size);
    for (size_t i = 0; i < tensor_byte_size; i++) {
      data_ptr[i] =
        (tensor.DataType() == mindspore::DataType::kNumberTypeFloat32) ? static_cast<char>(i) : static_cast<char>(0);
    }
  }
  batch_num_ = 1;
  return RET_OK;
}

int NetTrain::LoadStepInput(size_t step) {
  if (step >= batch_num_) {
    auto cur_batch = step + 1;
    MS_LOG(ERROR) << "Max input Batch is:" << batch_num_ << " but got batch :" << cur_batch;
    return RET_ERROR;
  }
  for (size_t i = 0; i < ms_inputs_for_api_.size(); i++) {
    auto cur_tensor = ms_inputs_for_api_.at(i);
    MS_ASSERT(cur_tensor != nullptr);
    auto tensor_data_size = cur_tensor.DataSize();
    auto input_data = cur_tensor.MutableData();
    MS_ASSERT(input_data != nullptr);
    memcpy_s(input_data, tensor_data_size, inputs_buf_[i].get() + step * tensor_data_size, tensor_data_size);
  }
  return RET_OK;
}

int NetTrain::ReadInputFile() {
  if (ms_inputs_for_api_.empty()) {
    return RET_OK;
  }

  if (this->flags_->in_data_type_ == kImage) {
    MS_LOG(ERROR) << "Not supported image input";
    return RET_ERROR;
  } else {
    for (size_t i = 0; i < ms_inputs_for_api_.size(); i++) {
      auto cur_tensor = ms_inputs_for_api_.at(i);
      MS_ASSERT(cur_tensor != nullptr);
      size_t size;
      std::string file_name = flags_->in_data_file_ + std::to_string(i + 1) + ".bin";
      auto bin_buf = ReadFile(file_name.c_str(), &size);
      if (bin_buf == nullptr) {
        MS_LOG(ERROR) << "ReadFile return nullptr";
        return RET_ERROR;
      }
      auto tensor_data_size = cur_tensor.DataSize();
      MS_ASSERT(tensor_data_size != 0);
      if (size == 0 || size % tensor_data_size != 0 || (batch_num_ != 0 && size / tensor_data_size != batch_num_)) {
        std::cerr << "Input binary file size error, required :N * " << tensor_data_size << ", in fact: " << size
                  << " ,file_name: " << file_name.c_str() << std::endl;
        MS_LOG(ERROR) << "Input binary file size error, required: N * " << tensor_data_size << ", in fact: " << size
                      << " ,file_name: " << file_name.c_str();
        delete bin_buf;
        return RET_ERROR;
      }
      inputs_buf_.emplace_back(bin_buf);
      inputs_size_.emplace_back(size);
      batch_num_ = size / tensor_data_size;
    }
  }
  return RET_OK;
}

int NetTrain::CompareOutput() {
  std::cout << "================ Comparing Forward Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  bool has_error = false;
  auto output_tensors = ms_model_->GetOutputs();
  if (output_tensors.empty()) {
    MS_LOG(ERROR) << "Cannot find output tensors, get model output failed";
    return RET_ERROR;
  }
  std::map<std::string, MSTensor> ordered_outputs;
  for (const auto &output_tensor : output_tensors) {
    ordered_outputs.insert({output_tensor.Name(), output_tensor});
  }
  int i = 1;
  mindspore::MSTensor tensor;
  for (auto &ordered_output : ordered_outputs) {
    tensor = ordered_output.second;
    std::cout << "output is tensor " << ordered_output.first << "\n";
    auto outputs = tensor.MutableData();
    size_t size;
    std::string output_file = flags_->data_file_ + std::to_string(i) + ".bin";
    auto bin_buf = std::unique_ptr<float[]>(ReadFileBuf(output_file.c_str(), &size));
    if (bin_buf == nullptr) {
      MS_LOG(ERROR) << "ReadFile return nullptr";
      std::cout << "ReadFile return nullptr" << std::endl;
      return RET_ERROR;
    }
    if (size != tensor.DataSize()) {
      MS_LOG(ERROR) << "Output buffer and output file differ by size. Tensor size: " << tensor.DataSize()
                    << ", read size: " << size;
      std::cout << "Output buffer and output file differ by size. Tensor size: " << tensor.DataSize()
                << ", read size: " << size << std::endl;
      return RET_ERROR;
    }
    float bias = CompareData<float>(bin_buf.get(), tensor.ElementNum(), reinterpret_cast<float *>(outputs));
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

int NetTrain::MarkPerformance() {
  MS_LOG(INFO) << "Running train loops...";
  std::cout << "Running train loops..." << std::endl;
  uint64_t time_min = 0xFFFFFFFFFFFFFFFF;
  uint64_t time_max = 0;
  uint64_t time_avg = 0;

  for (int i = 0; i < flags_->epochs_; i++) {
    auto start = GetTimeUs();
    for (size_t step = 0; step < batch_num_; step++) {
      MS_LOG(INFO) << "Run for epoch:" << i << " step:" << step;
      auto ret = LoadStepInput(step);
      if (ret != RET_OK) {
        return ret;
      }
      auto status = ms_model_->RunStep(before_call_back_, after_call_back_);
      if (status != mindspore::kSuccess) {
        MS_LOG(ERROR) << "Inference error " << status;
        std::cerr << "Inference error " << status;
        return RET_ERROR;
      }
    }

    auto end = GetTimeUs();
    auto time = end - start;
    time_min = std::min(time_min, time);
    time_max = std::max(time_max, time);
    time_avg += time;
  }

  if (flags_->time_profiling_) {
    const std::vector<std::string> per_op_name = {"opName", "avg(ms)", "percent", "calledTimes", "opTotalTime"};
    const std::vector<std::string> per_op_type = {"opType", "avg(ms)", "percent", "calledTimes", "opTotalTime"};
    PrintResult(per_op_name, op_times_by_name_);
    PrintResult(per_op_type, op_times_by_type_);
  }

  if (flags_->epochs_ > 0) {
    time_avg /= static_cast<size_t>(flags_->epochs_);
    MS_LOG(INFO) << "Model = " << flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
                 << ", NumThreads = " << flags_->num_threads_ << ", MinRunTime = " << time_min / 1000.0f
                 << ", MaxRuntime = " << time_max / 1000.0f << ", AvgRunTime = " << time_avg / 1000.0f;
    printf("Model = %s, NumThreads = %d, MinRunTime = %f ms, MaxRuntime = %f ms, AvgRunTime = %f ms\n",
           flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1).c_str(), flags_->num_threads_,
           time_min / 1000.0f, time_max / 1000.0f, time_avg / 1000.0f);
  }
  return RET_OK;
}

int NetTrain::MarkAccuracy(bool enforce_accuracy) {
  MS_LOG(INFO) << "MarkAccuracy";
  auto load_ret = LoadStepInput(0);
  if (load_ret != RET_OK) {
    return load_ret;
  }
  for (auto &msInput : ms_model_->GetInputs()) {
    switch (msInput.DataType()) {
      case mindspore::DataType::kNumberTypeFloat32:
        PrintInputData<float>(&msInput);
        break;
      case mindspore::DataType::kNumberTypeInt32:
        PrintInputData<int>(&msInput);
        break;
      default:
        MS_LOG(ERROR) << "Datatype " << static_cast<int>(msInput.DataType()) << " is not supported.";
        return RET_ERROR;
    }
  }
  auto status = ms_model_->RunStep(before_call_back_, after_call_back_);
  if (status != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Inference error " << status;
    std::cerr << "Inference error " << status << std::endl;
    return RET_ERROR;
  }

  auto ret = CompareOutput();
  if (ret == RET_TOO_BIG && !enforce_accuracy) {
    MS_LOG(INFO) << "Accuracy Error is big but not enforced";
    std::cout << "Accuracy Error is big but not enforced" << std::endl;
    return RET_OK;
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Compare output error " << ret;
    std::cerr << "Compare output error " << ret << std::endl;
    return ret;
  }
  return RET_OK;
}

int NetTrain::CreateAndRunNetworkForTrain(const std::string &filename, const std::string &bb_filename,
                                          const std::shared_ptr<mindspore::Context> &context,
                                          const std::shared_ptr<TrainCfg> &train_cfg, int epochs) {
  std::string model_name = filename.substr(filename.find_last_of(DELIM_SLASH) + 1);
  Status status;
  if (!bb_filename.empty()) {
    MS_LOG(INFO) << "Build transfer learning model from models files" << filename << " and " << bb_filename;
    std::cout << "Build transfer learning model from model file " << filename << " and " << bb_filename << std::endl;
    mindspore::Graph back_bone_graph;
    mindspore::Graph head_graph;
    status = mindspore::Serialization::Load(bb_filename, mindspore::kMindIR, &back_bone_graph);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "load back bone ms file failed. " << bb_filename;
      return RET_ERROR;
    }
    status = mindspore::Serialization::Load(filename, mindspore::kMindIR, &head_graph);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "load head ms file failed. " << filename;
      return RET_ERROR;
    }
    status = ms_model_->BuildTransferLearning(static_cast<GraphCell>(back_bone_graph),
                                              static_cast<GraphCell>(head_graph), context, train_cfg);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "build transfer learning failed. " << model_name;
      return RET_ERROR;
    }
  } else {
    MS_LOG(INFO) << "Build mindspore model from model file" << filename.c_str();
    std::cout << "Build mindspore model from model file" << filename.c_str() << std::endl;
    mindspore::Graph graph;
    status = mindspore::Serialization::Load(filename, mindspore::kMindIR, &graph);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "load ms file failed. " << model_name;
      return RET_ERROR;
    }
    status = ms_model_->Build(static_cast<GraphCell>(graph), context, train_cfg);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "build transfer learning failed. " << model_name;
      return RET_ERROR;
    }
  }
  if (epochs > 0) {
    if (flags_->virtual_batch_) {
      ms_model_->SetupVirtualBatch(epochs);
    }
    status = ms_model_->SetTrainMode(true);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "set train mode failed. ";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int NetTrain::CreateAndRunNetworkForInference(const std::string &filename,
                                              const std::shared_ptr<mindspore::Context> &context) {
  std::string model_name = filename.substr(filename.find_last_of(DELIM_SLASH) + 1);
  std::string filenamems = filename;
  if (filenamems.substr(filenamems.find_last_of('.') + 1) != "ms") {
    filenamems = filenamems + ".ms";
  }
  MS_LOG(INFO) << "start reading model file " << filenamems.c_str();
  std::cout << "start reading model file " << filenamems.c_str() << std::endl;
  auto status = ms_model_->Build(filenamems, mindspore::kMindIR, context);
  if (status != mindspore::kSuccess) {
    MS_LOG(ERROR) << "ms model build failed. " << model_name;
    return RET_ERROR;
  }
  return RET_OK;
}

void NetTrain::InitMSContext(const std::shared_ptr<Context> &context) {
  context->SetThreadNum(flags_->num_threads_);
  context->SetThreadAffinity(flags_->cpu_bind_mode_);
  auto cpu_context_info = std::make_shared<mindspore::CPUDeviceInfo>();
  cpu_context_info->SetEnableFP16(flags_->enable_fp16_);
  context->MutableDeviceInfo().emplace_back(cpu_context_info);
}

void NetTrain::InitTrainCfg(const std::shared_ptr<TrainCfg> &train_cfg) {
  if (flags_->loss_name_.empty()) {
    return;
  }
  std::vector<std::string> empty_loss_name;
  train_cfg->SetLossName(empty_loss_name);  // clear train_cfg's loss_name
  std::string delimiter = ",";
  size_t pos = 0;
  std::string token;
  while ((pos = flags_->loss_name_.find(delimiter)) != std::string::npos) {
    token = flags_->loss_name_.substr(0, pos);
    flags_->loss_name_.erase(0, pos + delimiter.length());  // change to delim without deletion
    std::vector<std::string> train_cfg_loss_name = train_cfg->GetLossName();
    train_cfg_loss_name.emplace_back(token);
    train_cfg->SetLossName(train_cfg_loss_name);
  }
  if (!(flags_->loss_name_.empty())) {
    std::vector<std::string> train_cfg_loss_name = train_cfg->GetLossName();
    train_cfg_loss_name.emplace_back(flags_->loss_name_);
    train_cfg->SetLossName(train_cfg_loss_name);
  }
}

int NetTrain::CreateAndRunNetwork(const std::string &filename, const std::string &bb_filename, bool is_train,
                                  int epochs, bool check_accuracy) {
  ms_model_ = std::make_shared<Model>();  // reset Model object
  if (ms_model_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create Model object";
    return RET_ERROR;
  }
  auto start_prepare_time = GetTimeUs();

  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "new context failed.";
    return RET_ERROR;
  }
  InitMSContext(context);

  auto train_cfg = std::make_shared<mindspore::TrainCfg>();
  if (train_cfg == nullptr) {
    MS_LOG(ERROR) << "new train cfg failed.";
    return RET_ERROR;
  }
  InitTrainCfg(train_cfg);

  int ret;
  if (is_train) {
    ret = CreateAndRunNetworkForTrain(filename, bb_filename, context, train_cfg, epochs);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CreateAndRunNetworkForTrain failed.";
      return RET_ERROR;
    }
  } else {
    ret = CreateAndRunNetworkForInference(filename, context);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CreateAndRunNetworkForInference failed.";
      return RET_ERROR;
    }
  }

  if (!flags_->resize_dims_.empty()) {
    std::vector<std::vector<int64_t>> resize_dims;
    (void)std::transform(flags_->resize_dims_.begin(), flags_->resize_dims_.end(), std::back_inserter(resize_dims),
                         [&](auto &shapes) { return this->ConverterToInt64Vector<int>(shapes); });
    auto status = ms_model_->Resize(ms_model_->GetInputs(), resize_dims);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "Input tensor resize failed.";
      std::cout << "Input tensor resize failed.";
      return RET_ERROR;
    }
  }

  ms_inputs_for_api_ = ms_model_->GetInputs();
  auto end_prepare_time = GetTimeUs();
  MS_LOG(INFO) << "PrepareTime = " << ((end_prepare_time - start_prepare_time) / kTHOUSAND) << " ms";
  std::cout << "PrepareTime = " << ((end_prepare_time - start_prepare_time) / kTHOUSAND) << " ms" << std::endl;
  // Load input
  MS_LOG(INFO) << "Load input data";
  auto status = LoadInput();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Load input data error";
    std::cout << "Load input data error" << std::endl;
    return status;
  }

  if ((epochs > 0) && is_train) {
    status = MarkPerformance();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run MarkPerformance error: " << status;
      std::cout << "Run MarkPerformance error: " << status << std::endl;
      return status;
    }
    SaveModels();  // save file if flags are on
  }
  if (!flags_->data_file_.empty()) {
    auto res = ms_model_->SetTrainMode(false);
    if (res != mindspore::kSuccess) {
      MS_LOG(ERROR) << "set eval mode failed. ";
      return RET_ERROR;
    }

    status = MarkAccuracy(check_accuracy);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run MarkAccuracy error: " << status;
      std::cout << "Run MarkAccuracy error: " << status << std::endl;
      return status;
    }
  }
  return RET_OK;
}

int NetTrain::SaveModels() {
  if (!flags_->export_file_.empty()) {
    if (flags_->bb_model_file_.empty()) {
      auto status = mindspore::Serialization::ExportModel(*ms_model_, mindspore::kMindIR, flags_->export_file_ + "_qt",
                                                          mindspore::QuantizationType::kWeightQuant, false);
      if (status != mindspore::kSuccess) {
        MS_LOG(ERROR) << "Export quantized model error " << flags_->export_file_ + "_qt";
        std::cout << "Export quantized model error " << flags_->export_file_ + "_qt" << std::endl;
        return RET_ERROR;
      }
    }
    auto status = mindspore::Serialization::ExportModel(*ms_model_, mindspore::kMindIR, flags_->export_file_,
                                                        QuantizationType::kNoQuant, false);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "Export non quantized model error " << flags_->export_file_;
      std::cout << "Export non quantized model error " << flags_->export_file_ << std::endl;
      return RET_ERROR;
    }
  }
  if (!flags_->inference_file_.empty()) {
    auto status = mindspore::Serialization::ExportModel(*ms_model_, mindspore::kMindIR, flags_->inference_file_ + "_qt",
                                                        QuantizationType::kWeightQuant, true);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "Export quantized inference model error " << flags_->inference_file_ + "_qt";
      std::cout << "Export quantized inference model error " << flags_->inference_file_ + "_qt" << std::endl;
      return RET_ERROR;
    }

    auto tick = GetTimeUs();
    status = mindspore::Serialization::ExportModel(*ms_model_, mindspore::kMindIR, flags_->inference_file_,
                                                   QuantizationType::kNoQuant, true);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "Export non quantized inference model error " << flags_->inference_file_;
      std::cout << "Export non quantized inference model error " << flags_->inference_file_ << std::endl;
      return RET_ERROR;
    }
    std::cout << "ExportInference() execution time is " << GetTimeUs() - tick << "us\n";
  }
  return RET_OK;
}

int NetTrain::InitDumpTensorDataCallbackParameter() {
  // before callback
  before_call_back_ = [&](const std::vector<mindspore::MSTensor> &before_inputs,
                          const std::vector<mindspore::MSTensor> &before_outputs, const MSCallBackParam &call_param) {
    auto dump_mode = dump_cfg_json_[dump::kSettings][dump::kMode].get<int>();
    auto input_output_mode = dump_cfg_json_[dump::kSettings][dump::kInputOutput].get<int>();
    auto kernels = dump_cfg_json_[dump::kSettings][dump::kKernels].get<std::vector<std::string>>();
    if (dump_mode == 0 || std::find(kernels.begin(), kernels.end(), call_param.node_name) != kernels.end()) {
      if (input_output_mode == 0 || input_output_mode == 1) {
        for (size_t i = 0; i < before_inputs.size(); i++) {
          auto ms_tensor = before_inputs.at(i);
          auto file_name = GenerateOutputFileName(&ms_tensor, call_param.node_name, "input", i);
          auto abs_file_path = dump_file_output_dir_ + "/" + file_name;
          if (WriteToBin(abs_file_path, ms_tensor.MutableData(), ms_tensor.DataSize()) != RET_OK) {  // save to file
            MS_LOG(ERROR) << "write tensor data to file failed.";
            return false;
          }
        }
      }
    }
    return true;
  };

  // after callback
  after_call_back_ = [&](const std::vector<mindspore::MSTensor> &after_inputs,
                         const std::vector<mindspore::MSTensor> &after_outputs, const MSCallBackParam &call_param) {
    auto dump_mode = dump_cfg_json_[dump::kSettings][dump::kMode].get<int>();
    auto input_output_mode = dump_cfg_json_[dump::kSettings][dump::kInputOutput].get<int>();
    auto kernels = dump_cfg_json_[dump::kSettings][dump::kKernels].get<std::vector<std::string>>();
    if (dump_mode == kDumpInputsAndOutputs ||
        std::find(kernels.begin(), kernels.end(), call_param.node_name) != kernels.end()) {
      if (input_output_mode == kDumpInputsAndOutputs || input_output_mode == kDumpOutputs) {
        for (size_t i = 0; i < after_outputs.size(); i++) {
          auto ms_tensor = after_outputs.at(i);
          auto file_name = GenerateOutputFileName(&ms_tensor, call_param.node_name, "output", i);
          auto abs_file_path = dump_file_output_dir_ + "/" + file_name;
          if (WriteToBin(abs_file_path, ms_tensor.MutableData(), ms_tensor.DataSize()) != RET_OK) {  // save to file
            MS_LOG(ERROR) << "write tensor data to file failed.";
            return false;
          }
        }
      }
    }
    return true;
  };
  return RET_OK;
}

int NetTrain::InitTimeProfilingCallbackParameter() {
  // before callback
  before_call_back_ = [&](const std::vector<mindspore::MSTensor> &before_inputs,
                          const std::vector<mindspore::MSTensor> &before_outputs,
                          const mindspore::MSCallBackParam &callParam) {
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
        std::fill(reinterpret_cast<int8_t *>(tensor.MutableData()),
                  reinterpret_cast<int8_t *>(tensor.MutableData()) + tensor.DataSize(), 0);
      }
    }
    return true;
  };

  // after callback
  after_call_back_ = [&](const std::vector<mindspore::MSTensor> &after_inputs,
                         const std::vector<mindspore::MSTensor> &after_outputs,
                         const mindspore::MSCallBackParam &call_param) {
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
        auto ms_tensor = after_inputs.at(i);
        CheckSum(&ms_tensor, call_param.node_type, i, "in");
      }
      for (size_t i = 0; i < after_outputs.size(); i++) {
        auto ms_tensor = after_outputs.at(i);
        CheckSum(&ms_tensor, call_param.node_type, i, "out");
      }
      std::cout << std::endl;
    }
    return true;
  };
  return RET_OK;
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

    stringBuf[kField1] = std::to_string(iter.second.second / flags_->epochs_);
    len = stringBuf[kField1].length();
    if (len > columnLenMax.at(kField1)) {
      columnLenMax.at(kField1) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[kField1]);

    stringBuf[kField2] = std::to_string(iter.second.second / op_cost_total_);
    len = stringBuf[kField2].length();
    if (len > columnLenMax.at(kField2)) {
      columnLenMax.at(kField2) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[kField2]);

    stringBuf[kField3] = std::to_string(iter.second.first);
    len = stringBuf[kField3].length();
    if (len > columnLenMax.at(kField3)) {
      columnLenMax.at(kField3) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[kField3]);

    stringBuf[kField4] = std::to_string(iter.second.second);
    len = stringBuf[kField4].length();
    if (len > columnLenMax.at(kField4)) {
      columnLenMax.at(kField4) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[kField4]);

    rows.push_back(columns);
  }

  printf("-------------------------------------------------------------------------\n");
  for (int i = 0; i < kFieldsToPrint; i++) {
    auto printBuf = title[i];
    if (printBuf.size() > columnLenMax.at(i)) {
      columnLenMax.at(i) = printBuf.size();
    }
    printBuf.resize(columnLenMax.at(i), ' ');
    printf("%s\t", printBuf.c_str());
  }
  printf("\n");
  for (auto &row : rows) {
    for (int j = 0; j < kFieldsToPrint; j++) {
      auto printBuf = row[j];
      printBuf.resize(columnLenMax.at(j), ' ');
      printf("%s\t", printBuf.c_str());
    }
    printf("\n");
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
