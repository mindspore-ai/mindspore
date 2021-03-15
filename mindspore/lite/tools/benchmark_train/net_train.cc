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
#include <utility>
#include "src/common/common.h"
#include "include/ms_tensor.h"
#include "include/context.h"
#include "src/runtime/runtime_api.h"
#include "include/version.h"

namespace mindspore {
namespace lite {
static const char *DELIM_COLON = ":";
static const char *DELIM_COMMA = ",";
static const char *DELIM_SLASH = "/";

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
  std::unique_ptr<float[]> buf((new (std::nothrow) float[*size / sizeof(float) + 1]));
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

int NetTrain::GenerateInputData() {
  for (auto tensor : ms_inputs_) {
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
      MS_LOG(ERROR) << "GenerateRandomData for inTensor failed:" << status;
      return status;
    }
  }
  return RET_OK;
}

int NetTrain::LoadInput() {
  if (flags_->in_data_file_.empty()) {
    auto status = GenerateInputData();
    if (status != RET_OK) {
      std::cerr << "Generate input data error " << status << std::endl;
      MS_LOG(ERROR) << "Generate input data error " << status;
      return status;
    }
  } else {
    auto status = ReadInputFile();
    if (status != RET_OK) {
      std::cerr << "ReadInputFile error, " << status << std::endl;
      MS_LOG(ERROR) << "ReadInputFile error, " << status;
      return status;
    }
  }
  return RET_OK;
}

int NetTrain::ReadInputFile() {
  if (ms_inputs_.empty()) {
    return RET_OK;
  }

  if (this->flags_->in_data_type_ == kImage) {
    MS_LOG(ERROR) << "Not supported image input";
    return RET_ERROR;
  } else {
    if (ms_inputs_.size() > flags_->input_data_list_.size()) {
      MS_LOG(ERROR) << "missing input files expecting " << ms_inputs_.size() << ",got "
                    << flags_->input_data_list_.size();
      return RET_ERROR;
    }
    for (size_t i = 0; i < ms_inputs_.size(); i++) {
      auto cur_tensor = ms_inputs_.at(i);
      MS_ASSERT(cur_tensor != nullptr);
      size_t size;
      char *bin_buf = ReadFile(flags_->input_data_list_[i].c_str(), &size);
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

int NetTrain::CompareOutput() {
  std::cout << "================ Comparing Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  bool has_error = false;
  auto tensors_list = session_->GetOutputs();
  if (tensors_list.empty()) {
    MS_LOG(ERROR) << "Cannot find output tensors, get model output failed";
    return RET_ERROR;
  }
  mindspore::tensor::MSTensor *tensor = nullptr;
  int i = 1;
  for (auto it = tensors_list.begin(); it != tensors_list.end(); ++it) {
    tensor = session_->GetOutputByTensorName(it->first);
    std::cout << "output is tensor " << it->first << "\n";
    auto outputs = tensor->MutableData();
    size_t size;
    std::string output_file = flags_->data_file_ + std::to_string(i) + ".bin";
    auto *bin_buf = ReadFileBuf(output_file.c_str(), &size);
    if (bin_buf == nullptr) {
      MS_LOG(ERROR) << "ReadFile return nullptr";
      return RET_ERROR;
    }
    if (size != tensor->Size()) {
      MS_LOG(ERROR) << "Output buffer and output file differ by size. Tensor size: " << tensor->Size()
                    << ", read size: " << size;
      return RET_ERROR;
    }
    float bias = CompareData<float>(bin_buf, tensor->ElementsNum(), reinterpret_cast<float *>(outputs));
    if (bias >= 0) {
      total_bias += bias;
      total_size++;
    } else {
      has_error = true;
      break;
    }
    i++;
    delete[] bin_buf;
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
      MS_LOG(ERROR) << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%";
      std::cerr << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%" << std::endl;
      return RET_ERROR;
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
    session_->BindThread(true);
    auto start = GetTimeUs();
    auto status =
      flags_->time_profiling_ ? session_->RunGraph(before_call_back_, after_call_back_) : session_->RunGraph();
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
    session_->BindThread(false);
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

int NetTrain::MarkAccuracy() {
  MS_LOG(INFO) << "MarkAccuracy";
  std::cout << "MarkAccuracy" << std::endl;
  for (auto &msInput : ms_inputs_) {
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
  session_->Eval();

  auto status = session_->RunGraph();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Inference error " << status;
    std::cerr << "Inference error " << status << std::endl;
    return status;
  }

  status = CompareOutput();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Compare output error " << status;
    std::cerr << "Compare output error " << status << std::endl;
    return status;
  }
  return RET_OK;
}

int NetTrain::RunExportedNet() {
  auto start_prepare_time = GetTimeUs();
  // Load graph
  std::string model_name = flags_->export_file_.substr(flags_->export_file_.find_last_of(DELIM_SLASH) + 1);

  MS_LOG(INFO) << "start reading exported model file";
  std::cout << "start reading exported model file" << std::endl;
  auto context = std::make_shared<Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running " << model_name.c_str();
    std::cerr << "New context failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }

  if (flags_->cpu_bind_mode_ == 2) {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = MID_CPU;
  } else if (flags_->cpu_bind_mode_ == 1) {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = HIGHER_CPU;
  } else {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = NO_BIND;
  }

  context->thread_num_ = flags_->num_threads_;
  session_ = session::TrainSession::CreateSession(flags_->export_file_.c_str(), context.get());
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "ExportedFile CreateSession failed while running " << model_name.c_str();
    std::cout << "CreateSession failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }
  ms_inputs_ = session_->GetInputs();
  auto end_prepare_time = GetTimeUs();
  MS_LOG(INFO) << "Exported model PrepareTime = " << (end_prepare_time - start_prepare_time) / 1000 << " ms";
  std::cout << "Exported model PrepareTime = " << (end_prepare_time - start_prepare_time) / 1000 << " ms" << std::endl;

  // Load input
  MS_LOG(INFO) << "start generate input data";
  auto status = LoadInput();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Generate input data error";
    return status;
  }

  if (!flags_->data_file_.empty()) {
    MS_LOG(INFO) << "Check accuracy for exported model";
    std::cout << "Check accuracy for exported model " << std::endl;
    status = MarkAccuracy();
    for (auto &data : data_) {
      data.second->shape.clear();
      data.second->data.clear();
      delete data.second;
    }
    data_.clear();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run MarkAccuracy on exported model error: " << status;
      std::cout << "Run MarkAccuracy on exported model error: " << status << std::endl;
      return status;
    }
  }
  return RET_OK;
}

int NetTrain::RunNetTrain() {
  auto start_prepare_time = GetTimeUs();
  // Load graph
  std::string model_name = flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1);

  MS_LOG(INFO) << "start reading model file";
  std::cout << "start reading model file" << std::endl;
  auto context = std::make_shared<Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running " << model_name.c_str();
    std::cerr << "New context failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }

  if (flags_->cpu_bind_mode_ == 2) {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = MID_CPU;
  } else if (flags_->cpu_bind_mode_ == 1) {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = HIGHER_CPU;
  } else {
    context->device_list_[0].device_info_.cpu_device_info_.cpu_bind_mode_ = NO_BIND;
  }
  context->device_list_[0].device_info_.cpu_device_info_.enable_float16_ = flags_->enable_fp16_;
  layer_checksum_ = flags_->layer_checksum_;
  context->thread_num_ = flags_->num_threads_;
  session_ = session::TrainSession::CreateSession(flags_->model_file_.c_str(), context.get());
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "RunNetTrain CreateSession failed while running " << model_name.c_str();
    std::cout << "RunNetTrain CreateSession failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }

  session_->Train();

  ms_inputs_ = session_->GetInputs();
  auto end_prepare_time = GetTimeUs();
  MS_LOG(INFO) << "PrepareTime = " << (end_prepare_time - start_prepare_time) / 1000 << " ms";
  std::cout << "PrepareTime = " << (end_prepare_time - start_prepare_time) / 1000 << " ms" << std::endl;

  // Load input
  MS_LOG(INFO) << "start generate input data";
  auto status = LoadInput();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Generate input data error";
    return status;
  }
  if (flags_->epochs_ > 0) {
    status = MarkPerformance();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run MarkPerformance error: " << status;
      std::cout << "Run MarkPerformance error: " << status << std::endl;
      return status;
    }
  }
  if (!flags_->data_file_.empty()) {
    status = MarkAccuracy();
    for (auto &data : data_) {
      data.second->shape.clear();
      data.second->data.clear();
      delete data.second;
    }
    data_.clear();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run MarkAccuracy error: " << status;
      std::cout << "Run MarkAccuracy error: " << status << std::endl;
      return status;
    }
  }
  if (!flags_->export_file_.empty()) {
    auto ret = session_->SaveToFile(flags_->export_file_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "SaveToFile error";
      std::cout << "Run SaveToFile error";
      return RET_ERROR;
    }
    delete session_;
    status = RunExportedNet();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run Exported model error: " << status;
      std::cout << "Run Exported model error: " << status << std::endl;
      return status;
    }
  }
  return RET_OK;
}

void NetTrainFlags::InitInputDataList() {
  char *saveptr1 = nullptr;
  char *input_list = new char[this->in_data_file_.length() + 1];
  snprintf(input_list, this->in_data_file_.length() + 1, "%s", this->in_data_file_.c_str());
  const char *split_c = ",";
  char *cur_input = strtok_r(input_list, split_c, &saveptr1);
  while (cur_input != nullptr) {
    input_data_list_.emplace_back(cur_input);
    cur_input = strtok_r(nullptr, split_c, &saveptr1);
  }
  delete[] input_list;
}

void NetTrainFlags::InitResizeDimsList() {
  std::string content;
  content = this->resize_dims_in_;
  std::vector<int64_t> shape;
  auto shape_strs = StringSplit(content, std::string(DELIM_COLON));
  for (const auto &shape_str : shape_strs) {
    shape.clear();
    auto dim_strs = StringSplit(shape_str, std::string(DELIM_COMMA));
    std::cout << "Resize Dims: ";
    for (const auto &dim_str : dim_strs) {
      std::cout << dim_str << " ";
      shape.emplace_back(static_cast<int64_t>(std::stoi(dim_str)));
    }
    std::cout << std::endl;
    this->resize_dims_.emplace_back(shape);
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
    if (layer_checksum_) {
      float *output = reinterpret_cast<float *>(after_outputs.at(0)->MutableData());
      float sum = 0;
      for (int i = 0; i < after_outputs.at(0)->ElementsNum(); i++) sum += output[i];
      std::cout << call_param.node_type << " shape= " << after_outputs.at(0)->shape() << " sum=" << sum << "\n";
    }
    return true;
  };
  return RET_OK;
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
  flags_->InitInputDataList();
  flags_->InitResizeDimsList();
  if (!flags_->resize_dims_.empty() && flags_->resize_dims_.size() != flags_->input_data_list_.size()) {
    MS_LOG(ERROR) << "Size of input resizeDims should be equal to size of input inDataPath";
    std::cerr << "Size of input resizeDims should be equal to size of input inDataPath" << std::endl;
    return RET_ERROR;
  }

  if (flags_->time_profiling_) {
    auto status = InitCallbackParameter();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Init callback Parameter failed.";
      std::cerr << "Init callback Parameter failed." << std::endl;
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int NetTrain::PrintResult(const std::vector<std::string> &title,
                          const std::map<std::string, std::pair<int, float>> &result) {
  std::vector<size_t> columnLenMax(5);
  std::vector<std::vector<std::string>> rows;

  for (auto &iter : result) {
    char stringBuf[5][100] = {};
    std::vector<std::string> columns;
    size_t len;

    len = iter.first.size();
    if (len > columnLenMax.at(0)) {
      columnLenMax.at(0) = len + 4;
    }
    columns.push_back(iter.first);

    len = snprintf(stringBuf[1], sizeof(stringBuf[1]), "%f", iter.second.second / flags_->epochs_);
    if (len > columnLenMax.at(1)) {
      columnLenMax.at(1) = len + 4;
    }
    columns.emplace_back(stringBuf[1]);

    len = snprintf(stringBuf[2], sizeof(stringBuf[2]), "%f", iter.second.second / op_cost_total_);
    if (len > columnLenMax.at(2)) {
      columnLenMax.at(2) = len + 4;
    }
    columns.emplace_back(stringBuf[2]);

    len = snprintf(stringBuf[3], sizeof(stringBuf[3]), "%d", iter.second.first);
    if (len > columnLenMax.at(3)) {
      columnLenMax.at(3) = len + 4;
    }
    columns.emplace_back(stringBuf[3]);

    len = snprintf(stringBuf[4], sizeof(stringBuf[4]), "%f", iter.second.second);
    if (len > columnLenMax.at(4)) {
      columnLenMax.at(4) = len + 4;
    }
    columns.emplace_back(stringBuf[4]);

    rows.push_back(columns);
  }

  printf("-------------------------------------------------------------------------\n");
  for (int i = 0; i < 5; i++) {
    auto printBuf = title[i];
    if (printBuf.size() > columnLenMax.at(i)) {
      columnLenMax.at(i) = printBuf.size();
    }
    printBuf.resize(columnLenMax.at(i), ' ');
    printf("%s\t", printBuf.c_str());
  }
  printf("\n");
  for (size_t i = 0; i < rows.size(); i++) {
    for (int j = 0; j < 5; j++) {
      auto printBuf = rows[i][j];
      printBuf.resize(columnLenMax.at(j), ' ');
      printf("%s\t", printBuf.c_str());
    }
    printf("\n");
  }
  return RET_OK;
}

NetTrain::~NetTrain() {
  for (auto iter : this->data_) {
    delete (iter.second);
  }
  this->data_.clear();
  if (session_ != nullptr) delete (session_);
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
