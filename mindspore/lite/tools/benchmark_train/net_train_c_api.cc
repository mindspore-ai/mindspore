/**
 * Copyright 2023-2023 Huawei Technologies Co., Ltd
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

#include "tools/benchmark_train/net_train_c_api.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace lite {
uint64_t g_op_begin_ = 0;
int g_op_call_times_total_ = 0;
float g_op_cost_total_ = 0.0f;
std::map<std::string, std::pair<int, float>> g_c_op_times_by_type_;
std::map<std::string, std::pair<int, float>> g_c_op_times_by_name_;

int NetTrainCApi::GenerateInputData() {
  for (size_t i = 0; i < ms_inputs_for_api_.handle_num; i++) {
    MSTensorHandle tensor = ms_inputs_for_api_.handle_list[i];
    auto data_type = MSTensorGetDataType(tensor);
    if (data_type == kMSDataTypeObjectTypeString) {
      MS_LOG(ERROR) << "Unsupported OH_AI_DATATYPE_OBJECTTYPE_STRING";
      return RET_ERROR;
    } else {
      (void)GenerateRandomData(static_cast<mindspore::MSTensor *>(tensor));
    }
  }
  return RET_OK;
}

int NetTrainCApi::SaveModels() {
  if (!flags_->export_file_.empty()) {
    if (flags_->bb_model_file_.empty()) {
      auto status = MSExportModel(ms_model_, kMSModelTypeMindIR, (flags_->export_file_ + "_qt").c_str(),
                                  kMSWEIGHT_QUANT, false, nullptr, 0);
      if (status != kMSStatusSuccess) {
        MS_LOG(ERROR) << "Export quantized model error " << flags_->export_file_ + "_qt";
        std::cout << "Export quantized model error " << flags_->export_file_ + "_qt" << std::endl;
        return RET_ERROR;
      }
    }
    auto status =
      MSExportModel(ms_model_, kMSModelTypeMindIR, (flags_->export_file_).c_str(), kMSNO_QUANT, false, nullptr, 0);

    if (status != kMSStatusSuccess) {
      MS_LOG(ERROR) << "Export non quantized model error " << flags_->export_file_;
      std::cout << "Export non quantized model error " << flags_->export_file_ << std::endl;
      return RET_ERROR;
    }
  }
  if (!flags_->inference_file_.empty()) {
    auto status = MSExportModel(ms_model_, kMSModelTypeMindIR, (flags_->inference_file_ + "_qt").c_str(),
                                kMSWEIGHT_QUANT, true, nullptr, 0);
    if (status != kMSStatusSuccess) {
      MS_LOG(ERROR) << "Export quantized inference model error " << flags_->inference_file_ + "_qt";
      std::cout << "Export quantized inference model error " << flags_->inference_file_ + "_qt" << std::endl;
      return RET_ERROR;
    }

    auto tick = GetTimeUs();
    status =
      MSExportModel(ms_model_, kMSModelTypeMindIR, (flags_->inference_file_).c_str(), kMSNO_QUANT, true, nullptr, 0);
    if (status != kMSStatusSuccess) {
      MS_LOG(ERROR) << "Export non quantized inference model error " << flags_->inference_file_;
      std::cout << "Export non quantized inference model error " << flags_->inference_file_ << std::endl;
      return RET_ERROR;
    }
    std::cout << "ExportInference() execution time is " << GetTimeUs() - tick << "us\n";
  }
  return RET_OK;
}

int NetTrainCApi::LoadStepInput(size_t step) {
  if (step >= batch_num_) {
    auto cur_batch = step + 1;
    MS_LOG(ERROR) << "Max input Batch is:" << batch_num_ << " but got batch :" << cur_batch;
    return RET_ERROR;
  }
  for (size_t i = 0; i < ms_inputs_for_api_.handle_num; i++) {
    MSTensorHandle cur_tensor = ms_inputs_for_api_.handle_list[i];
    MS_ASSERT(cur_tensor != nullptr);
    auto tensor_data_size = MSTensorGetDataSize(cur_tensor);
    auto input_data = MSTensorGetMutableData(cur_tensor);
    MS_ASSERT(input_data != nullptr);
    memcpy_s(input_data, tensor_data_size, inputs_buf_[i].get() + step * tensor_data_size, tensor_data_size);
  }
  return RET_OK;
}

int NetTrainCApi::ReadInputFile() {
  if (this->flags_->in_data_type_ == lite::kImage) {
    MS_LOG(ERROR) << "Unsupported image input";
    return RET_ERROR;
  } else {
    for (size_t i = 0; i < ms_inputs_for_api_.handle_num; i++) {
      MSTensorHandle tensor = ms_inputs_for_api_.handle_list[i];
      MS_ASSERT(tensor != nullptr);
      size_t size;
      std::string file_name = flags_->in_data_file_ + std::to_string(i + 1) + ".bin";
      auto bin_buf = lite::ReadFile(file_name.c_str(), &size);
      if (bin_buf == nullptr) {
        MS_LOG(ERROR) << "ReadFile failed";
        return RET_ERROR;
      }
      auto tensor_data_size = MSTensorGetDataSize(tensor);
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

int NetTrainCApi::InitDumpTensorDataCallbackParameter() {
  MS_LOG(ERROR) << "Unsupported feature.";
  return RET_ERROR;
}

int NetTrainCApi::InitTimeProfilingCallbackParameter() {
  before_call_back_ = TimeProfilingBeforeCallback;
  after_call_back_ = TimeProfilingAfterCallback;
  return RET_OK;
}

int NetTrainCApi::InitMSContext() {
  context_ = MSContextCreate();
  if (context_ == nullptr) {
    MS_LOG(INFO) << "OH_AI_ContextCreate failed";
    return RET_ERROR;
  }
  MSContextSetThreadNum(context_, flags_->num_threads_);
  MSContextSetThreadAffinityMode(context_, flags_->cpu_bind_mode_);

  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSDeviceInfoSetEnableFP16(cpu_device_info, flags_->enable_fp16_);
  MSContextAddDeviceInfo(context_, cpu_device_info);
  return RET_OK;
}

char **NetTrainCApi::TransStrVectorToCharArrays(const std::vector<std::string> &s) {
  char **char_arr = static_cast<char **>(malloc(s.size() * sizeof(char *)));
  for (size_t i = 0; i < s.size(); i++) {
    char_arr[i] = static_cast<char *>(malloc((s[i].size() + 1)));
    strcpy_s(char_arr[i], s[i].size() + 1, s[i].c_str());
  }
  return char_arr;
}

std::vector<std::string> NetTrainCApi::TransCharArraysToStrVector(char **c, const size_t &num) {
  std::vector<std::string> str;
  for (size_t i = 0; i < num; i++) {
    str.push_back(std::string(c[i]));
  }
  return str;
}

void NetTrainCApi::InitTrainCfg() {
  if (flags_->loss_name_.empty()) {
    return;
  }

  std::string delimiter = ",";
  size_t pos = 0;
  std::string token;
  train_cfg_ = MSTrainCfgCreate();
  size_t num = 0;
  std::vector<std::string> train_cfg_loss_name;
  MSTrainCfgSetLossName(train_cfg_, nullptr, train_cfg_loss_name.size());
  while ((pos = flags_->loss_name_.find(delimiter)) != std::string::npos) {
    token = flags_->loss_name_.substr(0, pos);
    flags_->loss_name_.erase(0, pos + delimiter.length());  // change to delim without deletion
    char **name = MSTrainCfgGetLossName(train_cfg_, &num);
    train_cfg_loss_name = TransCharArraysToStrVector(name, num);
    train_cfg_loss_name.push_back(token);
    char **loss_name = TransStrVectorToCharArrays(train_cfg_loss_name);
    MSTrainCfgSetLossName(train_cfg_, const_cast<const char **>(loss_name), train_cfg_loss_name.size());
    for (size_t i = 0; i < train_cfg_loss_name.size(); i++) {
      free(loss_name[i]);
    }
    free(loss_name);
    for (size_t i = 0; i < num; i++) {
      free(name[i]);
    }
    free(name);
  }
  if (!(flags_->loss_name_.empty())) {
    char **name = MSTrainCfgGetLossName(train_cfg_, &num);
    train_cfg_loss_name = TransCharArraysToStrVector(name, num);
    train_cfg_loss_name.push_back(flags_->loss_name_);
    char **loss_name = TransStrVectorToCharArrays(train_cfg_loss_name);
    MSTrainCfgSetLossName(train_cfg_, const_cast<const char **>(loss_name), train_cfg_loss_name.size());
    for (size_t i = 0; i < train_cfg_loss_name.size(); i++) {
      free(loss_name[i]);
    }
    free(loss_name);
    for (size_t i = 0; i < num; i++) {
      free(name[i]);
    }
    free(name);
  }
}

int NetTrainCApi::CreateAndRunNetworkForInference(const std::string &filename, const MSContextHandle &context) {
  std::string model_name = filename.substr(filename.find_last_of(DELIM_SLASH) + 1);
  std::string filenamems = filename;
  if (filenamems.substr(filenamems.find_last_of('.') + 1) != "ms") {
    filenamems = filenamems + ".ms";
  }
  MS_LOG(INFO) << "start reading model file " << filenamems.c_str();
  std::cout << "start reading model file " << filenamems.c_str() << std::endl;
  auto status =
    MSModelBuildFromFile(ms_model_, filenamems.c_str(), static_cast<MSModelType>(mindspore::kMindIR), context);
  if (status != kMSStatusSuccess) {
    MS_LOG(ERROR) << "ms model build failed. " << model_name;
    return RET_ERROR;
  }
  return RET_OK;
}

int NetTrainCApi::CreateAndRunNetworkForTrain(const std::string &filename, const std::string &bb_filename,
                                              const MSContextHandle &context, const MSTrainCfgHandle &train_cfg,
                                              int epochs) {
  std::string model_name = filename.substr(filename.find_last_of(DELIM_SLASH) + 1);
  MSStatus status;
  if (!bb_filename.empty()) {
    MS_LOG(ERROR) << "build transfer learning not supported. " << model_name;
    return RET_ERROR;
  } else {
    MS_LOG(INFO) << "Build mindspore model from model file" << filename.c_str();
    std::cout << "Build mindspore model from model file" << filename.c_str() << std::endl;
    status = MSTrainModelBuildFromFile(ms_model_, filename.c_str(), kMSModelTypeMindIR, context, train_cfg);
    if (status != kMSStatusSuccess) {
      MS_LOG(ERROR) << "build transfer learning failed. " << model_name;
      return RET_ERROR;
    }
  }
  if (epochs > 0) {
    if (flags_->virtual_batch_) {
      MSModelSetupVirtualBatch(ms_model_, epochs, -1.0f, -1.0f);
    }
    status = MSModelSetTrainMode(ms_model_, true);
    if (status != kMSStatusSuccess) {
      MS_LOG(ERROR) << "set train mode failed. ";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int NetTrainCApi::CompareOutput() {
  std::cout << "================ Comparing Forward Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  bool has_error = false;
  auto output_tensors_handle = MSModelGetOutputs(ms_model_);

  std::vector<mindspore::MSTensor> output_tensors;
  for (size_t i = 0; i < output_tensors_handle.handle_num; i++) {
    output_tensors.push_back(*static_cast<mindspore::MSTensor *>(output_tensors_handle.handle_list[i]));
  }
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

int NetTrainCApi::MarkPerformance() {
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
      auto status = MSRunStep(ms_model_, before_call_back_, after_call_back_);
      if (status != kMSStatusSuccess) {
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
    PrintResult(per_op_name, g_c_op_times_by_name_);
    PrintResult(per_op_type, g_c_op_times_by_type_);
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

int NetTrainCApi::MarkAccuracy(bool enforce_accuracy) {
  MS_LOG(INFO) << "MarkAccuracy";
  auto load_ret = LoadStepInput(0);
  if (load_ret != RET_OK) {
    return load_ret;
  }
  auto status = PrintInputData();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "PrintInputData failed, ret: " << status;
    return status;
  }
  status = MSRunStep(ms_model_, before_call_back_, after_call_back_);
  if (status != kMSStatusSuccess) {
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

int NetTrainCApi::CreateAndRunNetwork(const std::string &filename, const std::string &bb_filename, bool is_train,
                                      int epochs, bool check_accuracy) {
  auto start_prepare_time = GetTimeUs();

  int ret = InitMSContext();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitContext failed, ret: " << ret;
    return ret;
  }

  InitTrainCfg();
  ms_model_ = MSModelCreate();

  if (is_train) {
    ret = CreateAndRunNetworkForTrain(filename, bb_filename, context_, train_cfg_, epochs);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CreateAndRunNetworkForTrain failed.";
      return RET_ERROR;
    }
  } else {
    ret = CreateAndRunNetworkForInference(filename, context_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CreateAndRunNetworkForInference failed.";
      return RET_ERROR;
    }
  }

  ms_inputs_for_api_ = MSModelGetInputs(ms_model_);
  if (ms_inputs_for_api_.handle_list == nullptr) {
    MS_LOG(ERROR) << "OH_AI_ModelGetInputs failed, ret: ";
    return RET_ERROR;
  }

  if (!flags_->resize_dims_.empty()) {
    std::vector<MSShapeInfo> shape_infos;
    std::transform(flags_->resize_dims_.begin(), flags_->resize_dims_.end(), std::back_inserter(shape_infos),
                   [&](auto &shapes) {
                     MSShapeInfo shape_info;
                     shape_info.shape_num = shapes.size();
                     for (size_t i = 0; i < shape_info.shape_num; i++) {
                       shape_info.shape[i] = shapes[i];
                     }
                     return shape_info;
                   });
    auto status = MSModelResize(ms_model_, ms_inputs_for_api_, shape_infos.data(), shape_infos.size());
    if (status != kMSStatusSuccess) {
      MS_LOG(ERROR) << "Input tensor resize failed.";
      std::cout << "Input tensor resize failed.";
      return RET_ERROR;
    }
  }

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
    auto res = MSModelSetTrainMode(ms_model_, false);
    if (res != kMSStatusSuccess) {
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

int NetTrainCApi::PrintInputData() {
  constexpr int64_t kPrintDataNum = 20;
  for (size_t i = 0; i < ms_inputs_for_api_.handle_num; i++) {
    auto input = ms_inputs_for_api_.handle_list[i];
    std::cout << "InData" << i << ": ";
    auto data_type = static_cast<TypeId>(MSTensorGetDataType(input));
    if (data_type == TypeId::kObjectTypeString) {
      MS_LOG(ERROR) << "Unsupported OH_AI_DATATYPE_OBJECTTYPE_STRING.";
      return RET_ERROR;
    }
    auto tensor_data = MSTensorGetData(input);
    size_t print_num = std::min(MSTensorGetElementNum(input), kPrintDataNum);
    for (size_t j = 0; j < print_num; j++) {
      if (data_type == TypeId::kNumberTypeFloat32 || data_type == TypeId::kNumberTypeFloat) {
        std::cout << static_cast<const float *>(tensor_data)[j] << " ";
      } else if (data_type == TypeId::kNumberTypeInt8) {
        std::cout << static_cast<const int8_t *>(tensor_data)[j] << " ";
      } else if (data_type == TypeId::kNumberTypeUInt8) {
        std::cout << static_cast<const uint8_t *>(tensor_data)[j] << " ";
      } else if (data_type == TypeId::kNumberTypeInt32) {
        std::cout << static_cast<const int32_t *>(tensor_data)[j] << " ";
      } else if (data_type == TypeId::kNumberTypeInt64) {
        std::cout << static_cast<const int64_t *>(tensor_data)[j] << " ";
      } else if (data_type == TypeId::kNumberTypeBool) {
        std::cout << static_cast<const bool *>(tensor_data)[j] << " ";
      } else {
        MS_LOG(ERROR) << "Datatype: " << data_type << " is not supported.";
        return RET_ERROR;
      }
    }
    std::cout << std::endl;
  }
  return RET_OK;
}

int NetTrainCApi::PrintResult(const std::vector<std::string> &title,
                              const std::map<std::string, std::pair<int, float>> &result) {
  std::vector<size_t> columnLenMax(kFieldsToPrint);
  std::vector<std::vector<std::string>> rows;

  for (auto &iter : result) {
    std::string stringBuf[kFieldsToPrint];
    std::vector<std::string> columns;
    size_t len = 0;
    int index = 0;
    len = iter.first.size();
    if (len > columnLenMax.at(index)) {
      columnLenMax.at(index) = len + kPrintOffset;
    }
    columns.push_back(iter.first);

    index++;
    if (title[0] == "opName") {
      stringBuf[index] = std::to_string(iter.second.second / flags_->epochs_);
    } else {
      stringBuf[index] = std::to_string(iter.second.second / iter.second.first);
    }
    len = stringBuf[index].length();
    if (len > columnLenMax.at(index)) {
      columnLenMax.at(index) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[index]);

    index++;
    stringBuf[index] = std::to_string(iter.second.second / g_op_cost_total_);
    len = stringBuf[index].length();
    if (len > columnLenMax.at(index)) {
      columnLenMax.at(index) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[index]);

    index++;
    stringBuf[index] = std::to_string(iter.second.first);
    len = stringBuf[index].length();
    if (len > columnLenMax.at(index)) {
      columnLenMax.at(index) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[index]);

    index++;
    stringBuf[index] = std::to_string(iter.second.second);
    len = stringBuf[index].length();
    if (len > columnLenMax.at(index)) {
      columnLenMax.at(index) = len + kPrintOffset;
    }
    columns.emplace_back(stringBuf[index]);

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

bool TimeProfilingBeforeCallback(const MSTensorHandleArray inputs, const MSTensorHandleArray outputs,
                                 const MSCallBackParamC kernel_Info) {
  if (g_c_op_times_by_type_.find(kernel_Info.node_type) == g_c_op_times_by_type_.end()) {
    g_c_op_times_by_type_.insert(std::make_pair(kernel_Info.node_type, std::make_pair(0, 0.0f)));
  }
  if (g_c_op_times_by_name_.find(kernel_Info.node_name) == g_c_op_times_by_name_.end()) {
    g_c_op_times_by_name_.insert(std::make_pair(kernel_Info.node_name, std::make_pair(0, 0.0f)));
  }

  g_op_call_times_total_++;
  g_op_begin_ = mindspore::lite::GetTimeUs();
  return true;
}

bool TimeProfilingAfterCallback(const MSTensorHandleArray inputs, const MSTensorHandleArray outputs,
                                const MSCallBackParamC kernel_Info) {
  uint64_t opEnd = mindspore::lite::GetTimeUs();
  float cost = static_cast<float>(opEnd - g_op_begin_) / 1000.0f;
  g_op_cost_total_ += cost;
  g_c_op_times_by_type_[kernel_Info.node_type].first++;
  g_c_op_times_by_type_[kernel_Info.node_type].second += cost;
  g_c_op_times_by_name_[kernel_Info.node_name].first++;
  g_c_op_times_by_name_[kernel_Info.node_name].second += cost;
  return true;
}
}  // namespace lite
}  // namespace mindspore
