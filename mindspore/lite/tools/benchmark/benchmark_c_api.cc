/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/benchmark/benchmark_c_api.h"
#include <algorithm>
#include <map>
#include <string>
#include <utility>

using mindspore::lite::GetTimeUs;
using mindspore::lite::kFloatMSEC;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace tools {
int BenchmarkCApi::LoadInput() {
  if (flags_->in_data_file_.empty()) {
    auto status = GenerateInputData();
    if (status != 0) {
      std::cerr << "Generate input data error " << status << std::endl;
      MS_LOG(ERROR) << "Generate input data error " << status;
      return status;
    }
  } else {
    auto status = ReadInputFile();
    if (status != 0) {
      std::cerr << "ReadInputFile error, " << status << std::endl;
      MS_LOG(ERROR) << "ReadInputFile error, " << status;
      return status;
    }
  }
  return RET_OK;
}

int BenchmarkCApi::RunBenchmark() {
  auto start_prepare_time = GetTimeUs();
  int ret = InitContext();
  if (ret != RET_OK) {
    BENCHMARK_LOG_ERROR("InitContext failed, ret: " << ret);
    return ret;
  }
  model_ = MSModelCreate();
  ret = MSModelBuildFromFile(model_, flags_->model_file_.c_str(), kMSModelTypeMindIR, context_);
  if (ret != kMSStatusSuccess) {
    BENCHMARK_LOG_ERROR("MSModelBuildFromFile failed, ret: " << ret);
    return ret;
  }
  inputs_ = MSModelGetInputs(model_);
  if (inputs_.handle_list == nullptr) {
    BENCHMARK_LOG_ERROR("MSModelGetInputs failed, ret: " << ret);
    return ret;
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
    ret = MSModelResize(model_, inputs_, shape_infos.data(), inputs_.handle_num);
    if (ret != kMSStatusSuccess) {
      BENCHMARK_LOG_ERROR("MSModelResize failed, ret: " << ret);
      return ret;
    }
  }
  auto end_prepare_time = GetTimeUs();
  MS_LOG(INFO) << "PrepareTime = " << ((end_prepare_time - start_prepare_time) / kFloatMSEC) << " ms";
  std::cout << "PrepareTime = " << ((end_prepare_time - start_prepare_time) / kFloatMSEC) << " ms" << std::endl;

  ret = LoadInput();
  if (ret != kMSStatusSuccess) {
    BENCHMARK_LOG_ERROR("LoadInput failed, ret: " << ret)
    return ret;
  }
  if (!flags_->benchmark_data_file_.empty()) {
    ret = MarkAccuracy();
  } else {
    ret = MarkPerformance();
  }
  if (ret != kMSStatusSuccess) {
    BENCHMARK_LOG_ERROR("Run failed, ret: " << ret);
    return ret;
  }
  if (flags_->dump_tensor_data_) {
    BENCHMARK_LOG_ERROR("Dumped file is saved to : " + dump_file_output_dir_)
  }
  return RET_OK;
}

int BenchmarkCApi::InitContext() {
  constexpr int kFrequencyDefault = 3;
  context_ = MSContextCreate();
  if (context_ == nullptr) {
    BENCHMARK_LOG_ERROR("MSContextCreate failed");
    return RET_ERROR;
  }
  MSContextSetThreadNum(context_, flags_->num_threads_);
  MSContextSetEnableParallel(context_, flags_->enable_parallel_);
  MSContextSetThreadAffinityMode(context_, flags_->cpu_bind_mode_);

  if (flags_->device_ == "GPU") {
    MSDeviceInfoHandle gpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeGPU);
    MSDeviceInfoSetEnableFP16(gpu_device_info, flags_->enable_fp16_);
    MSContextAddDeviceInfo(context_, gpu_device_info);
  }
  if (flags_->device_ == "NPU") {
    MSDeviceInfoHandle npu_device_info = MSDeviceInfoCreate(kMSDeviceTypeKirinNPU);
    MSDeviceInfoSetFrequency(npu_device_info, kFrequencyDefault);
    MSContextAddDeviceInfo(context_, npu_device_info);
  }
  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSDeviceInfoSetEnableFP16(cpu_device_info, flags_->enable_fp16_);
  MSContextAddDeviceInfo(context_, cpu_device_info);
  return RET_OK;
}

int BenchmarkCApi::GenerateInputData() {
  for (size_t i = 0; i < inputs_.handle_num; i++) {
    MSTensorHandle tensor = inputs_.handle_list[i];
    auto data_type = MSTensorGetDataType(tensor);
    if (data_type == kMSDataTypeObjectTypeString) {
      BENCHMARK_LOG_ERROR("Unsupported kMSDataTypeObjectTypeString");
      return RET_ERROR;
    } else {
      auto data_ptr = MSTensorGetMutableData(tensor);
      auto data_size = MSTensorGetDataSize(tensor);
      (void)GenerateRandomData(data_size, data_ptr, static_cast<int>(data_type));
    }
  }
  return RET_OK;
}

int BenchmarkCApi::ReadInputFile() {
  if (this->flags_->in_data_type_ == lite::kImage) {
    BENCHMARK_LOG_ERROR("Unsupported image input");
    return RET_ERROR;
  } else {
    for (size_t i = 0; i < flags_->input_data_list_.size(); i++) {
      MSTensorHandle tensor = inputs_.handle_list[i];
      size_t size;
      auto bin_buf = lite::ReadFile(flags_->input_data_list_[i].c_str(), &size);
      if (bin_buf == nullptr) {
        BENCHMARK_LOG_ERROR("ReadFile failed");
        return RET_ERROR;
      }
      if (MSTensorGetDataType(tensor) == kMSDataTypeObjectTypeString) {
        BENCHMARK_LOG_ERROR("Unsupported kMSDataTypeObjectTypeString");
        return RET_ERROR;
      } else {
        auto tensor_data_size = MSTensorGetDataSize(tensor);
        if (tensor_data_size != size) {
          BENCHMARK_LOG_ERROR("Input file size error, required: " << tensor_data_size << ", in fact: " << size);
          delete[] bin_buf;
          return RET_ERROR;
        }
        auto input_data = MSTensorGetMutableData(tensor);
        if (input_data == nullptr) {
          BENCHMARK_LOG_ERROR("MSTensorGetMutableData failed");
          return RET_ERROR;
        }
        memcpy(input_data, bin_buf, size);
      }
      delete[] bin_buf;
    }
  }
  return RET_OK;
}

int BenchmarkCApi::MarkAccuracy() {
  MS_LOG(INFO) << "MarkAccuracy";
  std::cout << "MarkAccuracy" << std::endl;
  auto status = PrintInputData();
  if (status != RET_OK) {
    BENCHMARK_LOG_ERROR("PrintInputData failed, ret: " << status);
    return status;
  }
  status = MSModelPredict(model_, inputs_, &outputs_, before_call_back_, after_call_back_);
  if (status != kMSStatusSuccess) {
    BENCHMARK_LOG_ERROR("MSModelPredict failed, ret: " << status);
    return RET_ERROR;
  }
  status = ReadCalibData();
  if (status != RET_OK) {
    BENCHMARK_LOG_ERROR("ReadCalibData failed, ret: " << status);
    return status;
  }
  status = CompareOutput();
  if (status != RET_OK) {
    BENCHMARK_LOG_ERROR("CompareOutput failed, ret: " << status);
    return status;
  }
  return RET_OK;
}

int BenchmarkCApi::MarkPerformance() {
  MS_LOG(INFO) << "Running warm up loops...";
  std::cout << "Running warm up loops..." << std::endl;
  for (int i = 0; i < flags_->warm_up_loop_count_; i++) {
    auto ret = MSModelPredict(model_, inputs_, &outputs_, before_call_back_, after_call_back_);
    if (ret != kMSStatusSuccess) {
      BENCHMARK_LOG_ERROR("MSModelPredict failed, ret: " << kMSStatusSuccess);
      return RET_ERROR;
    }
  }

  MS_LOG(INFO) << "Running benchmark loops...";
  std::cout << "Running benchmark loops..." << std::endl;
  uint64_t time_min = 1000000;
  uint64_t time_max = 0;
  uint64_t time_avg = 0;

  for (int i = 0; i < flags_->loop_count_; i++) {
    auto start = GetTimeUs();
    auto ret = MSModelPredict(model_, inputs_, &outputs_, before_call_back_, after_call_back_);
    if (ret != kMSStatusSuccess) {
      BENCHMARK_LOG_ERROR("MSModelPredict failed, ret: " << kMSStatusSuccess);
      return RET_ERROR;
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

  if (flags_->loop_count_ > 0) {
    time_avg /= flags_->loop_count_;
    MS_LOG(INFO) << "Model = "
                 << flags_->model_file_.substr(flags_->model_file_.find_last_of(lite::DELIM_SLASH) + 1).c_str()
                 << ", NumThreads = " << flags_->num_threads_ << ", MinRunTime = " << time_min / lite::kFloatMSEC
                 << ", MaxRuntime = " << time_max / lite::kFloatMSEC
                 << ", AvgRunTime = " << time_avg / lite::kFloatMSEC;
    printf("Model = %s, NumThreads = %d, MinRunTime = %f ms, MaxRuntime = %f ms, AvgRunTime = %f ms\n",
           flags_->model_file_.substr(flags_->model_file_.find_last_of(lite::DELIM_SLASH) + 1).c_str(),
           flags_->num_threads_, time_min / lite::kFloatMSEC, time_max / lite::kFloatMSEC, time_avg / lite::kFloatMSEC);
  }
  return RET_OK;
}

int BenchmarkCApi::GetDataTypeByTensorName(const std::string &tensor_name) {
  return MSTensorGetDataType(MSModelGetOutputByTensorName(model_, tensor_name.c_str()));
}

int BenchmarkCApi::CompareOutput() {
  constexpr int kPercentageDivisor = 100;
  std::cout << "================ Comparing Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  for (const auto &calib_tensor : benchmark_data_) {
    std::string tensor_name = calib_tensor.first;
    MSTensorHandle tensor = MSModelGetOutputByTensorName(model_, tensor_name.c_str());
    if (tensor == nullptr) {
      BENCHMARK_LOG_ERROR("Get tensor failed, tensor name: " << tensor_name);
      return RET_ERROR;
    }
    int ret;
    if (static_cast<int>(MSTensorGetDataType(tensor)) == kObjectTypeString) {
      BENCHMARK_LOG_ERROR("Unsupported kMSDataTypeObjectTypeString");
      return RET_ERROR;
    } else {
      ret = CompareDataGetTotalBiasAndSize(tensor_name, tensor, &total_bias, &total_size);
    }
    if (ret != RET_OK) {
      BENCHMARK_LOG_ERROR("Error in CompareData");
      BENCHMARK_LOG_ERROR("=======================================================");
      return ret;
    }
  }
  float mean_bias;
  if (total_size != 0) {
    mean_bias = ((total_bias / float_t(total_size)) * kPercentageDivisor);
  } else {
    mean_bias = 0;
  }

  std::cout << "Mean bias of all nodes/tensors: " << mean_bias << "%" << std::endl;
  std::cout << "=======================================================" << std::endl << std::endl;

  if (mean_bias > this->flags_->accuracy_threshold_) {
    BENCHMARK_LOG_ERROR("Mean bias of all nodes/tensors is too big: " << mean_bias << "%");
    return RET_ERROR;
  }
  return RET_OK;
}

int BenchmarkCApi::CompareDataGetTotalBiasAndSize(const std::string &name, MSTensorHandle tensor, float *total_bias,
                                                  int *total_size) {
  auto tensor_data = MSTensorGetData(tensor);
  if (tensor_data == nullptr) {
    BENCHMARK_LOG_ERROR("MSTensorGetData failed.");
    return RET_ERROR;
  }
  size_t shape_num;
  const int64_t *shape = MSTensorGetShape(tensor, &shape_num);
  std::vector<int64_t> vec_shape(shape, shape + shape_num);
  float bias = 0;
  switch (static_cast<TypeId>(MSTensorGetDataType(tensor))) {
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32: {
      bias = CompareData<float, int64_t>(name, vec_shape, tensor_data);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      bias = CompareData<int8_t, int64_t>(name, vec_shape, tensor_data);
      break;
    }
    case TypeId::kNumberTypeUInt8: {
      bias = CompareData<uint8_t, int64_t>(name, vec_shape, tensor_data);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      bias = CompareData<int32_t, int64_t>(name, vec_shape, tensor_data);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      bias = CompareData<int16_t, int64_t>(name, vec_shape, tensor_data);
      break;
    }
    case TypeId::kNumberTypeBool: {
      bias = CompareData<bool, int64_t>(name, vec_shape, tensor_data);
      break;
    }
    default:
      BENCHMARK_LOG_ERROR("Unsupported data type" << static_cast<int>(MSTensorGetDataType(tensor)));
      return RET_ERROR;
  }
  if (bias < 0) {
    BENCHMARK_LOG_ERROR("CompareData failed, name: " << name);
    return RET_ERROR;
  }
  *total_bias += bias;
  *total_size += 1;
  return RET_OK;
}

int BenchmarkCApi::PrintInputData() {
  constexpr int64_t kPrintDataNum = 20;
  for (size_t i = 0; i < inputs_.handle_num; i++) {
    auto input = inputs_.handle_list[i];
    std::cout << "InData" << i << ": ";
    auto data_type = static_cast<TypeId>(MSTensorGetDataType(input));
    if (data_type == TypeId::kObjectTypeString) {
      BENCHMARK_LOG_ERROR("Unsupported kMSDataTypeObjectTypeString.");
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
        BENCHMARK_LOG_ERROR("Datatype: " << data_type << " is not supported.");
        return RET_ERROR;
      }
    }
    std::cout << std::endl;
  }
  return RET_OK;
}

int BenchmarkCApi::InitTimeProfilingCallbackParameter() {
  before_call_back_ = TimeBeforeCallback;
  after_call_back_ = TimeAfterCallback;
  return RET_OK;
}

int BenchmarkCApi::InitPerfProfilingCallbackParameter() {
  BENCHMARK_LOG_ERROR("Unsupported feature.");
  return RET_ERROR;
}

int BenchmarkCApi::InitPrintTensorDataCallbackParameter() {
  BENCHMARK_LOG_ERROR("Unsupported feature.");
  return RET_ERROR;
}
int BenchmarkCApi::InitDumpTensorDataCallbackParameter() {
  BENCHMARK_LOG_ERROR("Unsupported feature.");
  return RET_ERROR;
}
}  // namespace tools
}  // namespace mindspore

uint64_t g_op_begin_ = 0;
int g_op_call_times_total_ = 0;
float g_op_cost_total_ = 0.0f;
std::map<std::string, std::pair<int, float>> g_op_times_by_type_;
std::map<std::string, std::pair<int, float>> g_op_times_by_name_;

bool TimeBeforeCallback(const MSTensorHandleArray inputs, const MSTensorHandleArray outputs,
                        const MSCallBackParamC kernel_Info) {
  if (g_op_times_by_type_.find(kernel_Info.node_type) == g_op_times_by_type_.end()) {
    g_op_times_by_type_.insert(std::make_pair(kernel_Info.node_type, std::make_pair(0, 0.0f)));
  }
  if (g_op_times_by_name_.find(kernel_Info.node_name) == g_op_times_by_name_.end()) {
    g_op_times_by_name_.insert(std::make_pair(kernel_Info.node_name, std::make_pair(0, 0.0f)));
  }

  g_op_call_times_total_++;
  g_op_begin_ = mindspore::lite::GetTimeUs();
  return true;
}

bool TimeAfterCallback(const MSTensorHandleArray inputs, const MSTensorHandleArray outputs,
                       const MSCallBackParamC kernel_Info) {
  uint64_t opEnd = mindspore::lite::GetTimeUs();
  float cost = static_cast<float>(opEnd - g_op_begin_) / mindspore::lite::kFloatMSEC;
  g_op_cost_total_ += cost;
  g_op_times_by_type_[kernel_Info.node_type].first++;
  g_op_times_by_type_[kernel_Info.node_type].second += cost;
  g_op_times_by_name_[kernel_Info.node_name].first++;
  g_op_times_by_name_[kernel_Info.node_name].second += cost;
  return true;
}
