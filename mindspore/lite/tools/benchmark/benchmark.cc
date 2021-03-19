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

#include "tools/benchmark/benchmark.h"
#define __STDC_FORMAT_MACROS
#include <cinttypes>
#undef __STDC_FORMAT_MACROS
#include <algorithm>
#include <utility>
#include <functional>
#include "include/context.h"
#include "include/ms_tensor.h"
#include "include/version.h"
#include "src/common/common.h"
#include "src/runtime/runtime_api.h"
#ifdef ENABLE_ARM64
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>
#include <unistd.h>
#endif

namespace mindspore {
namespace lite {
static const char *DELIM_COLON = ":";
static const char *DELIM_COMMA = ",";
static const char *DELIM_SLASH = "/";

int Benchmark::GenerateRandomData(size_t size, void *data, TypeId data_type) {
  MS_ASSERT(data != nullptr);
  switch (data_type) {
    case kNumberTypeFloat32:
    case kNumberTypeFloat:
      FillInputData<float>(size, data, std::uniform_real_distribution<float>(0.1f, 1.0f));
      break;
    case kNumberTypeFloat64:
      FillInputData<double>(size, data, std::uniform_real_distribution<double>(0.1, 1.0));
      break;
    case kNumberTypeInt64:
      FillInputData<int64_t>(size, data, std::uniform_int_distribution<int64_t>(0, 1));
      break;
    case kNumberTypeInt:
    case kNumberTypeInt32:
      FillInputData<int32_t>(size, data, std::uniform_int_distribution<int32_t>(0, 1));
      break;
    case kNumberTypeInt16:
      FillInputData<int16_t>(size, data, std::uniform_int_distribution<int16_t>(0, 1));
      break;
    case kNumberTypeInt8:
      FillInputData<int8_t>(size, data, std::uniform_int_distribution<int8_t>(-127, 127));
      break;
    case kNumberTypeUInt8:
      FillInputData<uint8_t>(size, data, std::uniform_int_distribution<uint8_t>(0, 254));
      break;
    default:
      char *casted_data = static_cast<char *>(data);
      for (size_t i = 0; i < size; i++) {
        casted_data[i] = static_cast<char>(i);
      }
  }
  return RET_OK;
}

int Benchmark::GenerateInputData() {
  for (auto tensor : ms_inputs_) {
    MS_ASSERT(tensor != nullptr);
    auto input_data = tensor->MutableData();
    if (input_data == nullptr) {
      MS_LOG(ERROR) << "MallocData for inTensor failed";
      return RET_ERROR;
    }
    int status;
    if (tensor->data_type() == kObjectTypeString) {
      status = StringsToMSTensor({"you're the best."}, tensor);
    } else {
      status = GenerateRandomData(tensor->Size(), input_data, tensor->data_type());
    }
    if (status != RET_OK) {
      std::cerr << "GenerateRandomData for inTensor failed: " << status << std::endl;
      MS_LOG(ERROR) << "GenerateRandomData for inTensor failed:" << status;
      return status;
    }
  }
  return RET_OK;
}

int Benchmark::LoadInput() {
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

int Benchmark::ReadInputFile() {
  if (ms_inputs_.empty()) {
    return RET_OK;
  }

  if (this->flags_->in_data_type_ == kImage) {
    MS_LOG(ERROR) << "Not supported image input";
    return RET_ERROR;
  } else {
    for (size_t i = 0; i < flags_->input_data_list_.size(); i++) {
      auto cur_tensor = ms_inputs_.at(i);
      MS_ASSERT(cur_tensor != nullptr);
      size_t size;
      char *bin_buf = ReadFile(flags_->input_data_list_[i].c_str(), &size);
      if (bin_buf == nullptr) {
        MS_LOG(ERROR) << "ReadFile return nullptr";
        return RET_ERROR;
      }
      if (cur_tensor->data_type() == kObjectTypeString) {
        std::string str(bin_buf, size);
        auto ret = StringsToMSTensor({str}, cur_tensor);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "write strings to tensor failed";
          delete[] bin_buf;
          return RET_ERROR;
        }
      } else {
        auto tensor_data_size = cur_tensor->Size();
        if (size != tensor_data_size) {
          std::cerr << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size
                    << std::endl;
          MS_LOG(ERROR) << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size;
          delete[] bin_buf;
          return RET_ERROR;
        }
        auto input_data = cur_tensor->MutableData();
        if (input_data == nullptr) {
          MS_LOG(ERROR) << "input_data is nullptr.";
          return RET_ERROR;
        }
        memcpy(input_data, bin_buf, tensor_data_size);
      }
      delete[] bin_buf;
    }
  }
  return RET_OK;
}

// calibData is FP32
int Benchmark::ReadCalibData() {
  const char *calib_data_path = flags_->benchmark_data_file_.c_str();
  // read calib data
  std::ifstream in_file(calib_data_path);
  if (!in_file.good()) {
    std::cerr << "file: " << calib_data_path << " is not exist" << std::endl;
    MS_LOG(ERROR) << "file: " << calib_data_path << " is not exist";
    return RET_ERROR;
  }

  if (!in_file.is_open()) {
    std::cerr << "file: " << calib_data_path << " open failed" << std::endl;
    MS_LOG(ERROR) << "file: " << calib_data_path << " open failed";
    in_file.close();
    return RET_ERROR;
  }
  MS_LOG(INFO) << "Start reading calibData file";
  std::string line;
  std::string tensor_name;

  while (!in_file.eof()) {
    getline(in_file, line);
    std::stringstream string_line1(line);
    size_t dim = 0;
    string_line1 >> tensor_name >> dim;
    std::vector<size_t> dims;
    for (size_t i = 0; i < dim; i++) {
      size_t tmp_dim;
      string_line1 >> tmp_dim;
      dims.push_back(tmp_dim);
    }
    auto ret = ReadTensorData(in_file, tensor_name, dims);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Read tensor data failed, tensor name: " << tensor_name;
      return RET_ERROR;
    }
  }
  in_file.close();
  MS_LOG(INFO) << "Finish reading calibData file";
  return RET_OK;
}

int Benchmark::ReadTensorData(std::ifstream &in_file_stream, const std::string &tensor_name,
                              const std::vector<size_t> &dims) {
  std::string line;
  getline(in_file_stream, line);
  std::stringstream line_stream(line);
  if (this->benchmark_data_.find(tensor_name) != this->benchmark_data_.end()) {
    return RET_OK;
  }
  tensor::MSTensor *tensor = GetTensorByNameOrShape(tensor_name, dims);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Get tensor failed, tensor name: " << tensor_name;
    return RET_ERROR;
  }
  std::vector<float> data;
  std::vector<std::string> strings_data;
  size_t shape_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
  if (tensor->data_type() == kObjectTypeString) {
    strings_data.push_back(line);
    for (size_t i = 1; i < shape_size; i++) {
      getline(in_file_stream, line);
      strings_data.push_back(line);
    }
  } else {
    for (size_t i = 0; i < shape_size; i++) {
      float tmp_data;
      line_stream >> tmp_data;
      data.push_back(tmp_data);
    }
  }
  auto *check_tensor = new (std::nothrow) CheckTensor(dims, data, strings_data);
  if (check_tensor == nullptr) {
    MS_LOG(ERROR) << "New CheckTensor failed, tensor name: " << tensor_name;
    return RET_ERROR;
  }
  this->benchmark_data_.insert(std::make_pair(tensor_name, check_tensor));
  return RET_OK;
}

int Benchmark::CompareOutput() {
  std::cout << "================ Comparing Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  for (const auto &calib_tensor : benchmark_data_) {
    std::string node_or_tensor_name = calib_tensor.first;
    tensor::MSTensor *tensor = GetTensorByNameOrShape(node_or_tensor_name, calib_tensor.second->shape);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Get tensor failed, tensor name: " << node_or_tensor_name;
      return RET_ERROR;
    }
    int ret;
    if (tensor->data_type() == kObjectTypeString) {
      ret = CompareStringData(node_or_tensor_name, tensor);
    } else {
      ret = CompareDataGetTotalBiasAndSize(node_or_tensor_name, tensor, &total_bias, &total_size);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Error in CompareData";
      std::cerr << "Error in CompareData" << std::endl;
      std::cout << "=======================================================" << std::endl << std::endl;
      return ret;
    }
  }
  float mean_bias;
  if (total_size != 0) {
    mean_bias = total_bias / float_t(total_size) * 100;
  } else {
    mean_bias = 0;
  }

  std::cout << "Mean bias of all nodes/tensors: " << mean_bias << "%" << std::endl;
  std::cout << "=======================================================" << std::endl << std::endl;

  if (mean_bias > this->flags_->accuracy_threshold_) {
    MS_LOG(ERROR) << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%";
    std::cerr << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%" << std::endl;
    return RET_ERROR;
  }
  return RET_OK;
}

tensor::MSTensor *Benchmark::GetTensorByNodeShape(const std::vector<size_t> &node_shape) {
  std::vector<tensor::MSTensor *> match_tensors;
  std::vector<int> shape_vector;
  (void)std::transform(node_shape.begin(), node_shape.end(), std::back_inserter(shape_vector),
                       [](const size_t &value) { return static_cast<int>(value); });
  auto tensors = session_->GetOutputs();
  for (auto &out_tensor_pair : tensors) {
    if (out_tensor_pair.second->shape() == shape_vector) {
      match_tensors.emplace_back(out_tensor_pair.second);
    }
  }
  if (match_tensors.empty() || match_tensors.size() != 1) {
    MS_LOG(ERROR) << "get tensor by node shape failed";
    return nullptr;
  }
  return match_tensors.front();
}

tensor::MSTensor *Benchmark::GetTensorByNameOrShape(const std::string &node_or_tensor_name,
                                                    const std::vector<size_t> &dims) {
  tensor::MSTensor *tensor = nullptr;
  auto tensors = session_->GetOutputsByNodeName(node_or_tensor_name);
  if (tensors.empty() || tensors.size() != 1) {
    MS_LOG(INFO) << "Cannot find output node: " << node_or_tensor_name
                 << " or node has more than one output tensor, switch to GetOutputByTensorName";
    tensor = session_->GetOutputByTensorName(node_or_tensor_name);
    if (tensor == nullptr) {
      return GetTensorByNodeShape(dims);
    }
  } else {
    tensor = tensors.front();
  }
  return tensor;
}

int Benchmark::CompareStringData(const std::string &name, tensor::MSTensor *tensor) {
  auto iter = this->benchmark_data_.find(name);
  if (iter != this->benchmark_data_.end()) {
    std::vector<std::string> calib_strings = iter->second->strings_data;
    std::vector<std::string> output_strings = MSTensorToStrings(tensor);
    size_t compare_num = std::min(calib_strings.size(), output_strings.size());
    size_t print_num = std::min(compare_num, static_cast<size_t>(5));

    std::cout << "Data of node " << name << " : " << std::endl;
    for (size_t i = 0; i < compare_num; i++) {
      if (i < print_num) {
        std::cout << "  " << output_strings[i] << std::endl;
      }
      if (calib_strings[i] != output_strings[i]) {
        MS_LOG(ERROR) << "Compare failed, index: " << i;
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int Benchmark::CompareDataGetTotalBiasAndSize(const std::string &name, tensor::MSTensor *tensor, float *total_bias,
                                              int *total_size) {
  float bias = 0;
  auto mutableData = tensor->MutableData();
  if (mutableData == nullptr) {
    MS_LOG(ERROR) << "mutableData is nullptr.";
    return RET_ERROR;
  }
  switch (tensor->data_type()) {
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32: {
      bias = CompareData<float>(name, tensor->shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      bias = CompareData<int8_t>(name, tensor->shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeUInt8: {
      bias = CompareData<uint8_t>(name, tensor->shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      bias = CompareData<int32_t>(name, tensor->shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      bias = CompareData<int16_t>(name, tensor->shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeBool: {
      bias = CompareData<bool>(name, tensor->shape(), mutableData);
      break;
    }
    default:
      MS_LOG(ERROR) << "Datatype " << tensor->data_type() << " is not supported.";
      return RET_ERROR;
  }
  if (bias < 0) {
    MS_LOG(ERROR) << "CompareData failed, name: " << name;
    return RET_ERROR;
  }
  *total_bias += bias;
  *total_size += 1;
  return RET_OK;
}

int Benchmark::MarkPerformance() {
  MS_LOG(INFO) << "Running warm up loops...";
  std::cout << "Running warm up loops..." << std::endl;
  for (int i = 0; i < flags_->warm_up_loop_count_; i++) {
    auto status = session_->RunGraph();
    if (status != 0) {
      MS_LOG(ERROR) << "Inference error " << status;
      std::cerr << "Inference error " << status << std::endl;
      return status;
    }
  }

  MS_LOG(INFO) << "Running benchmark loops...";
  std::cout << "Running benchmark loops..." << std::endl;
  uint64_t time_min = 1000000;
  uint64_t time_max = 0;
  uint64_t time_avg = 0;

  for (int i = 0; i < flags_->loop_count_; i++) {
    session_->BindThread(true);
    auto start = GetTimeUs();
    auto status = (flags_->time_profiling_ || flags_->perf_profiling_)
                    ? session_->RunGraph(before_call_back_, after_call_back_)
                    : session_->RunGraph();
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
#ifdef ENABLE_ARM64
  } else if (flags_->perf_profiling_) {
    if (flags_->perf_event_ == "CACHE") {
      const std::vector<std::string> per_op_name = {"opName", "cache ref(k)", "cache ref(%)", "miss(k)", "miss(%)"};
      const std::vector<std::string> per_op_type = {"opType", "cache ref(k)", "cache ref(%)", "miss(k)", "miss(%)"};
      PrintPerfResult(per_op_name, op_perf_by_name_);
      PrintPerfResult(per_op_type, op_perf_by_type_);
    } else if (flags_->perf_event_ == "STALL") {
      const std::vector<std::string> per_op_name = {"opName", "frontend(k)", "frontend(%)", "backendend(k)",
                                                    "backendend(%)"};
      const std::vector<std::string> per_op_type = {"opType", "frontend(k)", "frontend(%)", "backendend(k)",
                                                    "backendend(%)"};
      PrintPerfResult(per_op_name, op_perf_by_name_);
      PrintPerfResult(per_op_type, op_perf_by_type_);
    } else {
      const std::vector<std::string> per_op_name = {"opName", "cycles(k)", "cycles(%)", "ins(k)", "ins(%)"};
      const std::vector<std::string> per_op_type = {"opType", "cycles(k)", "cycles(%)", "ins(k)", "ins(%)"};
      PrintPerfResult(per_op_name, op_perf_by_name_);
      PrintPerfResult(per_op_type, op_perf_by_type_);
    }
#endif
  }

  if (flags_->loop_count_ > 0) {
    time_avg /= flags_->loop_count_;
    MS_LOG(INFO) << "Model = " << flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
                 << ", NumThreads = " << flags_->num_threads_ << ", MinRunTime = " << time_min / 1000.0f
                 << ", MaxRuntime = " << time_max / 1000.0f << ", AvgRunTime = " << time_avg / 1000.0f;
    printf("Model = %s, NumThreads = %d, MinRunTime = %f ms, MaxRuntime = %f ms, AvgRunTime = %f ms\n",
           flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1).c_str(), flags_->num_threads_,
           time_min / 1000.0f, time_max / 1000.0f, time_avg / 1000.0f);
  }
  return RET_OK;
}

int Benchmark::MarkAccuracy() {
  MS_LOG(INFO) << "MarkAccuracy";
  std::cout << "MarkAccuracy" << std::endl;

  auto status = PrintInputData();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "PrintInputData error " << status;
    std::cerr << "PrintInputData error " << status << std::endl;
    return status;
  }
  status = session_->RunGraph();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Inference error " << status;
    std::cerr << "Inference error " << status << std::endl;
    return status;
  }
  status = ReadCalibData();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read calib data error " << status;
    std::cerr << "Read calib data error " << status << std::endl;
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

int Benchmark::PrintInputData() {
  for (size_t i = 0; i < ms_inputs_.size(); i++) {
    auto input = ms_inputs_[i];
    MS_ASSERT(input != nullptr);
    auto tensor_data_type = input->data_type();

    std::cout << "InData" << i << ": ";
    if (tensor_data_type == TypeId::kObjectTypeString) {
      std::vector<std::string> output_strings = MSTensorToStrings(input);
      size_t print_num = std::min(output_strings.size(), static_cast<size_t>(20));
      for (size_t j = 0; j < print_num; j++) {
        std::cout << output_strings[j] << std::endl;
      }
      continue;
    }
    size_t print_num = std::min(input->ElementsNum(), 20);
    const void *in_data = input->MutableData();
    if (in_data == nullptr) {
      MS_LOG(ERROR) << "in_data is nullptr.";
      return RET_ERROR;
    }

    for (size_t j = 0; j < print_num; j++) {
      if (tensor_data_type == TypeId::kNumberTypeFloat32 || tensor_data_type == TypeId::kNumberTypeFloat) {
        std::cout << static_cast<const float *>(in_data)[j] << " ";
      } else if (tensor_data_type == TypeId::kNumberTypeInt8) {
        std::cout << static_cast<const int8_t *>(in_data)[j] << " ";
      } else if (tensor_data_type == TypeId::kNumberTypeUInt8) {
        std::cout << static_cast<const uint8_t *>(in_data)[j] << " ";
      } else if (tensor_data_type == TypeId::kNumberTypeInt32) {
        std::cout << static_cast<const int32_t *>(in_data)[j] << " ";
      } else if (tensor_data_type == TypeId::kNumberTypeInt64) {
        std::cout << static_cast<const int64_t *>(in_data)[j] << " ";
      } else if (tensor_data_type == TypeId::kNumberTypeBool) {
        std::cout << static_cast<const bool *>(in_data)[j] << " ";
      } else {
        MS_LOG(ERROR) << "Datatype: " << tensor_data_type << " is not supported.";
        return RET_ERROR;
      }
    }
    std::cout << std::endl;
  }
  return RET_OK;
}

int Benchmark::RunBenchmark() {
  auto start_prepare_time = GetTimeUs();
  // Load graph
  std::string model_name = flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1);

  MS_LOG(INFO) << "start reading model file";
  std::cout << "start reading model file" << std::endl;
  size_t size = 0;
  char *graph_buf = ReadFile(flags_->model_file_.c_str(), &size);
  if (graph_buf == nullptr) {
    MS_LOG(ERROR) << "Read model file failed while running " << model_name.c_str();
    std::cerr << "Read model file failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }
  auto model = std::shared_ptr<Model>(lite::Model::Import(graph_buf, size));
  delete[](graph_buf);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model file failed while running " << model_name.c_str();
    std::cerr << "Import model file failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }
  auto context = std::make_shared<Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running " << model_name.c_str();
    std::cerr << "New context failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }

  auto &cpu_device_ctx = context->device_list_[0];
  if (flags_->cpu_bind_mode_ == MID_CPU || flags_->cpu_bind_mode_ == HIGHER_CPU) {
    cpu_device_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = CpuBindMode(flags_->cpu_bind_mode_);
  } else {
    cpu_device_ctx.device_info_.cpu_device_info_.cpu_bind_mode_ = NO_BIND;
  }
  cpu_device_ctx.device_info_.cpu_device_info_.enable_float16_ = flags_->enable_fp16_;

  if (flags_->device_ == "GPU") {
    DeviceContext gpu_device_ctx{DT_GPU, {false}};
    gpu_device_ctx.device_info_.gpu_device_info_.enable_float16_ = flags_->enable_fp16_;
    context->device_list_.push_back(gpu_device_ctx);
  }

  if (flags_->device_ == "NPU") {
    DeviceContext npu_device_ctx{DT_NPU};
    npu_device_ctx.device_info_.npu_device_info_.frequency_ = 3;
    context->device_list_.push_back(npu_device_ctx);
  }

  context->thread_num_ = flags_->num_threads_;

  session_ = session::LiteSession::CreateSession(context.get());
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "CreateSession failed while running ", model_name.c_str();
    std::cout << "CreateSession failed while running ", model_name.c_str();
    return RET_ERROR;
  }
  auto ret = session_->CompileGraph(model.get());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CompileGraph failed while running ", model_name.c_str();
    std::cout << "CompileGraph failed while running ", model_name.c_str();
    return ret;
  }
  if (!flags_->resize_dims_.empty()) {
    ret = session_->Resize(session_->GetInputs(), flags_->resize_dims_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Input tensor resize failed.";
      std::cout << "Input tensor resize failed.";
      return ret;
    }
  }
  if (model != nullptr) model->Free();

  ms_inputs_ = session_->GetInputs();
  auto end_prepare_time = GetTimeUs();
  MS_LOG(INFO) << "PrepareTime = " << (end_prepare_time - start_prepare_time) / 1000 << " ms";
  std::cout << "PrepareTime = " << (end_prepare_time - start_prepare_time) / 1000 << " ms" << std::endl;

  // Load input
  MS_LOG(INFO) << "start generate input data";
  auto status = LoadInput();
  if (status != 0) {
    MS_LOG(ERROR) << "Generate input data error";
    return status;
  }
  if (!flags_->benchmark_data_file_.empty()) {
    status = MarkAccuracy();
    for (auto &data : benchmark_data_) {
      data.second->shape.clear();
      data.second->data.clear();
      delete data.second;
      data.second = nullptr;
    }
    benchmark_data_.clear();
    if (status != 0) {
      MS_LOG(ERROR) << "Run MarkAccuracy error: " << status;
      std::cout << "Run MarkAccuracy error: " << status << std::endl;
      return status;
    }
  } else {
    status = MarkPerformance();
    if (status != 0) {
      MS_LOG(ERROR) << "Run MarkPerformance error: " << status;
      std::cout << "Run MarkPerformance error: " << status << std::endl;
      return status;
    }
  }
  return RET_OK;
}

void BenchmarkFlags::InitInputDataList() {
  char *input_list = new char[this->in_data_file_.length() + 1];
  snprintf(input_list, this->in_data_file_.length() + 1, "%s", this->in_data_file_.c_str());
  char *cur_input;
  const char *split_c = ",";
  cur_input = strtok(input_list, split_c);
  while (cur_input != nullptr) {
    input_data_list_.emplace_back(cur_input);
    cur_input = strtok(nullptr, split_c);
  }
  delete[] input_list;
}

void BenchmarkFlags::InitResizeDimsList() {
  std::string content;
  content = this->resize_dims_in_;
  std::vector<int> shape;
  auto shape_strs = StringSplit(content, std::string(DELIM_COLON));
  for (const auto &shape_str : shape_strs) {
    shape.clear();
    auto dim_strs = StringSplit(shape_str, std::string(DELIM_COMMA));
    std::cout << "Resize Dims: ";
    for (const auto &dim_str : dim_strs) {
      std::cout << dim_str << " ";
      shape.emplace_back(static_cast<int>(std::stoi(dim_str)));
    }
    std::cout << std::endl;
    this->resize_dims_.emplace_back(shape);
  }
}

int Benchmark::InitTimeProfilingCallbackParameter() {
  // before callback
  before_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                          const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                          const CallBackParam &callParam) {
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
                         const CallBackParam &call_param) {
    uint64_t opEnd = GetTimeUs();

    if (after_inputs.empty()) {
      MS_LOG(INFO) << "The num of after inputs is empty";
    }
    if (after_outputs.empty()) {
      MS_LOG(INFO) << "The num of after outputs is empty";
    }

    float cost = static_cast<float>(opEnd - op_begin_) / 1000.0f;
    if (flags_->device_ == "GPU") {
      auto gpu_param = reinterpret_cast<const GPUCallBackParam &>(call_param);
      cost = static_cast<float>(gpu_param.execute_time);
    }
    op_cost_total_ += cost;
    op_times_by_type_[call_param.node_type].first++;
    op_times_by_type_[call_param.node_type].second += cost;
    op_times_by_name_[call_param.node_name].first++;
    op_times_by_name_[call_param.node_name].second += cost;
    return true;
  };
  return RET_OK;
}
int Benchmark::InitPerfProfilingCallbackParameter() {
#ifndef ENABLE_ARM64
  MS_LOG(ERROR) << "Only support perf_profiling on arm64.";
  return RET_ERROR;
#else
  struct perf_event_attr pe, pe2;
  memset(&pe, 0, sizeof(struct perf_event_attr));
  memset(&pe2, 0, sizeof(struct perf_event_attr));
  pe.type = PERF_TYPE_HARDWARE;
  pe2.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(struct perf_event_attr);
  pe2.size = sizeof(struct perf_event_attr);
  pe.disabled = 1;
  pe2.disabled = 1;
  pe.exclude_kernel = 1;   // don't count kernel
  pe2.exclude_kernel = 1;  // don't count kernel
  pe.exclude_hv = 1;       // don't count hypervisor
  pe2.exclude_hv = 1;      // don't count hypervisor
  pe.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  pe2.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  if (flags_->perf_event_ == "CACHE") {
    pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
    pe2.config = PERF_COUNT_HW_CACHE_MISSES;
  } else if (flags_->perf_event_ == "STALL") {
    pe.config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;
    pe2.config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
  } else {
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe2.config = PERF_COUNT_HW_INSTRUCTIONS;
  }
  perf_fd = syscall(__NR_perf_event_open, pe, 0, -1, -1, 0);
  if (perf_fd == -1) {
    MS_LOG(ERROR) << "Failed to open perf event " << pe.config;
    return RET_ERROR;
  }
  perf_fd2 = syscall(__NR_perf_event_open, pe2, 0, -1, perf_fd, 0);
  if (perf_fd2 == -1) {
    MS_LOG(ERROR) << "Failed to open perf event " << pe2.config;
    return RET_ERROR;
  }
  struct PerfCount zero;
  zero.value[0] = 0;
  zero.value[1] = 0;
  // before callback
  before_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                          const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                          const CallBackParam &callParam) {
    if (before_inputs.empty()) {
      MS_LOG(INFO) << "The num of beforeInputs is empty";
    }
    if (before_outputs.empty()) {
      MS_LOG(INFO) << "The num of beforeOutputs is empty";
    }
    if (op_perf_by_type_.find(callParam.node_type) == op_perf_by_type_.end()) {
      op_perf_by_type_.insert(std::make_pair(callParam.node_type, std::make_pair(0, zero)));
    }
    if (op_perf_by_name_.find(callParam.node_name) == op_perf_by_name_.end()) {
      op_perf_by_name_.insert(std::make_pair(callParam.node_name, std::make_pair(0, zero)));
    }

    op_call_times_total_++;
    ioctl(perf_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    return true;
  };

  // after callback
  after_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                         const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                         const CallBackParam &call_param) {
    struct PerfResult res;
    ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    read(perf_fd, &res, sizeof(struct PerfResult));

    if (after_inputs.empty()) {
      MS_LOG(INFO) << "The num of after inputs is empty";
    }
    if (after_outputs.empty()) {
      MS_LOG(INFO) << "The num of after outputs is empty";
    }
    float cost1 = static_cast<float>(res.values[0].value);
    float cost2 = static_cast<float>(res.values[1].value);
    op_cost_total_ += cost1;
    op_cost2_total_ += cost2;
    op_perf_by_type_[call_param.node_type].first++;
    op_perf_by_type_[call_param.node_type].second.value[0] += cost1;
    op_perf_by_type_[call_param.node_type].second.value[1] += cost2;
    op_perf_by_name_[call_param.node_name].first++;
    op_perf_by_name_[call_param.node_name].second.value[0] += cost1;
    op_perf_by_name_[call_param.node_name].second.value[1] += cost2;
    return true;
  };
#endif
  return RET_OK;
}

int Benchmark::InitCallbackParameter() {
  int ret = RET_OK;
  if (flags_->time_profiling_) {
    ret = InitTimeProfilingCallbackParameter();
  } else if (flags_->perf_profiling_) {
    ret = InitPerfProfilingCallbackParameter();
  }
  return ret;
}

int Benchmark::Init() {
  if (this->flags_ == nullptr) {
    return 1;
  }
  MS_LOG(INFO) << "ModelPath = " << this->flags_->model_file_;
  MS_LOG(INFO) << "InDataPath = " << this->flags_->in_data_file_;
  MS_LOG(INFO) << "InDataType = " << this->flags_->in_data_type_in_;
  MS_LOG(INFO) << "LoopCount = " << this->flags_->loop_count_;
  MS_LOG(INFO) << "DeviceType = " << this->flags_->device_;
  MS_LOG(INFO) << "AccuracyThreshold = " << this->flags_->accuracy_threshold_;
  MS_LOG(INFO) << "WarmUpLoopCount = " << this->flags_->warm_up_loop_count_;
  MS_LOG(INFO) << "NumThreads = " << this->flags_->num_threads_;
  MS_LOG(INFO) << "Fp16Priority = " << this->flags_->enable_fp16_;
  MS_LOG(INFO) << "calibDataPath = " << this->flags_->benchmark_data_file_;
  std::cout << "ModelPath = " << this->flags_->model_file_ << std::endl;
  std::cout << "InDataPath = " << this->flags_->in_data_file_ << std::endl;
  std::cout << "InDataType = " << this->flags_->in_data_type_in_ << std::endl;
  std::cout << "LoopCount = " << this->flags_->loop_count_ << std::endl;
  std::cout << "DeviceType = " << this->flags_->device_ << std::endl;
  std::cout << "AccuracyThreshold = " << this->flags_->accuracy_threshold_ << std::endl;
  std::cout << "WarmUpLoopCount = " << this->flags_->warm_up_loop_count_ << std::endl;
  std::cout << "NumThreads = " << this->flags_->num_threads_ << std::endl;
  std::cout << "Fp16Priority = " << this->flags_->enable_fp16_ << std::endl;
  std::cout << "calibDataPath = " << this->flags_->benchmark_data_file_ << std::endl;
  if (this->flags_->loop_count_ < 1) {
    MS_LOG(ERROR) << "LoopCount:" << this->flags_->loop_count_ << " must be greater than 0";
    std::cerr << "LoopCount:" << this->flags_->loop_count_ << " must be greater than 0" << std::endl;
    return RET_ERROR;
  }

  if (this->flags_->num_threads_ < 1) {
    MS_LOG(ERROR) << "numThreads:" << this->flags_->num_threads_ << " must be greater than 0";
    std::cerr << "numThreads:" << this->flags_->num_threads_ << " must be greater than 0" << std::endl;
    return RET_ERROR;
  }
  static std::vector<std::string> CPU_BIND_MODE_MAP = {"NO_BIND", "HIGHER_CPU", "MID_CPU"};
  if (this->flags_->cpu_bind_mode_ >= 1) {
    MS_LOG(INFO) << "cpuBindMode = " << CPU_BIND_MODE_MAP[this->flags_->cpu_bind_mode_];
    std::cout << "cpuBindMode = " << CPU_BIND_MODE_MAP[this->flags_->cpu_bind_mode_] << std::endl;
  } else {
    MS_LOG(INFO) << "cpuBindMode = NO_BIND";
    std::cout << "cpuBindMode = NO_BIND" << std::endl;
  }

  this->flags_->in_data_type_ = this->flags_->in_data_type_in_ == "img" ? kImage : kBinary;

  if (!flags_->benchmark_data_type_.empty()) {
    if (data_type_map_.find(flags_->benchmark_data_type_) == data_type_map_.end()) {
      MS_LOG(ERROR) << "CalibDataType not supported: " << flags_->benchmark_data_type_.c_str();
      return RET_ERROR;
    }
    msCalibDataType = data_type_map_.at(flags_->benchmark_data_type_);
    MS_LOG(INFO) << "CalibDataType = " << flags_->benchmark_data_type_.c_str();
    std::cout << "CalibDataType = " << flags_->benchmark_data_type_.c_str() << std::endl;
  }

  if (flags_->model_file_.empty()) {
    MS_LOG(ERROR) << "modelPath is required";
    std::cerr << "modelPath is required" << std::endl;
    return 1;
  }
  flags_->InitInputDataList();
  flags_->InitResizeDimsList();
  if (!flags_->resize_dims_.empty() && !flags_->input_data_list_.empty() &&
      flags_->resize_dims_.size() != flags_->input_data_list_.size()) {
    MS_LOG(ERROR) << "Size of input resizeDims should be equal to size of input inDataPath";
    std::cerr << "Size of input resizeDims should be equal to size of input inDataPath" << std::endl;
    return RET_ERROR;
  }

  if (flags_->device_ != "CPU" && flags_->device_ != "GPU" && flags_->device_ != "NPU") {
    MS_LOG(ERROR) << "Device type:" << flags_->device_ << " is not supported.";
    std::cerr << "Device type:" << flags_->device_ << " is not supported." << std::endl;
    return RET_ERROR;
  }

  if (flags_->time_profiling_ || flags_->perf_profiling_) {
    if (flags_->time_profiling_ && flags_->perf_profiling_) {
      MS_LOG(INFO) << "time_profiling is enabled, will not run perf_profiling.";
    }
    auto status = InitCallbackParameter();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Init callback Parameter failed.";
      std::cerr << "Init callback Parameter failed." << std::endl;
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int Benchmark::PrintResult(const std::vector<std::string> &title,
                           const std::map<std::string, std::pair<int, float>> &result) {
  std::vector<size_t> columnLenMax(5);
  std::vector<std::vector<std::string>> rows;

  for (auto &iter : result) {
    char stringBuf[5][100] = {};
    std::vector<std::string> columns;
    size_t len = 0;

    len = iter.first.size();
    if (len > columnLenMax.at(0)) {
      columnLenMax.at(0) = len + 4;
    }
    columns.push_back(iter.first);

    len = snprintf(stringBuf[1], sizeof(stringBuf[1]), "%f", iter.second.second / float_t(flags_->loop_count_));
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
  for (auto &row : rows) {
    for (int j = 0; j < 5; j++) {
      auto printBuf = row[j];
      printBuf.resize(columnLenMax.at(j), ' ');
      printf("%s\t", printBuf.c_str());
    }
    printf("\n");
  }
  return RET_OK;
}

#ifdef ENABLE_ARM64
int Benchmark::PrintPerfResult(const std::vector<std::string> &title,
                               const std::map<std::string, std::pair<int, struct PerfCount>> &result) {
  std::vector<size_t> columnLenMax(5);
  std::vector<std::vector<std::string>> rows;

  for (auto &iter : result) {
    char stringBuf[5][100] = {};
    std::vector<std::string> columns;
    size_t len = 0;

    len = iter.first.size();
    if (len > columnLenMax.at(0)) {
      columnLenMax.at(0) = len + 4;
    }
    columns.push_back(iter.first);

    float tmp = float_t(flags_->num_threads_) * iter.second.second.value[0] / float_t(flags_->loop_count_) / 1000.0f;
    len = snprintf(stringBuf[1], sizeof(stringBuf[1]), "%.2f", tmp);
    if (len > columnLenMax.at(1)) {
      columnLenMax.at(1) = len + 4;
    }
    columns.emplace_back(stringBuf[1]);

    len = snprintf(stringBuf[2], sizeof(stringBuf[2]), "%f", iter.second.second.value[0] / op_cost_total_);
    if (len > columnLenMax.at(2)) {
      columnLenMax.at(2) = len + 4;
    }
    columns.emplace_back(stringBuf[2]);

    tmp = float_t(flags_->num_threads_) * iter.second.second.value[1] / float_t(flags_->loop_count_) / 1000.0f;
    len = snprintf(stringBuf[3], sizeof(stringBuf[3]), "%.2f", tmp);
    if (len > columnLenMax.at(3)) {
      columnLenMax.at(3) = len + 4;
    }
    columns.emplace_back(stringBuf[3]);

    len = snprintf(stringBuf[4], sizeof(stringBuf[4]), "%f", iter.second.second.value[1] / op_cost2_total_);
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
  for (auto &row : rows) {
    for (int j = 0; j < 5; j++) {
      auto printBuf = row[j];
      printBuf.resize(columnLenMax.at(j), ' ');
      printf("%s\t", printBuf.c_str());
    }
    printf("\n");
  }
  return RET_OK;
}
#endif

Benchmark::~Benchmark() {
  for (const auto &iter : this->benchmark_data_) {
    delete (iter.second);
  }
  this->benchmark_data_.clear();
  delete (session_);
}

int RunBenchmark(int argc, const char **argv) {
  BenchmarkFlags flags;
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

  Benchmark benchmark(&flags);
  auto status = benchmark.Init();
  if (status != 0) {
    MS_LOG(ERROR) << "Benchmark init Error : " << status;
    std::cerr << "Benchmark init Error : " << status << std::endl;
    return RET_ERROR;
  }

  status = benchmark.RunBenchmark();
  if (status != 0) {
    MS_LOG(ERROR) << "Run Benchmark "
                  << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
                  << " Failed : " << status;
    std::cerr << "Run Benchmark " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
              << " Failed : " << status << std::endl;
    return RET_ERROR;
  }

  MS_LOG(INFO) << "Run Benchmark " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
               << " Success.";
  std::cout << "Run Benchmark " << flags.model_file_.substr(flags.model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
            << " Success." << std::endl;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
