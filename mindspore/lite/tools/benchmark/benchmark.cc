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
#include <utility>
#include <functional>
#include <algorithm>
#include "include/context.h"
#include "include/ms_tensor.h"
#include "include/version.h"
#include "schema/model_generated.h"
#include "src/common/common.h"
#include "src/tensor.h"
#include "nnacl/nnacl_common.h"
#ifdef ENABLE_ARM64
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>
#include <unistd.h>
#endif

namespace mindspore {
namespace lite {
namespace {
constexpr float kNumUsPerMs = 1000.;
constexpr auto kBenchmarkInputNames = "MSLITE_BENCH_INPUT_NAMES";
}  // namespace

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
      status = GenerateRandomData(tensor->Size(), input_data, static_cast<float>(tensor->data_type()));
    }
    if (status != RET_OK) {
      std::cerr << "GenerateRandomData for inTensor failed: " << status << std::endl;
      MS_LOG(ERROR) << "GenerateRandomData for inTensor failed:" << status;
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

int Benchmark::GetDataTypeByTensorName(const std::string &tensor_name) {
  auto tensor = session_->GetOutputByTensorName(tensor_name);
  if (tensor != nullptr) {
    return tensor->data_type();
  } else {
    return kTypeUnknown;
  }
}

void Benchmark::InitContext(const std::shared_ptr<Context> &context) {
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
    npu_device_ctx.device_info_.npu_device_info_.frequency_ = AiModelDescription_Frequency_HIGH;
    context->device_list_.push_back(npu_device_ctx);
  }

  context->thread_num_ = flags_->num_threads_;
  context->enable_parallel_ = flags_->enable_parallel_;
}

int Benchmark::CompareOutput() {
  std::cout << "================ Comparing Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  // check the output tensor name.
  if (this->benchmark_tensor_names_ != session_->GetOutputTensorNames()) {
    MS_LOG(ERROR) << "The output tensor name is wrong.";
    return RET_ERROR;
  }
  for (const auto &calib_tensor : benchmark_data_) {
    std::string tensor_name = calib_tensor.first;
    tensor::MSTensor *tensor = session_->GetOutputByTensorName(tensor_name);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Get tensor failed, tensor name: " << tensor_name;
      return RET_ERROR;
    }
    int ret;
    if (tensor->data_type() == kObjectTypeString) {
      std::vector<std::string> output_strings = MSTensorToStrings(tensor);
      ret = CompareStringData(tensor_name, calib_tensor.second->strings_data, output_strings);
    } else {
      ret = CompareDataGetTotalBiasAndSize(tensor_name, tensor, &total_bias, &total_size);
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
      bias = CompareData<float, int>(name, tensor->shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      bias = CompareData<int8_t, int>(name, tensor->shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeUInt8: {
      bias = CompareData<uint8_t, int>(name, tensor->shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      bias = CompareData<int32_t, int>(name, tensor->shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      bias = CompareData<int16_t, int>(name, tensor->shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeBool: {
      bias = CompareData<bool, int>(name, tensor->shape(), mutableData);
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
    auto inputs = session_->GetInputs();
    for (auto tensor : inputs) {
      tensor->MutableData();  // prepare data
    }
    session_->BindThread(true);
    auto start = GetTimeUs();
    auto status = session_->RunGraph(before_call_back_, after_call_back_);
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
  status = session_->RunGraph(before_call_back_, after_call_back_);
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

  (void)InitContext(context);

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
  if (model != nullptr && !flags_->dump_tensor_data_) {
    model->Free();
  }

  ms_inputs_ = session_->GetInputs();
  auto end_prepare_time = GetTimeUs();
  MS_LOG(INFO) << "PrepareTime = " << static_cast<float>(end_prepare_time - start_prepare_time) / kNumUsPerMs << " ms";
  std::cout << "PrepareTime = " << static_cast<float>(end_prepare_time - start_prepare_time) / kNumUsPerMs << " ms"
            << std::endl;

  // Check input names
  if (CheckInputNames() != RET_OK) {
    MS_LOG(ERROR) << "Check input names failed.";
    return RET_ERROR;
  }

  // Load input
  MS_LOG(INFO) << "start generate input data";
  auto status = LoadInput();
  if (status != 0) {
    MS_LOG(ERROR) << "Generate input data error";
    return status;
  }
  if (!flags_->benchmark_data_file_.empty()) {
    status = MarkAccuracy();
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
  if (flags_->dump_tensor_data_) {
    std::cout << "Dumped file is saved to : " + dump_file_output_dir_ << std::endl;
  }
  return RET_OK;
}

int Benchmark::InitTimeProfilingCallbackParameter() {
  // before callback
  before_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                          const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                          const CallBackParam &call_param) {
    if (before_inputs.empty()) {
      MS_LOG(INFO) << "The num of beforeInputs is empty";
    }
    if (before_outputs.empty()) {
      MS_LOG(INFO) << "The num of beforeOutputs is empty";
    }
    if (op_times_by_type_.find(call_param.node_type) == op_times_by_type_.end()) {
      op_times_by_type_.insert(std::make_pair(call_param.node_type, std::make_pair(0, 0.0f)));
    }
    if (op_times_by_name_.find(call_param.node_name) == op_times_by_name_.end()) {
      op_times_by_name_.insert(std::make_pair(call_param.node_name, std::make_pair(0, 0.0f)));
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

    float cost = static_cast<float>(opEnd - op_begin_) / kNumUsPerMs;
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
                          const CallBackParam &call_param) {
    if (before_inputs.empty()) {
      MS_LOG(INFO) << "The num of beforeInputs is empty";
    }
    if (before_outputs.empty()) {
      MS_LOG(INFO) << "The num of beforeOutputs is empty";
    }
    if (op_perf_by_type_.find(call_param.node_type) == op_perf_by_type_.end()) {
      op_perf_by_type_.insert(std::make_pair(call_param.node_type, std::make_pair(0, zero)));
    }
    if (op_perf_by_name_.find(call_param.node_name) == op_perf_by_name_.end()) {
      op_perf_by_name_.insert(std::make_pair(call_param.node_name, std::make_pair(0, zero)));
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
    if (read(perf_fd, &res, sizeof(struct PerfResult)) == -1) {
      MS_LOG(ERROR) << "Failed to read perf_fd";
      return false;
    }

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

namespace {
template <typename T>
std::string DataToString(void *data, size_t data_number, size_t print_len = 40) {
  if (data == nullptr) {
    return "Data of tensor is nullptr";
  }
  auto t_data = static_cast<T *>(data);
  std::ostringstream sstream;
  for (size_t i = 0; i < print_len && i < data_number; i++) {
    sstream << " " << t_data[i];
  }
  return sstream.str();
}

std::string DumpMSTensor(tensor::MSTensor *tensor) {
  if (tensor == nullptr) {
    return "Tensor is nullptr";
  }
  std::ostringstream oss;
  oss << " DataType: " << tensor->data_type();
  oss << " Shape:";
  for (auto &dim : tensor->shape()) {
    oss << " " << dim;
  }
  oss << std::endl << " Data:";
  switch (tensor->data_type()) {
    case kNumberTypeFloat32: {
      oss << DataToString<float>(tensor->data(), tensor->ElementsNum());
    } break;
    case kNumberTypeFloat16: {
      oss << DataToString<int16_t>(tensor->data(), tensor->ElementsNum());
    } break;
    case kNumberTypeInt32: {
      oss << DataToString<int32_t>(tensor->data(), tensor->ElementsNum());
    } break;
    case kNumberTypeInt16: {
      oss << DataToString<int16_t>(tensor->data(), tensor->ElementsNum());
    } break;
    case kNumberTypeInt8: {
      oss << DataToString<int8_t>(tensor->data(), tensor->ElementsNum());
    } break;
    default:
      oss << "Unsupported data type to print";
      break;
  }
  return oss.str();
}

std::string GenerateOutputFileName(tensor::MSTensor *tensor, const std::string &op_name, const std::string &file_type,
                                   const size_t &idx) {
  std::string file_name = op_name;
  auto pos = file_name.find_first_of('/');
  while (pos != std::string::npos) {
    file_name.replace(pos, 1, ".");
    pos = file_name.find_first_of('/');
  }
  file_name += "_" + file_type + "_" + std::to_string(idx) + "_shape_";
  for (const auto &dim : tensor->shape()) {
    file_name += std::to_string(dim) + "_";
  }
  if (kTypeIdMap.find(tensor->data_type()) != kTypeIdMap.end()) {
    file_name += kTypeIdMap.at(tensor->data_type());
  }
  auto tensor_format = static_cast<lite::Tensor *>(tensor)->format();
  if (kTensorFormatMap.find(tensor_format) != kTensorFormatMap.end()) {
    file_name += "_" + kTensorFormatMap.at(tensor_format) + ".bin";
  }
  return file_name;
}
}  // namespace

int Benchmark::InitPrintTensorDataCallbackParameter() {
  // before callback
  before_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                          const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                          const CallBackParam &call_param) { return true; };

  // after callback
  after_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                         const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                         const CallBackParam &call_param) {
    std::cout << "================================================================" << std::endl;
    std::cout << call_param.node_name << " inputs : " << std::endl;
    for (auto ms_tensor : after_inputs) {
      std::cout << DumpMSTensor(ms_tensor) << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << call_param.node_name << " outputs : " << std::endl;
    for (const auto ms_tensor : after_outputs) {
      std::cout << DumpMSTensor(ms_tensor) << std::endl;
    }
    std::cout << "================================================================" << std::endl;
    return true;
  };
  return RET_OK;
}
int Benchmark::InitDumpTensorDataCallbackParameter() {
  // before callback
  before_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                          const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                          const CallBackParam &call_param) {
    auto dump_mode = dump_cfg_json_[dump::kSettings][dump::kMode].get<int>();
    auto input_output_mode = dump_cfg_json_[dump::kSettings][dump::kInputOutput].get<int>();
    auto kernels = dump_cfg_json_[dump::kSettings][dump::kKernels].get<std::vector<std::string>>();
    if (dump_mode == 0 || std::find(kernels.begin(), kernels.end(), call_param.node_name) != kernels.end()) {
      if (input_output_mode == DUMP_MODE_ALL || input_output_mode == DUMP_MODE_INPUT) {
        for (size_t i = 0; i < before_inputs.size(); i++) {
          auto ms_tensor = before_inputs.at(i);
          auto file_name = GenerateOutputFileName(ms_tensor, call_param.node_name, "input", i);
          auto abs_file_path = dump_file_output_dir_ + "/" + file_name;
          if (WriteToBin(abs_file_path, ms_tensor->data(), ms_tensor->Size()) != RET_OK) {  // save to file
            MS_LOG(ERROR) << "write tensor data to file failed.";
            return false;
          }
        }
      }
    }
    return true;
  };

  // after callback
  after_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                         const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                         const CallBackParam &call_param) {
    auto dump_mode = dump_cfg_json_[dump::kSettings][dump::kMode].get<int>();
    auto input_output_mode = dump_cfg_json_[dump::kSettings][dump::kInputOutput].get<int>();
    auto kernels = dump_cfg_json_[dump::kSettings][dump::kKernels].get<std::vector<std::string>>();
    if (dump_mode == 0 || std::find(kernels.begin(), kernels.end(), call_param.node_name) != kernels.end()) {
      if (input_output_mode == DUMP_MODE_ALL || input_output_mode == DUMP_MODE_OUTPUT) {
        for (size_t i = 0; i < after_outputs.size(); i++) {
          auto ms_tensor = after_outputs.at(i);
          auto file_name = GenerateOutputFileName(ms_tensor, call_param.node_name, "output", i);
          auto abs_file_path = dump_file_output_dir_ + "/" + file_name;
          if (WriteToBin(abs_file_path, ms_tensor->data(), ms_tensor->Size()) != RET_OK) {  // save to file
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

int Benchmark::CheckInputNames() {
  auto bench_inputs = std::getenv(kBenchmarkInputNames);
  if (bench_inputs == nullptr || std::string(bench_inputs).empty()) {
    MS_LOG(WARNING) << "The benchmark input names is not set.";
    return RET_OK;
  }
  auto input_names = StrSplit(bench_inputs, std::string(DELIM_COMMA));
  std::vector<std::string> ms_input_names(ms_inputs_.size());
  std::transform(ms_inputs_.begin(), ms_inputs_.end(), ms_input_names.begin(),
                 [](mindspore::tensor::MSTensor *input) { return input->tensor_name(); });
  if (ms_input_names != input_names) {
    MS_LOG(ERROR) << "The input names are not the same as ones of the original model.";
    return RET_ERROR;
  }
  return RET_OK;
}

Benchmark::~Benchmark() { delete (session_); }
}  // namespace lite
}  // namespace mindspore
