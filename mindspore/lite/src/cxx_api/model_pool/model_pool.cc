/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/cxx_api/model_pool/model_pool.h"
#include <unistd.h>
#include <future>
#include "src/common/log.h"
#include "include/lite_types.h"
#include "src/common/config_file.h"
#include "src/runtime/inner_allocator.h"
#include "src/common//file_utils.h"
#include "src/pack_weight_manager.h"
#include "src/runtime/numa_adapter.h"
#include "src/common/common.h"

namespace mindspore {
namespace {
constexpr int32_t kNumThreads = 4;
int GetCoreNum() {
  int core_num = 1;
#if defined(_MSC_VER) || defined(_WIN32)
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  core_num = sysinfo.dwNumberOfProcessors;
#else
  core_num = sysconf(_SC_NPROCESSORS_CONF);
#endif
  return core_num;
}

void SetNumaBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, int thread_num, int node_id) {
  if (UNLIKELY(thread_num == 0)) {
    MS_LOG(ERROR) << "thread num is zero.";
    return;
  }
  std::vector<int> cpu_list = numa::NUMAAdapter::GetInstance()->GetCPUList(node_id);
  auto cpu_num = cpu_list.size();
  if (cpu_num == 0) {
    return;
  }
  std::vector<int> bind_id;
  bind_id.reserve(thread_num);
  all_model_bind_list->reserve(cpu_num / thread_num + 1);
  bind_id.emplace_back(cpu_list[0]);
  for (size_t i = 1; i < cpu_num; ++i) {
    if (i % thread_num == 0) {
      all_model_bind_list->emplace_back(bind_id);
      bind_id.clear();
    }
    bind_id.emplace_back(cpu_list[i]);
  }
  if (!bind_id.empty()) {
    all_model_bind_list->emplace_back(bind_id);
  }
}
}  // namespace

void ModelPool::SetBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, int thread_num) {
  if (thread_num == 0) {
    MS_LOG(ERROR) << "thread num is zero.";
    return;
  }
  int core_num = GetCoreNum();
  int core_id = 0;
  for (size_t i = 0; i < num_models_; i++) {
    std::vector<int> bind_id;
    for (int j = 0; j < thread_num; j++) {
      if (core_id >= core_num) {
        core_id = 0;
      }
      bind_id.push_back(core_id);
      core_id++;
    }
    all_model_bind_list->push_back(bind_id);
  }
}

ModelPool *ModelPool::GetInstance() {
  static ModelPool instance;
  return &instance;
}

std::shared_ptr<Context> ModelPool::InitContext(const std::shared_ptr<RunnerConfig> &runner_config) {
  auto model_context = std::make_shared<mindspore::Context>();
  if (model_context == nullptr) {
    MS_LOG(ERROR) << "New context failed in ModelPool.";
    return nullptr;
  }
  if (runner_config != nullptr) {
    model_context = runner_config->context;
    auto device_list = model_context->MutableDeviceInfo();
    if (device_list.size() != 1) {
      MS_LOG(ERROR) << "model pool only support device num 1.";
      return nullptr;
    }
    auto device = device_list.front();
    if (device->GetDeviceType() != kCPU && device->GetDeviceType() != kGPU) {
      MS_LOG(ERROR) << "model pool only support cpu or gpu type.";
      return nullptr;
    }
    auto cpu_context = device->Cast<CPUDeviceInfo>();
    auto enable_fp16 = cpu_context->GetEnableFP16();
    if (enable_fp16) {
      MS_LOG(ERROR) << "model pool not support enable fp16.";
      return nullptr;
    }

    if (device->GetDeviceType() == kGPU) {
      num_models_ = 1;
    } else {
      num_models_ = GetCoreNum() / static_cast<int>(model_context->GetThreadNum());
    }
  } else {
    MS_LOG(DEBUG) << "use default config.";
    model_context->SetThreadNum(kNumThreads);
    model_context->SetEnableParallel(true);
    model_context->SetThreadAffinity(lite::HIGHER_CPU);
    auto &device_list = model_context->MutableDeviceInfo();
    auto device_info = std::make_shared<CPUDeviceInfo>();
    device_info->SetEnableFP16(false);
    device_list.push_back(device_info);
  }
  return model_context;
}

ModelPoolContex ModelPool::CreateModelContext(const std::shared_ptr<RunnerConfig> &runner_config) {
  auto model_context = InitContext(runner_config);
  if (model_context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return {};
  }
  if (model_context->GetThreadNum() < 1) {
    MS_LOG(ERROR) << "Invalid thread num " << model_context->GetThreadNum();
    return {};
  }
  int node_id = -1;
  if (numa::NUMAAdapter::GetInstance()->Available()) {
    node_id = 0;
    num_models_ =
      numa::NUMAAdapter::GetInstance()->GetCPUList(node_id).size() / static_cast<int>(model_context->GetThreadNum());
  } else {
    num_models_ = GetCoreNum() / static_cast<int>(model_context->GetThreadNum());
  }
  ModelPoolContex model_pool_context;
  std::vector<std::vector<int>> all_model_bind_list;
  if (model_context->GetThreadAffinityMode() == lite::HIGHER_CPU) {
    if (numa::NUMAAdapter::GetInstance()->Available()) {
      SetNumaBindStrategy(&all_model_bind_list, static_cast<int>(model_context->GetThreadNum()), node_id);
    } else {
      SetBindStrategy(&all_model_bind_list, static_cast<int>(model_context->GetThreadNum()));
    }
  } else if (model_context->GetThreadAffinityMode() == lite::MID_CPU) {
    MS_LOG(ERROR) << "not support bind MID_CPU.";
    return {};
  }
  for (size_t i = 0; i < num_models_; i++) {
    auto context = std::make_shared<Context>();
    if (context == nullptr) {
      MS_LOG(ERROR) << "New Context failed.";
      return {};
    }
    context->SetThreadNum(model_context->GetThreadNum());
    context->SetEnableParallel(model_context->GetEnableParallel());
    if (model_context->GetThreadAffinityMode() != lite::NO_BIND) {
      // bind by core id
      context->SetThreadAffinity(all_model_bind_list[i]);
    } else {
      // not bind core
      context->SetThreadAffinity(model_context->GetThreadAffinityMode());
    }
    auto &new_device_list = context->MutableDeviceInfo();
    std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
    device_info->SetEnableFP16(false);
    new_device_list.push_back(device_info);
    model_pool_context.push_back(context);
  }
  return model_pool_context;
}

std::vector<MSTensor> ModelPool::GetInputs() {
  if (model_inputs_.empty()) {
    MS_LOG(ERROR) << "model input is empty.";
    return {};
  }
  return model_inputs_;
}

std::vector<MSTensor> ModelPool::GetOutputs() {
  if (model_outputs_.empty()) {
    MS_LOG(ERROR) << "model output is empty.";
    return {};
  }
  return model_outputs_;
}

Status ModelPool::Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config,
                       const Key &dec_key, const std::string &dec_mode) {
  auto model_pool_context = CreateModelContext(runner_config);
  if (model_pool_context.empty()) {
    MS_LOG(ERROR) << "CreateModelContext failed, context is empty.";
    return kLiteError;
  }
  size_t size = 0;
  graph_buf_ = lite::ReadFile(model_path.c_str(), &size);
  if (graph_buf_ == nullptr) {
    MS_LOG(ERROR) << "read file failed.";
    return kLiteError;
  }
  lite::PackWeightManager::GetInstance()->InitWeightManagerByBuf(graph_buf_);
  std::shared_ptr<ModelThread> model_thread = nullptr;
  int node_id = -1;
  if (numa::NUMAAdapter::GetInstance()->Available()) {
    node_id = 0;
  }
  for (size_t i = 0; i < num_models_; i++) {
    model_thread = std::make_shared<ModelThread>();
    auto status = model_thread->Init(graph_buf_, size, model_pool_context[i], dec_key, dec_mode, node_id);
    if (status != kSuccess) {
      MS_LOG(ERROR) << " model thread init failed.";
      return kLiteError;
    }
    model_thread_vec_.push_back(std::thread(&ModelThread::Run, model_thread));
  }
  if (model_thread != nullptr) {
    model_inputs_ = model_thread->GetInputs();
    model_outputs_ = model_thread->GetOutputs();
  }
  return kSuccess;
}

Status ModelPool::SplitInputTensorByBatch(const std::vector<MSTensor> &inputs,
                                          std::vector<std::vector<MSTensor>> *new_inputs, size_t batch_split_num) {
  if (batch_split_num == 0) {
    MS_LOG(ERROR) << "batch_split_num is zero.";
    return kLiteError;
  }
  auto batch = inputs[0].Shape()[0];
  std::vector<size_t> split_batch;
  size_t batch_sum = 0;
  size_t per_batch = batch / batch_split_num;
  for (size_t i = 0; i < batch_split_num - 1; i++) {
    split_batch.push_back(per_batch);
    batch_sum += per_batch;
  }
  split_batch.push_back(batch - batch_sum);
  std::vector<std::vector<std::vector<int64_t>>> all_input_shape;
  std::vector<size_t> input_data_split_size(inputs.size(), 0);
  for (size_t k = 0; k < batch_split_num; k++) {  // do for batch
    std::vector<std::vector<int64_t>> inputs_shape;
    std::vector<MSTensor> new_inputs_tensor;
    for (size_t i = 0; i < inputs.size(); i++) {  // do for input
      std::vector<int64_t> shape;
      size_t input_size = split_batch[k];
      shape.push_back(split_batch[k]);
      for (size_t j = 1; j < inputs[i].Shape().size(); j++) {  // do for dims
        shape.push_back(inputs[i].Shape()[j]);
        input_size *= inputs[i].Shape()[j];
      }
      inputs_shape.push_back(shape);
      if (inputs[i].DataType() == static_cast<enum DataType>(kNumberTypeFloat32)) {
        if (input_size * sizeof(float) > MAX_MALLOC_SIZE) {
          MS_LOG(ERROR) << "malloc size is wrong.";
          return kLiteError;
        }
        auto data =
          reinterpret_cast<float *>(const_cast<MSTensor &>(inputs[i]).MutableData()) + input_data_split_size[i];
        auto new_tensor = MSTensor(inputs[i].Name(), static_cast<enum DataType>(kNumberTypeFloat32), shape, data,
                                   input_size * sizeof(float));
        if (new_tensor == nullptr) {
          MS_LOG(ERROR) << "create tensor failed.";
          return kLiteError;
        }
        new_inputs_tensor.push_back(new_tensor);
        input_data_split_size[i] += input_size;
      } else if (inputs[i].DataType() == static_cast<enum DataType>(kNumberTypeInt32)) {
        if (input_size * sizeof(int32_t) > MAX_MALLOC_SIZE) {
          MS_LOG(ERROR) << "malloc size is wrong.";
          return kLiteError;
        }
        auto data =
          reinterpret_cast<int32_t *>(const_cast<MSTensor &>(inputs[i]).MutableData()) + input_data_split_size[i];
        auto new_tensor = MSTensor(inputs[i].Name(), static_cast<enum DataType>(kNumberTypeInt32), shape, data,
                                   input_size * sizeof(int32_t));
        if (new_tensor == nullptr) {
          MS_LOG(ERROR) << "create tensor failed.";
          return kLiteError;
        }
        new_inputs_tensor.push_back(new_tensor);
        input_data_split_size[i] += input_size;
      } else {
        MS_LOG(ERROR) << "not support data type in split batch.";
        return kLiteError;
      }
    }
    new_inputs->push_back(new_inputs_tensor);
    all_input_shape.push_back(inputs_shape);
  }
  return kSuccess;
}

Status ModelPool::SplitOutputTensorByBatch(std::vector<std::vector<MSTensor>> *new_outputs,
                                           std::vector<MSTensor> *outputs, size_t batch_split_num) {
  if (batch_split_num == 0) {
    MS_LOG(ERROR) << "batch_split_num is zero.";
    return kLiteError;
  }
  for (size_t i = 0; i < batch_split_num; i++) {
    std::vector<MSTensor> new_output;
    for (size_t tensor_num_idx = 0; tensor_num_idx < outputs->size(); tensor_num_idx++) {
      if (outputs->at(tensor_num_idx).MutableData() != nullptr && outputs->at(tensor_num_idx).DataSize() != 0) {
        is_user_data_ = true;
        auto data = reinterpret_cast<float *>(outputs->at(tensor_num_idx).MutableData()) +
                    outputs->at(tensor_num_idx).Shape().at(0) / batch_split_num * i;
        auto out_tensor =
          MSTensor(outputs->at(tensor_num_idx).Name(), outputs->at(tensor_num_idx).DataType(), {}, data, 0);
        new_output.push_back(out_tensor);
      }
    }
    new_outputs->push_back(new_output);
  }
  return kSuccess;
}

Status ModelPool::ConcatPredictOutput(std::vector<std::vector<MSTensor>> *outputs, std::vector<MSTensor> *new_outputs) {
  if (outputs->empty()) {
    MS_LOG(ERROR) << "output is empty";
    return kLiteError;
  }
  for (size_t i = 0; i < outputs->at(0).size(); i++) {
    std::vector<int64_t> output_tensor_shape = outputs->at(0)[i].Shape();
    if (output_tensor_shape.empty()) {
      MS_LOG(ERROR) << "output_tensor_shape is empty";
      return kLiteError;
    }
    size_t all_data_size = 0;
    size_t all_batch_size = 0;
    std::vector<size_t> per_bacth_data_size;
    for (size_t batch = 0; batch < outputs->size(); batch++) {
      per_bacth_data_size.push_back(all_data_size);
      all_data_size += outputs->at(batch).at(i).DataSize();
      all_batch_size += outputs->at(batch).at(i).Shape().front();
    }
    output_tensor_shape[0] = all_batch_size;
    if (is_user_data_) {
      new_outputs->at(i).SetShape(output_tensor_shape);
      continue;
    }
    auto all_out_data = malloc(all_data_size);
    if (all_out_data == nullptr) {
      MS_LOG(ERROR) << "all_out_data is nullptr.";
      return kLiteError;
    }
    for (size_t j = 0; j < outputs->size(); j++) {
      void *out_data = outputs->at(j).at(i).MutableData();
      if (out_data == nullptr) {
        free(all_out_data);
        all_out_data = nullptr;
        MS_LOG(ERROR) << "output data is nullptr.";
        return kLiteError;
      }
      memcpy(reinterpret_cast<float *>(all_out_data) + per_bacth_data_size[j] / sizeof(float),
             reinterpret_cast<float *>(out_data), outputs->at(j)[i].DataSize());
    }
    auto new_tensor = mindspore::MSTensor::CreateTensor(outputs->at(0)[i].Name(), outputs->at(0)[i].DataType(),
                                                        output_tensor_shape, all_out_data, all_data_size);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "create tensor failed.";
      return kLiteError;
    }
    if (all_out_data != nullptr) {
      free(all_out_data);
      all_out_data = nullptr;
    }
    new_outputs->push_back(*new_tensor);
    delete new_tensor;
  }
  return kSuccess;
}

Status ModelPool::FreeSplitTensor(std::vector<std::vector<MSTensor>> *new_inputs,
                                  std::vector<std::vector<MSTensor>> *new_outputs) {
  for (size_t i = 0; i < new_inputs->size(); i++) {
    for (size_t j = 0; j < new_inputs->at(i).size(); j++) {
      new_inputs->at(i).at(j).SetData(nullptr);
    }
  }
  new_inputs->clear();
  if (is_user_data_) {
    for (size_t i = 0; i < new_outputs->size(); i++) {
      for (size_t j = 0; j < new_outputs->at(i).size(); j++) {
        new_outputs->at(i).at(j).SetData(nullptr);
      }
    }
    new_outputs->clear();
  }
  return kSuccess;
}

Status ModelPool::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  mtx_split_task_.lock();
  auto wait_model_num = PredictTaskQueue::GetInstance()->GetWaitModelNum();
  auto batch = inputs[0].Shape()[0];
  if (PredictTaskQueue::GetInstance()->GetTaskNum() == 0 && wait_model_num > 1 && batch >= wait_model_num) {
    size_t batch_split_num = PredictTaskQueue::GetInstance()->GetWaitModelNum();
    PredictTaskQueue::GetInstance()->DecreaseWaitModelNum(batch_split_num);
    std::vector<std::vector<MSTensor>> new_inputs;
    std::vector<std::vector<MSTensor>> new_outputs;
    auto status = SplitInputTensorByBatch(inputs, &new_inputs, batch_split_num);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "model pool split input tensor by batch failed.";
      return kLiteError;
    }
    status = SplitOutputTensorByBatch(&new_outputs, outputs, batch_split_num);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "model pool split output tensor by batch failed.";
      return kLiteError;
    }

    std::vector<std::shared_ptr<PredictTask>> tasks;
    for (size_t i = 0; i < batch_split_num; i++) {
      auto predict_task = std::make_shared<PredictTask>(&new_inputs[i], &new_outputs.at(i), before, after);
      PredictTaskQueue::GetInstance()->PushPredictTask(predict_task);
      tasks.push_back(predict_task);
    }
    mtx_split_task_.unlock();
    for (size_t i = 0; i < batch_split_num; i++) {
      PredictTaskQueue::GetInstance()->WaitUntilPredictActive(tasks[i]);
    }
    status = ConcatPredictOutput(&new_outputs, outputs);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "ConcatPredictOutput failed.";
      return kLiteError;
    }
    status = FreeSplitTensor(&new_inputs, &new_outputs);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "free split tensor failed.";
      return kLiteError;
    }
  } else {
    if (wait_model_num == 1) {
      PredictTaskQueue::GetInstance()->DecreaseWaitModelNum(1);
    }
    auto predict_task = std::make_shared<PredictTask>(&inputs, outputs, before, after);
    PredictTaskQueue::GetInstance()->PushPredictTask(predict_task);
    mtx_split_task_.unlock();
    PredictTaskQueue::GetInstance()->WaitUntilPredictActive(predict_task);
  }
  return kSuccess;
}

ModelPool::~ModelPool() {
  if (graph_buf_ != nullptr) {
    delete[] graph_buf_;
    graph_buf_ = nullptr;
  }
  for (auto &th : model_thread_vec_) {
    if (th.joinable()) {
      th.join();
    }
  }
}
}  // namespace mindspore
