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
#include "src/runtime/cxx_api/model_pool/model_pool.h"
#include <unistd.h>
#include <future>
#include "src/common/log_adapter.h"
#include "include/lite_types.h"
#include "src/common/config_file.h"
#include "src/runtime/inner_allocator.h"
#include "src/common/file_utils.h"
#include "src/runtime/pack_weight_manager.h"
#include "src/runtime/numa_adapter.h"
#include "src/common/common.h"

namespace mindspore {
namespace {
constexpr int32_t kNumThreads = 8;
constexpr int kNumDeviceInfo = 2;
constexpr int kNumMaxTaskQueueSize = 1000;
constexpr int kNumBindMode = lite::HIGHER_CPU;
constexpr int32_t kNumInterOpParallel = 4;
constexpr bool is_enable_parallel = true;
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
}  // namespace

Status ModelPool::SetNumaBindStrategy(std::vector<std::vector<int>> *all_model_bind_list,
                                      std::vector<int> *numa_node_id, int thread_num) {
  if (MS_UNLIKELY(thread_num == 0)) {
    MS_LOG(ERROR) << "thread num is zero.";
    return kLiteError;
  }
  if (thread_num * static_cast<int>(workers_num_) > GetCoreNum()) {
    MS_LOG(ERROR) << "thread num or worker num is wrong ,not support param.";
    return kLiteNotSupport;
  }
  for (size_t i = 0; i < workers_num_;) {
    std::vector<int> cpu_list = numa::NUMAAdapter::GetInstance()->GetCPUList(used_numa_node_num_);
    if (static_cast<int>(cpu_list.size()) < thread_num) {
      MS_LOG(ERROR) << "one numa node do not have enough cpu core for bind thread.";
      return kLiteError;
    }
    for (size_t j = 0; j < cpu_list.size() / thread_num; j++) {
      std::vector<int> bind_id;
      bind_id.insert(bind_id.begin(), cpu_list.begin() + j * thread_num, cpu_list.begin() + (j + 1) * thread_num);
      all_model_bind_list->push_back(bind_id);
      numa_node_id->push_back(used_numa_node_num_);
      i++;
    }
    used_numa_node_num_++;
  }
  return kSuccess;
}

void ModelPool::SetBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, std::vector<int> *numa_node_id,
                                int thread_num) {
  if (thread_num == 0) {
    MS_LOG(ERROR) << "thread num is zero.";
    return;
  }
  int core_num = GetCoreNum();
  int core_id = 0;
  for (size_t i = 0; i < workers_num_; i++) {
    std::vector<int> bind_id;
    for (int j = 0; j < thread_num; j++) {
      if (core_id >= core_num) {
        core_id = 0;
      }
      bind_id.push_back(core_id);
      core_id++;
    }
    all_model_bind_list->push_back(bind_id);
    numa_node_id->push_back(-1);
  }
}

Status ModelPool::SetDefaultOptimalModelNum(const std::shared_ptr<mindspore::Context> &context) {
  if (context->GetThreadNum() <= 0) {
    MS_LOG(ERROR) << "The number of threads in the context is set to less than 1.";
    return kLiteError;
  }
  if (use_numa_bind_mode_) {
    // now only supports the same number of cores per numa node
    // do not use if there are extra cores
    int one_numa_node_cpu_size = numa::NUMAAdapter::GetInstance()->GetCPUList(0).size();
    if (context->GetThreadNum() > one_numa_node_cpu_size) {
      MS_LOG(ERROR) << "thread num more than numa node cpu cores.";
      return kLiteError;
    } else {
      workers_num_ = one_numa_node_cpu_size / static_cast<int>(context->GetThreadNum()) * numa_node_num_;
    }
  } else {
    // each model binds all kernels in order
    workers_num_ = GetCoreNum() / static_cast<int>(context->GetThreadNum());
  }
  return kSuccess;
}

std::shared_ptr<mindspore::Context> ModelPool::GetDefaultContext() {
  MS_LOG(DEBUG) << "use default config.";
  auto context = std::make_shared<Context>();
  context->SetThreadNum(kNumThreads);
  context->SetEnableParallel(is_enable_parallel);
  context->SetInterOpParallelNum(kNumInterOpParallel);
  context->SetThreadAffinity(kNumBindMode);
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<CPUDeviceInfo>();
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "device_info is nullptr.";
    return nullptr;
  }
  device_info->SetEnableFP16(false);
  device_list.push_back(device_info);
  return context;
}

std::shared_ptr<Context> ModelPool::InitUserDefineContext(const std::shared_ptr<RunnerConfig> &runner_config) {
  auto context = runner_config->GetContext();
  if (context == nullptr) {
    MS_LOG(ERROR) << "user set config context nullptr.";
    return nullptr;
  }
  auto device_list = context->MutableDeviceInfo();
  if (device_list.size() > kNumDeviceInfo) {
    MS_LOG(ERROR) << "model pool only support device CPU or GPU.";
    return nullptr;
  }
  for (size_t i = 0; i < device_list.size(); i++) {
    auto device = device_list[i];
    if (device->GetDeviceType() != kCPU && device->GetDeviceType() != kGPU) {
      MS_LOG(ERROR) << "model pool only support cpu or gpu type.";
      return nullptr;
    }
    if (device->GetDeviceType() == kGPU) {
      return context;
    } else if (device->GetDeviceType() == kCPU) {
      auto cpu_context = device->Cast<CPUDeviceInfo>();
      auto enable_fp16 = cpu_context->GetEnableFP16();
      if (enable_fp16) {
        MS_LOG(ERROR) << "model pool not support enable fp16.";
        return nullptr;
      }
      if (context->GetThreadNum() <= 0) {
        MS_LOG(ERROR) << "Invalid thread num " << context->GetThreadNum();
        return nullptr;
      }
    }
  }
  return context;
}

std::shared_ptr<Context> ModelPool::InitContext(const std::shared_ptr<RunnerConfig> &runner_config) {
  std::shared_ptr<mindspore::Context> context = nullptr;
  if (runner_config != nullptr && runner_config->GetContext() != nullptr) {
    use_numa_bind_mode_ = numa::NUMAAdapter::GetInstance()->Available() &&
                          runner_config->GetContext()->GetThreadAffinityMode() == lite::HIGHER_CPU;
    numa_node_num_ = numa::NUMAAdapter::GetInstance()->NodesNum();
    context = InitUserDefineContext(runner_config);
  } else {
    use_numa_bind_mode_ = numa::NUMAAdapter::GetInstance()->Available();
    numa_node_num_ = numa::NUMAAdapter::GetInstance()->NodesNum();
    context = GetDefaultContext();
  }
  if (context == nullptr) {
    MS_LOG(ERROR) << "Init context failed. status=" << kLiteNullptr;
    return nullptr;
  }
  return context;
}

Status ModelPool::SetModelBindMode(std::vector<std::vector<int>> *all_model_bind_list, std::vector<int> *numa_node_id,
                                   std::shared_ptr<Context> model_context) {
  if (numa::NUMAAdapter::GetInstance()->Available()) {
    auto status =
      SetNumaBindStrategy(all_model_bind_list, numa_node_id, static_cast<int>(model_context->GetThreadNum()));
    if (status != kSuccess) {
      MS_LOG(ERROR) << "SetNumaBindStrategy failed.";
      return kLiteError;
    }
  } else {
    SetBindStrategy(all_model_bind_list, numa_node_id, static_cast<int>(model_context->GetThreadNum()));
  }
  return kSuccess;
}

Status ModelPool::SetWorkersNum(const std::shared_ptr<RunnerConfig> &runner_config,
                                const std::shared_ptr<Context> &context) {
  if ((runner_config != nullptr && runner_config->GetWorkersNum() == 0) || runner_config == nullptr) {
    // the user does not define the number of models, the default optimal number of models is used
    auto status = SetDefaultOptimalModelNum(context);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "SetDefaultOptimalModelNum failed.";
      return kLiteError;
    }
  } else if (runner_config != nullptr && runner_config->GetWorkersNum() > 0) {
    // User defined number of models
    workers_num_ = runner_config->GetWorkersNum();
  } else {
    MS_LOG(ERROR) << "user set worker num" << runner_config->GetWorkersNum() << " < 0";
    return kLiteError;
  }
  return kSuccess;
}

ModelPoolConfig ModelPool::CreateModelPoolConfig(const std::shared_ptr<RunnerConfig> &runner_config) {
  ModelPoolConfig model_pool_config;
  auto init_context = InitContext(runner_config);
  if (init_context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr.";
    return model_pool_config;
  }
  auto status = SetWorkersNum(runner_config, init_context);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "set worker num failed.";
    return model_pool_config;
  }
  auto device_num = init_context->MutableDeviceInfo().size();
  if (device_num == 0) {
    MS_LOG(ERROR) << "device_info is null.";
    return {};
  }
  if (device_num > 1) {
    // GPU
    auto worker_config = std::make_shared<WorkerConfig>();
    if (worker_config == nullptr) {
      MS_LOG(ERROR) << "new worker config failed.";
      return model_pool_config;
    }
    worker_config->context = init_context;
    worker_config->numa_id = -1;
    used_numa_node_num_ = 1;
    workers_num_ = 1;
    model_pool_config.push_back(worker_config);
    return model_pool_config;
  }
  // init all bind list
  std::vector<std::vector<int>> all_model_bind_list;
  std::vector<int> numa_node_id;
  if (init_context->GetThreadAffinityMode() == lite::HIGHER_CPU) {
    status = SetModelBindMode(&all_model_bind_list, &numa_node_id, init_context);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "SetModelBindMode failed.";
      return {};
    }
  } else if (init_context->GetThreadAffinityMode() == lite::MID_CPU) {
    MS_LOG(ERROR) << "not support bind MID_CPU.";
    return {};
  }

  // create all model context
  for (size_t i = 0; i < workers_num_; i++) {
    auto context = std::make_shared<Context>();
    if (context == nullptr) {
      MS_LOG(ERROR) << "New Context failed.";
      return {};
    }
    auto worker_config = std::make_shared<WorkerConfig>();
    if (worker_config == nullptr) {
      MS_LOG(ERROR) << "new worker config failed.";
      return {};
    }
    context->SetThreadNum(init_context->GetThreadNum());
    context->SetEnableParallel(init_context->GetEnableParallel());
    context->SetInterOpParallelNum(init_context->GetInterOpParallelNum());
    if (init_context->GetThreadAffinityMode() != lite::NO_BIND) {
      // bind by core id
      worker_config->numa_id = numa_node_id[i];
      context->SetThreadAffinity(all_model_bind_list[i]);
    } else {
      // not bind core , not use numa
      worker_config->numa_id = -1;
      context->SetThreadAffinity(init_context->GetThreadAffinityMode());
    }
    auto &new_device_list = context->MutableDeviceInfo();
    std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
    if (device_info == nullptr) {
      MS_LOG(ERROR) << "device_info is nullptr.";
      return {};
    }
    std::shared_ptr<Allocator> allocator = nullptr;
    if (init_context->MutableDeviceInfo().front()->GetAllocator() == nullptr) {
      allocator = std::make_shared<DynamicMemAllocator>(worker_config->numa_id);
    } else {
      allocator = init_context->MutableDeviceInfo().front()->GetAllocator();
    }
    if (numa_allocator_.find(worker_config->numa_id) == numa_allocator_.end()) {
      numa_allocator_.insert(std::make_pair(worker_config->numa_id, allocator));
    }
    device_info->SetAllocator(allocator);
    device_info->SetEnableFP16(false);
    new_device_list.push_back(device_info);
    if (runner_config != nullptr) {
      worker_config->config_info = runner_config->GetConfigInfo();
    }
    worker_config->context = context;
    model_pool_config.push_back(worker_config);
  }
  return model_pool_config;
}

std::vector<MSTensor> ModelPool::GetInputs() {
  std::vector<MSTensor> inputs;
  if (model_pool_inputs_.empty()) {
    MS_LOG(ERROR) << "model input is empty.";
    return {};
  }
  for (size_t i = 0; i < model_pool_inputs_.size(); i++) {
    auto tensor = mindspore::MSTensor::CreateTensor(model_pool_inputs_.at(i).Name(),
                                                    model_pool_inputs_.at(i).DataType(), {}, nullptr, 0);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "create tensor failed.";
      return {};
    }
    tensor->SetShape(model_pool_inputs_.at(i).Shape());
    inputs.push_back(*tensor);
    delete tensor;
  }
  return inputs;
}

std::vector<MSTensor> ModelPool::GetOutputs() {
  std::vector<MSTensor> outputs;
  if (model_pool_outputs_.empty()) {
    MS_LOG(ERROR) << "model output is empty.";
    return {};
  }
  for (size_t i = 0; i < model_pool_outputs_.size(); i++) {
    auto tensor =
      mindspore::MSTensor::CreateTensor(model_pool_outputs_.at(i).Name(), model_pool_outputs_.at(i).DataType(),
                                        model_pool_outputs_.at(i).Shape(), nullptr, 0);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "create tensor failed.";
      return {};
    }
    outputs.push_back(*tensor);
    delete tensor;
  }
  return outputs;
}

Status ModelPool::Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config) {
  // create model pool config
  auto model_pool_config = CreateModelPoolConfig(runner_config);
  if (model_pool_config.empty()) {
    MS_LOG(ERROR) << "CreateModelPoolConfig failed, context is empty.";
    return kLiteError;
  }
  // create task queue for model pool
  predict_task_queue_ = std::make_shared<PredictTaskQueue>();
  if (predict_task_queue_ == nullptr) {
    MS_LOG(ERROR) << "create PredictTaskQueue failed, predict task queue is nullptr.";
    return kLiteNullptr;
  }
  if (use_numa_bind_mode_) {
    predict_task_queue_->SetTaskQueueNum(used_numa_node_num_);
  } else {
    predict_task_queue_->SetTaskQueueNum(1);
  }
  // read model by path and init packed weight by buffer
  size_t size = 0;
  auto graph_buf = lite::ReadFile(model_path.c_str(), &size);
  if (graph_buf == nullptr) {
    MS_LOG(ERROR) << "read file failed.";
    return kLiteNullptr;
  }
  // create worker
  std::shared_ptr<ModelWorker> model_worker = nullptr;
  if (model_pool_config.size() != workers_num_) {
    MS_LOG(ERROR) << "model pool config size is wrong.";
    return kLiteError;
  }
  for (size_t i = 0; i < workers_num_; i++) {
    int numa_node_id = model_pool_config[i]->numa_id;
    auto ret = lite::PackWeightManager::GetInstance()->InitByBuf(graph_buf, size, numa_node_id);
    MS_CHECK_FALSE_MSG(ret != kSuccess, kLiteError, "InitWeightManagerByBuf failed.");
    auto new_model_buf = lite::PackWeightManager::GetInstance()->GetNumaModelBuf(graph_buf, numa_node_id);
    MS_CHECK_TRUE_MSG(new_model_buf != nullptr, kLiteError, "get model buf is nullptr from PackWeightManager");
    model_worker = std::make_shared<ModelWorker>();
    if (model_worker == nullptr) {
      MS_LOG(ERROR) << "model worker is nullptr.";
      return kLiteNullptr;
    }
    int task_queue_id = numa_node_id != -1 ? numa_node_id : 0;
    predict_task_queue_->IncreaseWaitModelNum(1, task_queue_id);
    worker_thread_vec_.push_back(std::thread(&ModelWorker::CreateThreadWorker, model_worker, new_model_buf, size,
                                             model_pool_config[i], predict_task_queue_, &create_worker_success_));
    all_model_worker_.push_back(model_worker);
  }
  for (size_t i = 0; i < workers_num_; i++) {
    auto work = all_model_worker_[i];
    work->WaitCreateWorkerDone();
    if (!create_worker_success_) {
      MS_LOG(ERROR) << "init failed.";
      return kLiteError;
    }
  }
  // init model pool input and output
  if (model_worker != nullptr) {
    model_pool_inputs_ = model_worker->GetInputs();
    model_pool_outputs_ = model_worker->GetOutputs();
  }
  if (graph_buf != nullptr) {
    delete[] graph_buf;
    graph_buf = nullptr;
  }
  return kSuccess;
}

Status ModelPool::UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config) {
  for (auto &worker : all_model_worker_) {
    auto status = worker->UpdateConfig(section, config);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "model pool update config failed, status=" << status;
      return status;
    }
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

Status ModelPool::ConcatPredictOutput(std::vector<std::vector<MSTensor>> *outputs, std::vector<MSTensor> *new_outputs,
                                      int numa_id) {
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
    if (all_data_size > MAX_MALLOC_SIZE || all_data_size == 0) {
      MS_LOG(ERROR) << "malloc size is wrong.";
      return kLiteError;
    }
    int numa_allocator_id = used_numa_node_num_ ? numa_id : -1;
    auto all_out_data = numa_allocator_[numa_allocator_id]->Malloc(all_data_size);
    if (all_out_data == nullptr) {
      MS_LOG(ERROR) << "all_out_data is nullptr.";
      return kLiteError;
    }
    for (size_t j = 0; j < outputs->size(); j++) {
      void *out_data = outputs->at(j).at(i).MutableData();
      if (out_data == nullptr) {
        MS_LOG(ERROR) << "alloc addr: " << numa_allocator_[numa_allocator_id] << "  numa id: " << numa_id;
        numa_allocator_[numa_allocator_id]->Free(all_out_data);
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
      numa_allocator_[numa_allocator_id]->Free(all_out_data);
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

std::shared_ptr<ModelWorker> ModelPool::GetMaxWaitWorkerNum(int *max_wait_worker_node_id, int *max_wait_worker_num) {
  *max_wait_worker_node_id = 0;
  *max_wait_worker_num = predict_task_queue_->GetWaitModelNum(0);
  for (int i = 1; i < used_numa_node_num_; i++) {
    int worker_num = predict_task_queue_->GetWaitModelNum(i);
    if (*max_wait_worker_num < worker_num) {
      *max_wait_worker_num = worker_num;
      *max_wait_worker_node_id = i;
    }
  }
  if (*max_wait_worker_num > 0 && !use_split_batch_) {
    for (auto &worker : all_model_worker_) {
      if (worker->IsAvailable()) {
        return worker;
      }
    }
  }
  return nullptr;
}

Status ModelPool::PredictBySplitBatch(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                      const MSKernelCallBack &before, const MSKernelCallBack &after,
                                      int max_wait_worker_node_id) {
  size_t batch_split_num = predict_task_queue_->GetWaitModelNum(max_wait_worker_node_id);
  predict_task_queue_->DecreaseWaitModelNum(batch_split_num, max_wait_worker_node_id);
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
    if (predict_task == nullptr) {
      MS_LOG(ERROR) << "predict task is nullptr.";
      return kLiteNullptr;
    }
    predict_task_queue_->PushPredictTask(predict_task, max_wait_worker_node_id);
    tasks.push_back(predict_task);
  }
  predict_task_mutex_.unlock();
  for (size_t i = 0; i < batch_split_num; i++) {
    predict_task_queue_->WaitUntilPredictActive(tasks[i]);
  }
  status = ConcatPredictOutput(&new_outputs, outputs, max_wait_worker_node_id);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "ConcatPredictOutput failed.";
    return kLiteError;
  }
  status = FreeSplitTensor(&new_inputs, &new_outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "free split tensor failed.";
    return kLiteError;
  }
  predict_task_queue_->IncreaseWaitModelNum(batch_split_num, max_wait_worker_node_id);
  return kSuccess;
}

Status ModelPool::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  predict_task_mutex_.lock();
  int max_wait_worker_node_id = 0;
  int max_wait_worker_num = 0;
  auto available_worker = GetMaxWaitWorkerNum(&max_wait_worker_node_id, &max_wait_worker_num);
  if (inputs.size() == 0 || inputs.front().Shape().size() == 0) {
    MS_LOG(ERROR) << "inputs is invalid. input size: " << inputs.size();
    return kLiteError;
  }
  auto batch = inputs[0].Shape()[0];
  if (use_split_batch_ && predict_task_queue_->GetTaskNum(max_wait_worker_node_id) == 0 && max_wait_worker_num > 1 &&
      batch >= max_wait_worker_num) {
    // split batch
    auto status = PredictBySplitBatch(inputs, outputs, before, after, max_wait_worker_node_id);
    if (status != kSuccess) {
      predict_task_mutex_.unlock();
      MS_LOG(ERROR) << "do split batch failed. ret=" << status;
      return kLiteError;
    }
    return kSuccess;
  } else if (available_worker != nullptr) {
    // dispatch tasks directly to workers
    predict_task_mutex_.unlock();
    auto ret = available_worker->Predict(inputs, outputs, before, after);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "direct predict failed.";
      return kLiteError;
    }
    return kSuccess;
  } else {
    // do predict
    if (predict_task_queue_->GetTaskNum(max_wait_worker_node_id) > kNumMaxTaskQueueSize) {
      MS_LOG(ERROR) << "The number of waiting tasks in the queue exceeds the limit, ret=" << kLiteServiceDeny;
      predict_task_mutex_.unlock();
      return kLiteServiceDeny;
    }
    predict_task_queue_->DecreaseWaitModelNum(1, max_wait_worker_node_id);
    auto predict_task = std::make_shared<PredictTask>(&inputs, outputs, before, after);
    if (predict_task == nullptr) {
      MS_LOG(ERROR) << "predict_task is nullptr.";
      predict_task_mutex_.unlock();
      return kLiteNullptr;
    }
    predict_task_queue_->PushPredictTask(predict_task, max_wait_worker_node_id);
    predict_task_mutex_.unlock();
    predict_task_queue_->WaitUntilPredictActive(predict_task);
    predict_task_queue_->IncreaseWaitModelNum(1, max_wait_worker_node_id);
  }
  return kSuccess;
}

ModelPool::~ModelPool() {
  if (predict_task_queue_ != nullptr) {
    predict_task_queue_->SetPredictTaskDone();
  }
  for (auto &th : worker_thread_vec_) {
    if (th.joinable()) {
      th.join();
    }
  }
}
}  // namespace mindspore
