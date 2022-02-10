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
}  // namespace

void ModelPool::SetBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, int thread_num) {
  if (thread_num == 0) {
    MS_LOG(ERROR) << "thread num is zero.";
    return;
  }
  int core_num = GetCoreNum();
  num_models_ = core_num / thread_num;
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
    model_context = runner_config->model_ctx;
    if (runner_config->num_model == 0) {
      MS_LOG(ERROR) << "num of model is zero.";
      return nullptr;
    }
    num_models_ = runner_config->num_model;
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
    if (device->GetDeviceType() == kGPU) {
      num_models_ = 1;
    }
    auto cpu_context = device->Cast<CPUDeviceInfo>();
    auto enable_fp16 = cpu_context->GetEnableFP16();
    if (enable_fp16) {
      MS_LOG(ERROR) << "model pool not support enable fp16.";
      return nullptr;
    }
  } else {
    MS_LOG(DEBUG) << "use default config.";
    num_models_ = GetCoreNum() / static_cast<int>(model_context->GetThreadNum());
    model_context->SetThreadNum(kNumThreads);
    model_context->SetEnableParallel(false);
    model_context->SetThreadAffinity(lite::NO_BIND);
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
  if (model_context->GetThreadNum() == 0) {
    MS_LOG(ERROR) << "thread num is zero.";
    return {};
  }
  ModelPoolContex model_pool_context;
  std::vector<std::vector<int>> all_model_bind_list;
  if (model_context->GetThreadAffinityMode() == lite::HIGHER_CPU) {
    SetBindStrategy(&all_model_bind_list, static_cast<int>(model_context->GetThreadNum()));
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
  std::shared_ptr<ModelThread> model_thread = nullptr;
  for (size_t i = 0; i < num_models_; i++) {
    model_thread = std::make_shared<ModelThread>();
    auto status = model_thread->Init(model_path, model_pool_context[i], dec_key, dec_mode);
    model_thread_vec_.push_back(std::thread(&ModelThread::Run, model_thread));
  }
  if (model_thread != nullptr) {
    model_inputs_ = model_thread->GetInputs();
    model_outputs_ = model_thread->GetOutputs();
  }
  return kSuccess;
}

Status ModelPool::SplitTensorByBatch(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                     std::vector<std::vector<MSTensor>> *new_inputs) {
  auto batch = inputs[0].Shape()[0];
  if (batch % batch_split_num_ != 0) {
    MS_LOG(DEBUG) << "Can not split input tensor.";
    return kLiteSuccessExit;
  }
  std::vector<std::vector<std::vector<int64_t>>> all_input_shape;
  for (size_t k = 0; k < batch_split_num_; k++) {  // do for batch
    std::vector<std::vector<int64_t>> inputs_shape;
    std::vector<MSTensor> new_inputs_tensor;
    for (size_t i = 0; i < inputs.size(); i++) {  // do for input
      std::vector<int64_t> shape;
      size_t input_size = batch / batch_split_num_;
      shape.push_back(batch / batch_split_num_);
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
        void *data = malloc(input_size * sizeof(float));
        if (data == nullptr) {
          MS_LOG(ERROR) << "malloc failed.";
          return kLiteError;
        }
        memcpy(reinterpret_cast<float *>(data),
               reinterpret_cast<float *>(const_cast<MSTensor &>(inputs[i]).MutableData()) + input_size * k,
               input_size * sizeof(float));
        auto new_tensor = mindspore::MSTensor::CreateTensor(
          inputs[i].Name(), static_cast<enum DataType>(kNumberTypeFloat32), shape, data, input_size * sizeof(float));
        if (new_tensor == nullptr) {
          MS_LOG(ERROR) << "create tensor failed.";
          return kLiteError;
        }
        new_inputs_tensor.push_back(*new_tensor);
        free(data);
      } else if (inputs[i].DataType() == static_cast<enum DataType>(kNumberTypeInt32)) {
        if (input_size * sizeof(int32_t) > MAX_MALLOC_SIZE) {
          MS_LOG(ERROR) << "malloc size is wrong.";
          return kLiteError;
        }
        void *data = malloc(input_size * sizeof(int32_t));
        if (data == nullptr) {
          MS_LOG(ERROR) << "malloc failed.";
          return kLiteError;
        }
        memcpy(reinterpret_cast<int32_t *>(data),
               reinterpret_cast<int32_t *>(const_cast<MSTensor &>(inputs[i]).MutableData()) + input_size * k,
               input_size * sizeof(int32_t));
        auto new_tensor = mindspore::MSTensor::CreateTensor(
          inputs[i].Name(), static_cast<enum DataType>(kNumberTypeInt32), shape, data, input_size * sizeof(int32_t));
        if (new_tensor == nullptr) {
          MS_LOG(ERROR) << "create tensor failed.";
          return kLiteError;
        }
        new_inputs_tensor.push_back(*new_tensor);
        free(data);
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
    output_tensor_shape[0] *= batch_split_num_;
    if (all_out_data != nullptr) {
      free(all_out_data);
      all_out_data = nullptr;
    }
    all_out_data = malloc(outputs->at(0).at(i).DataSize() * batch_split_num_);
    if (all_out_data == nullptr) {
      MS_LOG(ERROR) << "all_out_data is nullptr.";
      return kLiteError;
    }
    for (size_t j = 0; j < batch_split_num_; j++) {
      void *out_data = outputs->at(j)[i].MutableData();
      if (out_data == nullptr) {
        MS_LOG(ERROR) << "output data is nullptr.";
        return kLiteError;
      }
      memcpy(reinterpret_cast<float *>(all_out_data) + outputs->at(j)[i].ElementNum() * j,
             reinterpret_cast<float *>(out_data), outputs->at(j)[i].DataSize());
    }
    auto new_tensor =
      mindspore::MSTensor::CreateTensor(outputs->at(0)[i].Name(), outputs->at(i)[0].DataType(), output_tensor_shape,
                                        all_out_data, outputs->at(0)[i].DataSize() * batch_split_num_);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "create tensor failed.";
      return kLiteError;
    }
    new_outputs->push_back(*new_tensor);
  }
  return kSuccess;
}

Status ModelPool::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  mtx_split_task_.lock();
  auto batch = inputs[0].Shape()[0];
  if (batch % batch_split_num_ == 0 && PredictTaskQueue::GetInstance()->GetTaskNum() == 0 &&
      batch_split_num_ <= static_cast<size_t>(PredictTaskQueue::GetInstance()->GetWaitModelNum())) {
    batch_split_num_ = PredictTaskQueue::GetInstance()->GetWaitModelNum();
    PredictTaskQueue::GetInstance()->DecreaseWaitModelNum(batch_split_num_);
    std::vector<std::vector<MSTensor>> new_inputs;
    std::vector<std::vector<MSTensor>> new_outputs;
    auto status = SplitTensorByBatch(inputs, outputs, &new_inputs);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "model pool predict failed.";
      return kLiteError;
    }
    for (size_t i = 0; i < batch_split_num_; i++) {
      std::vector<MSTensor> new_output;
      new_outputs.push_back(new_output);
    }
    std::vector<std::shared_ptr<PredictTask>> tasks;
    for (size_t i = 0; i < batch_split_num_; i++) {
      auto predict_task = std::make_shared<PredictTask>(&new_inputs[i], &new_outputs[i], before, after);
      PredictTaskQueue::GetInstance()->PushPredictTask(predict_task);
      tasks.push_back(predict_task);
    }
    mtx_split_task_.unlock();
    for (size_t i = 0; i < batch_split_num_; i++) {
      PredictTaskQueue::GetInstance()->WaitUntilPredictActive(tasks[i]);
    }
    status = ConcatPredictOutput(&new_outputs, outputs);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "ConcatPredictOutput failed.";
      return kLiteError;
    }
  } else {
    auto predict_task = std::make_shared<PredictTask>(&inputs, outputs, before, after);
    PredictTaskQueue::GetInstance()->PushPredictTask(predict_task);
    mtx_split_task_.unlock();
    PredictTaskQueue::GetInstance()->WaitUntilPredictActive(predict_task);
  }
  return kSuccess;
}

ModelPool::~ModelPool() {
  for (auto &th : model_thread_vec_) {
    if (th.joinable()) {
      th.join();
    }
  }
  if (all_out_data != nullptr) {
    free(all_out_data);
    all_out_data = nullptr;
  }
}
}  // namespace mindspore
