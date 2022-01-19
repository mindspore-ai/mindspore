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
#ifdef USING_SERVING
#include "src/cxx_api/model/model_pool.h"
#include <unistd.h>
#include <future>
#include "src/common/log.h"
#include "include/lite_types.h"
#include "src/common/config_file.h"
namespace mindspore {
void ModelPool::SetBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, int thread_num) {
  int core_num = sysconf(_SC_NPROCESSORS_CONF);
  if (thread_num == 0) {
    MS_LOG(ERROR) << "thread num is zero.";
    return;
  }
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

Status ModelPool::InitContext(const std::shared_ptr<mindspore::Context> &context,
                              std::map<std::string, std::map<std::string, std::string>> *all_config_info) {
  if (all_config_info->size() != 1) {
    MS_LOG(ERROR) << "all_config_info size should be 1";
    return kLiteError;
  }
  for (auto &item : *all_config_info) {
    auto config = item.second;
    auto num_thread = atoi(config["num_thread"].c_str());
    auto bind_mode = atoi(config["bind_mode"].c_str());
    context->SetThreadNum(num_thread);
    context->SetThreadAffinity(bind_mode);
  }
  context->SetEnableParallel(false);
  auto &device_list = context->MutableDeviceInfo();
  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_info->SetEnableFP16(false);
  device_list.push_back(device_info);
  return kSuccess;
}

ModelPoolContex ModelPool::CreateModelContext(const std::string &config_path) {
  std::map<std::string, std::map<std::string, std::string>> all_config_info;
  auto ret = lite::GetAllSectionInfoFromConfigFile(config_path, &all_config_info);
  if (ret != 0) {
    MS_LOG(ERROR) << "GetAllSectionInfoFromConfigFile failed.";
    return {};
  }
  auto model_context = std::make_shared<mindspore::Context>();
  if (model_context == nullptr) {
    MS_LOG(ERROR) << "model context is nullptr.";
    return {};
  }
  auto status = InitContext(model_context, &all_config_info);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "InitMSContext failed.";
    return {};
  }
  auto device_list = model_context->MutableDeviceInfo();
  if (device_list.size() != 1) {
    MS_LOG(ERROR) << "model pool only support device num 1.";
    return {};
  }
  auto device = device_list.front();
  if (device->GetDeviceType() != kCPU) {
    MS_LOG(ERROR) << "model pool only support cpu type.";
    return {};
  }
  auto cpu_context = device->Cast<CPUDeviceInfo>();
  auto enable_fp16 = cpu_context->GetEnableFP16();
  if (enable_fp16) {
    MS_LOG(ERROR) << "model pool not support enable fp16.";
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

Status ModelPool::Run() {
  std::unique_lock<std::mutex> model_lock(mtx_model_queue_);
  while (model_pool_queue_.empty()) {
    cv_model_.wait(model_lock);
  }
  auto model = model_pool_queue_.front();
  model_pool_queue_.pop();
  model_lock.unlock();
  std::unique_lock<std::mutex> data_lock(mtx_data_queue_);
  if (model_data_queue_.empty()) {
    MS_LOG(ERROR) << "model data queue is empty";
    return kLiteError;
  }
  auto model_data = model_data_queue_.front();
  model_data_queue_.pop();
  auto inputs = model_data->inputs;
  auto outputs = model_data->outputs;
  auto before = model_data->before;
  auto after = model_data->after;
  auto status = model->Predict(*inputs, outputs, before, after);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model predict failed.";
    return status;
  }
  mtx_model_queue_.lock();
  model_pool_queue_.push(model);
  cv_model_.notify_one();
  mtx_model_queue_.unlock();
  return kSuccess;
}

Status ModelPool::Init(const std::string &model_path, const std::string &config_path, const Key &dec_key,
                       const std::string &dec_mode) {
  auto model_pool_context = CreateModelContext(config_path);
  for (size_t i = 0; i < num_models_; i++) {
    auto model = std::make_shared<ModelThread>();
    auto status = model->Init(model_path, model_pool_context[i], dec_key, dec_mode);
    model_pool_queue_.push(model);
  }
  return kSuccess;
}

Status ModelPool::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  {
    std::unique_lock<std::mutex> data_lock(mtx_data_queue_);
    auto model_data = std::make_shared<ModelData>();
    model_data->inputs = &inputs;
    model_data->outputs = outputs;
    model_data->before = before;
    model_data->after = after;
    model_data_queue_.push(model_data);
  }
  auto future_status = std::async(std::launch::async, &ModelPool::Run, this);
  auto status = future_status.get();
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model run failed in model pool predict.";
    return status;
  }
  return kSuccess;
}
}  // namespace mindspore
#endif
