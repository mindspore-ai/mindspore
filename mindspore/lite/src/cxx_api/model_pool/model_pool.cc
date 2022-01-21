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
namespace mindspore {
void ModelPool::SetBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, int thread_num) {
  int core_num = 1;
#if defined(_MSC_VER) || defined(_WIN32)
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  core_num = sysinfo.dwNumberOfProcessors;
#else
  core_num = sysconf(_SC_NPROCESSORS_CONF);
#endif
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

std::vector<MSTensor> ModelPool::GetInputs() {
  if (model_inputs_.empty()) {
    MS_LOG(ERROR) << "model input is empty.";
    return {};
  }
  return model_inputs_;
}

Status ModelPool::Init(const std::string &model_path, const std::string &config_path, const Key &dec_key,
                       const std::string &dec_mode) {
  auto model_pool_context = CreateModelContext(config_path);
  for (size_t i = 0; i < num_models_; i++) {
    auto model_thread = std::make_shared<ModelThread>();
    auto status = model_thread->Init(model_path, model_pool_context[i], dec_key, dec_mode);
    if (model_inputs_.empty()) {
      model_inputs_ = model_thread->GetInputs();
    }
    model_thread_vec_.push_back(std::thread(&ModelThread::Run, model_thread));
  }
  return kSuccess;
}

Status ModelPool::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  outputs->clear();
  auto predict_task = std::make_shared<PredictTask>(&inputs, outputs, before, after);
  PredictTaskQueue::GetInstance()->PushPredictTask(predict_task);
  PredictTaskQueue::GetInstance()->WaitUntilPredictActive(outputs);
  return kSuccess;
}

ModelPool::~ModelPool() {
  for (auto &th : model_thread_vec_) {
    if (th.joinable()) {
      th.join();
    }
  }
}
}  // namespace mindspore
