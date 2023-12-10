/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/extendrt/cxx_api/llm_engine/post_process_model_manager.h"
#include <memory>
#include <algorithm>
#include <map>
#include <string>

namespace mindspore {
namespace llm {

std::shared_ptr<mindspore::ModelImpl> PostProcessModelManager::GetModel(GenerateParameters *param) {
  if (param == nullptr) {
    return nullptr;
  }

  if (param->do_sample) {
    if (argmax_model_ == nullptr) {
      argmax_model_ = CreateArgMaxModel(param);
    }
    return argmax_model_;
  }
  if (topk_and_topp_model_ == nullptr) {
    topk_and_topp_model_ = CreateTopKTopPModel(param);
  }
  return topk_and_topp_model_;
}

FuncGraphPtr PostProcessModelManager::CreateArgMaxFuncGraph() {
  MS_LOG(ERROR) << "Failed to create arg max func graph";
  return nullptr;
}

FuncGraphPtr PostProcessModelManager::CreatTopKTopPFuncGraph() {
  MS_LOG(ERROR) << "Failed to create topk topp func graph";
  return nullptr;
}

std::shared_ptr<mindspore::Context> PostProcessModelManager::CreateContext() {
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed";
    return nullptr;
  }
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "New AscendDeviceInfo failed";
    return nullptr;
  }
  device_info->SetDeviceID(0);
  device_info->SetProvider("ge");
  device_list.push_back(device_info);

  return context;
}

std::shared_ptr<mindspore::ModelImpl> PostProcessModelManager::CreateArgMaxModel(GenerateParameters *param) {
  if (param == nullptr) {
    return nullptr;
  }

  auto func = CreateArgMaxFuncGraph();
  if (func == nullptr) {
    return nullptr;
  }

  auto argmax_model = std::make_shared<mindspore::ModelImpl>();
  if (argmax_model == nullptr) {
    MS_LOG(ERROR) << "New argmax model impl_ failed.";
    return nullptr;
  }

  auto status = argmax_model->Build(func, CreateContext());
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to build argmax model";
    return nullptr;
  }
  return argmax_model;
}

std::shared_ptr<mindspore::ModelImpl> PostProcessModelManager::CreateTopKTopPModel(GenerateParameters *param) {
  if (param == nullptr) {
    return nullptr;
  }

  auto func = CreatTopKTopPFuncGraph();
  if (func == nullptr) {
    return nullptr;
  }

  auto model = std::make_shared<mindspore::ModelImpl>();
  if (model == nullptr) {
    MS_LOG(ERROR) << "New topk topp model impl_ failed.";
    return nullptr;
  }

  auto status = model->Build(func, CreateContext());
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to build topk topp model";
    return nullptr;
  }
  return model;
}

}  // namespace llm
}  // namespace mindspore
