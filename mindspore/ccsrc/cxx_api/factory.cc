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

#include "cxx_api/factory.h"
namespace mindspore {
GraphImplFactory &GraphImplFactory::Instance() {
  std::call_once(once_flag_, []() {
    if (instance_ == nullptr) {
      instance_ = std::make_shared<GraphImplFactory>();
    }
  });
  return *instance_;
}

void GraphImplFactory::Register(const std::string &device_name, GraphImplCreator &&creator) {
  MS_LOG(DEBUG) << "Start register graph impl for " << device_name;
  (void)creators_.emplace_back(std::move(creator));
}

std::shared_ptr<GraphCell::GraphImpl> GraphImplFactory::Create(enum DeviceType device_type) {
  for (auto &item : creators_) {
    MS_EXCEPTION_IF_NULL(item);
    auto val = item();
    MS_EXCEPTION_IF_NULL(val);
    if (val->CheckDeviceSupport(device_type)) {
      return val;
    }
  }
  MS_LOG(WARNING) << "Unsupported device target " << device_type;
  return nullptr;
}

ModelImplFactory &ModelImplFactory::Instance() {
  std::call_once(once_flag_, []() {
    if (instance_ == nullptr) {
      instance_ = std::make_shared<ModelImplFactory>();
    }
  });
  return *instance_;
}

void ModelImplFactory::Register(const std::string &device_name, ModelImplCreator &&creator) {
  MS_LOG(DEBUG) << "Start register model for " << device_name;
  (void)creators_.emplace_back(std::move(creator));
}

std::shared_ptr<ModelImpl> ModelImplFactory::Create(enum DeviceType device_type) {
  for (auto &item : creators_) {
    MS_EXCEPTION_IF_NULL(item);
    auto val = item();
    if (val->CheckDeviceSupport(device_type)) {
      return val;
    }
  }
  MS_LOG(WARNING) << "Unsupported device target " << device_type;
  return nullptr;
}
}  // namespace mindspore
