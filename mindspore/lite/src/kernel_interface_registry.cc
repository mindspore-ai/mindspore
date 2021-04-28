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
#include "src/kernel_interface_registry.h"
#include "src/kernel_interface.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/common/version_manager.h"
#include "schema/model_generated.h"

using mindspore::kernel::KernelInterfaceCreator;
using mindspore::schema::PrimitiveType_MAX;
using mindspore::schema::PrimitiveType_MIN;
namespace mindspore {
namespace lite {
namespace {
static const auto kMaxKernelNum = PrimitiveType_MAX - PrimitiveType_MIN;
}

bool KernelInterfaceRegistry::CheckReg(const lite::Model::Node *node) {
  if (VersionManager::GetInstance()->GetSchemaVersion() == SCHEMA_V0) {
    return false;
  }
  auto primitive = static_cast<const schema::Primitive *>(node->primitive_);
  if (primitive == nullptr) {
    return false;
  }

  auto op_type = primitive->value_type();
  if (op_type == schema::PrimitiveType_Custom) {
    return std::any_of(custom_interfaces_.begin(), custom_interfaces_.end(), [node](auto &&item) {
      if (item.second[node->name_] != nullptr) {
        return true;
      }
      return false;
    });
  }

  return std::any_of(kernel_interfaces_.begin(), kernel_interfaces_.end(),
                     [op_type, &mutex = this->mutex_](auto &&item) {
                       std::unique_lock<std::mutex> lock(mutex);
                       if (item.second[op_type] != nullptr) {
                         return true;
                       }
                       return false;
                     });
}

int KernelInterfaceRegistry::CustomReg(const std::string &provider, const std::string &op_type,
                                       KernelInterfaceCreator creator) {
  custom_interfaces_[provider][op_type] = creator;
  return RET_OK;
}

kernel::KernelInterface *KernelInterfaceRegistry::GetKernelInterface(const std::string &provider, int op_type) {
  if (op_type < PrimitiveType_MIN || op_type > kMaxKernelNum) {
    MS_LOG(ERROR) << "reg op_type invalid!op_type: " << op_type << ", max value: " << kMaxKernelNum;
    return nullptr;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  auto iter = kernel_interfaces_.find(provider);
  if (iter == kernel_interfaces_.end()) {
    return nullptr;
  }

  auto creator = iter->second[op_type];
  if (creator != nullptr) {
    return creator();
  }
  return nullptr;
}

int KernelInterfaceRegistry::Reg(const std::string &provider, int op_type, KernelInterfaceCreator creator) {
  if (op_type < PrimitiveType_MIN || op_type > kMaxKernelNum) {
    MS_LOG(ERROR) << "reg op_type invalid!op_type: " << op_type << ", max value: " << kMaxKernelNum;
    return RET_ERROR;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  auto iter = kernel_interfaces_.find(provider);
  if (iter == kernel_interfaces_.end()) {
    kernel_interfaces_[provider] =
      reinterpret_cast<KernelInterfaceCreator *>(malloc(kMaxKernelNum * sizeof(KernelInterfaceCreator)));
    if (kernel_interfaces_[provider] == nullptr) {
      MS_LOG(ERROR) << "malloc kernel dev delegate creator fail!";
      return RET_ERROR;
    }
  }

  kernel_interfaces_[provider][op_type] = creator;
  return RET_OK;
}

KernelInterfaceRegistry::~KernelInterfaceRegistry() {
  for (auto &&item : kernel_interfaces_) {
    free(item.second);
    item.second = nullptr;
  }
}
}  // namespace lite
}  // namespace mindspore
