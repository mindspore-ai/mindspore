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

using mindspore::kernel::KernelInterfaceCreator;
using mindspore::schema::PrimitiveType_MAX;
using mindspore::schema::PrimitiveType_MIN;
namespace mindspore {
namespace lite {
namespace {
static const auto kMaxKernelNum = PrimitiveType_MAX - PrimitiveType_MIN + 1;
}

int KernelInterfaceRegistry::Reg(const std::string &vendor, const int &op_type, KernelInterfaceCreator creator) {
  auto vendor_hash = std::hash<std::string>{}(vendor);
  auto iter = kernel_interfaces_.find(vendor_hash);
  if (iter == kernel_interfaces_.end()) {
    kernel_interfaces_[vendor_hash] =
      reinterpret_cast<KernelInterfaceCreator *>(malloc(kMaxKernelNum * sizeof(KernelInterfaceCreator)));
    if (kernel_interfaces_[vendor_hash] == nullptr) {
      MS_LOG(ERROR) << "malloc kernel dev delegate creator fail!";
      return RET_ERROR;
    }
  }
  if (op_type < PrimitiveType_MIN || op_type > kMaxKernelNum) {
    MS_LOG(ERROR) << "reg op_type invalid!op_type: " << op_type << ", max value: " << kMaxKernelNum;
    return RET_ERROR;
  }
  kernel_interfaces_[vendor_hash][op_type] = creator;
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore
