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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_COMMON_SOMAS_ALLOCATOR_H
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_COMMON_SOMAS_ALLOCATOR_H

#include <vector>
#include <string>
#include <map>
#include <utility>
#include <memory>
#include "backend/common/somas/somas.h"
#include "include/backend/device_type.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
class CommonSomasAllocator {
 public:
  void set_mem_base_addr(uint8_t *mem_base_addr) { mem_base_addr_ = mem_base_addr; }
  static bool Assign(const session::KernelGraph &graph);
  uint8_t *GetNodeOutputPtr(const AnfNodePtr &node, size_t index) const;
  uint8_t *GetNodeWorkSpacePtr(const AnfNodePtr &node, size_t index) const;

 private:
  // Memory base addr
  uint8_t *mem_base_addr_{nullptr};
  static std::string GetTargetFromContext() {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    return context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  }
};
using CommonSomasAllocatorPtr = std::shared_ptr<CommonSomasAllocator>;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_COMMON_SOMAS_ALLOCATOR_H
