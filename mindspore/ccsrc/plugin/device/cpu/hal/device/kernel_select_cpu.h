/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_KERNEL_SELECT_CPU_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_KERNEL_SELECT_CPU_H_

#include <utility>
#include <string>
#include <vector>

#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "kernel/common_utils.h"
#include "kernel/kernel_build_info.h"
#include "kernel/graph_kernel_info.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
namespace cpu {
using kernel::DataType;
bool IsVmapNotSupported(const CNodePtr &node);
BACKEND_EXPORT std::pair<std::string, ExceptionType> SetKernelInfoWithMsg(const CNodePtr &apply_kernel_ptr);

class BACKEND_EXPORT CPUGraphKernelInfo : public GraphKernelInfo {
 public:
  CPUGraphKernelInfo() = default;
  virtual ~CPUGraphKernelInfo() = default;
  void SetKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type) override;
};

REG_GRAPH_KERNEL_INFO(kCPUDevice, CPUGraphKernelInfo);
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_CPU_KERNEL_SELECT_CPU_H_
