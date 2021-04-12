/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_SELECT_ASCEND_ANFALGO_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_SELECT_ASCEND_ANFALGO_H_
#include "ir/anf.h"
#include "backend/kernel_compiler/kernel_build_info.h"
namespace mindspore {
namespace device {
namespace ascend {
enum KernelSelectStatus {
  kNoMatched = -1,
  kStatusAllMatched = 0,
  kStatusReducePrecision = 1,
  kStatusRaisePrecision = 2,
};
KernelSelectStatus SelectKernelInfo(const CNodePtr &kernel_node,
                                    KernelType kernel_type = KernelType::UNKNOWN_KERNEL_TYPE);
void SetTensorDeviceInfo(const CNodePtr &kernel_node);
void SelectGraphKernelInfo(const CNodePtr &kernel_node, const FuncGraphPtr &func_graph);
void SetKernelInfo(const CNodePtr &kernel_node, KernelType kernel_type);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_KERNEL_SELECT_ASCEND_ANFALGO_H_
