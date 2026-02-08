/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SYMMETRIC_MEMORY_KERNEL_BUILD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SYMMETRIC_MEMORY_KERNEL_BUILD_H_

#include <memory>
#include <string>
#include <vector>
#include "kernel/ascend/visible.h"
#include "primitive/other_op_name.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
static const std::set<std::string> enableSymmetricMemoryNodeList = {
  kCreateSymmetricMemoryOpName, kPutMemOpName,         kGetMemOpName, kSignalOpOpName,
  kPutMemSignalOpName,          kSignalWaitUntilOpName};
OPS_ASCEND_API KernelModPtr SymmetricMemoryKernelBuild(const AnfNodePtr &anf_node);
OPS_ASCEND_API void GetValidKernelBuildInfoWithSymmetricMemoryFormat(const AnfNodePtr &node,
                                                                     std::vector<std::string> *input_formats,
                                                                     std::vector<std::string> *output_formats);
OPS_ASCEND_API bool IsEnableSymmetricMemoryNode(const AnfNodePtr &node);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SYMMETRIC_MEMORY_KERNEL_BUILD_H_
