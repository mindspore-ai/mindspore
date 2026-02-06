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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCCL_KERNEL_METADATA_ANFALGO_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCCL_KERNEL_METADATA_ANFALGO_H_
#include <vector>
#include <memory>
#include "kernel/ascend/visible.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"

namespace mindspore {
namespace kernel {
OPS_ASCEND_API void HcclMetadataInfo(const CNodePtr &kernel_node,
                                     std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCCL_KERNEL_METADATA_ANFALGO_H_
