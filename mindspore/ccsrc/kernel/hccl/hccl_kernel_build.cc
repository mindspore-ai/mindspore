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

#include "kernel/hccl/hccl_kernel_build.h"

#include <string>
#include <memory>
#include <algorithm>

#include "kernel/hccl/hccl_kernel.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
KernelModPtr HcclOpBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string opname = AnfAlgo::GetCNodeName(anf_node);
  MS_LOG(INFO) << "Hccl op [" << opname << "]";
  auto kerPtr = HcclKernelFactory::Get(opname);
  if (kerPtr == nullptr) {
    MS_LOG(ERROR) << "Hccl can't find Kernel[" << opname << "]";
    return nullptr;
  }
  if (!kerPtr->Init(anf_node)) {
    MS_LOG(ERROR) << "Kernel initialize failed!";
    return nullptr;
  }
  return kerPtr;
}
}  // namespace kernel
}  // namespace mindspore
