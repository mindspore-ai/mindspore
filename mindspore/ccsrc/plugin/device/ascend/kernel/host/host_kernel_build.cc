/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/host/host_kernel_build.h"
#include <string>
#include "plugin/device/ascend/kernel/host/host_kernel_mod.h"
#include "include/common/utils/anfalgo.h"
#include "utils/log_adapter.h"
namespace mindspore {
namespace kernel {
KernelModPtr HostOpBuild(const std::shared_ptr<AnfNode> &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string opname = common::AnfAlgo::GetCNodeName(anf_node);
  MS_LOG(INFO) << "Host op [" << opname << "]";
  auto kerPtr = HostKernelFactory::Get(opname);
  if (kerPtr == nullptr) {
    MS_LOG(ERROR) << "Host can't find Kernel[" << opname << "]";
    return nullptr;
  }
  if (!kerPtr->Init(anf_node)) {
    MS_LOG(ERROR) << "Host Kernel initialize failed!";
    return nullptr;
  }
  return kerPtr;
}
}  // namespace kernel
}  // namespace mindspore
