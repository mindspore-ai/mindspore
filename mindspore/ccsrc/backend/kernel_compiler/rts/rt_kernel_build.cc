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

#include "backend/kernel_compiler/rts/rt_kernel_build.h"
#include <string>
#include <memory>
#include <algorithm>
#include "backend/kernel_compiler/rts/rt_kernel.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
KernelModPtr RtOpBuild(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::string op_name = AnfAlgo::GetCNodeName(anf_node);
  (void)std::transform(op_name.begin(), op_name.end(), op_name.begin(), ::tolower);
  MS_LOG(INFO) << "Op Name(tolower)[" << op_name << "]";
  auto ker_ptr = RtKernelFactory::Create(op_name);
  MS_EXCEPTION_IF_NULL(ker_ptr);
  if (!ker_ptr->Init(anf_node)) {
    MS_LOG(ERROR) << "Rt Op initialize failed!";
    return nullptr;
  }

  return ker_ptr;
}
}  // namespace kernel
}  // namespace mindspore
