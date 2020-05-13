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

#include "kernel/rts/rt_kernel.h"

namespace mindspore {
namespace kernel {
void RtKernelFactory::Registe(const std::string &name, RtKernelCreater &&fun) {
  (void)fmap_.emplace(name, std::move(fun));
}

std::shared_ptr<RtKernel> RtKernelFactory::Create(const std::string &name) {
  const auto &map = Get().fmap_;
  auto it = map.find(name);
  if (it != map.end() && it->second) {
    return (it->second)();
  }
  return nullptr;
}

RtKernelFactory &RtKernelFactory::Get() {
  static RtKernelFactory _this;
  return _this;
}

RtKernel::RtKernel() {}

RtKernel::~RtKernel() {}

bool RtKernel::Init(const mindspore::AnfNodePtr & /*anf_node*/) { return true; }

const std::vector<size_t> &RtKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &RtKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &RtKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }
}  // namespace kernel
}  // namespace mindspore
