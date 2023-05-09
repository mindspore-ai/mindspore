/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/bisheng/delegate.h"

namespace mindspore {
void BishengDelegate::ReplaceNodes(const std::shared_ptr<FuncGraph> &graph) {
  // need to implementation
}

bool BishengDelegate::IsDelegateNode(const std::shared_ptr<AnfNode> &node) {
  // need to implementation
  return true;
}

std::shared_ptr<kernel::BaseKernel> BishengDelegate::CreateKernel(const std::shared_ptr<AnfNode> &node) {
  // need to implementation
  return nullptr;
}
}  // namespace mindspore
