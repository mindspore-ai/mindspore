/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/parallel/multi_node_split.h"
#include "tools/optimizer/parallel/multi_conv_info.h"
namespace mindspore {
namespace opt {

int MultiNodeSplitProxy::InitResource() {
  switch (split_mode_) {
    case SplitN:
      multi_node_split_ = std::make_shared<MultiConvSplitN>(strategy_, primitive_type_, fmk_type_, num_);
      return RET_OK;
    case SplitH:
      multi_node_split_ = std::make_shared<MultiConvSplitH>(strategy_, primitive_type_, fmk_type_, num_);
      return RET_OK;
    case SplitCIN:
      multi_node_split_ = std::make_shared<MultiConvSplitCIN>(strategy_, primitive_type_, fmk_type_, num_);
      return RET_OK;
    case SplitCOUT:
      multi_node_split_ = std::make_shared<MultiConvSplitCOUT>(strategy_, primitive_type_, fmk_type_, num_);
      return RET_OK;
    default:
      return RET_ERROR;
  }
}

int MultiNodeSplitProxy::FreeResource() {
  multi_node_split_ = nullptr;
  return RET_OK;
}

AnfNodePtr MultiNodeSplitProxy::DoSplit(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  int ret = InitResource();
  if (ret != RET_OK) {
    return node;
  }
  auto res_node = multi_node_split_->DoSplit(func_graph, node);
  ret = FreeResource();
  if (ret != RET_OK) {
    return node;
  }
  return res_node;
}

}  // namespace opt
}  // namespace mindspore
