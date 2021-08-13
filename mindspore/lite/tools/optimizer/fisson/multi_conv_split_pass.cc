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

#include "tools/optimizer/fisson/multi_conv_split_pass.h"
#include <string>
#include <memory>
#include "utils/utils.h"
#include "base/base.h"
#include "ops/fusion/conv2d_fusion.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/parallel/split_strategy.h"

using mindspore::converter::FmkType;
using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;
namespace mindspore {
namespace opt {

std::string MultiConvSplitPass::IsMultiParallelConvNode(const AnfNodePtr &node) const {
  std::string parallel_name;
  for (const auto &parallel_prim : kParallelOpNames) {
    if (CheckPrimitiveType(node, parallel_prim.first.first)) {
      return parallel_prim.second;
    }
  }
  return parallel_name;
}

const BaseRef MultiConvSplitPass::DefinePattern() const {
  auto conv1_var = std::make_shared<CondVar>(IsParallelSplitConvNode);
  auto conv1_other_var = std::make_shared<SeqVar>();
  VectorRef res = VectorRef({conv1_var, conv1_other_var});
  int32_t idx = 1;
  while (idx < num_) {
    auto tmp_var = std::make_shared<CondVar>(IsParallelSplitConvNode);
    auto tmp_other_var = std::make_shared<SeqVar>();
    res = VectorRef({tmp_var, res, tmp_other_var});
    idx++;
  }
  return res;
}

const AnfNodePtr MultiConvSplitPass::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  auto device_type_attr = cnode->GetAttr(mindspore::ops::kDeviceType);
  auto device_type = (device_type_attr != nullptr) ? GetValue<int32_t>(device_type_attr) : kDeviceTypeNone;
  if (device_type != kDeviceTypeNone) {
    return node;
  }
  auto parallel_name = IsMultiParallelConvNode(node);
  if (parallel_name.empty()) {
    return node;
  }
  // if current node has more than two outputs node, we do not split it.
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users_iter = manager->node_users().find(node);
  if (node_users_iter == manager->node_users().end()) {
    return node;
  }
  auto output_info_list = node_users_iter->second;
  if (output_info_list.size() > kDefaultBatch) {
    return node;
  }

  std::shared_ptr<MultiNodeSplitProxy> multi_node_split_proxy =
    std::make_shared<MultiNodeSplitProxy>(strategys_.at(parallel_name), primitive_type_, fmk_type_, num_);
  return multi_node_split_proxy->DoSplit(func_graph, node);
}

}  // namespace opt
}  // namespace mindspore
