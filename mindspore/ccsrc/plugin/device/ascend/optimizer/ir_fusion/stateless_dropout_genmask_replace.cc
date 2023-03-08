/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/stateless_dropout_genmask_replace.h"

#include <string>
#include <vector>
#include <memory>
#include <set>
#include "utils/hash_set.h"
#include "backend/common/optimizer/const_input_to_attr.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_info.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/optimizer/optimizer_factory.h"

namespace mindspore::opt {
namespace {
constexpr auto kAttrSeed0 = "Seed0";
constexpr auto kAttrSeed1 = "Seed1";
constexpr size_t kDropoutGenMaskInputTensorNum = 2;
}  // namespace

const BaseRef StatelessDropOutGenMaskReplace::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kDropoutGenMaskOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr StatelessDropOutGenMaskReplace::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto cnode = CheckAnfNodeIfCNodeAndInputSize(node, kDropoutGenMaskInputTensorNum);
  if (!common::AnfAlgo::HasNodeAttr(kAttrSeed0, cnode) || !common::AnfAlgo::HasNodeAttr(kAttrSeed1, cnode)) {
    MS_LOG(INFO) << "Node [" << node->fullname_with_scope() << "] has no seed0 or seed1 attr, quit fusion.";
    return nullptr;
  }

  // create seed0, seed1, offset valuenode input
  auto seed0 = common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrSeed0);
  auto seed1 = common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrSeed1);
  auto input_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  ValueNodePtr seed0_node = nullptr;
  ValueNodePtr seed1_node = nullptr;
  if (input_dtype == kNumberTypeInt32) {
    seed0_node = kernel_graph->NewValueNode(std::make_shared<tensor::Tensor>(static_cast<int32_t>(seed0), kInt32));
    seed1_node = kernel_graph->NewValueNode(std::make_shared<tensor::Tensor>(static_cast<int32_t>(seed1), kInt32));
  } else {
    seed0_node = kernel_graph->NewValueNode(std::make_shared<tensor::Tensor>(seed0, kInt64));
    seed1_node = kernel_graph->NewValueNode(std::make_shared<tensor::Tensor>(seed1, kInt64));
  }
  MS_EXCEPTION_IF_NULL(seed0_node);
  MS_EXCEPTION_IF_NULL(seed1_node);
  auto offset_node = CreateShapeValueNode(func_graph, {0, 0}, true);
  MS_EXCEPTION_IF_NULL(offset_node);
  // create StatelessDropOutGenMask
  std::vector<AnfNodePtr> new_inputs{NewValueNode(std::make_shared<Primitive>(kStatelessDropOutGenMaskOpName))};
  (void)new_inputs.insert(new_inputs.cend(), cnode->inputs().cbegin() + 1, cnode->inputs().cend());
  new_inputs.push_back(seed0_node);
  new_inputs.push_back(seed1_node);
  new_inputs.push_back(offset_node);
  CNodePtr new_cnode = NewCNode(new_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_abstract(cnode->abstract());
  new_cnode->set_scope(cnode->scope());
  new_cnode->set_primal_attrs(cnode->primal_attrs());
  new_cnode->set_attrs(cnode->attrs());
  return new_cnode;
}
}  // namespace mindspore::opt
