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
#include "pre_activate/pass/convert_const_input_to_attr.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>

#include "pre_activate/pass/const_input_to_attr_registry.h"
#include "utils/utils.h"
#include "utils/context/ms_context.h"
#include "operator/ops.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
void ConstInputToAttr(const CNodePtr &cnode, const std::unordered_set<size_t> &input_attrs) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> new_inputs;
  std::vector<std::string> new_input_names;
  auto primitive = AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  auto input_names = primitive->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr in cnode[" + cnode->DebugString() + "]";
    return;
  }
  auto input_names_vec = GetValue<std::vector<std::string>>(input_names);
  auto inputs = cnode->inputs();
  new_inputs.push_back(inputs[0]);
  bool need_update = false;
  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_attrs.find(i) != input_attrs.end() && input_node->isa<ValueNode>()) {
      auto value_node = input_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      MS_LOG(DEBUG) << "start erase input[" << i << "] of cnode[" + cnode->DebugString() + "]";
      if (i >= input_names_vec.size()) {
        MS_LOG(EXCEPTION) << "index " << i << " is larger than input names size [" << input_names_vec.size() << "]";
      }
      primitive->set_attr(input_names_vec[i], value_node->value());
      need_update = true;
    } else {
      new_inputs.push_back(input_node);
      if (i < input_names_vec.size()) {
        new_input_names.push_back(input_names_vec[i]);
      }
    }
  }
  if (need_update) {
    // Update cnode's inputs
    cnode->set_inputs(new_inputs);
    // Update cnode's input_names attr
    primitive->set_attr(kAttrInputNames, MakeValue(new_input_names));
  }
}
}  // namespace

const AnfNodePtr ConvertConstInputToAttr::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  if (node == nullptr || !AnfAlgo::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();

  ConstInputToAttrInfoRegister reg;
  if (!ConstInputToAttrInfoRegistry::Instance().GetRegisterByOpName(AnfAlgo::GetCNodeName(cnode), &reg)) {
    return nullptr;
  }
  ConstInputToAttr(cnode, reg.GetConstInputAttrInfo());
  return cnode;
}

void ConvertConstInputToAttr::Init() {
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimCast->name(), {1});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimConv2DBackpropInput->name(), {2});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimConv2DBackpropFilter->name(), {2});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimReshape->name(), {1});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimReduceMax->name(), {1});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimReduceMin->name(), {1});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimReduceSum->name(), {1});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimReduceMean->name(), {1});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimGatherV2->name(), {2});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimTranspose->name(), {1});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimUnsortedSegmentSum->name(), {2});
  ConstInputToAttrInfoRegistry::Instance().Register(prim::kPrimOneHot->name(), {1});
  ConstInputToAttrInfoRegistry::Instance().Register(kUnsortedSegmentProdOpName, {2});
  ConstInputToAttrInfoRegistry::Instance().Register(kUnsortedSegmentMinOpName, {2});
  ConstInputToAttrInfoRegistry::Instance().Register(kSimpleMeanGradOpName, {1});
  ConstInputToAttrInfoRegistry::Instance().Register(kMeanGradOpName, {1});
  ConstInputToAttrInfoRegistry::Instance().Register(kSliceOpName, {1, 2});
  ConstInputToAttrInfoRegistry::Instance().Register(kSliceGradOpName, {2, 3});
  ConstInputToAttrInfoRegistry::Instance().Register(kTileOpName, {1});
  ConstInputToAttrInfoRegistry::Instance().Register(kScatterNdOpName, {2});
  ConstInputToAttrInfoRegistry::Instance().Register(kStridedSliceAssignOpName, {1, 2, 3});
  ConstInputToAttrInfoRegistry::Instance().Register(kStridedSliceOpName, {1, 2, 3});
  ConstInputToAttrInfoRegistry::Instance().Register(kStridedSliceGradOpName, {1, 2, 3, 4});
  ConstInputToAttrInfoRegistry::Instance().Register(kFlattenGradOpName, {1});
  ConstInputToAttrInfoRegistry::Instance().Register(kExpandDimsOpName, {1});
  ConstInputToAttrInfoRegistry::Instance().Register(kSplitOpName, {0});
  ConstInputToAttrInfoRegistry::Instance().Register(kTopKOpName, {1});
  ConstInputToAttrInfoRegistry::Instance().Register(kSparseApplyAdagradOpName, {2});
  ConstInputToAttrInfoRegistry::Instance().Register(kResizeNearestNeighborGrad, {1});
}
}  // namespace opt
}  // namespace mindspore
