/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ge/convert_pad_v3_paddings.h"
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include "mindspore/core/ops/nn_ops.h"
#include "mindspore/core/ops/op_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
constexpr auto kLength8 = 8;
constexpr auto kStep2 = 2;

bool ConvertBasePaddings::HasDynPaddings(const CNodePtr &cnode) const {
  auto input_paddings = common::AnfAlgo::GetInputNode(cnode, kIndex1);
  MS_EXCEPTION_IF_NULL(input_paddings);
  auto paddings_abstract = input_paddings->abstract();
  MS_EXCEPTION_IF_NULL(paddings_abstract);
  auto paddings_value = paddings_abstract->GetValue();
  MS_EXCEPTION_IF_NULL(paddings_value);
  auto input_paddings_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, kIndex1);
  if (input_paddings_type_id == kNumberTypeInt32) {
    auto paddings_array_value = ops::GetArrayValue<int32_t>(paddings_value);
    return !paddings_array_value.has_value();
  }
  auto paddings_array_value = ops::GetArrayValue<int64_t>(paddings_value);
  return !paddings_array_value.has_value();
}

template <typename T, TypeId type_id>
const AnfNodePtr ConvertBasePaddings::OptimizePaddingsValue(const FuncGraphPtr &graph,
                                                            const AbstractBasePtr &ori_paddings,
                                                            const bool &paddings_contiguous, const size_t &dst_length,
                                                            bool force_length8) const {
  std::vector<T> paddings_data;
  auto paddings_type = ori_paddings->GetType();
  MS_EXCEPTION_IF_NULL(paddings_type);
  if (paddings_type->template isa<TensorType>()) {
    auto paddings_value = ori_paddings->GetValue();
    MS_EXCEPTION_IF_NULL(paddings_value);
    auto paddings_array_value = ops::GetArrayValue<T>(paddings_value);
    paddings_data = paddings_array_value.value().ToVector();
  } else {
    auto paddings_value = ops::GetArrayValue<T>(ori_paddings);
    paddings_data = paddings_value->ToVector();
  }
  if (!paddings_contiguous) {
    auto tmp = paddings_data;
    for (size_t i = 0; i < paddings_data.size(); i++) {
      if (i % kStep2 == 0) {
        paddings_data[i] = tmp[i / kStep2];
      } else {
        paddings_data[i] = tmp[(i + paddings_data.size()) / kStep2];
      }
    }
  }
  // (0, 1, 2, 3, 4, 5, 6, 7) -> (6, 7, 4, 5, 2, 3, 0, 1)
  std::reverse(paddings_data.begin(), paddings_data.end());
  for (size_t i = 1; i < paddings_data.size(); i += kStep2) {
    std::swap(paddings_data[i - 1], paddings_data[i]);
  }
  // (1, 2, 3, 4) -> (0, 0, 0, 0, 1, 2, 3, 4)
  std::vector<T> opt_paddings_data(dst_length);
  auto offset = opt_paddings_data.size() - paddings_data.size();
  std::transform(paddings_data.begin(), paddings_data.end(), opt_paddings_data.begin() + offset,
                 [](const T &val) { return val; });
  // For ge::PadV3Grad, the length of paddings is required to be 8
  if (force_length8 && dst_length <= kLength8) {
    for (size_t i = 0; i < kLength8 - dst_length; i++) {
      opt_paddings_data.push_back(0);
    }
  }
  if (!paddings_contiguous) {
    auto opt_paddings_size = opt_paddings_data.size();
    std::vector<T> tmp_l;
    std::vector<T> tmp_r;
    for (size_t i = 0; i < opt_paddings_size; i++) {
      if (i % kStep2 == 0) {
        tmp_l.template emplace_back(opt_paddings_data[i]);
      } else {
        tmp_r.template emplace_back(opt_paddings_data[i]);
      }
    }
    opt_paddings_data.clear();
    std::transform(tmp_l.begin(), tmp_l.end(), std::back_inserter(opt_paddings_data), [](const T &val) { return val; });
    std::transform(tmp_r.begin(), tmp_r.end(), std::back_inserter(opt_paddings_data), [](const T &val) { return val; });
  }
  // Create ValueNode
  auto extend_paddings = CreateValueNodeWithKernelInfo(graph, MakeValue(opt_paddings_data));
  return extend_paddings;
}

const AnfNodePtr ConvertBasePaddings::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto input_x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, kIndex0);
  auto input_paddings_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, kIndex1);
  auto opt_paddings_size = 2 * input_x_shape.size();

  if (HasDynPaddings(cnode)) {
    MS_EXCEPTION(TypeError) << "While running in Ascend, the input [paddings] of PadV3 is required to be constant, but "
                               "that is dynamic in node["
                            << node->fullname_with_scope() << "]";
  } else {
    auto prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    auto paddings_contiguous = GetValue<bool>(prim->GetAttr("paddings_contiguous"));
    // ge::padV3 only support that the length of `paddings` is twice than the rank of `x`
    auto input_paddings = common::AnfAlgo::GetInputNode(cnode, kIndex1);
    MS_EXCEPTION_IF_NULL(input_paddings);
    auto paddings_abstract = input_paddings->abstract();
    MS_EXCEPTION_IF_NULL(paddings_abstract);
    auto paddings_type = paddings_abstract->GetType();
    MS_EXCEPTION_IF_NULL(paddings_type);

    auto paddings_value_node =
      CreatePaddingsNode(graph, paddings_abstract, paddings_contiguous, opt_paddings_size, input_paddings_type_id);
    MS_EXCEPTION_IF_NULL(paddings_value_node);
    cnode->set_input(kIndex2, paddings_value_node);
  }
  auto is_expand = ExpandInputXDims(graph, cnode);
  if (is_expand) {
    ReduceOutputDims(graph, cnode);
  }
  return node;
}

bool ConvertPadV3GradPaddings::ExpandInputXDims(const FuncGraphPtr &graph, const CNodePtr &node) const {
  auto input_x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, kIndex0);
  auto input_x_rank = input_x_shape.size();
  // For ge::PadV3Grad, input x must be no less than 4-dimension
  if (input_x_rank >= 4) {
    return false;
  }
  // Expand shape to 4 dimensions
  auto new_shape = input_x_shape;
  for (size_t i = 0; i < kDim4 - input_x_rank; i++) {
    (void)new_shape.emplace_back(1);
  }
  // Replace the x with Reshape
  auto input_x_node = common::AnfAlgo::GetInputNode(node, kIndex0);
  MS_EXCEPTION_IF_NULL(input_x_node);
  auto reshape_node = mindspore::common::CreateReshapeNode(graph, input_x_node, new_shape);
  MS_EXCEPTION_IF_NULL(reshape_node);
  node->set_input(kIndex1, reshape_node);
  return true;
}

void ConvertPadV3GradPaddings::ReduceOutputDims(const FuncGraphPtr &graph, const CNodePtr &node) const {
  auto output_shape = common::AnfAlgo::GetOutputInferShape(node, kIndex0);
  auto reshape_node = mindspore::common::CreateReshapeNode(graph, node, output_shape);
  MS_EXCEPTION_IF_NULL(reshape_node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(node, reshape_node);
}

const BaseRef ConvertPadV3Paddings::DefinePattern() const {
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimPadV3, inputs});
}

const BaseRef ConvertPadV3GradPaddings::DefinePattern() const {
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimPadV3Grad, inputs});
}
}  // namespace opt
}  // namespace mindspore
