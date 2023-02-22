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

#include "plugin/device/ascend/optimizer/ge/avg_pool_grad_for_ge.h"

#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr char kPoolDataFormatAttrName[] = "format";
constexpr char kPoolKernelSizeAttrName[] = "kernel_size";
constexpr char kPoolStridesAttrName[] = "strides";
constexpr char kPoolPadModeAttrName[] = "pad_mode";
constexpr size_t kAvgPoolGradInputXIndex = 1;
constexpr size_t kAvgPoolGradInputOriginOutIndex = 2;
constexpr size_t kAvgPoolGradInputGradIndex = 3;
}  // namespace

const BaseRef AvgPoolGradForGE::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VarPtr x3 = std::make_shared<Var>();
  return VectorRef({prim::kPrimAvgPoolGrad, x1, x2, x3});
}

const AnfNodePtr AvgPoolGradForGE::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto avg_pool_grad_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(avg_pool_grad_node);
  auto origin_prim = common::AnfAlgo::GetCNodePrimitive(avg_pool_grad_node);
  MS_EXCEPTION_IF_NULL(origin_prim);
  auto format_value = origin_prim->GetAttr(kPoolDataFormatAttrName);
  std::string format;
  if (format_value == nullptr) {
    format = "NCHW";
  } else {
    format = GetValue<std::string>(format_value);
  }
  auto pad_mode_value = origin_prim->GetAttr(kPoolPadModeAttrName);
  auto pad_mode_type = pad_mode_value->type()->type_id();
  std::string pad_mode;
  if (pad_mode_type == TypeId::kNumberTypeInt64) {
    auto pad_value = GetValue<int64_t>(pad_mode_value);
    pad_mode = pad_value == 1 ? "SAME" : "VALID";
  } else {
    pad_mode = GetValue<std::string>(pad_mode_value);
  }
  auto origin_shape = avg_pool_grad_node->input(kAvgPoolGradInputXIndex)->Shape();
  if (origin_shape->IsDynamic()) {
    MS_LOG(EXCEPTION) << "Do not support dynamic AvgPoolGrad in GE backend";
  }
  auto shape_vector = origin_shape->cast<abstract::ShapePtr>()->shape();
  std::vector<int32_t> value_node_data;
  (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(value_node_data), LongToInt);
  auto origin_shape_value = MakeValue(value_node_data);
  auto origin_shape_node = NewValueNode(origin_shape_value);
  origin_shape_node->set_abstract(origin_shape_value->ToAbstract());
  auto new_avg_pool_node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kAvgPoolGradGeOpName)),
                                            origin_shape_node, avg_pool_grad_node->input(kAvgPoolGradInputGradIndex)});
  MS_EXCEPTION_IF_NULL(new_avg_pool_node);
  common::AnfAlgo::CopyNodeAttr(kPoolKernelSizeAttrName, avg_pool_grad_node, new_avg_pool_node);
  common::AnfAlgo::CopyNodeAttr(kPoolStridesAttrName, avg_pool_grad_node, new_avg_pool_node);
  common::AnfAlgo::SetNodeAttr(kPoolDataFormatAttrName, MakeValue(format), new_avg_pool_node);
  common::AnfAlgo::SetNodeAttr(kPoolPadModeAttrName, MakeValue(pad_mode), new_avg_pool_node);
  new_avg_pool_node->set_abstract(node->abstract());
  return new_avg_pool_node;
}
}  // namespace opt
}  // namespace mindspore
