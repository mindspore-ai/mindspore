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

#include "frontend/optimizer/irpass/ge/avg_pool_grad_for_ge.h"

#include "pybind_api/pybind_patch.h"
#include "pybind_api/ir/tensor_py.h"
#include "pipeline/pynative/base.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "include/common/utils/python_adapter.h"
#include "mindspore/core/mindapi/ir/common.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
constexpr char kPoolDataFormatAttrName[] = "data_format";
constexpr char kPoolKernelSizeAttrName[] = "kernel_size";
constexpr char kPoolStridesAttrName[] = "strides";
constexpr char kPoolPadModeAttrName[] = "pad_mode";
constexpr char kOpsGradFunctionName[] = "mindspore.ops.operations._grad_ops";
constexpr size_t kAvgPoolGradInputXIndex = 1;
constexpr size_t kAvgPoolGradInputOriginOutIndex = 2;
constexpr size_t kAvgPoolGradInputGradIndex = 3;
}  // namespace
AnfNodePtr AvgPoolGradForGE::operator()(const OptimizerPtr &opt, const AnfNodePtr &node) {
  Reset();
  AnfVisitor::Match(prim::kPrimAvgPoolGrad, {IsNode, IsNode, IsNode})(node);

  if (!is_match_ && node->func_graph() == nullptr) {
    return nullptr;
  }

  auto avg_pool_grad_node = node->cast<CNodePtr>();
  auto origin_prim = GetValueNode<PrimitivePtr>(avg_pool_grad_node->input(0));
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
    pad_mode = pad_value ? "SAME" : "VALID";
  } else {
    pad_mode = GetValue<std::string>(pad_mode_value);
  }
  auto origin_shape = avg_pool_grad_node->input(kAvgPoolGradInputXIndex)->Shape();
  if (origin_shape->IsDynamic()) {
    MS_LOG(EXCEPTION) << "Do not support dynamic AvgPoolGrad in GE backend";
  } else {
    auto shape_vector = origin_shape->cast<abstract::ShapePtr>()->shape();
    std::vector<int32_t> value_node_data;
    std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(value_node_data), LongToInt);
    auto origin_shape_value = MakeValue(value_node_data);
    auto origin_shape_node = NewValueNode(origin_shape_value);
    origin_shape_node->set_abstract(origin_shape_value->ToAbstract());

    auto avg_pool_grad_ge_obj = python_adapter::GetPyFn(kOpsGradFunctionName, "AvgPoolGradGe")();

    const auto &adapter = py::cast<PrimitivePyAdapterPtr>(avg_pool_grad_ge_obj);
    MS_EXCEPTION_IF_NULL(adapter);
    auto attached_prim = adapter->attached_primitive();
    if (attached_prim == nullptr) {
      attached_prim = std::make_shared<PrimitivePy>(avg_pool_grad_ge_obj, adapter);
      adapter->set_attached_primitive(attached_prim);
    }
    auto new_prim = attached_prim->cast<PrimitivePtr>();

    new_prim->set_attr(kPoolKernelSizeAttrName, origin_prim->GetAttr(kPoolKernelSizeAttrName));
    new_prim->set_attr(kPoolStridesAttrName, origin_prim->GetAttr(kPoolStridesAttrName));
    new_prim->set_attr(kPoolDataFormatAttrName, MakeValue(format));
    new_prim->set_attr(kPoolPadModeAttrName, MakeValue(pad_mode));

    auto new_avg_pool_node = node->func_graph()->NewCNode(
      new_prim, {origin_shape_node, avg_pool_grad_node->input(kAvgPoolGradInputGradIndex)});
    new_avg_pool_node->set_abstract(node->abstract());
    return new_avg_pool_node;
  }
  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
