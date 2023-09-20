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

#include "plugin/device/ascend/optimizer/mindir/reduce_axis_update.h"
#include <vector>
#include <memory>
#include "mindspore/core/ops/math_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kReduceInputNum = 2;
constexpr size_t kXInputIndex = 1;
constexpr size_t kAxisInputIndex = 2;
constexpr auto r_reduce = "r_reduce";
constexpr auto m_reduce = "m_reduce";
constexpr auto kXs = "Xs";
constexpr auto kV = "V";
constexpr auto v_axis = "axis";

tensor::TensorPtr CreateTensor(const std::vector<int64_t> &values) {
  auto type_ptr = kInt64;
  auto data_length = sizeof(int64_t);
  std::vector<int64_t> tensor_shape = {SizeToLong(values.size())};
  MS_EXCEPTION_IF_NULL(type_ptr);
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_ptr->type_id(), tensor_shape);
  MS_EXCEPTION_IF_NULL(tensor);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, type_ptr};
  tensor->set_device_info(device_info);
  auto data_ptr = tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto buffer_size = values.size() * data_length;
  if (buffer_size != 0) {
    auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(tensor->data().nbytes()), values.data(), buffer_size);
    if (ret_code != EOK) {
      MS_LOG(EXCEPTION) << "Failed to copy data into tensor, memcpy_s errorno: " << ret_code;
    }
  }
  return tensor;
}
}  // namespace

bool ReduceAxisUpdate::IsReduce(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (IsPrimitive(node, prim::kPrimReduceMin) || IsPrimitive(node, prim::kPrimReduceMax) ||
        IsPrimitive(node, prim::kPrimReduceMean) || IsPrimitive(node, prim::kPrimReduceSum) ||
        IsPrimitive(node, prim::kPrimReduceProd) || IsPrimitive(node, prim::kPrimReduceAll) ||
        IsPrimitive(node, prim::kPrimReduceAny)) {
      return true;
    }
  }

  return false;
}

bool ReduceAxisUpdate::IsAxisEmpty(const ValueNodePtr &axis_node) const {
  MS_EXCEPTION_IF_NULL(axis_node);
  const ValuePtr &value = axis_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<ValueTuple>()) {
    auto tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple);
    return tuple->size() == 0;
  } else if (value->isa<ValueList>()) {
    auto list = value->cast<ValueListPtr>();
    MS_EXCEPTION_IF_NULL(list);
    return list->size() == 0;
  } else if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    return tensor->DataSize() == 0;
  }

  return false;
}

bool ReduceAxisUpdate::IsInputScalar(const AnfNodePtr &x_node) const {
  MS_EXCEPTION_IF_NULL(x_node);
  auto x_shape_ptr = x_node->Shape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  ShapeVector x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  return x_shape.empty();
}

namespace {
constexpr size_t kAxisIndex = 2;
bool IsAxisEmptySequence(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() <= kAxisIndex) {
    return false;
  }
  const auto &axis_node = cnode->input(kAxisIndex);

  if (axis_node == nullptr || (!axis_node->isa<ValueNode>())) {
    return false;
  }
  const auto &value_node = axis_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &value = value_node->value();
  if (value == nullptr) {
    return false;
  }
  if (value->isa<ValueSequence>()) {
    const auto &value_sequence = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(value_sequence);
    return value_sequence->size() == 0;
  } else if (value->isa<tensor::Tensor>()) {
    const auto &tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    const auto &shapes = tensor->shape();
    return shapes.size() == 1 && shapes[0] == 0;
  }
  return false;
}
}  // namespace

bool ReduceAxisUpdate::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "Reduce node is " << node->DebugString() << ".";

  // In control flow, empty tuples are sometimes set to dynamic len which are considered dynamic shapes, but they
  // are not actually needed, so empty tuple scenarios are excluded here.
  if (common::AnfAlgo::IsNodeOutputDynamicShape(node) && (!IsAxisEmptySequence(node))) {
    MS_LOG(INFO) << "The dimension of " << node->DebugString() << " is unknown.";
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto input_num = common::AnfAlgo::GetInputNum(cnode);
  if (input_num < kReduceInputNum) {
    MS_LOG(EXCEPTION) << "The input tensor size[" << input_num << "] of node ["
                      << cnode->DebugString() + "] is not equal to " << kReduceInputNum
                      << trace::DumpSourceLines(cnode);
  }

  const auto &inputs = cnode->inputs();
  const AnfNodePtr &input_x = inputs.at(kXInputIndex);
  const AnfNodePtr &input_axis = inputs.at(kAxisInputIndex);
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_axis);
  MS_LOG(INFO) << "X input is " << input_x->DebugString() << ".";
  MS_LOG(INFO) << "Axis input is " << input_axis->DebugString() << ".";

  auto axis_value_node = input_axis->cast<ValueNodePtr>();
  if (axis_value_node == nullptr || (!IsAxisEmpty(axis_value_node) && !IsInputScalar(input_x))) {
    MS_LOG(INFO) << "Axis input of node " << node->fullname_with_scope() << " is not value node or axis is not empty.";
    return false;
  } else {
    MS_LOG(INFO) << "Axis of node " << node->fullname_with_scope() << " is empty.";
  }
  return true;
}

AnfNodePtr BuildAxis(const PatternMap &m) {
  auto node = m.Get(m_reduce);
  MS_EXCEPTION_IF_NULL(node);
  ShapeVector x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  size_t x_dim_len = x_shape.size();
  MS_LOG(INFO) << "Input x dim len: " << x_dim_len;
  std::vector<int64_t> axis = {0};
  for (size_t i = 1; i < x_dim_len; ++i) {
    (void)axis.emplace_back(SizeToLong(i));
    MS_LOG(INFO) << "x dim: " << x_shape[i];
  }
  ValuePtr new_value = MakeValue(CreateTensor(axis));
  MS_EXCEPTION_IF_NULL(new_value);
  auto new_axis_node = std::make_shared<ValueNode>(new_value);
  MS_EXCEPTION_IF_NULL(new_axis_node);
  new_axis_node->set_abstract(new_value->ToAbstract());

  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  new_axis_node->set_kernel_info(kernel_info);
  std::shared_ptr<kernel::KernelBuildInfo::KernelBuildInfoBuilder> builder =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  kernel_info->set_select_kernel_build_info(builder->Build());
  MS_EXCEPTION_IF_NULL(kernel_info->select_kernel_build_info());
  kernel_info->GetMutableSelectKernelBuildInfo()->SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  kernel_info->GetMutableSelectKernelBuildInfo()->SetOutputsFormat({kOpFormat_DEFAULT});
  kernel_info->GetMutableSelectKernelBuildInfo()->SetOutputsDeviceType({TypeId::kNumberTypeInt64});
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddValueNodeToGraph(new_axis_node);
  return new_axis_node;
}

AnfNodePtr BuildReduce(const PatternMap &m, const AnfNodePtr &) {
  auto anf = m.Get(m_reduce);
  MS_EXCEPTION_IF_NULL(anf);
  auto cnode = anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  cnode->set_input(kAxisInputIndex, m.Get(v_axis));
  return cnode;
}

void ReduceAxisUpdate::DefineSrcPattern(SrcPattern *src_pattern) {
  (void)(*src_pattern).AddVar(kV, IsReduce).AddSeqVar(kXs).AddCNode(m_reduce, {kV, kXs});
}

void ReduceAxisUpdate::DefineDstPattern(DstPattern *dst_pattern) {
  auto reduce_input = Unpacking(kXs);
  reduce_input.at(kAxisInputIndex - 1) = v_axis;
  (void)(*dst_pattern).AddValueNode(v_axis, BuildAxis).AddCNode(r_reduce, {kV, reduce_input}, BuildReduce);
}
}  // namespace opt
}  // namespace mindspore
