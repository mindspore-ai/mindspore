/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "frontend/expander/bprop/bprop_irbuilder.h"

#include <algorithm>
#include <limits>
#include <vector>
#include "frontend/expander/bprop/grad_ops/common_utils.h"
#include "include/common/expander/core/node.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"
#include "ops/array_ops.h"
#include "ops/sequence_op_name.h"
#include "ops/tensor_to_scalar.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace expander {
namespace bprop {
NodePtrList BpropBuilder::Run(const NodePtrList &inputs, const mindspore::HashMap<std::string, ValuePtr> &attrs,
                              const BpropHandle &handle, const std::string &instance_name) {
  inputs_ptr_ = &inputs;
  attrs_ptr_ = &attrs;
  instance_name_ = instance_name;
  return handle.func(this);
}

class BroadcastGradientArgsShapeCalc : public ShapeCalcFunctor {
 public:
  // cppcheck-suppress unknownMacro
  DECLARE_SHAPE_CALC("ShapeCalc_BroadcastGradientArgs", BroadcastGradientArgsShapeCalc)
  explicit BroadcastGradientArgsShapeCalc(size_t shift)
      : ShapeCalcFunctor("ShapeCalc_BroadcastGradientArgs"), shift_(shift) {}
  ValuePtr ToValue() const override { return MakeValue(shift_); }
  void FromValue(const ValuePtr &value) override { shift_ = GetValue<size_t>(value); }
  ShapeArray Calc(const ShapeArray &inputs) const override {
    auto shape_x = inputs.at(kIndex0);
    auto shape_y = inputs.at(kIndex1);
    if (shift_ == 0) {
      return BroadcastGradientArgsInferValue(shape_x, shape_y);
    } else {
      auto shift1 = SizeToLong(std::min(shift_, shape_x.size()));
      auto shift2 = SizeToLong(std::min(shift_, shape_y.size()));
      ShapeVector broadcast_shape_of_x(shape_x.begin(), shape_x.end() - shift1);
      ShapeVector broadcast_shape_of_y(shape_y.begin(), shape_y.end() - shift2);
      return BroadcastGradientArgsInferValue(broadcast_shape_of_x, broadcast_shape_of_y);
    }
  }
  std::vector<int64_t> Infer(const ShapeArray &, const HashSet<size_t> &) const override {
    constexpr int64_t kShapeDimAny = -1;
    return {kShapeDimAny, kShapeDimAny};
  }

 protected:
  size_t shift_{0};
};
REG_FUNCTOR("ShapeCalc_BroadcastGradientArgs", BroadcastGradientArgsShapeCalc);

NodePtrList BpropBuilder::BroadcastGradientArgs(const NodePtr &s0, const NodePtr &s1, size_t shift) {
  auto check_shp_valid_func = [shift](size_t, const ShapeVector &shape) -> bool {
    auto offset = std::min(shift, shape.size());
    return !(IsDynamicRank(shape) || IsDynamic(ShapeVector{shape.begin(), shape.end() - offset}));
  };

  return ShapeCalc(std::make_shared<BroadcastGradientArgsShapeCalc>(shift), {s0, s1}, {}, check_shp_valid_func);
}

ValuePtr BpropBuilder::GetAttr(const std::string &attr) const {
  auto iter = attrs_ptr_->find(attr);
  if (iter != attrs_ptr_->end()) {
    return iter->second;
  }
  MS_LOG(WARNING) << "The attr " << attr << " does not exist in op " << name();
  return nullptr;
}

ValuePtr BpropBuilder::GetAttr(const NodePtr &node, const std::string &attr) const {
  auto p = GetCNodePrimitive(node->get());
  MS_EXCEPTION_IF_NULL(p);
  return p->GetAttr(attr);
}

int64_t BpropBuilder::GetSize(const NodePtr &node) const {
  auto shape = GetShape(node);
  return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

std::string BpropBuilder::GetTargetFromContext() const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
}

bool BpropBuilder::IsGraphMode() const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode);
}

NodePtr BpropBuilder::TensorGetItem(const NodePtr &node, int64_t idx) {
  auto data_shape = GetShape(node);
  auto n = data_shape.size();
  constexpr const size_t kMaxDims = 8;
  if (n < 1 || n > kMaxDims) {
    MS_EXCEPTION(ValueError) << "Expect Tensor to have dimension between 1 and " << kMaxDims << ", but got: " << n;
  }
  std::vector<int64_t> begin_strides(n, 0);
  std::vector<int64_t> end_strides = data_shape;
  std::vector<int64_t> step_strides(n, 1);
  begin_strides[0] = CheckRange(idx, data_shape[0]);
  end_strides[0] = begin_strides[0] + 1;
  constexpr int64_t begin_mask = 252;  // sum 2^i, i in [2, 8)
  constexpr int64_t end_mask = 252;
  constexpr int64_t ellipsis_mask = 0;
  constexpr int64_t new_axis_mask = 0;
  constexpr int64_t shrink_axis_mask = 1;
  return StridedSlice(node, EmitValue(MakeValue(begin_strides)), EmitValue(MakeValue(end_strides)),
                      EmitValue(MakeValue(step_strides)), begin_mask, end_mask, ellipsis_mask, new_axis_mask,
                      shrink_axis_mask);
}

NodePtr BpropBuilder::StridedSlice(const NodePtr &x, const std::map<int64_t, std::vector<int64_t>> &slices) {
  auto data_shape = GetShape(x);
  auto n = data_shape.size();
  std::vector<int64_t> begin_strides(n, 0);
  std::vector<int64_t> end_strides = data_shape;
  std::vector<int64_t> step_strides(n, 1);
  size_t shrink_axis_mask = 0;
  size_t end_mask = 0;
  constexpr size_t one = 1;
  auto zero = MakeValue<int64_t>(0);
  for (const auto &[_axis, slice] : slices) {
    auto axis = LongToSize(CheckRange(_axis, static_cast<int64_t>(n)));
    if (slice.size() >= kDim2) {
      begin_strides[axis] = slice[kIndex0];
      end_strides[axis] = slice[kIndex1];
      if (end_strides[axis] == LLONG_MAX) {
        end_mask |= (one << axis);
      }
      if (slice.size() >= kDim3) {
        step_strides[axis] = slice[kIndex2];
      }
    } else {
      if (slice.size() == 1) {
        begin_strides[axis] = slice[kIndex0];
        end_strides[axis] = begin_strides[axis] + 1;
        shrink_axis_mask |= (one << axis);
      }
    }
  }
  return StridedSlice(x, Value(begin_strides), Value(end_strides), Value(step_strides), 0, SizeToLong(end_mask), 0, 0,
                      SizeToLong(shrink_axis_mask));
}

DEF_PURE_SHAPE_CALC(g_dyn_size)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray { return {{abstract::ShapeSize(inputs.at(0))}}; })
  .SetInfer([](const ShapeArray &, const HashSet<size_t> &) -> ShapeVector { return {1}; });

// This function will be removed, not recommended to use.
NodePtr BpropBuilder::DynSize(const NodePtr &node) {
  if (!IsDynamic(GetShape(node))) {
    return Tensor(GetSize(node), kInt64);
  }
  return SequenceToTensor(ShapeCalc(g_dyn_size, {node})[0]);
}

// This function will be removed, not recommended to use.
NodePtr BpropBuilder::DynSize(const NodePtr &node, const TypePtr &type) {
  return Cast(SequenceToTensor(DynSize(node)), type);
}

// This function will be removed, not recommended to use.
NodePtr BpropBuilder::DynSize(const NodePtr &node, TypeId type_id) {
  return Cast(SequenceToTensor(DynSize(node)), type_id);
}

NodePtr BpropBuilder::SequenceToTensor(const NodePtr &node, const TypePtr &dtype) {
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    if (node->input_type() == InputType::kConstant) {
      return Tensor(GetIntList(node), dtype);
    }
    if (abs->isa<abstract::AbstractTuple>()) {
      return Emit(kTupleToTensorOpName, {node, Value(static_cast<int64_t>(dtype->type_id()))});
    } else {
      return Emit(kListToTensorOpName, {node, Value(dtype)});
    }
  }
  return node;
}

NodePtr BpropBuilder::TensorToSequence(const NodePtr &node, const AbstractBasePtr &abs, const TypePtr &dtype) {
  if (node->abstract()->isa<abstract::AbstractTensor>()) {
    if (abs->isa<abstract::AbstractTuple>()) {
      if (node->input_type() == InputType::kConstant) {
        return EmitValue(MakeValue(GetIntList(node)));
      }
      return Emit(kTensorToTupleOpName, {node});
    } else {
      if (node->input_type() == InputType::kConstant) {
        auto vec = GetIntList(node);
        std::vector<ValuePtr> value_list;
        (void)std::transform(vec.begin(), vec.end(), std::back_inserter(value_list),
                             [](int64_t ele) { return MakeValue(ele); });
        return EmitValue(std::make_shared<ValueList>(value_list));
      }
      return Emit(kTensorToListOpName, {node});
    }
  }
  return node;
}

NodePtr BpropBuilder::SequenceSetItem(const NodePtr &node, const NodePtr &index, const NodePtr &value) {
  auto abs = node->abstract();
  if (abs->isa<abstract::AbstractTuple>()) {
    return Emit(kTupleSetItemOpName, {node, index, value});
  }
  return Emit(kListSetItemOpName, {node, index, value});
}

NodePtr BpropBuilder::SequenceSlice(const NodePtr &node, const NodePtr &start, const NodePtr &stop,
                                    const NodePtr &step) {
  return Emit(kSequenceSliceOpName, {node, start, stop, step});
}

NodePtr BpropBuilder::TensorToScalar(const NodePtr &node) {
  if (node->input_type() == InputType::kConstant) {
    auto value = GetIntList(node);
    if (value.size() != 1) {
      MS_LOG(EXCEPTION) << "For TensorToScalar, the input value should have only one element, but got " << value.size();
    }
    return Value(value[0]);
  }
  return Emit(ops::kNameTensorToScalar, {node});
}

NodePtr IrBuilder::EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) {
  AnfNodePtrList cnode_inputs = {NewValueNode(prim)};
  cnode_inputs.reserve(inputs.size() + 1);
  (void)std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(cnode_inputs), [](const NodePtr &no) {
    MS_EXCEPTION_IF_NULL(no);
    return no->get();
  });
  auto cnode = func_graph_->NewCNode(cnode_inputs);
  if (scope_ != nullptr) {
    cnode->set_scope(scope_);
  }
  auto node = NewIrNode(cnode->cast<AnfNodePtr>());
  infer_->Infer(node);
  return node;
}

NodePtr IrBuilder::EmitValue(const ValuePtr &value) {
  auto node = NewIrNode(NewValueNode(value));
  infer_->Infer(node);
  return node;
}

NodePtr IrBuilder::Conditional(const NodePtr &cond, const BlockFunc &true_case, const BlockFunc &false_case) {
  CtrlFlowBlock cfb(this, this->func_graph());
  this->func_graph()->set_flag(kFlagIsControlFlow, true);
  return cfb.IfThenElse(cond, true_case, false_case);
}

NodePtr IrBuilder::While(const NodePtr &cond, const BlockFunc &body, const NodePtrList &init_list) {
  CtrlFlowBlock cfb(this, this->func_graph());
  this->func_graph()->set_flag(kFlagIsControlFlow, true);
  return cfb.While(cond, body, init_list);
}
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
