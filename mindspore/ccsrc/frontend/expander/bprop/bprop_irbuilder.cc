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
#include "include/common/utils/utils.h"
#include "ops/array_ops.h"
#include "ops/sequence_op_name.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace expander {
namespace bprop {
NodePtrList BpropIRBuilder::Run(const NodePtrList &inputs, const mindspore::HashMap<std::string, ValuePtr> &attrs,
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
      return bprop::BroadcastGradientArgs(shape_x, shape_y);
    } else {
      ShapeVector broadcast_shape_of_x(shape_x.begin(), shape_x.end() - shift_);
      ShapeVector broadcast_shape_of_y(shape_y.begin(), shape_y.end() - shift_);
      return bprop::BroadcastGradientArgs(broadcast_shape_of_x, broadcast_shape_of_y);
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

NodePtrList BpropIRBuilder::BroadcastGradientArgs(const NodePtr &s0, const NodePtr &s1, size_t shift) {
  auto check_shp_valid_func = [shift](size_t, const ShapeVector &shape) -> bool {
    return !(IsDynamicRank(shape) || IsDynamic(ShapeVector{shape.begin(), shape.end() - shift}));
  };

  return ShapeCalc(std::make_shared<BroadcastGradientArgsShapeCalc>(shift), {s0, s1}, {}, check_shp_valid_func);
}

ValuePtr BpropIRBuilder::GetAttr(const std::string &attr) const {
  auto iter = attrs_ptr_->find(attr);
  if (iter != attrs_ptr_->end()) {
    return iter->second;
  }
  MS_LOG(WARNING) << "The attr " << attr << " does not exist in op " << name();
  return nullptr;
}

NodePtr BpropIRBuilder::GetInput(size_t i) const {
  if (i >= inputs_ptr_->size()) {
    MS_LOG(EXCEPTION) << "For " << name_ << ", the index " << i << " is out of range of inputs size "
                      << inputs_ptr_->size();
  }
  return (*inputs_ptr_)[i];
}

ValuePtr BpropIRBuilder::GetAttr(const NodePtr &node, const std::string &attr) const {
  auto p = GetCNodePrimitive(node->get());
  MS_EXCEPTION_IF_NULL(p);
  return p->GetAttr(attr);
}

int64_t BpropIRBuilder::GetSize(const NodePtr &node) const {
  auto shape = GetShape(node);
  return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

std::string BpropIRBuilder::GetTargetFromContext() const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
}

bool BpropIRBuilder::IsGraphMode() const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode);
}

NodePtr BpropIRBuilder::TensorGetItem(const NodePtr &node, int64_t idx) {
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
  return Emit(
    kStridedSliceOpName,
    {node, EmitValue(MakeValue(begin_strides)), EmitValue(MakeValue(end_strides)), EmitValue(MakeValue(step_strides))},
    {{kAttrBeginMask, MakeValue(begin_mask)},
     {kAttrEndMask, MakeValue(end_mask)},
     {kAttrEllipsisMask, MakeValue(ellipsis_mask)},
     {kAttrNewAxisMask, MakeValue(new_axis_mask)},
     {kAttrShrinkAxisMask, MakeValue(shrink_axis_mask)}});
}

NodePtr BpropIRBuilder::StridedSlice(const NodePtr &x, const std::map<int64_t, std::vector<int64_t>> &slices) {
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
  return Emit(kStridedSliceOpName, {x, Value(begin_strides), Value(end_strides), Value(step_strides)},
              {{kAttrBeginMask, zero},
               {kAttrEndMask, MakeValue(SizeToLong(end_mask))},
               {kAttrEllipsisMask, zero},
               {kAttrNewAxisMask, zero},
               {kAttrShrinkAxisMask, MakeValue(SizeToLong(shrink_axis_mask))}});
}

DEF_PURE_SHAPE_CALC(g_dyn_size)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray { return {{abstract::ShapeSize(inputs.at(0))}}; })
  .SetInfer([](const ShapeArray &, const HashSet<size_t> &) -> ShapeVector { return {1}; });
NodePtr BpropIRBuilder::DynSize(const NodePtr &node) {
  if (!IsDynamic(GetShape(node))) {
    return Value(GetSize(node));
  }
  return ShapeCalc(g_dyn_size, {node})[0];
}

NodePtr BpropIRBuilder::SequenceToTensor(const NodePtr &node, const TypePtr &dtype) {
  auto abs = node->abstract();
  if (abs->isa<abstract::AbstractSequence>()) {
    if (node->isa<ValueNode>()) {
      return Tensor(GetIntList(node), dtype);
    }
    if (abs->isa<abstract::AbstractTuple>()) {
      return Emit(kTupleToTensorOpName, {node, Value(dtype)});
    } else {
      return Emit(kListToTensorOpName, {node, Value(dtype)});
    }
  }
  return node;
}

NodePtr BpropIRBuilder::TensorToSequence(const NodePtr &node, const AbstractBasePtr &abs, const TypePtr &dtype) {
  if (node->abstract()->isa<abstract::AbstractTensor>()) {
    if (abs->isa<abstract::AbstractTuple>()) {
      if (node->isa<ValueNode>()) {
        return EmitValue(MakeValue(GetIntList(node)));
      }
      return Emit(kTensorToTupleOpName, {node});
    } else {
      if (node->isa<ValueNode>()) {
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

NodePtr BpropIRBuilder::SequenceSetItem(const NodePtr &node, const NodePtr &index, const NodePtr &value) {
  auto abs = node->abstract();
  if (abs->isa<abstract::AbstractTuple>()) {
    return Emit(kTupleSetItemOpName, {node, index, value});
  }
  return Emit(kListSetItemOpName, {node, index, value});
}

NodePtr BpropIRBuilder::SequenceSlice(const NodePtr &node, const NodePtr &start, const NodePtr &stop,
                                      const NodePtr &step) {
  return Emit(kSequenceSliceOpName, {node, start, stop, step});
}
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
