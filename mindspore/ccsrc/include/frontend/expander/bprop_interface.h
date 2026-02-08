/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_FRONTEND_EXPANDER_BPROP_INTERFACE_H_
#define MINDSPORE_CCSRC_INCLUDE_FRONTEND_EXPANDER_BPROP_INTERFACE_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <functional>

#include "mindspore/ccsrc/include/utils/expander/node.h"
#include "mindspore/ccsrc/include/utils/expander/emitter.h"

namespace mindspore {
namespace expander {
namespace bprop {
class BpropBuilder;
using BpropBuilderFunc = std::function<NodePtrList(BpropBuilder *)>;

class FRONTEND_EXPORT PynativeCallback {
 public:
  virtual const std::string &opname() const = 0;
  virtual ValuePtr *GetInput(size_t index) const = 0;
  virtual ValuePtrList *GetInputs() const = 0;
  virtual ValuePtr *GetOutput() const = 0;
  virtual bool IsNotRequiresGrad(size_t index) const = 0;

  /// \brief Free device address of tensor without changing its shape and dtype.
  /// \param value[in/out] the tensor value. after calling, the value will be replaced by a new object.
  virtual void FreeDeviceAddress(ValuePtr *value) const = 0;
  /// \brief Free device address of input tensors
  /// \param indices index of inputs to be free. if empty, ALL inputs will be free.
  void FreeInputDeviceAddress(const std::vector<size_t> &indices = {}) const;

  /// \brief Free device address of output tensors
  /// \param indices index of outputs to be free. if empty, ALL outputs will be free.
  void FreeOutputDeviceAddress(const std::vector<size_t> &indices = {}) const;

  /// \brief Free device address of input and output tensors
  /// \param inputs_idx index of inputs to be free. if empty, ALL inputs will be free.
  /// \param outputs_idx index of outputs to be free. if empty, ALL outputs will be free.
  void FreeIODeviceAddress(const std::vector<size_t> &inputs_idx, const std::vector<size_t> &outputs_idx) const {
    FreeInputDeviceAddress(inputs_idx);
    FreeOutputDeviceAddress(outputs_idx);
  }
  /// \brief Deprecated. free input and/or output tensors.
  void DeprecatedFreeDeviceAddress(const mindspore::HashSet<size_t> &indices) const;
};
using FreeUselessValueFunc = std::function<void(const PynativeCallback &)>;
using CloneInplaceInputFunc = std::function<bool(const PynativeCallback &)>;

struct BpropHandle {
  BpropBuilderFunc func = nullptr;
  mindspore::HashSet<size_t> unused_inputs;
  FreeUselessValueFunc free_useless_value_func = nullptr;
  CloneInplaceInputFunc clone_inplace_input_func = nullptr;
};

class FRONTEND_EXPORT BpropBuilder : public Emitter {
 public:
  BpropBuilder(const std::string &name, const ExpanderInferPtr &infer);

  /// \brief Run irbuilder to generate a graph
  NodePtrList Run(const NodePtrList &inputs, const mindspore::HashMap<std::string, ValuePtr> &attrs,
                  const BpropHandle &handle, const std::string &instance_name);
  ValuePtr GetAttr(const std::string &attr) const;
  template <typename S>
  S GetAttr(const std::string &attr) const {
    return GetValue<S>(GetAttr(attr));
  }
  const mindspore::HashMap<std::string, ValuePtr> &GetAttrs() const { return *attrs_ptr_; }
  inline const NodePtr &GetInput(size_t i) const {
    if (MS_UNLIKELY(i >= inputs_ptr_->size())) {
      MS_LOG(EXCEPTION) << "For " << name_ << ", the index " << i << " is out of range of inputs size "
                        << inputs_ptr_->size();
    }
    return (*inputs_ptr_)[i];
  }
  inline const NodePtrList &GetInputs() const { return *inputs_ptr_; }

  NodePtrList BroadcastGradientArgs(const NodePtr &s0, const NodePtr &s1, size_t shift = 0LL);

  // For node that has single output
  ShapeVector GetShape(const NodePtr &node) const { return node->shape(); }
  size_t GetRank(const NodePtr &node) const { return GetShape(node).size(); }
  // For node that has multiple outputs
  std::vector<ShapeVector> GetShapes(const NodePtr &node) const { return node->shapes(); }
  TypePtr GetDtype(const NodePtr &node) const { return node->dtype(); }
  TypeId GetDtypeId(const NodePtr &node) const { return GetDtype(node)->type_id(); }
  ValuePtr GetAttr(const NodePtr &node, const std::string &attr) const;
  int64_t GetSize(const NodePtr &node) const;
  NodePtr DynSize(const NodePtr &node, const TypePtr &type);
  NodePtr DynSize(const NodePtr &node, TypeId type_id);
  NodePtr DynSize(const NodePtr &node);
  NodePtr Range(const NodePtr &limit) { return Range(Value<int64_t>(0), limit, Value<int64_t>(1)); }
  NodePtr Range(const NodePtr &start, const NodePtr &limit, const NodePtr &delta, int64_t max_len = 1000000) {
    return Emit("Range", {start, limit, delta, Value(max_len)});
  }

  NodePtr SequenceToTensor(const NodePtr &node, const TypePtr &dtype = kInt64);
  NodePtr TensorToSequence(const NodePtr &node, const AbstractBasePtr &abs, const TypePtr &dtype = kInt64);
  NodePtr SequenceSetItem(const NodePtr &node, const NodePtr &index, const NodePtr &value);
  NodePtr SequenceSlice(const NodePtr &node, const NodePtr &start, const NodePtr &stop, const NodePtr &step);
  NodePtr TensorToScalar(const NodePtr &node);

  std::string name() const { return name_; }
  std::string GetTargetFromContext() const;
  bool IsGraphMode() const;

  // Tensor getitem by a single integer number on the outermost axis.
  NodePtr TensorGetItem(const NodePtr &node, int64_t idx);

  // get a tensor slice like python.
  // case 1: x[...,2]   => StridedSlice(x, {{-1,{2}}})
  // case 2: x[2, ..., 1:3]   => StridedSlice(x, {{0,{2}}, {-1,{1,3}}})
  // case 3: x[..., 0:3:2, 0::2, :]   => StridedSlice(x, {{-3,{0,3,2}}, {-2,{0,LLONG_MAX,2}}})
  NodePtr StridedSlice(const NodePtr &x, const std::map<int64_t, std::vector<int64_t>> &slices);

  NodePtr StridedSlice(const NodePtr &dout, const NodePtr &begin, const NodePtr &end, const NodePtr &strides,
                       const NodePtr &begin_mask, const NodePtr &end_mask, const NodePtr &ellipsis_mask,
                       const NodePtr &new_axis_mask, const NodePtr &shrink_axis_mask) {
    return Emit("StridedSlice",
                {dout, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask});
  }

  NodePtr StridedSlice(const NodePtr &dout, const NodePtr &begin, const NodePtr &end, const NodePtr &strides,
                       int64_t begin_mask = 0, int64_t end_mask = 0, int64_t ellipsis_mask = 0,
                       int64_t new_axis_mask = 0, int64_t shrink_axis_mask = 0) {
    return StridedSlice(dout, begin, end, strides, Value(begin_mask), Value(end_mask), Value(ellipsis_mask),
                        Value(new_axis_mask), Value(shrink_axis_mask));
  }

  std::string GetInstanceName() const { return instance_name_; }
  NodePtr TanhGrad(const NodePtr &y, const NodePtr &dy) { return Emit("TanhGrad", {y, dy}); }
  virtual NodePtr OutZeros(const NodePtr &node) { return ZerosLike(node); }

 protected:
  std::string name_;
  std::string instance_name_;
  const NodePtrList *inputs_ptr_{nullptr};
  const mindspore::HashMap<std::string, ValuePtr> *attrs_ptr_{nullptr};
};

class FRONTEND_EXPORT BpropExpander {
 public:
  static void FreeUselessValues(const PynativeCallback &cb);
  static bool IsCloneInplaceInput(const PynativeCallback &cb);
};

#ifdef _MSC_VER
class FRONTEND_EXPORT WinBpropRegister {
 public:
  WinBpropRegister();
  ~WinBpropRegister() {}
  void DoNothing() const {}
};
#endif

FRONTEND_EXPORT const BpropHandle *GetBpropIRBuilder(const std::string &name);
}  // namespace bprop
}  // namespace expander

using expander::bprop::BpropExpander;
#ifdef _MSC_VER
using expander::bprop::WinBpropRegister;
#endif
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_FRONTEND_EXPANDER_BPROP_INTERFACE_H_
