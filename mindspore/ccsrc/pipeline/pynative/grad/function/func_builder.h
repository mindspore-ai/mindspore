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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_BUILDER_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_BUILDER_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include "utils/hash_map.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "pipeline/pynative/grad/function/func_pass.h"

namespace mindspore::pynative::autograd {
using NodePtr = expander::NodePtr;
using NodePtrList = expander::NodePtrList;
using BpropBuilder = expander::bprop::BpropBuilder;

class FuncBuilder : public BpropBuilder {
 public:
  FuncBuilder(const std::string &name, std::string device_target, const expander::ExpanderInferPtr &infer = nullptr)
      : BpropBuilder(name, infer), device_target_(device_target) {
    pass_forward_ = std::make_shared<bprop_pass::FuncPassForward>(this, std::move(device_target));
  }
  ~FuncBuilder() override = default;
  NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) override;
  NodePtr EmitValue(const ValuePtr &value) override;
  // Override Stack to flatten tuple input.
  NodePtr Stack(const NodePtr &x, const ValuePtr &axis) override;
  NodePtr Stack(const NodePtrList &x, int64_t axis) override;
  NodePtr BatchNormGrad(const NodePtrList &inputs) override;
  NodePtr SparseSoftmaxCrossEntropyWithLogits(const NodePtrList &inputs, const expander::DAttr &attrs,
                                              const NodePtr &out, const NodePtr &dout, bool is_graph_mode) override;
  NodePtr Depend(const NodePtr &value, const NodePtr &expr) override;
  NodePtr TupleGetItem(const NodePtr &input, size_t i) override;
  NodePtr TupleGetItem(const NodePtr &input, const NodePtr &index) override;
  NodePtr MakeTuple(const NodePtrList &inputs) override;
  NodePtr MakeList(const NodePtrList &inputs) override;
  NodePtr OutZeros(const NodePtr &node) override;

  ValuePtr Ones(const ValuePtr &value);
  ValuePtr Zeros(const ValuePtr &value);
  ValuePtr Add(const ValuePtr &input, const ValuePtr &other);
  void SetInputs(std::string instance_name, const std::vector<NodePtr> *inputs,
                 mindspore::HashMap<std::string, ValuePtr> *attrs_ptr);

 private:
  NodePtrList FlattenNode(const NodePtr &input);
  ValuePtr FillZeros(const ValuePtr &value, const abstract::AbstractBasePtr &abs);
  std::string device_target_;
  bprop_pass::FuncPassForwardPtr pass_forward_;
};
using FuncBuilderPtr = std::shared_ptr<FuncBuilder>;
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_BUILDER_H_
