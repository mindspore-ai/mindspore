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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_GRAD_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_GRAD_H_

#include <memory>
#include <utility>
#include <map>
#include <vector>
#include <string>
#include <tuple>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/grad/variable.h"
#include "pipeline/pynative/grad/ir/ir_bprop.h"
#include "pipeline/pynative/grad/auto_grad.h"
#include "pipeline/pynative/grad/function/func_builder.h"

namespace mindspore::pynative::autograd {
using NodePtr = expander::NodePtr;
using NodePtrList = expander::NodePtrList;

class FuncBackwardNode : public BackwardNode {
 public:
  FuncBackwardNode(const string &name, expander::bprop::BpropBuilderFunc func,
                   mindspore::HashMap<std::string, ValuePtr> attrs, ValuePtrList op_inputs,
                   AbstractBasePtrList input_abstract, ValuePtr op_output, size_t output_size,
                   AbstractBasePtr out_abstract, std::vector<InputType> grad_type)
      : BackwardNode(name, output_size),
        attrs_(std::move(attrs)),
        op_inputs_(std::move(op_inputs)),
        input_abstract_(std::move(input_abstract)),
        grad_type_(std::move(grad_type)),
        op_output_(std::move(op_output)),
        out_abstract_(std::move(out_abstract)),
        func_(std::move(func)) {}
  ~FuncBackwardNode() override = default;
  TensorPtrList CallBackward(const TensorPtrList &grads) override;
  NodePtrList PreProcess(const TensorPtrList &dout, FuncBuilder *emitter);
  const expander::bprop::BpropBuilderFunc &func() { return func_; }
  void set_attrs(const mindspore::HashMap<std::string, ValuePtr> &attrs) { attrs_ = attrs; }

 private:
  mindspore::HashMap<std::string, ValuePtr> attrs_;
  ValuePtrList op_inputs_;
  abstract::AbstractBasePtrList input_abstract_;
  std::vector<InputType> grad_type_;
  ValuePtr op_output_;
  abstract::AbstractBasePtr out_abstract_;
  expander::bprop::BpropBuilderFunc func_;
};

class HookBackwardNode : public BackwardNode {
 public:
  HookBackwardNode(const string &name, PrimitivePyPtr prim, VectorRef &&args, size_t output_size)
      : BackwardNode(name, output_size), prim_(std::move(prim)), args_(args) {}
  TensorPtrList CallBackward(const TensorPtrList &grads) override;

 private:
  PrimitivePyPtr prim_;
  VectorRef args_;
};

class GraphBackwardNode : public BackwardNode {
 public:
  explicit GraphBackwardNode(const string &name, size_t output_size, bool is_control_flow, std::string cache_key,
                             FuncGraphPtr func_graph, const VectorRef &args)
      : BackwardNode(name, output_size),
        is_control_flow_(is_control_flow),
        cache_key_(std::move(cache_key)),
        func_graph_(std::move(func_graph)),
        args_(args) {}
  TensorPtrList CallBackward(const TensorPtrList &grads) override;
  ValuePtr op_output_;
  bool jit_out_has_dict_{false};

 private:
  bool is_control_flow_{false};
  std::string cache_key_{false};
  FuncGraphPtr func_graph_;
  VectorRef args_;
};

class GraphRoot : public BackwardNode {
 public:
  explicit GraphRoot(const string &name) : BackwardNode(name) {}
  ~GraphRoot() override = default;
  TensorPtrList CallBackward(const TensorPtrList &grads) override { return grads; }
  TensorPtrList BuildFlattenSensGradient(const ValuePtrList &sens_gradient) const;
};

class FakeBackwardNode : public BackwardNode {
 public:
  explicit FakeBackwardNode(const string &name) : BackwardNode(name) {}
  ~FakeBackwardNode() override = default;
  TensorPtrList CallBackward(const TensorPtrList &grads) override {
    MS_LOG(EXCEPTION) << "Illegal primitive " << name() << "'s bprop not defined";
  }
};

class FuncGrad : public AutoGrad {
 public:
  FuncGrad(const ValuePtrList &input_param_values, size_t op_num_in_bprop_graph, bool grad_by_value);
  ~FuncGrad() override = default;

  bool KPynativeOp(const GradParamPtr &grad_param) override;
  // Update top cell output, record last_node
  void UpdateOutputNodeOfTopCell(const ValuePtr &sens_out) override;
  // Reverse connect jit or higher order sub bprop funcgraph
  bool KPynativeWithFProp(const GradParamPtr &grad_param) override;

  ValuePtr Finish(const TensorPtrList &weights, const std::vector<size_t> &grad_position, const GradAttr &grad_attr,
                  const ValuePtr &sens = nullptr);

 private:
  void BackPropagate();
  void BuildForwardLastNode(const ValuePtr &sens_gradient);
  OrderedSet<FuncVariablePtr>::reverse_iterator GetLastNodeReverseIter();
  void ConstructParameterNodes(const ValuePtrList &inputs);

  BackwardNodePtr BuildFuncBackwardNode(const PrimitivePtr &prim, const expander::bprop::BpropBuilderFunc &func,
                                        const ValuePtrList &flatten_inputs, const OpGradInfoPtr &op_grad_info);
  BackwardNodePtr BuildCustomBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                          const OpGradInfoPtr &op_grad_info);
  BackwardNodePtr BuildHookBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                        const OpGradInfoPtr &op_grad_info);
  BackwardNodePtr BuildGraphBackwardNode(const GradParamPtr &grad_param);
  ValuePtr GetGrads(const tensor::TensorPtrList &weights, const std::vector<size_t> &grad_position,
                    const GradAttr &grad_attr);
  ValuePtr GetInputGrads(bool grad_all_inputs, bool get_by_position, const std::vector<size_t> &grad_position);
  ValuePtr GetWeightGrads(bool grad_weights, const TensorPtrList &weights, bool weight_param_is_tuple);
  ValuePtr GetWeightGrad(const tensor::TensorPtr &weight);
  void ClearGrads(const TensorPtrList &weights);
  ValuePtrList OnsLike(const ValuePtr &value);
  void CheckSensShapeAndType(const ValuePtr &sens_gradient);
  void PruningGradGraph(const TensorPtrList &weights, const GradAttr &grad_attr,
                        const std::vector<size_t> &grad_position);
  std::shared_ptr<FuncBuilder> func_impl_;
  OrderedSet<FuncVariablePtr> variable_set_;
  std::vector<std::pair<ValuePtr, FuncVariablePtr>> cell_inputs_;
  std::vector<tensor::TensorPtr> weights_used_in_graph_;
  ValuePtr sens_value_{nullptr};
  FuncVariablePtr last_variable_{nullptr};
  TensorPtrList root_gradients_;
};
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_GRAD_H_
