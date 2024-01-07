/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_META_GRAD_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_META_GRAD_H_

#include <memory>
#include <utility>
#include <map>
#include <vector>
#include <string>
#include <tuple>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "frontend/expander/bprop/bprop.h"
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/grad/variable.h"
#include "pipeline/pynative/grad/function/func_builder.h"

namespace mindspore::pynative::autograd {
using TensorPtrList = tensor::TensorPtrList;
struct GradAttr {
  GradAttr(bool get_all, bool get_by_list, bool sens_param, bool get_by_position, bool weight_param_is_tuple)
      : grad_all_inputs(get_all),
        grad_weights(get_by_list),
        has_sens(sens_param),
        get_by_position(get_by_position),
        weight_param_is_tuple(weight_param_is_tuple) {}

  bool grad_all_inputs;
  bool grad_weights;
  bool has_sens;
  bool get_by_position;
  bool weight_param_is_tuple;
};

using NodePtr = expander::NodePtr;
using NodePtrList = expander::NodePtrList;
class FuncBackwardNode : public BackwardNode {
 public:
  FuncBackwardNode(const string &name, const expander::bprop::BpropBuilderFunc &func,
                   const mindspore::HashMap<std::string, ValuePtr> &attrs, const ValuePtrList &op_inputs,
                   const ValuePtr &op_output, size_t output_size, const std::vector<InputType> &grad_type)
      : BackwardNode(name, output_size),
        func_(func),
        attrs_(attrs),
        op_inputs_(op_inputs),
        op_output_(op_output),
        grad_type_(grad_type) {}
  ~FuncBackwardNode() override = default;
  TensorPtrList CallBackward(const TensorPtrList &grads) override;
  NodePtrList PreProcess(const TensorPtrList &dout, FuncBuilder *emitter);
  TensorPtrList LazeUpdateZeroGradient(const TensorPtrList &dout, FuncBuilder *emitter);
  const expander::bprop::BpropBuilderFunc &func() { return func_; }
  void set_attrs(const mindspore::HashMap<std::string, ValuePtr> &attrs) { attrs_ = attrs; }

 private:
  expander::bprop::BpropBuilderFunc func_;
  mindspore::HashMap<std::string, ValuePtr> attrs_;
  std::vector<ValuePtr> op_inputs_;
  ValuePtr op_output_;
  std::vector<InputType> grad_type_;
};

class HookBackwardNode : public BackwardNode {
 public:
  HookBackwardNode(const string &name, const PrimitivePyPtr &prim, const VectorRef &args, size_t output_size)
      : BackwardNode(name, output_size), prim_(prim), args_(args) {}
  TensorPtrList CallBackward(const TensorPtrList &grads) override;

 private:
  PrimitivePyPtr prim_;
  VectorRef args_;
};

class GraphRoot : public BackwardNode {
 public:
  explicit GraphRoot(const string &name) : BackwardNode(name) {}
  ~GraphRoot() override = default;
  TensorPtrList CallBackward(const TensorPtrList &grads) override { return grads; }
  TensorPtrList BuildFlattenSensGradient(const ValuePtrList &sens_gradient) const;
};

class AutoGradCell {
 public:
  AutoGradCell(const ValuePtrList &input_param_values, size_t op_num_in_bprop_graph, bool grad_by_value);
  ~AutoGradCell() = default;
  bool KPynativeOp(const GradParamPtr &grad_param);
  // Update top cell output, record last_node
  void UpdateOutputNodeOfTopCell(const ValuePtr &sens_out);

  ValuePtr Finish(const TensorPtrList &weights, const std::vector<size_t> &grad_position, const GradAttr &grad_attr,
                  const ValuePtr &sens = nullptr);
  // to do
  // Reverse connect jit or higher order sub bprop funcgraph
  bool KPynativeWithFProp(const GradParamPtr &grad_param) { return true; }

 private:
  void BackPropagate();
  void BuildForwardLastNode(const ValuePtr &sens_gradient);
  OrderedSet<VariablePtr>::reverse_iterator GetLastNodeReverseIter();
  void ConstructParameterNodes(const ValuePtrList &inputs);

  static BackwardNodePtr BuildFuncBackwardNode(const PrimitivePtr &prim, const expander::bprop::BpropBuilderFunc &func,
                                               const ValuePtrList &flatten_inputs, const OpGradInfoPtr &op_grad_info);
  static BackwardNodePtr BuildCustomBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                                 const OpGradInfoPtr &op_grad_info);
  static BackwardNodePtr BuildHookBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                               const OpGradInfoPtr &op_grad_info);
  ValuePtr GetGrads(const tensor::TensorPtrList &weights, const std::vector<size_t> &grad_position,
                    const GradAttr &grad_attr);
  ValuePtr GetInputGrads(bool grad_all_inputs, bool get_by_position, const std::vector<size_t> &grad_position);
  ValuePtr GetWeightGrads(bool grad_weights, const TensorPtrList &weights, bool weight_param_is_tuple);
  ValuePtr GetWeightGrad(const tensor::TensorPtr &weight);
  void ClearGrads(const TensorPtrList &weights);
  ValuePtrList OnsLike(const ValuePtr &value);
  // To do, combin code.
  void CheckSensShapeAndType(const ValuePtr &sens_gradient);
  std::unique_ptr<FuncBuilder> func_impl_;
  OrderedSet<VariablePtr> variable_set_;
  std::vector<std::pair<ValuePtr, VariablePtr>> cell_inputs_;
  ValuePtr sens_value_{nullptr};
  VariablePtr last_variable_{nullptr};
  TensorPtrList root_gradients_;
  bool grad_by_value_{true};
  std::string device_target_;
};
using AutoGradCellPtr = std::shared_ptr<AutoGradCell>;
void ClearPyNativeAutoGradStaticRes();
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_META_GRAD_H_