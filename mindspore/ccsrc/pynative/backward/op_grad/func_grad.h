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

#include <cstdint>
#include <memory>
#include <utility>
#include <map>
#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "pynative/utils/base.h"
#include "include/utils/pynative/variable.h"
#include "pynative/backward/hook/custom_function.h"
#include "pynative/backward/hook/function_py.h"
#include "pynative/backward/op_grad/func_builder.h"

namespace mindspore::pynative::autograd {
class FuncBackwardNode : public BackwardNode {
 public:
  FuncBackwardNode(string name, expander::bprop::BpropBuilderFunc func, FuncBuilderPtr emitter,
                   HashMap<std::string, ValuePtr> attrs, abstract::AbstractBasePtrList inputs_abs,
                   std::vector<InputType> input_value_grad_type, abstract::AbstractBasePtr out_abs, size_t output_size)
      : BackwardNode(std::move(name), output_size),
        func_(std::move(func)),
        emitter_(std::move(emitter)),
        attrs_(std::move(attrs)),
        inputs_abs_(std::move(inputs_abs)),
        input_value_grad_type_(std::move(input_value_grad_type)),
        out_abs_(std::move(out_abs)) {}
  ~FuncBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  NodePtrList PreProcess(const ValuePtrList &dout, const FuncBuilderPtr &emitter);
  ValuePtrList PostProcess(const ValuePtrList &gradient_value) override;
  const expander::bprop::BpropBuilderFunc &grad_func() { return func_; }
  void set_attrs(const mindspore::HashMap<std::string, ValuePtr> &attrs) { attrs_ = attrs; }
  const HashMap<std::string, ValuePtr> &attrs() const { return attrs_; }
  void SetSavedInputs(const std::vector<ValuePtr> &saved_inputs) { saved_inputs_ = saved_inputs; }
  void SetSavedOutput(const ValuePtr &saved_output) { saved_output_ = saved_output; }
  void Release() override;

 protected:
  expander::bprop::BpropBuilderFunc func_;
  FuncBuilderPtr emitter_;
  HashMap<std::string, ValuePtr> attrs_;
  ValuePtrList saved_inputs_;
  ValuePtr saved_output_;
  abstract::AbstractBasePtrList inputs_abs_;
  std::vector<InputType> input_value_grad_type_;
  abstract::AbstractBasePtr out_abs_;
};
using FuncBackwardNodePtr = std::shared_ptr<FuncBackwardNode>;

class HookBackwardNode : public BackwardNode {
 public:
  HookBackwardNode(const string &name, PrimitivePyPtr prim, size_t output_size, abstract::AbstractBasePtr out_abstract)
      : BackwardNode(name, output_size), prim_(std::move(prim)), out_abstract_(std::move(out_abstract)) {}
  ~HookBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  void Release() override;
  void SetSavedValues(ValuePtrList saved_values) { saved_values_ = std::move(saved_values); }

 private:
  PrimitivePyPtr prim_;
  ValuePtrList saved_values_;
  abstract::AbstractBasePtr out_abstract_;
};

struct GradientContext;

class GraphBackwardNode : public BackwardNode {
 public:
  explicit GraphBackwardNode(const string &name, FuncGraphPtr func_graph, const VectorRef &args,
                             const VectorRef &added_args, SavedNodePtr saved_output, size_t output_size,
                             std::string cache_key, bool is_control_flow, bool is_jit_graph, bool jit_out_has_dict)
      : BackwardNode(name, output_size),
        func_graph_(std::move(func_graph)),
        args_(args),
        added_args_(added_args),
        saved_output_(std::move(saved_output)),
        cache_key_(std::move(cache_key)),
        graph_call_condition_(is_control_flow, is_jit_graph, jit_out_has_dict, true) {}
  ~GraphBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  // Update nullptr grad.
  ValuePtrList LazeUpdateZeroGradient(const ValuePtrList &dout, FuncBuilder *func_builder, const ValuePtr &output);
  void Release() override;

 private:
  FuncGraphPtr func_graph_;
  VectorRef args_;
  VectorRef added_args_;
  SavedNodePtr saved_output_;
  std::string cache_key_{false};
  GraphCallCondition graph_call_condition_;
};

class LeafNode : public BackwardNode {
 public:
  explicit LeafNode(const string &name, tensor::TensorPtr leaf_tensor, std::vector<int64_t> shape, TypePtr dtype,
                    bool is_parameter = true, bool should_execute = true)
      : BackwardNode(name, UINT64_MAX, 1),
        leaf_tensor_(std::move(leaf_tensor)),
        shape_(std::move(shape)),
        dtype_(std::move(dtype)),
        is_parameter_(is_parameter),
        should_execute_(should_execute) {
    add_output_metadata(leaf_tensor);
  }
  ~LeafNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  bool IsLeaf() override { return true; }
  ValuePtr Zeros(const std::shared_ptr<FuncBuilder> &func_impl);
  bool should_execute() const { return should_execute_; }
  bool is_parameter() const { return is_parameter_; }

 private:
  std::weak_ptr<Tensor> leaf_tensor_;
  std::vector<int64_t> shape_;
  TypePtr dtype_;
  bool is_parameter_;
  bool should_execute_;
};

class GraphRoot : public BackwardNode {
 public:
  explicit GraphRoot(const string &name) : BackwardNode(name) {}
  ~GraphRoot() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override { return grads; }
};

class FakeBackwardNode : public BackwardNode {
 public:
  explicit FakeBackwardNode(const string &name, size_t output_size = 1) : BackwardNode(name, output_size) {}
  ~FakeBackwardNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override {
    MS_LOG(EXCEPTION) << "Illegal primitive " << name() << "'s bprop not defined";
  }
};

class CopySliceNode : public BackwardNode {
 public:
  CopySliceNode(std::string name, BackwardNodePtr inplace_op_func, FuncBuilderPtr emitter, size_t output_size,
                const tensor::TensorPtr &base, const tensor::TensorPtr &output)
      : BackwardNode(std::move(name), output_size),
        inplace_func_(std::move(inplace_op_func)),
        emitter_(std::move(emitter)) {
    base_ = TensorMeta(base->shape(), base->Dtype(), base->stride(), base->storage_offset());
    MS_EXCEPTION_IF_NULL(output->storage_info());
    output_ = TensorMeta(output->shape(), output->Dtype(), output->stride(), output->storage_offset());
    add_output_metadata(base);
  }
  ~CopySliceNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  ValuePtrList CallBackwardImpl(const NodePtr &grad_node);
  void Release() override;

 private:
  BackwardNodePtr inplace_func_;
  FuncBuilderPtr emitter_;
  TensorMeta base_;
  TensorMeta output_;
};

/// Update next edge of inputs and set gradient info to output tensor.
/// \param grad_param
void KPynativeOp(const GradParamPtr &grad_param);

// Reverse connect jit or higher order sub bprop funcgraph
bool KPynativeWithFProp(const GradParamPtr &grad_param);

// Update next edge of inputs and set gradient info to output tensor for custom bprop cell.
void CallCustomBprop(const CustomContext &context);

// Save get and update variable of tensor.
BackwardNodePtr SafeGetGradNodeImpl(const tensor::TensorPtr &tensor);

// When view tensor encounter inplace op, we need use rebase variable to update base tensor gradient edge,
// and update view tensor backward info.
void RebaseVariable(const OpGradInfoPtr &op_grad_info, const BackwardNodePtr &func_node,
                    const tensor::TensorPtr &output_tensor, size_t output_index);

// Update next edges info for backward node.
void UpdateNextEdges(const BackwardNodePtr &grad_node, const ValuePtrList &inputs);

// Update version for view's output
void UpdateVersion(const tensor::TensorPtr &output);

// Build func node for common op
FuncBackwardNodePtr BuildFuncBackwardNode(const PrimitivePtr &prim, const expander::bprop::BpropBuilderFunc &func,
                                          const ValuePtrList &flatten_inputs, const OpGradInfoPtr &op_grad_info,
                                          size_t flatten_output_size);

// Build custom backward node info like custom bprop cell.
BackwardNodePtr BuildCustomBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                        const OpGradInfoPtr &op_grad_info, size_t flatten_output_size);

// Build hook backward node info.
BackwardNodePtr BuildHookBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                      const OpGradInfoPtr &op_grad_info, size_t flatten_output_size);

// Build fake node which does not has definition of backward.
BackwardNodePtr BuildFakeBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                      const OpGradInfoPtr &op_grad_info, size_t flatten_output_size);

// Build jit grad graph node. PyNative mode with jit mode.
BackwardNodePtr BuildGraphBackwardNode(const GradParamPtr &grad_param);

// Build python custom function node info.
void CallCustomPyFunction(const std::shared_ptr<FunctionContext> &context);

// Build cpp custom function node info.
void CallCustomCFunction(const ValuePtrList &flatten_outputs, const TensorPtrSet &input_base_tensors,
                         const TensorPtrSet &dirty_tensors, const TensorPtrSet &non_diff_tensors,
                         const ValuePtrList &inputs, const std::vector<InputType> &input_value_grad_type,
                         const BackwardNodePtr &node);

tensor::TensorPtrList SearchUnusedParameters(const tensor::TensorPtrList &outputs,
                                             const tensor::TensorPtrList &total_params);

struct GradientContext {
  struct CapturedGradient {
    CapturedGradient() = default;
    explicit CapturedGradient(bool need_capture_grad) : need_capture(need_capture_grad) {}
    void SetGradient(const tensor::TensorPtr &tensor_grad) { grad = tensor_grad; }
    void SetNeedCapture(bool need_capture_grad) { need_capture = need_capture_grad; }
    tensor::TensorPtr grad{nullptr};
    bool need_capture{false};
  };
  using CapturedGradientVec = std::vector<CapturedGradient>;
  GradientContext() : requires_backward(false), captured_grad(nullptr) {}
  explicit GradientContext(bool requires_backward, std::unique_ptr<CapturedGradientVec> capture_grad = nullptr)
      : requires_backward(requires_backward), captured_grad(std::move(capture_grad)) {}
  [[nodiscard]] bool ShouldExecute() const { return requires_backward || captured_grad != nullptr; }
  bool requires_backward;
  std::unique_ptr<CapturedGradientVec> captured_grad{nullptr};
};

struct NodeStatus {
  NodeStatus(BackwardNode *backward_node, bool processed) : grad_node(backward_node), is_processed(processed) {}
  BackwardNode *grad_node;
  bool is_processed{false};
};

struct CompareNode {
  bool operator()(const BackwardNodePtr &lhs, const BackwardNodePtr &rhs) const {
    return lhs->seq_id() < rhs->seq_id();
  }
};

class AutoDiff : public AutoDiffInterface {
 public:
  /// Constructor
  /// \param output
  /// \param high_order
  /// \param is_run_recompute
  AutoDiff(const ValuePtr &output, bool keep_graph, bool high_order, bool is_run_recompute);
  DISABLE_COPY_AND_ASSIGN(AutoDiff)
  /// Calculate gradients of leaf nodes from outputs.
  /// \param inputs
  /// \param weights
  /// \param grad_position
  /// \param grad_attr
  /// \param collect_default_weights
  /// \param has_aux
  /// \param sens
  /// \return grads of weights and inputs.
  ValuePtr RunGradFunc(const ValuePtrList &inputs, const tensor::TensorPtrList &weights,
                       const std::vector<size_t> &grad_position, const GradAttr &grad_attr,
                       bool collect_default_weights, bool has_aux, const ValuePtr &sens = nullptr);

  ValuePtr RunBackward(const ValuePtrList &inputs, const ValuePtr &sens, bool accumulate_grad);
  /// Check the given node is in exec grad graph.
  /// \param node
  /// \return true if in grad graph
  bool IsInExecGraph(const BackwardNodePtr &node) const override;
  /// Add the given node to exec grad graph.
  /// \param node
  void AddNodeToExecGraph(const BackwardNodePtr &node) override;
  /// Get engine id.
  /// \param autodiff engine id.
  size_t CurrentAutoDiffEngineId() override;
  /// Add final callback
  /// \param callback
  void AddFinalCallback(std::function<void()> callback);
  /// Run final callback
  void RunFinalCallback() const;
  /// Clear resource of AutoDiff engine.
  void Clear();

 private:
  /// Check grad func input sens shape and type.
  /// \param sens_gradient
  void CheckSensShapeAndType(const ValuePtr &sens_gradient);

  /// Construct grad graph root node.
  /// \param sens_gradient
  void BuildGraphRoot(const ValuePtr &sens_gradient, bool has_aux);

  /// Pruning input grad of grad graph, if input do not need grad.
  /// \param inputs
  /// \param grad_attr
  /// \param grad_position
  void PruningInput(const ValuePtrList &inputs, const GradAttr &grad_attr, const std::vector<size_t> &grad_position);
  /// Pruning input grad of grad graph
  /// \param inputs
  void PruningInput(const ValuePtrList &inputs, bool accumulate_grad);
  /// Pruning grad graph.
  /// \param inputs
  /// \param weights
  /// \param grad_attr
  /// \param grad_position
  void PruningGradGraph(const ValuePtrList &inputs, const std::vector<BackwardNodePtr> &weights,
                        const GradAttr &grad_attr, const std::vector<size_t> &grad_position);

  void PruningGradNode();

  /// Compute in degree of grad node from grad graph,
  /// which used for judging node whether can be execute.
  void ComputeNodeInDegree();
  /// Update dependencies for grad node which accept None gradient, to keep dependencies correct.
  /// \param root
  /// \param input_buffer
  /// \param queue
  /// \param dependencies
  /// \return
  void UpdateDependencies(const BackwardNodePtr &root,
                          const mindspore::HashMap<BackwardNode *, ValuePtrList> &input_buffer,
                          std::priority_queue<BackwardNodePtr, std::vector<BackwardNodePtr>, CompareNode> *queue,
                          std::unordered_map<BackwardNode *, int32_t> *dependencies);
  /// Get grad nodes from weight tensor
  /// \param weights
  /// \param grad_attr
  /// \param collect_default_weights
  /// \return weights node
  std::vector<BackwardNodePtr> GetWeightsNode(const tensor::TensorPtrList &weights, const ValuePtrList &inputs,
                                              const GradAttr &grad_attr, bool collect_default_weights);

  /// Get default weights node when user set weights=None,
  /// we need collect weights which participate in forward function default.
  /// \param weights
  /// \param grad_attr
  /// \param collect_default_weights
  /// \return weights node
  std::vector<BackwardNodePtr> GetDefaultWeightsNode(const BackwardNodePtr &graph_root,
                                                     const std::vector<BackwardNodePtr> &inputs_node);

  /// Execute grad graph to get leaf node grads.
  void BackPropagate();

  /// Check leaf node not participate in backward procedure but has tensor hook,
  /// this is just for fix zeros3 bug.
  /// \param fn
  /// \return grad
  ValuePtr LeafNodeNotInGradButHasTensorHook(const std::shared_ptr<LeafNode> &fn) const;

  /// Ones op for sens.
  /// \param sens
  /// \return values
  ValuePtrList OnsLike(const ValuePtrList &sens);

  /// Get grads from leaf nodes.
  /// \param inputs
  /// \param weights
  /// \param grad_position
  /// \param grad_attr
  /// \return grads
  ValuePtr GetGrads(const ValuePtrList &inputs, const std::vector<BackwardNodePtr> &weights,
                    const std::vector<size_t> &grad_position, const GradAttr &grad_attr);
  /// Get grads of input leaf nodes.
  /// \param inputs
  /// \param grad_all_inputs
  /// \param get_by_position
  /// \param grad_position
  /// \return grads
  ValuePtr GetInputGrads(const ValuePtrList &inputs, bool grad_all_inputs, bool get_by_position,
                         const std::vector<size_t> &grad_position);

  /// Get grad from tensor.
  /// \param val
  /// \param get_grad_from_tensor whether get grad from tensor.
  /// \param run_tensor_hook whether execute hook of leaf node which not in grad graph.
  /// \return grad
  ValuePtr GetTensorGrad(const ValuePtr &val, bool get_grad_from_tensor, bool is_run_backward_api);
  /// Get grad from grad node.
  /// \param grad_node
  /// \return grad
  ValuePtr GetLeafNodeGrad(const BackwardNodePtr &grad_node);
  /// Get grads from weights node.
  /// \param grad_weights
  /// \param weights
  /// \param weight_param_is_tuple
  /// \return
  ValuePtr GetWeightGrads(bool grad_weights, const std::vector<BackwardNodePtr> &weights, bool weight_param_is_tuple);
  inline void CaptureTensorGrads(const ValuePtrList &gradient_in,
                                 const std::unordered_map<BackwardNode *, GradientContext>::iterator &ctx_iter);
  std::unordered_map<BackwardNode *, GradientContext> gradient_contexts_;
  std::unordered_map<BackwardNode *, int32_t> node_in_degree_;
  std::unordered_set<BackwardNode *> node_used_in_graph_;
  std::vector<std::function<void()>> final_callbacks_{};
  ValuePtrList flatten_sens_out_{};
  ValuePtr output_{nullptr};
  ValuePtrList root_gradients_{};
  BackwardNodePtr graph_root_{nullptr};
  std::shared_ptr<FuncBuilder> func_impl_;
  device::DeviceType device_target_;
  size_t engine_id_;
  bool keep_graph_{false};
  bool high_order_{false};
  bool is_run_recompute_{false};
};
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_FUNC_GRAD_H_
