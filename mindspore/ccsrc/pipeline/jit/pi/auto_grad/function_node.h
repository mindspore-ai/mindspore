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

#ifndef MINDSPORE_PI_JIT_FUNCTION_NODE_H_
#define MINDSPORE_PI_JIT_FUNCTION_NODE_H_

#include <atomic>
#include <memory>
#include <set>
#include <string>
#include "pipeline/jit/pi/auto_grad/backward_function.h"
#include "pipeline/jit/pi/auto_grad/edge.h"
#include "pipeline/jit/pi/auto_grad/function_context.h"
#include "pipeline/jit/pi/auto_grad/native_backward_function.h"
#include "pipeline/pynative/pynative_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace pijit {
namespace grad {
namespace py = pybind11;
using Convert = pynative::PyNativeAlgo::DataConvert;

/// \brief FunctionNode is a class, which represent a way to calculate the gradient.
class FunctionNode : public FunctionContext {
 public:
  /// \brief The constructor of FunctionNode.
  ///
  /// \param[in] tensor The tensor that is asked to calculate the gradient.
  ///
  /// \return The instance of FunctionNode.
  explicit FunctionNode(const py::object &tensor) : FunctionContext(Convert::PyObjToValue(tensor)), tensor_(tensor) {}

  /// \brief The constructor of FunctionNode.
  ///
  /// \param[in] tensor The tensor that is asked to calculate the gradient.
  /// \param[in] prim The calculation that the tensor as input.
  /// \param[in] out The output of the calculation that the tensor as input.
  ///
  /// \return The instance of FunctionNode.
  explicit FunctionNode(const py::object &tensor, const py::object &prim, const py::object &out)
      : FunctionContext(Convert::PyObjToValue(prim), Convert::PyObjToValue(out)),
        tensor_(tensor),
        backward_func_(NativeBackwardFunc::GetInstance(Convert::PyObjToValue(prim)->cast<PrimitivePtr>())) {}

  /// \brief Destructor.
  virtual ~FunctionNode() = default;

  /// \brief Release all resource.
  void CleanResource();

  /// \brief Determine whether the python object has attribute `requires_grad`.
  ///
  /// \param[in] obj The python object.
  ///
  /// \return The result of the python object's attribute `requires_grad`.
  static bool HasAttrReqGrad(const py::handle &obj) { return py::hasattr(obj, "requires_grad"); }

  /// \brief Determine whether the python object has attribute `requires_grad`, and the value is True.
  ///
  /// \param[in] obj The python object.
  ///
  /// \return The result of the python object's attribute `requires_grad`.
  static bool IsRequiresGradient(const py::handle &obj);

  /// \brief Determine whether the python object has attribute `grad_fn`.
  ///
  /// \param[in] obj The python object.
  ///
  /// \return The result whether the python object has attribute `grad_fn`.
  static bool HasGradFunc(const py::handle &obj);

  /// \brief Create a new function node.
  ///
  /// \param[in] tensor The tensor mounted by function node.
  /// \param[in] prim The forward execution function.
  /// \param[in] out The output of the forward execution function.
  /// \param[in] inputs The input of the forward execution function.
  ///
  /// \return The instance of function node.
  static FunctionNodePtr CreateFunctionNode(const py::object &tensor, const py::object &prim, const py::object &out,
                                            const py::list &inputs);

  /// \brief The static method to record the executed primitive during forward execution.
  ///
  /// \param[in] prim The executed primitive.
  /// \param[in] out The output of the executed primitive.
  /// \param[in] inputs The inputs of the executed primitive.
  static void RecordPrimitive(const py::object &prim, const py::object &out, const py::list &inputs);

  /// \brief Get the tensor that is asked to calculate the gradient.
  ///
  /// \return The tensor that is asked to calculate the gradient.
  const py::object &GetTensor() const { return tensor_; }

  /// \brief Set the inputs of the function node.
  ///
  /// \param[in] inputs The inputs.
  void SetInputs(const py::list &inputs);

  /// \brief Get the bprop function graph.
  ///
  /// \return The bprop function graph.
  const FuncGraphPtr &GetBpropFunction() const { return grad_fn_; }

  /// \brief Generate the bprop function.
  void GenerateBropFunction();

  /// \brief Start gradient calculation.
  void ApplyNative();

  /// \brief Get the called functions in the previous/next step.
  ///
  /// \return The called functions in the previous/next step.
  const EdgePtrList &GetNextEdges() const { return edges_; }

  /// \brief Set the called functions in the previous/next step.
  ///
  /// \param[in] edges The called functions.
  void SetNextEdges(const EdgePtrList &edges) { edges_ = edges; }

  /// \brief Add a called function in the previous/next step.
  ///
  /// \param[in] node The called function.
  /// \param[in] index The index of the input.
  void AddNextEdge(const FunctionNodePtr &node, size_t index) {
    edges_.push_back(std::make_shared<Edge>(node, index));
    node->dependences_.insert(shared_from_base<FunctionNode>());
  }

  /// \brief Synchronize gradient value ​​to python object.
  void SyncGradToPyObject();

  /// \brief Generate the grad value of function.
  ///
  /// \param[in] grad The default gradient value of the function node.
  ///
  /// \note This function node must be the tensor who call backward from python.
  void Apply(const py::object &grad);

  /// \brief Generate the description of the tree nodes.
  std::string ToString() const;

 private:
  /// \brief Generate the grad value of function.
  ///
  /// \param[in] dout The gradient of the output.
  void ApplyInner(const ValuePtr &dout);

  /// \brief Calculate the gradient of the next layer function node.
  ///
  /// \param[in] grad_values The the gradient values.
  void ApplyEdges(const ValuePtrList &grad_values);

  /// \brief Update data dependencies.
  void UpdateDependence();

  /// \brief Notify the function node that the gradient data is ready.
  ///
  /// \param[in] node The function node been notified.
  /// \param[in] dout The gradient data.
  void Notify(const FunctionNodePtr &node, const ValuePtr &dout);

  /// \brief Accumulate the delta of the gradient.
  ///
  /// \param[in] dout The delta of the gradient.
  /// \param[in] index The index of the gradient.
  void AccumulateGradient(const ValuePtr &dout, size_t index);

  /// \brief Determine whether the current function node can start gradient calculation
  bool IsReady() const { return depend_cnt_.load() == dependences_.size(); }

  /// \brief Dump the function node and its children.
  ///
  /// \param[in] ss The string stream.
  /// \param[in] prefix The prefix string for this node.
  void Dump(std::stringstream &ss, const std::string &prefix) const;

  /// \brief The called function.
  py::object tensor_;
  /// \brief the function used to calculate the gradient.
  BackwardFuncPtr backward_func_;
  /// \brief The bprop function.
  FuncGraphPtr grad_fn_;
  /// \brief The accumulate function.
  FuncGraphPtr acc_fn_;
  /// \brief The called functions in the previous/next step.
  EdgePtrList edges_;
  /// \brief The mutex for accumulate the delta of the gradient.
  std::mutex mutex_;
  /// \brief Used to locate the position of the tensor in multiple outputs.
  size_t index_{0};
  /// \brief Mark whether the current node is used in the reverse calculation.
  bool is_in_reverse_chain_{false};
  /// \brief Dependency data of the current node in gradient calculation.
  std::set<FunctionNodePtr> dependences_;
  /// \brief The dependency status.
  std::atomic<size_t> depend_cnt_{0};
};

using FunctionNodePtr = std::shared_ptr<FunctionNode>;
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
#endif  // MINDSPORE_PI_JIT_FUNCTION_NODE_H_
