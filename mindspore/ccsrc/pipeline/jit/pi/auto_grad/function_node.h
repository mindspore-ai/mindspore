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

#include <memory>
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/pi/auto_grad/edge.h"
#include "pipeline/jit/pi/auto_grad/function_context.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pybind11/stl.h"
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
  explicit FunctionNode(const py::object &tensor)
      : FunctionContext(
          Convert::PyObjToValue(tensor),
          TensorConstructUtils::CreateZerosTensor(GetStubTensorInfo(tensor).second, GetStubTensorInfo(tensor).first)),
        tensor_(tensor) {}

  /// \brief The constructor of FunctionNode.
  ///
  /// \param[in] tensor The tensor that is asked to calculate the gradient.
  ///
  /// \return The instance of FunctionNode.
  explicit FunctionNode(const py::object &tensor, const py::object &prim, const py::object &out)
      : FunctionContext(
          Convert::PyObjToValue(prim), Convert::PyObjToValue(out),
          TensorConstructUtils::CreateZerosTensor(GetStubTensorInfo(tensor).second, GetStubTensorInfo(tensor).first)),
        tensor_(tensor) {}

  /// \brief Destructor.
  virtual ~FunctionNode() = default;

  /// \brief The static method to record the executed primitive during forward execution.
  ///
  /// \param[in] prim The executed primitive.
  /// \param[in] out The output of the executed primitive.
  /// \param[in] inputs The inputs of the executed primitive.
  static void RecordPrimitive(const py::object &prim, const py::object &out, const py::list &inputs);

  // \brief The method to init field of function node.
  ///
  /// \param[in] prim The executed primitive.
  /// \param[in] inputs The inputs of the executed primitive.
  void InitDataField(const py::object &prim, const py::list &inputs);

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
  void AddNextEdge(const FunctionNodePtr &node) { edges_.push_back(std::make_shared<Edge>(node)); }

  /// \brief Save the grad value to python object.
  ///
  /// \param[in] grad The grad value.
  void SaveGradToPyObject(const py::object &grad);

  /// \brief Generate the grad value of function.
  ///
  /// \param[in] grad The default gradient value of the function node.
  ///
  /// \note This function node must be the tensor who call backward from python.
  void Apply(const py::object &grad);

 private:
  /// \brief Generate the grad value of function.
  ///
  /// \param[in] dout The gradient of the output.
  void ApplyInner(const ValuePtr &dout);

  /// \brief The called function.
  py::object tensor_;
  /// \brief The bprop function.
  FuncGraphPtr grad_fn_;
  /// \brief The accumulate function.
  FuncGraphPtr acc_fn_;
  /// \brief The called functions in the previous/next step.
  EdgePtrList edges_;
  /// \brief The index.
  size_t index_{0};
};

using FunctionNodePtr = std::shared_ptr<FunctionNode>;
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
#endif  // MINDSPORE_PI_JIT_FUNCTION_NODE_H_
