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

#ifndef MINDSPORE_JIT_GRAPH_FUNCTION_NODE_H_
#define MINDSPORE_JIT_GRAPH_FUNCTION_NODE_H_

#include <memory>
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/pi/auto_grad/edge.h"
#include "pipeline/jit/pi/auto_grad/function_context.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace jit {
namespace grad {
namespace py = pybind11;
/// \brief FunctionNode is a class, which represent a function call.
class FunctionNode : public FunctionContext {
 public:
  /// \brief The constructor of FunctionNode.
  ///
  /// \param[in] fn The called function.
  ///
  /// \return The instance of FunctionNode.
  explicit FunctionNode(const py::object &fn) : FunctionContext(), fn_(fn) {}

  /// \brief Destructor.
  virtual ~FunctionNode() = default;

  static FunctionNodePtr GetInstance(const py::object &fn) { return std::make_shared<FunctionNode>(fn); }

  /// \brief Add a input at the end of the input list.
  ///
  /// \param[in] input The input.
  void AddInput(const py::object &input);

  /// \brief Set the output of the function node.
  ///
  /// \param[in] output The output.
  void SetOutput(const py::object &output);

  /// \brief Get the called function.
  ///
  /// \return The called function.
  const py::object &GetFunction() const { return fn_; }

  /// \brief Get the bprop function graph.
  ///
  /// \return The bprop function graph.
  const FuncGraphPtr &GetBpropFunction() const { return grad_fn_; }

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
  void AddNextEdge(const FunctionNodePtr &node, size_t index) { edges_.push_back(std::make_shared<Edge>(node, index)); }

  /// \brief Get the python object grad.
  ///
  /// \return The python object grad.
  const py::object GetPyObjectGrad() const { return ValueToPyData(GetGrad()); }

  /// \brief Generate the bprop function.
  void GenBropFunction();

  /// \brief Generate the grad value of function.
  void Apply(const py::object &grad = py::none());

 private:
  /// \brief The called function.
  py::object fn_;
  /// \brief The bprop function.
  FuncGraphPtr grad_fn_;
  /// \brief The called functions in the previous/next step.
  EdgePtrList edges_;
};

using FunctionNodePtr = std::shared_ptr<FunctionNode>;

inline void RegFunctionNodes(const py::module *m) {
  (void)py::class_<FunctionNode, FunctionNodePtr>(*m, "FunctionNode_")
    .def_static("get_instance", &FunctionNode::GetInstance, py::arg("fn"), "Function node instance.")
    .def("set_inputs", &FunctionNode::SetInputs, py::arg("inputs"), "Set the inputs of function node.")
    .def("add_input", &FunctionNode::AddInput, py::arg("input"), "Add a input at the end of the input list.")
    .def("set_output", &FunctionNode::SetOutput, py::arg("output"), "Set the output of the function node.")
    .def("add_next_edge", &FunctionNode::AddNextEdge, py::arg("node"), py::arg("index"), "Add a edge.")
    .def("apply", &FunctionNode::Apply, py::arg("grad"), "Calculate the gradient of the function node.")
    .def("get_grad", &FunctionNode::GetPyObjectGrad, "Get the gradient of the function node.");
}
}  // namespace grad
}  // namespace jit
}  // namespace mindspore
#endif  // MINDSPORE_JIT_GRAPH_FUNCTION_NODE_H_
