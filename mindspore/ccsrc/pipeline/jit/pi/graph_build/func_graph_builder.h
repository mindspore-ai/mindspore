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

#ifndef MINDSPORE_PI_JIT_GRAPH_BUILD_FUNC_GRAPH_BUILDER_H_
#define MINDSPORE_PI_JIT_GRAPH_BUILD_FUNC_GRAPH_BUILDER_H_

#include <vector>
#include <memory>
#include <string>
#include "ir/value.h"
#include "ops/sequence_ops.h"
#include "pipeline/jit/ps/parse/parse_base.h"

namespace mindspore {
class FuncGraphBuilder {
 public:
  FuncGraphBuilder() : graph_(std::make_shared<FuncGraph>()) {}
  virtual ~FuncGraphBuilder() { py_obj_to_node_.clear(); }

  /// \brief Add an input parameter to the graph.
  ///
  /// \param[in] obj The input python object.
  ///
  /// \return If the input is a tensor, return a fake tensor python object, else return the origin python object.
  py::object AddInput(const py::object &obj);

  /// \brief Add a cnode to the graph.
  ///
  /// \param[in] callable_obj The callable python object.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The python object of the infer result.
  py::object AddNode(const py::object &callable_obj, const std::vector<py::object> &inputs_obj);

  /// \brief Add a cnode to the graph.
  ///
  /// \param[in] callable_value The callable value.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The python object of the infer result.
  py::object AddNode(const ValuePtr &callable_value, const std::vector<py::object> &inputs_obj);

  /// \brief Add a binary operation cnode to the graph.
  ///
  /// \param[in] opcode The binary operation code.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The python object of the infer result.
  py::object AddMultiNode(const std::string &opcode, const std::vector<py::object> &inputs_obj);

  /// \brief Add an output node to the graph.
  ///
  /// \param[in] output_obj The output python object.
  ///
  /// \return Return true if the output object can be used as the output of the graph.
  bool AddOutput(const py::object &output_obj);

  /// \brief Remove an output node of the graph.
  ///
  /// \param[in] output_obj The output python object.
  void RemoveOutput(const py::object &output_obj);

  /// \brief Get the callable python primitive or function.
  ///
  /// \param[in] obj The method of a python object.
  ///
  /// \return Return the corresponding primitive of function of the func.
  static py::object ConvertMethod(const py::object &obj);

  /// \brief Get the callable python primitive, meta_func_graph or function.
  ///
  /// \param[in] obj The python object of a function.
  ///
  /// \return Return the corresponding primitive of function of the func.
  static py::object ConvertFunction(const py::object &obj);

  /// \brief Check if the python object can be converted to a cnode directly.
  ///
  /// \param[in] obj A python object.
  ///
  /// \return Return true if the python object can be converted to a cnode directly.
  static bool CheckCallable(const py::object &obj);

  /// \brief Check if the python object is a function which can be constantly folded.
  ///
  /// \param[in] obj A python object.
  ///
  /// \return Return true if the python object is a function which can be constantly folded.
  static bool CanConstantFoldFunc(const py::object &obj);

  /// \brief Set the final outputs and get the graph.
  ///
  /// \return The graph constructed.
  FuncGraphPtr graph();

  /// \brief Set the name of the func_graph.
  ///
  /// \param[in] name The func_graph name to set.
  void SetGraphName(const std::string &name);

  static ValuePtr ConvertPyObjToValue(const py::object &obj);

  static AbstractBasePtr EvalValue(const ValuePtr &value, const AbstractBasePtrList &inputs_abs_list);

  static py::object ConvertToPyObj(const AbstractBasePtr &abs);

 private:
  static bool CheckCallable(const ValuePtr &value, const AbstractBasePtr &abs);

  static bool CheckGraphOutput(const AbstractBasePtr &abs);

  py::object AddFgCallNode(const FuncGraphPtr &fg, const std::vector<py::object> &inputs_obj);

  bool GetInputNodesAndAbstracts(const ValuePtr &callable_value, const std::vector<py::object> &inputs_obj,
                                 std::vector<AnfNodePtr> *input_node_list,
                                 std::vector<AbstractBasePtr> *input_abs_list);

  static AbstractBasePtr DoInferAndCheck(const ValuePtr &callable_value,
                                         const std::vector<AbstractBasePtr> &input_abs_list);

  py::object TryToAddNode(const ValuePtr &callable_value, const std::vector<py::object> &inputs_obj);

  FuncGraphPtr graph_{nullptr};
  bool has_set_output_{false};
  HashMap<PyObject *, AnfNodePtr> py_obj_to_node_;
  std::vector<AnfNodePtr> output_nodes_;
};
using FuncGraphBuilderPtr = std::shared_ptr<FuncGraphBuilder>;
}  // namespace mindspore
#endif  // MINDSPORE_PI_JIT_GRAPH_BUILD_FUNC_GRAPH_BUILDER_H_
