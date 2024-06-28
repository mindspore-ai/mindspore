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
#include "pipeline/jit/ps/parse/parse.h"

namespace mindspore {
class FuncGraphBuilder;
using FuncGraphBuilderPtr = std::shared_ptr<FuncGraphBuilder>;

class FuncGraphBuilder {
 public:
  explicit FuncGraphBuilder(bool is_top = false) : graph_(std::make_shared<FuncGraph>()) {
    if (is_top) {
      parse::Parser::UpdateTopFuncGraph(graph_);
      mng_ = Manage(graph_, true);
      graph_->set_manager(mng_);
    }
  }
  virtual ~FuncGraphBuilder() { py_obj_to_node_.clear(); }

  /// \brief Add an input parameter to the graph.
  ///
  /// \param[in] obj The input python object.
  ///
  /// \return If the input is a tensor, return a fake tensor python object, else return the origin python object.
  py::object AddSubGraphInput(const py::object &obj);

  /// \brief Add an input parameter to the top graph.
  ///
  /// \param[in] packed_inputs The input python object for top graph.
  ///
  /// \return True if add top graph success, otherwise false.
  bool AddTopGraphInputs(std::vector<py::object> packed_inputs);

  FuncGraphManagerPtr manager() const { return mng_; }

  void set_manager(const FuncGraphManagerPtr &mng) { mng_ = mng; }

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

  /// \brief Add a python object to graph.
  ///
  /// \param[in] object The python object add to graph.
  ///
  /// \return Indicate whether the python object add to graph successfully.
  bool AddAttrPythonObject(const py::object &object);

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
  /// \param[in] is_top_graph Indicate whether the graph to add output is top graph.
  ///
  /// \return Return true if the output object can be used as the output of the graph.
  bool AddOutput(const py::object &output_obj, bool is_top_graph = true);

  /// \brief Remove an output node of the graph.
  ///
  /// \param[in] output_obj The output python object.
  void RemoveOutput(const py::object &output_obj);

  /// \brief Clear all output node of the graph.
  void ClearOutputNodes() { output_nodes_.clear(); }

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

  /// \brief Check if the python object is valid as the callable object in graph.
  ///
  /// \param[in] obj A python object.
  ///
  /// \return Return true if the python object is valid as the callable object in graph.
  static bool ValidateCallableObject(const py::object &obj);

  /// \brief Set the final outputs and get the graph.
  ///
  /// \return The graph constructed.
  FuncGraphPtr graph();

  /// \brief Clear abstract for nodes.
  void ClearNodeAbstract();

  /// \brief Set the name of the func_graph.
  ///
  /// \param[in] name The func_graph name to set.
  void SetGraphName(const std::string &name);

  static ValuePtr ConvertPyObjToValue(const py::object &obj);

  static AbstractBasePtr EvalValue(const ValuePtr &value, const AbstractBasePtrList &inputs_abs_list);

  using PyTensorConverter = std::function<py::object(const py::object &)>;
  static py::object ConvertToPyObj(const AbstractBasePtr &abs);
  static py::object ConvertToPyObj(const AbstractBasePtr &abs, const PyTensorConverter &tensor_convert_func);

  void AddPrevBuilder(const FuncGraphBuilderPtr &builder);

  const std::vector<FuncGraphBuilder *> &prev_builders() const { return prev_builders_; }

  AnfNodePtr ReadLocalVariable(const py::object &obj);

  bool AddLocalVariable(const py::object &obj);

  py::object BuildGradNetNode(const ValuePtr &callable_value, const py::object &callable_obj,
                              const std::vector<py::object> &inputs_obj);

  py::object BuildGradNode(const py::str &key, const std::vector<py::object> &inputs, bool need_unpack);

 private:
  static bool CheckCallable(const ValuePtr &value, const AbstractBasePtr &abs);

  static bool CheckGraphOutput(const AbstractBasePtr &abs);

  AnfNodePtr GetNodeByObject(const py::object &obj, bool generate_value_node = true);

  AnfNodePtr ConvertObjToNode(const py::object &input_obj);

  AnfNodePtr ConvertParameterTupleToNode(const py::object &input_obj);

  py::object AddFgCallNode(const FuncGraphPtr &fg, const std::vector<py::object> &inputs_obj);

  bool GetInputNodesAndAbstracts(const ValuePtr &callable_value, const std::vector<py::object> &inputs_obj,
                                 std::vector<AnfNodePtr> *input_node_list,
                                 std::vector<AbstractBasePtr> *input_abs_list);

  static AbstractBasePtr DoInferAndCheck(const ValuePtr &callable_value,
                                         const std::vector<AbstractBasePtr> &input_abs_list);

  CNodePtr DoPrimitiveInferAndCheck(const PrimitivePtr &primitive, const AnfNodePtrList &input_node_list,
                                    const AbstractBasePtrList &args_abs_list);
  CNodePtr AddPrimitiveCNode(const PrimitivePtr &primitive, const AnfNodePtrList &input_node_list,
                             const AbstractBasePtrList &args_abs_list);

  static AbstractBasePtr GetAbstractOf(const AnfNodePtr &node);

  py::object TryToAddNode(const ValuePtr &callable_value, const std::vector<py::object> &inputs_obj);

  py::object ConvertToPyTensorOrParameter(const py::object &cpp_tensor);

  static bool CheckInvalidCellListDictMethod(const py::object &obj);

  bool AddTopGraphArgsInputs(const py::object &object);

  bool AddTopGraphVargsInputs(const py::object &vargs);

  bool AddTopGraphKwargsInputs(const py::object &vargs);

  py::object HandleGrad(const py::str &key, const std::vector<py::object> &inputs, bool need_unpack);

  FuncGraphPtr graph_{nullptr};
  bool has_set_output_{false};
  HashMap<PyObject *, AnfNodePtr> py_obj_to_node_;
  std::vector<AnfNodePtr> output_nodes_;

  // Store all previous builders for subgraph call and control flow.
  std::vector<FuncGraphBuilder *> prev_builders_;

  FuncGraphManagerPtr mng_;
};
}  // namespace mindspore
#endif  // MINDSPORE_PI_JIT_GRAPH_BUILD_FUNC_GRAPH_BUILDER_H_
