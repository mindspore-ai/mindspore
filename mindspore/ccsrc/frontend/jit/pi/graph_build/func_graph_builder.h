/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_BUILD_FUNC_GRAPH_BUILDER_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_BUILD_FUNC_GRAPH_BUILDER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ir/value.h"
#include "primitive/sequence_ops.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "frontend/jit/ps/parse/parse.h"
#include "frontend/jit/pi/graph_capture/abstract_wrapper.h"

namespace mindspore {
namespace pijit {
class FuncGraphBuilder;
using FuncGraphBuilderPtr = std::shared_ptr<FuncGraphBuilder>;
class AbstractWrapper;
using AbstractWrapperPtr = std::shared_ptr<AbstractWrapper>;
using CallableGraph = std::function<PyObject *(PyObject *, PyObject *)>;

class FuncGraphBuilder {
 public:
  explicit FuncGraphBuilder(bool is_top = false);
  virtual ~FuncGraphBuilder() { key_to_node_.clear(); }

  /// \brief Add single arg input to top graph.
  ///
  /// \param[in] object Arg python object input for top graph.
  ///
  /// \return The AbstractWrapperPtr for top arg input.
  AbstractWrapperPtr AddTopGraphArgInput(const py::object &object);

  /// \brief Add vargs input to top graph.
  ///
  /// \param[in] object Vargs python object input for top graph.
  ///
  /// \return The AbstractWrapperPtr for top vargs input.
  AbstractWrapperPtr AddTopGraphVargsInputs(const py::object &vargs);

  /// \brief Add kwargs input to top graph.
  ///
  /// \param[in] object Kwargs python object input for top graph.
  ///
  /// \return The AbstractWrapperPtr for top kwargs input.
  AbstractWrapperPtr AddTopGraphKwargsInputs(const py::object &vargs);

  /// \brief Add an input parameter to the subgraph.
  ///
  /// \param[in] abstract_wrapper The key to find node in function graph builder.
  ///
  /// \return The AbstractWrapperPtr for subgraph input.
  AbstractWrapperPtr AddSubGraphInput(const AbstractWrapperPtr abstract_wrapper);

  /// \brief Add a cnode to the graph.
  ///
  /// \param[in] callable_obj The callable python object.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The abstract wrapper  of the infer result.
  AbstractWrapperPtr AddNode(const py::object &callable_obj, const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add a cnode to the graph.
  ///
  /// \param[in] callable_value The callable value.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The abstract wrapper of the infer result.
  AbstractWrapperPtr AddNode(const ValuePtr &callable_value, const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add a cnode to the graph with abstract, no need to evaluate.
  ///
  /// \param[in] inputs The inputs of new node.
  /// \param[in] inputs_obj The abstract of new node.
  ///
  /// \return The abstract wrapper of the new node.
  AbstractWrapperPtr AddNodeWithAbstract(const AnfNodePtrList &inputs, const AbstractBasePtr &abstract);

  /// \brief Add a cnode to the graph with graph is parsed in ast and byte code is CallFunctionEx.
  ///
  /// \param[in] callable_obj The callable python object.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The abstract wrapper of the infer result.
  AbstractWrapperPtr AddNodeCallFunctionEx(const py::object &callable_obj,
                                           const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add a cnode to the graph with graph is parsed in ast and byte code is CallFunctionEx.
  ///
  /// \param[in] callable_value The callable value.
  /// \param[in] inputs_obj The input python objects.
  ///
  /// \return The abstract wrapper of the infer result.
  AbstractWrapperPtr AddNodeCallFunctionEx(const ValuePtr &callable_value,
                                           const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add a cnode to the graph with graph is parsed in ast and byte code is CallFunctionKw.
  ///
  /// \param[in] callable_obj The callable python object.
  /// \param[in] inputs_obj The input python objects.
  /// \param[in] kw_names The input python objects.
  ///
  /// \return The abstract wrapper of the infer result.
  AbstractWrapperPtr AddNodeCallFunctionKw(const py::object &callable_obj,
                                           const AbstractWrapperPtrList &inputs_abstract_wrapper,
                                           const py::object &kw_names);

  /// \brief Add a cnode to the graph with graph is parsed in ast and byte code is CallFunctionKw.
  ///
  /// \param[in] callable_value The callable value.
  /// \param[in] inputs_obj The input python objects.
  /// \param[in] kw_names The input python objects.
  ///
  /// \return The abstract wrapper of the infer result.
  AbstractWrapperPtr AddNodeCallFunctionKw(const ValuePtr &callable_value,
                                           const AbstractWrapperPtrList &inputs_abstract_wrapper,
                                           const py::object &kw_names);

  /// \brief Add a python object to graph.
  ///
  /// \param[in] object The python object add to graph.
  ///
  /// \return Indicate whether the python object add to graph successfully.
  AbstractWrapperPtr AddAttrPythonObject(const py::object &object);

  /// \brief Add a binary operation cnode to the graph.
  ///
  /// \param[in] opcode The binary operation code.
  /// \param[in] inputs_abstract_wrapper The abstract wrapper for inputs.
  ///
  /// \return The python object of the infer result.
  AbstractWrapperPtr AddMultiNode(const std::string &name, const AbstractWrapperPtrList &inputs_abstract_wrapper);

  /// \brief Add an output node to the graph.
  ///
  /// \param[in] output_obj The output python object.
  /// \param[in] is_top_graph Indicate whether the graph to add output is top graph.
  ///
  /// \return Return true if the output object can be used as the output of the graph.
  bool AddOutput(const AbstractWrapperPtr &abstract_wrapper, bool is_top_graph = true);

  /// \brief Clear all output node of the graph.
  void ClearOutputNodes() { output_nodes_.clear(); }

  /// \brief Get number of output_nodes_.
  size_t GetOutputSize() const { return output_nodes_.size(); }

  /// \brief Get the callable python primitive or function.
  ///
  /// \param[in] obj The method of a python object.
  ///
  /// \return Return the corresponding primitive of function of the func.
  static py::object ConvertMethod(const py::object &obj);
  static py::object ConvertMethod(const std::string &class_name, const std::string &method_name);

  /// \brief Get the callable python primitive, meta_func_graph or function.
  ///
  /// \param[in] obj The python object of a function.
  ///
  /// \return Return the corresponding primitive of function of the func.
  static py::object ConvertFunction(const py::object &obj);

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
  /// \param[in] force Allows getting the graph when the outputs have not yet been added.
  ///
  /// \return The graph constructed.
  FuncGraphPtr graph(bool force = false);

  /// \brief Clear abstract for nodes.
  void ClearNodeAbstract();

  /// \brief Set the name of the func_graph.
  ///
  /// \param[in] name The func_graph name to set.
  void SetGraphName(const std::string &name);

  /// \brief Get manager for associated graph.
  ///
  /// \return The manager for function graph.
  FuncGraphManagerPtr manager() const { return mng_; }

  /// \brief Set manager for associated graph.
  ///
  /// \param[in] mng The manager to set.
  void set_manager(const FuncGraphManagerPtr &mng) {
    mng_ = mng;
    graph_->set_manager(mng_);
  }

  /// \brief Add single prev builder.
  ///
  /// \param[in] builder The prev builder to add.
  void AddPrevBuilder(const FuncGraphBuilderPtr &builder);

  /// \brief Get all prev builders.
  ///
  /// \return All pref builders for current builder.
  const std::vector<FuncGraphBuilder *> &prev_builders() const { return prev_builders_; }

  /// \brief Update value for key in key_to_node_ with node.
  ///
  /// \param[in] key The key to update.
  /// \param[in] node The new value for key.
  void UpdateNodesMap(const AbstractWrapperPtr &key, const AnfNodePtr &node);

  /// \brief Get origin input number for top graph.
  ///
  /// \return Origin input number for top graph.
  size_t origin_top_input_num() const { return origin_top_input_num_; }

  /// \brief Find node for wrapper, only in local builder scope.
  ///
  /// \param[in] abstract_wrapper The wrapper key to find node.
  ///
  /// \return The result node.
  AnfNodePtr ReadLocalVariable(const AbstractWrapperPtr &abstract_wrapper);

  /// \brief Find node for wrapper in local and all prev builder.
  ///
  /// \param[in] abstract_wrapper The wrapper key to find node.
  ///
  /// \return The result node.
  AnfNodePtr FindNodeByWrapper(const AbstractWrapperPtr &abstract_wrapper);

  /// \brief Find node for wrapper in local and all prev builder. If not found and the wrapper is
  ///        constant, build a value node for wrapper.
  ///
  /// \param[in] abstract_wrapper The wrapper key to find or build node.
  ///
  /// \return The result node.
  AnfNodePtr FindOrCreateNodeByWrapper(const AbstractWrapperPtr &abstract_wrapper);

  /// \brief Add a constant node for python object.
  ///
  /// \param[in] obj The python object to build node.
  ///
  /// \return The wrapper for corresponding node.
  AbstractWrapperPtr AddLocalVariable(const py::object &obj);

  /// \brief Add a custom node to the graph.
  ///
  /// \param[in] wrapper The abstract wrapper corresponding to the node.
  /// \param[in] node The node will be added.
  ///
  /// \note Nodes created during the conversion of Dict nodes need to be added to the graph using this method.
  void AddLocalVariableNode(const AbstractWrapperPtr &wrapper, const AnfNodePtr &node);

  void EraseCandidateIsolatedNode(const AnfNodePtr &node);

  AbstractWrapperPtr AddAttributeInput(const py::object &object);

  /// \brief Save the phase and the callable of the func_graph.
  ///
  /// \param[in] result The phase and the callable.
  void SetCompileResult(const std::pair<std::string, CallableGraph> &result) { compile_result_ = result; }
  /// \brief Get the phase and the callable of the func_graph.
  ///
  /// \return The phase and the callable.
  const std::pair<std::string, CallableGraph> &GetCompileResult() const { return compile_result_; }

 private:
  AnfNodePtr ConvertObjToNode(const py::object &input_obj);
  AnfNodePtr ConvertParameterTupleToNode(const py::object &input_obj);
  AnfNodePtr ConvertPyTupleListToNode(const py::object &obj);
  AnfNodePtr ConvertPyDictToNode(const py::dict &dict);

  AbstractWrapperPtr AddNodeWithAbstract(const ValuePtr &value, const AbstractWrapperPtrList &inputs_abstract_wrapper,
                                         const AbstractBasePtr &abstract);

  bool GetInputNodesAndAbstracts(const ValuePtr &callable_value, const AbstractWrapperPtrList &inputs_abstract_wrapper,
                                 AnfNodePtrList *input_node_list, AbstractBasePtrList *input_abs_list);

  CNodePtr DoPrimitiveInferAndCheck(const PrimitivePtr &primitive, const AnfNodePtrList &input_node_list,
                                    const AbstractBasePtrList &args_abs_list);
  CNodePtr AddPrimitiveCNode(const PrimitivePtr &primitive, const AnfNodePtrList &input_node_list,
                             const AbstractBasePtrList &args_abs_list);

  AbstractWrapperPtr TryToAddNode(const ValuePtr &callable_value,
                                  const AbstractWrapperPtrList &inputs_abstract_wrapper);

  void MarkNodeIsolated(const AnfNodePtr &node, bool force);

  AnfNodePtr GenerateOutputNode();

  AnfNodePtr AttachIsolatedNode(const AnfNodePtr &node) const;

  bool has_set_output_{false};
  size_t origin_top_input_num_{0};

  FuncGraphPtr graph_{nullptr};
  FuncGraphManagerPtr mng_{nullptr};

  HashMap<AbstractWrapperPtr, AnfNodePtr> key_to_node_;
  std::vector<AnfNodePtr> output_nodes_;

  // Store all isolated nodes for graph which should be appended to the output of graph.
  std::vector<AnfNodePtr> isolated_nodes_;

  // Store all previous builders for subgraph call and control flow.
  std::vector<FuncGraphBuilder *> prev_builders_;

  std::pair<std::string, CallableGraph> compile_result_;
};
}  // namespace pijit
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_BUILD_FUNC_GRAPH_BUILDER_H_
