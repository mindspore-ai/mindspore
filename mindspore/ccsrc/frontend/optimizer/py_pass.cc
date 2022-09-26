/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "frontend/optimizer/py_pass.h"
#include <deque>
#include <vector>

#include "utils/hash_set.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "pybind_api/ir/primitive_py.h"
#include "ir/scalar.h"
#include "ir/graph_utils.h"
#include "pipeline/jit/parse/parse_base.h"
#include "pipeline/jit/resource.h"
#include "frontend/optimizer/py_pass_manager.h"
#include "utils/info.h"

namespace mindspore {
namespace opt {
namespace python_pass {
namespace internal {
const char PARAMETER_MODULE[] = "mindspore.common.parameter";
const char PARAMETER_CLASS[] = "Parameter";
const char SET_PARAM[] = "__setattr__";
AnfNodePtr ProcessSinglePattern(const PatternPtr &pattern, const MatchResultPtr &res, const FuncGraphPtr &func_graph,
                                const FuncGraphPtr &top_graph);
AnfNodePtr BuildTarget(const PatternPtr &pattern, const FuncGraphPtr &func_graph, const FuncGraphPtr &top_graph,
                       const MatchResultPtr &res);
void ReflectParamBackToPython(const AnfNodePtr &param, const string &param_name, const tensor::TensorPtr &default_input,
                              bool requires_grad, bool layerwise_parallel);

bool IsTraversable(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (node->isa<CNode>() || node->isa<Parameter>()) {
    return true;
  }
  if (IsValueNode<FuncGraph>(node) || IsValueNode<RefKey>(node)) {
    return true;
  }
  return false;
}

AnfNodePtr BuildPrimitive(const PatternPtr &pattern) {
  // Build up AnfNode from primitive
  auto prim_pattern = pattern->cast_ptr<Prim>();
  MS_EXCEPTION_IF_NULL(prim_pattern);
  PrimitivePyPtr prim = prim_pattern->matched_primitive();
  MS_EXCEPTION_IF_NULL(prim);
  // Make value node out of primitives
  return std::make_shared<ValueNode>(prim);
}

AnfNodePtr BuildNewTensor(const PatternPtr &pattern) {
  // Build a ValueNode from TensorPtr
  auto new_tensor_pattern = pattern->cast_ptr<NewTensor>();
  MS_EXCEPTION_IF_NULL(new_tensor_pattern);
  auto input_tensor = new_tensor_pattern->input_tensor();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return std::make_shared<ValueNode>(input_tensor);
}

AnfNodePtr BuildPrimitiveValueNode(const PatternPtr &pattern, const MatchResultPtr &res, const FuncGraphPtr &fg,
                                   const FuncGraphPtr &top_graph) {
  auto call_pattern = pattern->cast_ptr<Call>();
  MS_EXCEPTION_IF_NULL(call_pattern);
  auto prim = call_pattern->prim_value();
  if (prim != nullptr) {
    return std::make_shared<ValueNode>(prim);
  }
  auto prim_pattern = call_pattern->prim_pattern();
  MS_EXCEPTION_IF_NULL(prim_pattern);
  return ProcessSinglePattern(prim_pattern, res, fg, top_graph);
}

AnfNodePtr BuildNewParameter(const PatternPtr &pattern, const MatchResultPtr &res, const FuncGraphPtr &top_graph) {
  auto new_para_pattern = pattern->cast_ptr<NewParameter>();
  MS_EXCEPTION_IF_NULL(new_para_pattern);
  if (!new_para_pattern->built()) {
    static int64_t parameter_id = 0;
    auto para_name = new_para_pattern->para_name() + new_para_pattern->unique_name() + std::to_string(parameter_id++);
    auto para_node = std::make_shared<Parameter>(top_graph);
    MS_EXCEPTION_IF_NULL(para_node);
    para_node->set_name(para_name);
    // Set function graph
    para_node->set_func_graph(top_graph);
    // Set Debug Info
    auto debug_info = std::make_shared<NodeDebugInfo>(para_name);
    para_node->set_debug_info(debug_info);
    // Set abstract
    auto default_value = new_para_pattern->default_tensor();
    MS_EXCEPTION_IF_NULL(default_value);
    para_node->set_abstract(default_value->ToAbstract()->Broaden());
    res->add_entry(pattern, para_node);
    top_graph->add_parameter(para_node);
    // Reflect back to Cell._params
    internal::ReflectParamBackToPython(para_node, para_name, default_value, new_para_pattern->requires_grad(),
                                       new_para_pattern->layerwise_parallel());
    MS_LOG(WARNING) << "Adding parameter: " + para_node->ToString() + " parameter name:" + para_node->name();
    new_para_pattern->set_built(true);
    return para_node;
  } else {
    // Built, fetch the node
    auto para_node = res->get_node(pattern);
    MS_EXCEPTION_IF_NULL(para_node);
    return para_node;
  }
}

AnfNodePtr BuildImmNode(const PatternPtr &pattern) {
  auto imm_pattern = pattern->cast_ptr<Imm>();
  MS_EXCEPTION_IF_NULL(imm_pattern);
  auto value = imm_pattern->value();
  auto scalar_value_ptr = std::make_shared<Int64Imm>(value);
  return std::make_shared<ValueNode>(scalar_value_ptr);
}

AnfNodePtr ProcessSinglePattern(const PatternPtr &pattern, const MatchResultPtr &res, const FuncGraphPtr &func_graph,
                                const FuncGraphPtr &top_graph) {
  auto target_node = res->get_node(pattern);
  if (target_node != nullptr) {
    // If pattern is NewParameter, check whether it shouldn't last and is not built
    auto new_para = pattern->cast_ptr<NewParameter>();
    if (new_para == nullptr || new_para->should_last() || new_para->built()) {
      return target_node;
    }
  }
  // Build up new node from pattern
  if (pattern->isa<Prim>()) {
    return BuildPrimitive(pattern);
  } else if (pattern->isa<NewTensor>()) {
    return BuildNewTensor(pattern);
  } else if (pattern->isa<Call>()) {
    return BuildPrimitiveValueNode(pattern, res, func_graph, top_graph);
  } else if (pattern->isa<NewParameter>()) {
    // Add new parameter to top graph instead of current graph
    return BuildNewParameter(pattern, res, top_graph);
  } else if (pattern->isa<Imm>()) {
    return BuildImmNode(pattern);
  }
  MS_LOG(EXCEPTION) << "Cannot find or build target node, pattern: " + pattern->unique_name() + "\n";
}

AnfNodePtr ProcessComplexPatternFirstInput(const PatternPtr &pattern, const MatchResultPtr &res,
                                           const FuncGraphPtr &func_graph, const FuncGraphPtr &top_graph) {
  if (pattern->isa<Call>()) {
    return BuildPrimitiveValueNode(pattern, res, func_graph, top_graph);
  }
  return nullptr;
}

AnfNodePtr BuildTarget(const PatternPtr &pattern, const FuncGraphPtr &func_graph, const FuncGraphPtr &top_graph,
                       const MatchResultPtr &res) {
  auto target_inputs = pattern->inputs();
  if (target_inputs.size() == 0) {
    auto new_anf_node = ProcessSinglePattern(pattern, res, func_graph, top_graph);
    if (new_anf_node != nullptr) {
      res->add_entry(pattern, new_anf_node);
    }
    return new_anf_node;
  }
  // Build up the AnfNode in a recursive manner
  std::vector<AnfNodePtr> new_inputs;
  auto prim_value_node = ProcessComplexPatternFirstInput(pattern, res, func_graph, top_graph);
  MS_EXCEPTION_IF_NULL(prim_value_node);
  new_inputs.push_back(prim_value_node);
  for (auto &iter : target_inputs) {
    if (iter == pattern) {
      MS_LOG(EXCEPTION) << "Circle references. Got pattern: " + pattern->unique_name() + "\n";
    }
    auto input_node = BuildTarget(iter, func_graph, top_graph, res);
    if (input_node == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to build input node for pattern : " + iter->unique_name() + "\n";
    }
    new_inputs.push_back(input_node);
  }
  auto new_c_node = func_graph->NewCNode(new_inputs);
  res->add_entry(pattern, new_c_node);
  return new_c_node;
}

void ReflectParamBackToPython(const AnfNodePtr &param, const string &param_name, const tensor::TensorPtr &default_input,
                              bool requires_grad, bool layerwise_parallel) {
  // 1. Get current cell object
  auto ppm = opt::python_pass::PyPassManager::GetInstance();
  auto resource = ppm->GetResource();
  py::object top_cell = resource->source_input();
  if (py::isinstance<py::none>(top_cell)) {
    MS_LOG(EXCEPTION) << "Failed to get top cell from resource.";
  }
  // 2. Clone default_input tensor
  MS_EXCEPTION_IF_NULL(default_input);
  auto default_tensor = std::make_shared<tensor::Tensor>(default_input->data_type(), default_input->shape_c(),
                                                         default_input->data_c(), default_input->Size());
  // 3. New a Parameter object with the above-specified args
  py::object parameter_class = py::module::import(PARAMETER_MODULE).attr(PARAMETER_CLASS);
  py::object new_parameter = parameter_class(default_tensor, param_name, requires_grad, layerwise_parallel);
  // 4. Add the new python Parameter object to Cell's _params attributes
  top_cell.attr(SET_PARAM)(param_name, new_parameter);
  // 5. Set default_param for param_node
  ValuePtr param_value = nullptr;
  bool converted = parse::ConvertData(new_parameter, &param_value, false);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Failed to convert new parameter to ValuePtr.";
  }
  MS_EXCEPTION_IF_NULL(param);
  auto param_node = param->cast_ptr<Parameter>();
  MS_EXCEPTION_IF_NULL(param_node);
  param_node->set_default_param(param_value);
}

void Reset(const PatternPtr &pattern) {
  MS_EXCEPTION_IF_NULL(pattern);
  if (pattern->isa<Prim>()) {
    auto prim_pattern = pattern->cast_ptr<Prim>();
    prim_pattern->reset();
  } else if (pattern->isa<NewParameter>()) {
    auto new_param_pattern = pattern->cast_ptr<NewParameter>();
    new_param_pattern->reset();
  } else if (pattern->isa<Call>()) {
    auto call_with_pattern = pattern->cast_ptr<Call>();
    for (const auto &sub_pattern : call_with_pattern->inputs()) {
      Reset(sub_pattern);
    }
  }
}
}  // namespace internal

AnfNodePtr PythonPass::Run(const FuncGraphPtr &func_graph, const FuncGraphPtr &top_graph, const AnfNodePtr &node,
                           const MatchResultPtr &res) {
  auto match_res = src_pattern_->match(node);
  if (match_res != nullptr) {
    res->merge(match_res);
    auto new_node = internal::BuildTarget(dst_pattern_, func_graph, top_graph, res);
    internal::Reset(dst_pattern());
    return new_node;
  }
  internal::Reset(src_pattern());
  return nullptr;
}

bool PythonPass::Run(const FuncGraphPtr &func_graph, const MatchResultPtr &res) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dst_pattern_);
  if (src_pattern_ == nullptr) {
    // Add NewParameter
    auto new_para_pattern = dst_pattern_->cast_ptr<NewParameter>();
    if (new_para_pattern == nullptr) {
      MS_LOG(EXCEPTION) << "Expect NewParameter pattern for target if src pattern is null.";
    }
    auto para_name = new_para_pattern->para_name() + new_para_pattern->unique_name();
    auto para_node = std::make_shared<Parameter>(func_graph);
    MS_EXCEPTION_IF_NULL(para_node);
    para_node->set_name(para_name);
    // Set function graph
    para_node->set_func_graph(func_graph);
    // Set Debug Info
    auto debug_info = std::make_shared<NodeDebugInfo>(para_name);
    para_node->set_debug_info(debug_info);
    // Set abstract
    auto default_value = new_para_pattern->default_tensor();
    MS_EXCEPTION_IF_NULL(default_value);
    para_node->set_abstract(default_value->ToAbstract()->Broaden());
    res->add_entry(dst_pattern_, para_node);
    func_graph->add_parameter(para_node);
    // Reflect back to Cell._params
    internal::ReflectParamBackToPython(para_node, para_name, default_value, new_para_pattern->requires_grad(),
                                       new_para_pattern->layerwise_parallel());
    MS_LOG(WARNING) << "[Gen]Adding parameter: " + para_node->ToString() + " parameter name:" + para_node->name();
    return true;
  }
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto func_graphs = manager->func_graphs();
  bool changes = false;
  for (auto &fg : func_graphs) {
    manager->AddFuncGraph(fg);
    auto graph_nodes_sorted = TopoSort(fg->output());
    // Traverse once
    for (auto &node : graph_nodes_sorted) {
      AnfNodePtr new_node = Run(fg, func_graph, node, res);
      if (new_node != nullptr && new_node != node) {
        MS_LOG(WARNING) << "Matched";
        (void)manager->Replace(node, new_node);
        changes = true;
      }
    }
  }
  return changes;
}
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore
