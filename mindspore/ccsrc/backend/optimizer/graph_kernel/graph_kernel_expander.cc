/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/graph_kernel/graph_kernel_expander.h"

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/optimizer/graph_kernel/substitute_dropout.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "mindspore/core/ir/graph_utils.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pybind_api/ir/primitive_py.h"
#include "runtime/device/kernel_info.h"
#include "vm/segment_runner.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kJsonKeyExpandInfo = "expand_info";

#define GET_VALUE_FOR_JSON(JSON, VALUE, VALUE_ELEM, TYPE_NAME, TYPE) \
  if (VALUE_ELEM->isa<TYPE_NAME>()) {                                \
    JSON = GetValue<TYPE>(VALUE);                                    \
  }

nlohmann::json ExpandAttrJsonInfo(const CNodePtr &cnode) {
  nlohmann::json attrs_json;
  if (auto prim = GetCNodePrimitive(cnode); prim != nullptr) {
    auto attrs = prim->attrs();
    for (const auto &[k, v] : attrs) {
      nlohmann::json attr_json;
      MS_LOG(DEBUG) << "attr key is : " << k << " and value type is : " << v->type_name();
      GET_VALUE_FOR_JSON(attr_json[k], v, v, Int32Imm, int);
      GET_VALUE_FOR_JSON(attr_json[k], v, v, Int64Imm, int64_t);
      GET_VALUE_FOR_JSON(attr_json[k], v, v, UInt32Imm, uint32_t);
      GET_VALUE_FOR_JSON(attr_json[k], v, v, UInt64Imm, uint64_t);
      GET_VALUE_FOR_JSON(attr_json[k], v, v, FP32Imm, float);
      GET_VALUE_FOR_JSON(attr_json[k], v, v, FP64Imm, double);
      GET_VALUE_FOR_JSON(attr_json[k], v, v, BoolImm, bool);
      GET_VALUE_FOR_JSON(attr_json[k], v, v, StringImm, std::string);

      if (v->isa<ValueList>() || v->isa<ValueTuple>()) {
        auto vec = v->isa<ValueList>() ? v->cast<ValueListPtr>()->value() : v->cast<ValueTuplePtr>()->value();
        if (!vec.empty()) {
          MS_LOG(DEBUG) << "value type is : " << vec[0]->type_name();
          GET_VALUE_FOR_JSON(attr_json[k], v, vec[0], Int32Imm, std::vector<int>);
          GET_VALUE_FOR_JSON(attr_json[k], v, vec[0], Int64Imm, std::vector<int64_t>);
          GET_VALUE_FOR_JSON(attr_json[k], v, vec[0], UInt32Imm, std::vector<uint32_t>);
          GET_VALUE_FOR_JSON(attr_json[k], v, vec[0], UInt64Imm, std::vector<uint64_t>);
          GET_VALUE_FOR_JSON(attr_json[k], v, vec[0], FP32Imm, std::vector<float>);
          GET_VALUE_FOR_JSON(attr_json[k], v, vec[0], FP64Imm, std::vector<double>);
          GET_VALUE_FOR_JSON(attr_json[k], v, vec[0], StringImm, std::vector<std::string>);
        }
      }
      if (!attr_json.empty()) {
        attrs_json.push_back(attr_json);
      }
    }
  }
  return attrs_json;
}

bool ExpandJsonInfo(const CNodePtr &cnode, nlohmann::json *kernel_json) {
  MS_EXCEPTION_IF_NULL(kernel_json);
  if (kernel_json->find(kJsonKeyExpandInfo) != kernel_json->end()) {
    return false;
  }

  nlohmann::json expand_info;
  expand_info[kernel::kJsonKeyAttr] = ExpandAttrJsonInfo(cnode);
  expand_info[kernel::kJsonKeyName] = AnfAlgo::GetCNodeName(cnode);
  expand_info[kernel::kJsonKeyProcess] = kernel::GetProcessorStr(cnode);
  std::vector<nlohmann::json> inputs_info;
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(cnode); ++i) {
    nlohmann::json input_info;
    input_info[kernel::kJsonKeyFormat] = AnfAlgo::GetInputFormat(cnode, i);
    input_info[kernel::kJsonKeyInferShape] = AnfAlgo::GetPrevNodeOutputInferShape(cnode, i);
    input_info[kernel::kJsonKeyShape] = AnfAlgo::GetInputDeviceShape(cnode, i);
    input_info[kernel::kJsonKeyInferDataType] =
      kernel::TypeId2String(AnfAlgo::GetPrevNodeOutputInferDataType(cnode, i));
    input_info[kernel::kJsonKeyDataType] = kernel::TypeId2String(AnfAlgo::GetInputDeviceDataType(cnode, i));
    inputs_info.push_back(input_info);
  }
  expand_info[kernel::kJsonKeyInputDesc] = inputs_info;

  std::vector<nlohmann::json> outputs_info;
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(cnode); ++i) {
    nlohmann::json output_info;
    output_info[kernel::kJsonKeyFormat] = AnfAlgo::GetOutputFormat(cnode, i);
    output_info[kernel::kJsonKeyInferShape] = AnfAlgo::GetOutputInferShape(cnode, i);
    output_info[kernel::kJsonKeyShape] = AnfAlgo::GetOutputDeviceShape(cnode, i);
    output_info[kernel::kJsonKeyInferDataType] = kernel::TypeId2String(AnfAlgo::GetOutputInferDataType(cnode, i));
    output_info[kernel::kJsonKeyDataType] = kernel::TypeId2String(AnfAlgo::GetOutputDeviceDataType(cnode, i));
    outputs_info.push_back(output_info);
  }
  expand_info[kernel::kJsonKeyOutputDesc] = outputs_info;
  (*kernel_json)[kJsonKeyExpandInfo] = expand_info;
  return true;
}
}  // namespace

FuncGraphPtr GraphKernelExpander::CreateExpandFuncGraph(const CNodePtr &node) {
  nlohmann::json kernel_json;
  if (!ExpandJsonInfo(node, &kernel_json)) {
    MS_LOG(ERROR) << "Expand json info to: " << node->DebugString(2) << " failed, ori_json:\n" << kernel_json.dump();
    return nullptr;
  }
  auto node_desc_str = kernel_json.dump();

  // call graph kernel ops generator.
  MS_LOG(DEBUG) << "CallPyFn: [" << kGetGraphKernelOpExpander << "] with input json:\n" << node_desc_str;
  auto ret = parse::python_adapter::CallPyFn(kGraphKernelModule, kGetGraphKernelOpExpander, node_desc_str);
  // parse result.
  if (py::isinstance<py::none>(ret)) {
    MS_LOG(ERROR) << "CallPyFn: [" << kGetGraphKernelOpExpander << "] return invalid result, input json:\n"
                  << node_desc_str;
    return nullptr;
  }
  std::string kernel_desc_str = py::cast<std::string>(ret);
  if (kernel_desc_str.empty()) {
    MS_LOG(ERROR) << "Jump expand node: " << node->fullname_with_scope();
    return nullptr;
  }
  // decode json to func_graph.
  std::vector<AnfNodePtr> ori_inputs(node->inputs().begin() + 1, node->inputs().end());
  return JsonDescToAnf(kernel_desc_str, ori_inputs);
}

void GraphKernelExpander::EliminateRedundantParameters(const FuncGraphPtr &func_graph, AnfNodePtrList *inputs) {
  const auto &ori_parameter = func_graph->parameters();
  auto todos = TopoSort(func_graph->get_return());
  std::unordered_set<AnfNodePtr> used_param;
  for (auto node : todos) {
    if (node->isa<Parameter>()) {
      used_param.insert(node);
    }
  }
  if (used_param.size() == ori_parameter.size()) {
    return;
  }
  AnfNodePtrList new_parameter, new_inputs;
  for (size_t i = 0; i < ori_parameter.size(); ++i) {
    if (used_param.count(ori_parameter[i])) {
      new_parameter.push_back(ori_parameter[i]);
      new_inputs.push_back((*inputs)[i]);
    }
  }
  func_graph->set_parameters(new_parameter);
  *inputs = std::move(new_inputs);
}

AnfNodePtr GraphKernelExpander::CreateExpandGraphKernel(const FuncGraphPtr &func_graph,
                                                        const FuncGraphPtr &new_func_graph, const CNodePtr &node) {
  std::vector<AnfNodePtr> inputs(node->inputs().begin() + 1, node->inputs().end());
  AnfNodePtrList kernel_nodes;
  AnfNodePtrList outputs;
  EliminateRedundantParameters(new_func_graph, &inputs);
  kernel::GetValidKernelNodes(new_func_graph, &kernel_nodes);
  kernel::GetFuncGraphOutputNodes(new_func_graph, &outputs);
  auto graph_kernel_node = CreateNewFuseCNode(func_graph, new_func_graph, inputs, outputs);
  SetNewKernelInfo(graph_kernel_node, new_func_graph, inputs, outputs, AnfAlgo::GetProcessor(node));
  std::string graph_kernel_flag;
  std::for_each(kernel_nodes.begin(), kernel_nodes.end(), [&graph_kernel_flag](const AnfNodePtr &node) {
    static_cast<void>(graph_kernel_flag.append(AnfAlgo::GetCNodeName(node)).append("_"));
  });
  MS_LOG(DEBUG) << "Expand node: " << node->fullname_with_scope() << " with: " << graph_kernel_flag;
  return graph_kernel_node;
}

bool GraphKernelExpander::DoExpand(const FuncGraphPtr &func_graph) {
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  std::reverse(todos.begin(), todos.end());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  for (const auto &n : todos) {
    auto node = n->cast<CNodePtr>();
    if (node == nullptr || IsKeepBasicNode(node) || !AnfAlgo::IsRealKernel(node) || AnfAlgo::IsGraphKernel(node) ||
        !CanExpand(node)) {
      continue;
    }

    MS_LOG(INFO) << "Expand process node: " << node->fullname_with_scope();
    auto new_func_graph = CreateExpandFuncGraph(node);
    if (new_func_graph == nullptr) {
      MS_LOG(ERROR) << "Decode fused nodes failed, " << node->fullname_with_scope();
      continue;
    }
    mng->AddFuncGraph(new_func_graph);
    MS_LOG(DEBUG) << "decode fused nodes success.";

    auto graph_kernel_node = CreateExpandGraphKernel(func_graph, new_func_graph, node);
    new_func_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(AnfAlgo::GetCNodeName(node)));

    // replace origin node.
    (void)mng->Replace(node, graph_kernel_node);

    ToPrimitive(AnfAlgo::GetCNodeFuncGraphPtr(graph_kernel_node));
    changed = true;
  }
  return changed;
}

void GraphKernelExpander::ToPrimitive(const FuncGraphPtr &func_graph) const {
  auto todos = TopoSort(func_graph->get_return());
  std::reverse(todos.begin(), todos.end());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  for (const auto &n : todos) {
    auto cnode = n->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }

    auto origin_prim = AnfAlgo::GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(origin_prim);
    if (!origin_prim->isa<PrimitivePy>()) {
      continue;
    }
    cnode->set_input(0, std::make_shared<ValueNode>(std::make_shared<Primitive>(*origin_prim)));
  }
}

bool GraphKernelExpander::Run(const FuncGraphPtr &func_graph) {
  expand_ops_ = GetExpandOps();
  MS_EXCEPTION_IF_NULL(func_graph);
  if (expand_ops_.count(prim::kPrimGkDropout) > 0) {
    std::shared_ptr<Pass> pass = std::make_shared<opt::SubstituteDropout>();
    pass->Run(func_graph);
  }
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  return DoExpand(func_graph);
}
}  // namespace opt
}  // namespace mindspore
