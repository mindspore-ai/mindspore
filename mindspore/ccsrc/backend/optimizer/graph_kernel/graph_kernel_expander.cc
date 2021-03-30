/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <set>
#include <utility>
#include <vector>

#include "utils/context/graph_kernel_flags.h"
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
std::vector<PrimitivePtr> GetExpandOps() {
  std::vector<PrimitivePtr> expand_ops = {
    prim::kPrimSquare,
    prim::kPrimGeLUGrad,
#if ENABLE_D
    prim::kPrimTile,
    prim::kPrimSqrtGrad,
    prim::kPrimClipByNormNoDivSum,
    prim::kLambApplyOptimizerAssign,
#elif ENABLE_GPU
    prim::kPrimBiasAdd,
    prim::kPrimBiasAddGrad,
    prim::kPrimGeLU,
    prim::kPrimFusedAdam,
    prim::kPrimFusedAdamWeightDecay,
    prim::kPrimBatchNorm,
    prim::kPrimBatchNormGrad,
    prim::kPrimReduceMean,
    prim::kPrimMaximumGrad,
    prim::kPrimMinimumGrad,
    prim::kPrimDropout,
    prim::kPrimDropoutGrad,
    prim::kPrimSoftmax,
    prim::kPrimLayerNorm,
    prim::kPrimLayerNormGrad,
    prim::kPrimRelu,
    prim::kPrimReluGrad,
    prim::kPrimSigmoid,
    prim::kPrimSigmoidGrad,
    prim::kPrimSigmoidCrossEntropyWithLogits,
    prim::kPrimSigmoidCrossEntropyWithLogitsGrad,
    prim::kPrimSoftmaxCrossEntropyWithLogits,
    prim::kPrimAssignAdd,
#endif
  };
  const auto &flags = context::GraphKernelFlags::GetInstance();
  OpListFilter(&expand_ops, flags.enable_expand_ops_only, flags.enable_expand_ops, flags.disable_expand_ops);
  return expand_ops;
}
}  // namespace

bool DefaultExpander::ExpandJsonInfo(const AnfNodePtr &node, nlohmann::json *kernel_json) {
  DumpOption dump_option;
  dump_option.extract_opinfo_from_anfnode = true;
  kernel::AkgKernelJsonGenerator json_generator(dump_option);
  return json_generator.CollectJson(node, kernel_json);
}

FuncGraphPtr DefaultExpander::CreateExpandFuncGraph(const CNodePtr &node) {
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
    return nullptr;
  }
  // decode json to func_graph.
  std::vector<AnfNodePtr> ori_inputs(node->inputs().begin() + 1, node->inputs().end());
  return JsonDescToAnf(kernel_desc_str, ori_inputs);
}

void DefaultExpander::EliminateRedundantParameters(const FuncGraphPtr &func_graph, AnfNodePtrList *inputs) {
  const auto &ori_parameter = func_graph->parameters();
  auto todos = TopoSort(func_graph->get_return());
  std::set<AnfNodePtr> used_param;
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

AnfNodePtr DefaultExpander::CreateExpandGraphKernel(const FuncGraphPtr &new_func_graph, const CNodePtr &old_node) {
  auto func_graph = old_node->func_graph();
  std::vector<AnfNodePtr> inputs(old_node->inputs().begin() + 1, old_node->inputs().end());
  AnfNodePtrList kernel_nodes;
  AnfNodePtrList outputs;
  EliminateRedundantParameters(new_func_graph, &inputs);
  kernel::GetValidKernelNodes(new_func_graph, &kernel_nodes);
  kernel::GetFuncGraphOutputNodes(new_func_graph, &outputs);
  auto graph_kernel_node = CreateNewFuseCNode(func_graph, new_func_graph, inputs, outputs);
  SetNewKernelInfo(graph_kernel_node, new_func_graph, inputs, outputs, AnfAlgo::GetProcessor(old_node));
  MS_LOG(DEBUG) << "Expand node: " << old_node->fullname_with_scope()
                << " with: " << graph_kernel_node->fullname_with_scope();
  return graph_kernel_node;
}

AnfNodePtr DefaultExpander::Run(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_func_graph = CreateExpandFuncGraph(cnode);
  if (new_func_graph == nullptr) {
    return nullptr;
  }
  new_func_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(AnfAlgo::GetCNodeName(cnode)));
  auto graph_kernel_node = CreateExpandGraphKernel(new_func_graph, cnode);
  if (AnfAlgo::GetOutputTensorNum(node) != AnfAlgo::GetOutputTensorNum(graph_kernel_node)) {
    MS_LOG(ERROR) << "The output num of composite node (" << AnfAlgo::GetOutputTensorNum(graph_kernel_node)
                  << ") does not match the original basic node (" << AnfAlgo::GetOutputTensorNum(node) << ")."
                  << node->fullname_with_scope();
    return nullptr;
  }
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

    MS_LOG(INFO) << "Expanding node: " << node->fullname_with_scope();
    auto new_node = GetExpander(node)->Run(node);
    if (new_node == nullptr) {
      MS_LOG(INFO) << "Skipped node: " << node->fullname_with_scope();
      continue;
    }
    (void)mng->Replace(node, new_node);
    changed = true;
  }
  return changed;
}

ExpanderPtr GraphKernelExpander::GetExpander(const AnfNodePtr &node) {
  std::vector<std::pair<PrimitivePtr, ExpanderPtr>> expanders = {
    {prim::kPrimDropout, std::make_shared<DropoutExpander>()},
  };
  for (auto &e : expanders) {
    if (IsPrimitiveCNode(node, e.first)) {
      return e.second;
    }
  }
  return std::make_shared<DefaultExpander>();
}

bool GraphKernelExpander::Run(const FuncGraphPtr &func_graph) {
  expand_ops_ = GetExpandOps();
  return DoExpand(func_graph);
}
}  // namespace opt
}  // namespace mindspore
