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
#include <algorithm>

#include "utils/context/graph_kernel_flags.h"
#include "backend/kernel_compiler/akg/akg_kernel_json_generator.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/optimizer/graph_kernel/split_umonad.h"
#include "backend/optimizer/graph_kernel/substitute_dropout.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "mindspore/core/ir/graph_utils.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pybind_api/ir/primitive_py.h"
#include "runtime/device/kernel_info.h"
#include "vm/segment_runner.h"
#include "backend/optimizer/graph_kernel/expanders/expander_factory.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kAssignInputIdx = 1;
constexpr size_t kLambOptimizerInputIdx = 12;
constexpr size_t kLambWeightInputIdx = 4;

std::vector<PrimitivePtr> GetExpandOps() {
  std::vector<PrimitivePtr> expand_ops = {
    prim::kPrimAddN,
    prim::kPrimAssignAdd,
    prim::kPrimErfc,
    prim::kPrimExpandDims,
    prim::kPrimGeLU,
    prim::kPrimGeLUGrad,
    prim::kPrimSquare,
    prim::kPrimTile,
#if ENABLE_D
    prim::kLambApplyOptimizerAssign,
    prim::kLambApplyWeightAssign,
    prim::kPrimClipByNormNoDivSum,
    prim::kPrimSqrtGrad,
    prim::kSoftmaxGradExt,
    prim::kFusedMulAdd,
#elif ENABLE_GPU
    prim::kPrimBatchMatMul,
    prim::kPrimBiasAdd,
    prim::kPrimBiasAddGrad,
    prim::kPrimDropout,
    prim::kPrimDropoutGrad,
    prim::kPrimFusedAdam,
    prim::kPrimFusedAdamWeightDecay,
    prim::kPrimMaximumGrad,
    prim::kPrimMinimumGrad,
    prim::kPrimLayerNorm,
    prim::kPrimLayerNormGrad,
    prim::kPrimLogSoftmax,
    prim::kPrimLogSoftmaxGrad,
    prim::kPrimMatMul,
    prim::kPrimReduceMean,
    prim::kPrimRelu,
    prim::kPrimReluGrad,
    prim::kPrimSigmoid,
    prim::kPrimSigmoidGrad,
    prim::kPrimSigmoidCrossEntropyWithLogits,
    prim::kPrimSigmoidCrossEntropyWithLogitsGrad,
    prim::kPrimSlice,
    prim::kPrimSoftmax,
    prim::kPrimSoftmaxCrossEntropyWithLogits,
    prim::kPrimSquaredDifference,
    prim::kPrimSqueeze,
    prim::kPrimEqualCount,
    prim::kPrimSquareSumAll,
    prim::kPrimIdentityMath,
    prim::kPrimOnesLike,
#endif
  };
  const auto &flags = context::GraphKernelFlags::GetInstance();
  OpListFilter(&expand_ops, flags.enable_expand_ops_only, flags.enable_expand_ops, flags.disable_expand_ops);
  return expand_ops;
}
}  // namespace

bool PyExpander::ExpandJsonInfo(const AnfNodePtr &node, nlohmann::json *kernel_json) {
  DumpOption dump_option;
  dump_option.extract_opinfo_from_anfnode = true;
  kernel::AkgKernelJsonGenerator json_generator(dump_option);
  return json_generator.CollectJson(node, kernel_json);
}

FuncGraphPtr PyExpander::CreateExpandFuncGraph(const CNodePtr &node) {
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
  return JsonDescToAnf(kernel_desc_str);
}

FuncGraphPtr DefaultExpander::CreateExpandFuncGraph(const CNodePtr &node) {
  auto expander_ptr = expanders::OpExpanderFactory::Instance().GetExpander(AnfAlgo::GetCNodeName(node));
  if (expander_ptr == nullptr) {
    return PyExpander::CreateExpandFuncGraph(node);
  }
  expanders::BaseInfoList inputs(node->size() - 1);
  expanders::BaseInfoList outputs(AnfAlgo::GetOutputTensorNum(node));
  for (size_t i = 0; i < inputs.size(); i++) {
    auto shape = AnfAlgo::GetInputDeviceShape(node, i);
    std::transform(shape.begin(), shape.end(), std::back_inserter(inputs[i].shape), SizeToLong);
    inputs[i].type = AnfAlgo::GetInputDeviceDataType(node, i);
    inputs[i].format = AnfAlgo::GetInputFormat(node, i);
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    auto shape = AnfAlgo::GetOutputDeviceShape(node, i);
    std::transform(shape.begin(), shape.end(), std::back_inserter(outputs[i].shape), SizeToLong);
    outputs[i].type = AnfAlgo::GetOutputDeviceDataType(node, i);
    outputs[i].format = AnfAlgo::GetOutputFormat(node, i);
  }
  auto &attrs = AnfAlgo::GetCNodePrimitive(node)->attrs();
  try {
    auto litegraph = expander_ptr->Run(inputs, outputs, attrs, kernel::GetStrProcessorFromContext());
    return LiteGraph2AnfGraph(litegraph);
  } catch (const graphkernel::GKException &e) {
    MS_LOG(INFO) << e.what() << ", undo expanding this op";
    return nullptr;
  }
}

AnfNodePtr PyExpander::CreateExpandGraphKernel(const FuncGraphPtr &new_func_graph, const CNodePtr &old_node) {
  auto func_graph = old_node->func_graph();
  std::vector<AnfNodePtr> inputs(old_node->inputs().begin() + 1, old_node->inputs().end());
  AnfNodePtrList kernel_nodes;
  AnfNodePtrList outputs;
  EliminateRedundantParameters(new_func_graph, &inputs);
  kernel::GetValidKernelNodes(new_func_graph, &kernel_nodes);
  kernel::GetFuncGraphOutputNodes(new_func_graph, &outputs);
  auto graph_kernel_node = CreateNewFuseCNode(func_graph, new_func_graph, inputs, outputs);
  SetNewKernelInfo(graph_kernel_node, new_func_graph, inputs, outputs);
  MS_LOG(DEBUG) << "Expand node: " << old_node->fullname_with_scope()
                << " with: " << graph_kernel_node->fullname_with_scope();
  return graph_kernel_node;
}

AnfNodePtr PyExpander::Run(const AnfNodePtr &node) {
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

ExpanderPtr GraphKernelExpander::GetExpander(const AnfNodePtr &node) {
  std::vector<std::pair<PrimitivePtr, ExpanderPtr>> expanders = {
    {prim::kPrimDropout, std::make_shared<DropoutExpander>()},
    {prim::kPrimAssignAdd, std::make_shared<OpUMonadExpander>(kAssignInputIdx)},
    {prim::kPrimAssignSub, std::make_shared<OpUMonadExpander>(kAssignInputIdx)},
    {prim::kLambApplyOptimizerAssign, std::make_shared<OpUMonadExpander>(kLambOptimizerInputIdx)},
    {prim::kLambApplyWeightAssign, std::make_shared<OpUMonadExpander>(kLambWeightInputIdx)},
  };

  for (auto &e : expanders) {
    if (IsPrimitiveCNode(node, e.first)) {
      return e.second;
    }
  }
  return std::make_shared<DefaultExpander>();
}

bool GraphKernelExpander::DoExpand(const FuncGraphPtr &func_graph) {
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  std::reverse(todos.begin(), todos.end());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  for (const auto &n : todos) {
    auto node = n->cast<CNodePtr>();
    if (node == nullptr || AnfAlgo::IsGraphKernel(node) || IsKeepBasicNode(node) || !AnfAlgo::IsRealKernel(node) ||
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

bool GraphKernelComplexExpander::CanExpand(const CNodePtr &node) const {
  bool has_complex = false;
  auto all_inputs_type = AnfAlgo::GetAllInputDeviceTypes(node);
  for (size_t i = 0; i < all_inputs_type.size(); ++i) {
    if (all_inputs_type[i] == kNumberTypeFloat64 || all_inputs_type[i] == kNumberTypeComplex64) {
      has_complex = true;
      break;
    }
  }
  return has_complex;
}

// Just test for complex op, then will be deleted
ExpanderPtr GraphKernelComplexExpander::GetExpander(const AnfNodePtr &node) {
  return std::make_shared<ComplexOpExpander>();
}
bool ComplexOpExpander::ExpandJsonInfo(const AnfNodePtr &node, nlohmann::json *kernel_json) {
  auto cnode = node->cast<CNodePtr>();
  auto all_inputs_type = AnfAlgo::GetAllInputDeviceTypes(cnode);
  for (size_t i = 0; i < all_inputs_type.size(); ++i) {
    if (all_inputs_type[i] == kNumberTypeFloat64 || all_inputs_type[i] == kNumberTypeComplex64) {
      all_inputs_type[i] = kNumberTypeComplex64;
    }
  }

  auto all_outputs_type = AnfAlgo::GetAllOutputDeviceTypes(cnode);
  for (size_t i = 0; i < all_outputs_type.size(); ++i) {
    if (all_outputs_type[i] == kNumberTypeFloat64) {
      all_outputs_type[i] = kNumberTypeComplex64;
    }
  }
  auto all_inputs_format = AnfAlgo::GetAllInputFormats(cnode);
  auto all_outputs_format = AnfAlgo::GetAllOutputFormats(cnode);
  auto graph_sel_info =
    BuildSelectKernelBuildInfo(all_inputs_format, all_inputs_type, all_outputs_format, all_outputs_type);
  AnfAlgo::SetSelectKernelBuildInfo(graph_sel_info, cnode.get());
  std::vector<size_t> original_shape = AnfAlgo::GetOutputInferShape(cnode, 0);
  ShapeVector real_shape;
  std::copy(original_shape.begin(), original_shape.end(), std::back_inserter(real_shape));
  auto complex_shape_ptr = std::make_shared<abstract::Shape>(abstract::Shape(real_shape));
  TypeId complex_type = kNumberTypeComplex64;
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(complex_type), complex_shape_ptr);
  cnode->set_abstract(abstract);
  if (!DefaultExpander::ExpandJsonInfo(cnode, kernel_json)) return false;
  (*kernel_json)["name"] = std::string("C") + AnfAlgo::GetCNodeName(cnode);
  return true;
}
bool GraphKernelExpander::Run(const FuncGraphPtr &func_graph) {
  expand_ops_ = GetExpandOps();
  return DoExpand(func_graph);
}
bool GraphKernelComplexExpander::Run(const FuncGraphPtr &func_graph) { return DoExpand(func_graph); }
}  // namespace opt
}  // namespace mindspore
