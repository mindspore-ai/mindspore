/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "common/graph_kernel/adapter/graph_kernel_expander_with_py.h"

#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "utils/ms_context.h"
#include "utils/context/graph_kernel_flags.h"
#include "kernel/akg/akg_kernel_json_generator.h"
#include "common/graph_kernel/graph_kernel_helper.h"
#include "common/graph_kernel/split_umonad.h"
#include "common/graph_kernel/substitute_dropout.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "mindspore/core/ir/graph_utils.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pybind_api/ir/primitive_py.h"

namespace mindspore::graphkernel {
constexpr size_t kAssignInputIdx = 1;
constexpr size_t kLambOptimizerInputIdx = 12;
constexpr size_t kLambWeightInputIdx = 4;
constexpr size_t kRandomInputIdx = 1;
constexpr size_t kAdamInputIdx = 10;

bool PyExpander::ExpandJsonInfo(const AnfNodePtr &node, nlohmann::json *kernel_json) {
  DumpOption dump_option;
  dump_option.extract_opinfo_from_anfnode = true;
  AkgKernelJsonGenerator json_generator(dump_option);
  return json_generator.CollectJson(node, kernel_json);
}

FuncGraphPtr PyExpander::CreateExpandFuncGraph(const CNodePtr &node) {
  nlohmann::json kernel_json;
  if (!ExpandJsonInfo(node, &kernel_json)) {
    constexpr int recursive_level = 2;
    MS_LOG(ERROR) << "Expand json info to: " << node->DebugString(recursive_level) << " failed, ori_json:\n"
                  << kernel_json.dump();
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

ExpanderPtr GraphKernelExpanderWithPy::GetExpander(const AnfNodePtr &node) {
  std::vector<std::pair<PrimitivePtr, ExpanderPtr>> expanders = {
    {prim::kPrimDropout, std::make_shared<DropoutExpander>()},
    {prim::kPrimAssignAdd, std::make_shared<OpUMonadExpander>(kAssignInputIdx)},
    {prim::kLambApplyOptimizerAssign, std::make_shared<OpUMonadExpander>(kLambOptimizerInputIdx)},
    {prim::kLambApplyWeightAssign, std::make_shared<OpUMonadExpander>(kLambWeightInputIdx)},
    {prim::kPrimStandardNormal, std::make_shared<OpUMonadExpander>(kRandomInputIdx)},
    {prim::kPrimAdam, std::make_shared<OpUMonadExpander>(kAdamInputIdx)},
    {prim::kPrimAddN, std::make_shared<PyExpander>()},
    {prim::kPrimBatchNorm, std::make_shared<PyExpander>()},
    {prim::kPrimBatchNormGrad, std::make_shared<PyExpander>()},
    {prim::kPrimBiasAddGrad, std::make_shared<PyExpander>()},
    {prim::kPrimClipByNormNoDivSum, std::make_shared<PyExpander>()},
    {prim::kPrimConv2D, std::make_shared<PyExpander>()},
    {prim::kPrimDropoutGrad, std::make_shared<PyExpander>()},
    {prim::kPrimEqualCount, std::make_shared<PyExpander>()},
    {prim::kPrimErfc, std::make_shared<PyExpander>()},
    {prim::kPrimFusedAdam, std::make_shared<PyExpander>()},
    {prim::kPrimFusedAdamWeightDecay, std::make_shared<PyExpander>()},
    {prim::kPrimGather, std::make_shared<PyExpander>()},
    {prim::kPrimGeLU, std::make_shared<PyExpander>()},
    {prim::kPrimGeLUGrad, std::make_shared<PyExpander>()},
    {prim::kPrimIdentityMath, std::make_shared<PyExpander>()},
    {prim::kPrimLayerNorm, std::make_shared<PyExpander>()},
    {prim::kPrimLayerNormGrad, std::make_shared<PyExpander>()},
    {prim::kPrimLogSoftmax, std::make_shared<PyExpander>()},
    {prim::kPrimLogSoftmaxGrad, std::make_shared<PyExpander>()},
    {prim::kPrimMatMul, std::make_shared<PyExpander>()},
    {prim::kPrimMaximumGrad, std::make_shared<PyExpander>()},
    {prim::kPrimMinimumGrad, std::make_shared<PyExpander>()},
    {prim::kPrimOnesLike, std::make_shared<PyExpander>()},
    {prim::kPrimReduceMean, std::make_shared<PyExpander>()},
    {prim::kPrimRelu, std::make_shared<PyExpander>()},
    {prim::kPrimReluGrad, std::make_shared<PyExpander>()},
    {prim::kPrimSigmoid, std::make_shared<PyExpander>()},
    {prim::kPrimSigmoidCrossEntropyWithLogits, std::make_shared<PyExpander>()},
    {prim::kPrimSigmoidCrossEntropyWithLogitsGrad, std::make_shared<PyExpander>()},
    {prim::kPrimSigmoidGrad, std::make_shared<PyExpander>()},
    {prim::kPrimSlice, std::make_shared<PyExpander>()},
    {prim::kPrimSoftmax, std::make_shared<PyExpander>()},
    {prim::kPrimSoftmaxCrossEntropyWithLogits, std::make_shared<PyExpander>()},
    {prim::kSoftmaxGradExt, std::make_shared<PyExpander>()},
    {prim::kPrimSqrtGrad, std::make_shared<PyExpander>()},
    {prim::kPrimSquare, std::make_shared<PyExpander>()},
    {prim::kPrimSquaredDifference, std::make_shared<PyExpander>()},
    {prim::kSquareSumV1, std::make_shared<PyExpander>()},
    {prim::kPrimSquareSumAll, std::make_shared<PyExpander>()},
    {prim::kPrimSqueeze, std::make_shared<PyExpander>()},
    {prim::kPrimTanhGrad, std::make_shared<PyExpander>()},
    {prim::kPrimTile, std::make_shared<PyExpander>()},
  };

  for (auto &e : expanders) {
    if (IsPrimitiveCNode(node, e.first)) {
      return e.second;
    }
  }
  return std::make_shared<DefaultExpander>();
}

bool GraphKernelComplexExpander::CanExpand(const CNodePtr &node) const {
  bool has_complex = false;
  auto all_inputs_type = AnfAlgo::GetAllInputDeviceTypes(node);
  for (size_t i = 0; i < all_inputs_type.size(); ++i) {
    if (all_inputs_type[i] == kNumberTypeComplex64) {
      has_complex = true;
      break;
    }
  }
  return has_complex;
}

ExpanderPtr GraphKernelComplexExpander::GetExpander(const AnfNodePtr &) {
  return std::make_shared<ComplexOpExpander>();
}
bool ComplexOpExpander::ExpandJsonInfo(const AnfNodePtr &node, nlohmann::json *kernel_json) {
  auto cnode = node->cast<CNodePtr>();
  if (!PyExpander::ExpandJsonInfo(cnode, kernel_json)) return false;
  (*kernel_json)["name"] = std::string("C") + AnfAlgo::GetCNodeName(cnode);
  return true;
}
bool GraphKernelComplexExpander::Run(const FuncGraphPtr &func_graph) { return DoExpand(func_graph); }
}  // namespace mindspore::graphkernel
