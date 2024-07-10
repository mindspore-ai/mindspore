/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/expander/utils.h"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include "ops/nn_op_name.h"
#include "ops/structure_ops.h"
#include "ops/op_def.h"
#include "ops/math_ops.h"
#include "ops/array_ops.h"
#include "mindspore/core/utils/anf_utils.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "frontend/expander/bprop/bprop.h"
#include "pybind_api/ir/primitive_py.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
/* namespace to support expander */
namespace expander {
namespace {
const std::map<std::string, std::vector<std::string>> op2attrs = {
  {kBroadcastOpName, {kAttrShape}},
  {kReduceMaxOpName, {kAttrKeepDims}},
  {kReduceMinOpName, {kAttrKeepDims}},
  {kReduceSumOpName, {kAttrKeepDims}},
  {kMatMulOpName, {kTransposeA, kTransposeB}},
  {kConcatOpName, {kAttrAxis}},
  {kSqueezeOpName, {kAttrAxis}},
  {kOneHotOpName, {kAttrAxis}},
  {kSoftmaxOpName, {kAttrAxis}},
  {kSplitOpName, {kAttrAxis}},
  {kLayerNormOpName, {kAttrBeginNormAxis, kAttrBeginParamsAxis, kAttrEpsilon}},
  {kStridedSliceOpName, {kAttrBeginMask, kAttrEndMask, kAttrEllipsisMask, kAttrNewAxisMask, kAttrShrinkAxisMask}},
  {kLayerNormGradOpName, {kAttrBeginNormAxis, kAttrBeginParamsAxis}},
  {kLayerNormGradGradOpName, {kAttrBeginNormAxis, kAttrBeginParamsAxis}},
  {kBiasAddOpName, {kAttrDataFormat}},
  {kBiasAddGradOpName, {kAttrDataFormat}},
  {kStackOpName, {kAttrAxis}},
  {kBatchMatMulOpName, {kTransposeA, kTransposeB}}};
}  // namespace

ValuePtr ConvertPrimToPrimPy(const PrimitivePtr &primc) {
  if (primc == nullptr || primc->isa<PrimitivePy>()) {
    return nullptr;
  }
  // If it is primitive function, no need convert because primitive function are all C++ infer.
  if (mindspore::ops::IsPrimitiveFunction(primc->name())) {
    return nullptr;
  }
  if (abstract::GetFrontendPrimitiveInferImpl(primc).has_value()) {
    return nullptr;
  }
  if (primc->isa<prim::DoSignaturePrimitive>()) {
    return nullptr;
  }
  const auto &primpy_cache = OpPrimPyRegister::GetInstance().GetPrimPyMap();
  if (auto it = primpy_cache.find(primc->name()); it != primpy_cache.end()) {
    return it->second;
  }
  parallel::OperatorAttrs attrs;
  const auto iter = op2attrs.find(primc->name());
  if (iter != op2attrs.end()) {
    for (auto &attr : iter->second) {
      if (primc->HasAttr(attr)) {
        (void)attrs.emplace_back(std::pair{attr, primc->GetAttr(attr)});
      } else {
        MS_LOG(WARNING) << primc->name() << " op do not have attr: " << attr;
        return nullptr;
      }
    }
  }
  auto new_prim = parallel::CreateOpInstance(attrs, primc->name(), "");
  MS_EXCEPTION_IF_NULL(new_prim);
  (void)new_prim->cast<PrimitivePtr>()->SetAttrs(primc->attrs());
  // prim can be cached when prim has no attrs
  constexpr size_t kOnlyIONames = 2;
  if ((primc->attrs().size() == kOnlyIONames) && primc->HasAttr("input_names") && primc->HasAttr("output_names")) {
    OpPrimPyRegister::GetInstance().SetPrimPyMap(primc->name(), new_prim);
  }
  return new_prim;
}

class PrimpyConverter {
 public:
  bool Run(const FuncGraphPtr &graph) {
    MS_EXCEPTION_IF_NULL(graph);
    (void)visited_graphs_.insert(graph);
    auto todos = TopoSort(graph->get_return());
    auto mng = Manage({graph}, false);
    for (const auto &node : todos) {
      if (node->isa<ValueNode>()) {
        auto sub_graph = node->cast<ValueNodePtr>()->value()->cast<FuncGraphPtr>();
        if (sub_graph != nullptr && visited_graphs_.count(sub_graph) == 0) {
          (void)Run(sub_graph);
          continue;
        }
      }
      if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
        continue;
      }
      auto primitive = GetCNodePrimitive(node);
      auto new_prim = ConvertPrimToPrimPy(primitive);
      AnfNodePtrList inputs = {NewValueNode(new_prim)};
      auto cnode = dyn_cast_ptr<CNode>(node);
      auto cnode_inputs = cnode->inputs();
      (void)inputs.insert(inputs.cend(), cnode_inputs.cbegin() + 1, cnode_inputs.cend());
      auto new_cnode = graph->NewCNodeInOrder(inputs);
      (void)mng->Replace(node, new_cnode);
    }
    return true;
  }

 private:
  std::set<FuncGraphPtr> visited_graphs_;
};

bool ConvertPrimToPrimPy(const FuncGraphPtr &graph) {
  PrimpyConverter c;
  return c.Run(graph);
}

void ClearAllCache() { bprop::ClearBpropOpGraphMap(); }
}  // namespace expander
}  // namespace mindspore
