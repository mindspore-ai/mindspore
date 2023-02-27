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

#include "frontend/optimizer/expander.h"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include "mindspore/core/utils/anf_utils.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "pybind_api/ir/primitive_py.h"
#include "common/graph_kernel/adapter/expander.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
namespace {
const std::map<std::string, std::vector<std::string>> op2attrs = {
  {prim::kPrimBroadcastTo->name(), {kAttrShape}},
  {prim::kPrimReduceMax->name(), {kAttrKeepDims}},
  {prim::kPrimReduceMin->name(), {kAttrKeepDims}},
  {prim::kPrimReduceSum->name(), {kAttrKeepDims}},
  {prim::kPrimMatMul->name(), {kTransposeA, kTransposeB}},
  {prim::kPrimConcat->name(), {kAttrAxis}},
  {prim::kPrimSqueeze->name(), {kAttrAxis}},
  {prim::kPrimOneHot->name(), {kAttrAxis}},
  {prim::kPrimSoftmax->name(), {kAttrAxis}},
  {prim::kPrimSplit->name(), {kAttrAxis}},
  {prim::kPrimLayerNorm->name(), {kAttrBeginNormAxis, kAttrBeginParamsAxis, kAttrEpsilon}},
  {prim::kPrimStridedSlice->name(),
   {kAttrBeginMask, kAttrEndMask, kAttrEllipsisMask, kAttrNewAxisMask, kAttrShrinkAxisMask}},
  {prim::kPrimLayerNormGrad->name(), {kAttrBeginNormAxis, kAttrBeginParamsAxis}},
  {prim::kPrimLayerNormGradGrad->name(), {kAttrBeginNormAxis, kAttrBeginParamsAxis}},
  {prim::kPrimBiasAdd->name(), {kAttrDataFormat}},
  {prim::kPrimBiasAddGrad->name(), {kAttrDataFormat}},
  {prim::kPrimStack->name(), {kAttrAxis}},
  {prim::kPrimBatchMatMul->name(), {kTransposeA, kTransposeB}}};
}  // namespace

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
      if (primitive == nullptr || primitive->isa<PrimitivePy>()) {
        continue;
      }
      if (abstract::GetFrontendPrimitiveInferImpl(primitive).has_value()) {
        continue;
      }
      if (primitive->isa<prim::DoSignaturePrimitive>()) {
        continue;
      }
      parallel::OperatorAttrs attrs;
      const auto iter = op2attrs.find(primitive->name());
      if (iter != op2attrs.end()) {
        for (auto &attr : iter->second) {
          if (primitive->HasAttr(attr)) {
            (void)attrs.emplace_back(std::pair{attr, primitive->GetAttr(attr)});
          } else {
            MS_LOG(WARNING) << primitive->name() << " op do not have attr: " << attr;
            return false;
          }
        }
      }
      auto new_prim = parallel::CreateOpInstance(attrs, primitive->name(), "");
      (void)new_prim->cast_ptr<Primitive>()->SetAttrs(primitive->attrs());
      AnfNodePtrList inputs = {NewValueNode(new_prim)};
      auto cnode = dyn_cast_ptr<CNode>(node);
      (void)inputs.insert(inputs.cend(), cnode->inputs().cbegin() + 1, cnode->inputs().cend());
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

using graphkernel::ExpanderDecorator;
using graphkernel::ExpanderPtr;
class PrimToPrimPyDecorator : public ExpanderDecorator {
 public:
  explicit PrimToPrimPyDecorator(const ExpanderPtr &decorated) : ExpanderDecorator(decorated) {}
  ~PrimToPrimPyDecorator() override = default;
  static ExpanderPtr Creator(const ExpanderPtr &decorated) {
    return std::static_pointer_cast<Expander>(std::make_shared<PrimToPrimPyDecorator>(decorated));
  }
  AnfNodePtr Run(const AnfNodePtr &node) override {
    auto new_node = decorated_->Run(node);
    if (new_node == nullptr) {
      return nullptr;
    }
    auto new_cnode = dyn_cast<CNode>(new_node);
    auto expand_fg = GetCNodeFuncGraph(new_cnode);
    if (!ConvertPrimToPrimPy(expand_fg)) {
      return nullptr;
    }
    new_cnode->set_input(0, NewValueNode(expand_fg));
    return new_cnode;
  }
};

AnfNodePtr TryExpandCNodeFE(const AnfNodePtr &node) {
  if (!graphkernel::CanExpandFallback(node)) {
    return nullptr;
  }
  using graphkernel::InputToAttrDeco;
  auto primitive = GetCNodePrimitive(node);
  if (primitive == nullptr) {
    return nullptr;
  }
  auto expander = graphkernel::GetExpander(node);
  std::map<std::string, graphkernel::ExpanderCreatorFuncList> creators = {
    {prim::kPrimExpandDims->name(), {InputToAttrDeco::Creator}},
    {prim::kPrimReshape->name(), {InputToAttrDeco::Creator}},
    {prim::kPrimReduceMean->name(), {InputToAttrDeco::Creator}},
    {prim::kPrimGather->name(), {InputToAttrDeco::Creator}},
    {kTileOpName, {InputToAttrDeco::Creator}},
    {kSliceOpName, {InputToAttrDeco::Creator}},
  };
  const auto &iter = creators.find(GetCNodePrimitive(node)->name());
  if (iter != creators.end()) {
    expander = graphkernel::WrapExpander(expander, iter->second);
  }
  expander = graphkernel::AttrToInputDeco::Creator(expander);
  expander = PrimToPrimPyDecorator::Creator(expander);
  auto new_node = expander->Run(node);
  auto expand_fg = GetCNodeFuncGraph(new_node);
  if (expand_fg == nullptr) {
    return nullptr;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("expand_fe_" + GetCNodeFuncName(node->cast<CNodePtr>()) + ".ir", expand_fg);
  }
#endif
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
