/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/symbol_engine_optimizer.h"

#include <vector>
#include <memory>
#include "ir/pattern_matcher.h"
#include "ir/functor.h"
#include "ops/array_ops.h"
#include "ops/math_ops.h"
#include "include/common/utils/utils.h"
#include "backend/common/graph_kernel/symbol_engine/symbol.h"
#include "backend/common/graph_kernel/symbol_engine/utils.h"
#include "backend/common/graph_kernel/adapter/symbol_engine_builder.h"

namespace mindspore {
namespace opt {
namespace irpass {
SymbolEnginePtr GetSymbolEngine(const AnfNodePtr &node) {
  auto fg = node->func_graph();
  auto symbol_engine_attr = fg->get_attr(kAttrSymbolEngine);
  if (symbol_engine_attr == nullptr) {
    return nullptr;
  }
  auto symbol_engine = symbol_engine_attr->cast<SymbolEnginePtr>();
  MS_EXCEPTION_IF_NULL(symbol_engine);
  return symbol_engine;
}

bool SymbolEngineBuilder::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &) {
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    (void)graphkernel::BuildSymbolEngine(func_graph);
  } catch (std::exception &e) {
    MS_LOG(WARNING) << "Build symbol engine failed. message: " << e.what();
  }
  return false;
}

bool RemoveSymbolEngineAttr::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &) {
  auto nodes = TopoSort(func_graph->output(), SuccDeeperSimple, AlwaysInclude);
  for (auto &node : nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    FuncGraphPtr sub_fg;
    if (IsPrimitiveCNode(cnode, prim::kPrimPartial) && IsValueNode<FuncGraph>(cnode->input(1))) {
      sub_fg = GetValue<FuncGraphPtr>(GetValueNode(cnode->input(1)));
    } else {
      sub_fg = GetCNodeFuncGraph(cnode);
    }
    if (sub_fg != nullptr) {
      sub_fg->erase_flag(kAttrSymbolEngine);
    }
  }
  func_graph->erase_flag(kAttrSymbolEngine);
  return false;
}

AnfNodePtr ElimShapeCalcOnBroadcastArgsGrad::operator()(const OptimizerPtr &opt, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> dout;
  PatternNode<AnfNodePtr> shape_calc;
  PatternNode<AnfNodePtr> shape;
  PatternNode<AnfNodePtr> keepdims;
  PatternNode<AnfNodePtr> skipmode;
  PConstant idx0(node, false, 0, true);
  PConstant idx1(node, false, 1, true);
  MATCH_REPLACE_IF(
    node,
    PPrimitive(prim::kPrimReduceSum, dout,
               PPrimitive(prim::kPrimTensorToTuple, PPrimitive(prim::kPrimTupleGetItem, shape_calc, idx0)), keepdims,
               skipmode),
    dout, Check(opt, shape_calc.GetNode(node), kIndex1));
  MATCH_REPLACE_IF(
    node,
    PPrimitive(prim::kPrimReduceSum, dout,
               PPrimitive(prim::kPrimTensorToTuple, PPrimitive(prim::kPrimTupleGetItem, shape_calc, idx1)), keepdims,
               skipmode),
    dout, Check(opt, shape_calc.GetNode(node), kIndex2));
  return nullptr;
}

bool ElimShapeCalcOnBroadcastArgsGrad::Check(const OptimizerPtr &opt, const AnfNodePtr &shape_calc,
                                             size_t input_index) {
  auto mng = opt->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto &users = mng->node_users();
  auto symbol_engine = GetSymbolEngine(shape_calc);
  if (symbol_engine == nullptr) {
    return false;
  }

  auto shapecalc_node = shape_calc->cast<CNodePtr>();
  constexpr const size_t shapecalc_size = 3;
  if (shapecalc_node == nullptr || !IsPrimitiveCNode(shapecalc_node, prim::kPrimShapeCalc) ||
      shapecalc_node->size() != shapecalc_size) {
    return false;
  }
  auto input_node = shapecalc_node->input(input_index);
  auto shapecalc_functor = common::AnfAlgo::GetNodeAttr<ShapeCalcBaseFunctorPtr>(shapecalc_node, kAttrFunctor);
  MS_EXCEPTION_IF_NULL(shapecalc_functor);
  if (shapecalc_functor->name() != "ShapeCalc_BroadcastGradientArgs") {
    // only support the broadcast gradient condition
    return false;
  }
  auto fwd_unique_id = shapecalc_node->primal_attrs().find(kPrimalAttrForwardUniqueId);
  if (fwd_unique_id == shapecalc_node->primal_attrs().end()) {
    // only support bprop node
    return false;
  }
  AnfNodePtr fwd_node = nullptr;
  for (auto &user : users[input_node]) {
    auto user_cnode = user.first->cast<CNodePtr>();
    if (user_cnode == nullptr) {
      continue;
    }
    if (auto uniq_id = user_cnode->primal_attrs().find(kPrimalAttrUniqueId);
        uniq_id != user_cnode->primal_attrs().end()) {
      if (*uniq_id->second == *fwd_unique_id->second) {
        fwd_node = user.first;
        break;
      }
    }
  }
  if (fwd_node == nullptr) {
    return false;
  }

  auto input_shape = symbol_engine->QuerySymbolicShape(input_node);
  auto output_shape = symbol_engine->QuerySymbolicShape(fwd_node);
  auto ret = CheckSymbolEqual(input_shape, output_shape, GetValue<size_t>(shapecalc_functor->ToValue()));
  if (ret) {
    MS_LOG(INFO) << "For " << shape_calc->DebugString() << " (" << shape_calc->fullname_with_scope() << ")"
                 << " generated by BroadcastGradientArgs. The gradient for input " << input_index
                 << " is unnecessary, which can be eliminated. grad symbol: " << input_shape->ToString()
                 << ". out symbol: " << output_shape->ToString();
  }
  return ret;
}

bool ElimShapeCalcOnBroadcastArgsGrad::CheckSymbolEqual(const graphkernel::symbol::ListSymbolPtr &input_shape,
                                                        const graphkernel::symbol::ListSymbolPtr &output_shape,
                                                        size_t shift) {
  if (input_shape == nullptr && output_shape == nullptr) {
    return false;
  }
  if (input_shape->size() < output_shape->size()) {
    return false;
  }
  for (size_t i = input_shape->size(); i > shift; i--) {
    auto inp = input_shape->symbols()[input_shape->size() - i];
    if (i <= output_shape->size() && !inp->EqualsTo(output_shape->symbols()[output_shape->size() - i])) {
      return false;
    }
  }
  return true;
}

AnfNodePtr ElimNotEffectiveNode::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  static const PrimitiveSet supports_op = {prim::kPrimReshape, prim::kPrimReduceSum, prim::kPrimReduceMax,
                                           prim::kPrimReduceMin};
  if (!IsOneOfPrimitiveCNode(node, supports_op)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto symbol_engine = GetSymbolEngine(node);
  if (symbol_engine == nullptr) {
    return nullptr;
  }
  auto input_node = node->cast<CNodePtr>()->input(1);
  auto input_shape = symbol_engine->QuerySymbolicShape(input_node);
  auto output_shape = symbol_engine->QuerySymbolicShape(node);
  if (input_shape != nullptr && input_shape->EqualsTo(output_shape)) {
    MS_LOG(INFO) << "For node " << node->DebugString() << " (" << node->fullname_with_scope()
                 << "), the input shape and output shape is same, which can be eliminated.";
    return input_node;
  }
  return nullptr;
}

AnfNodePtr OptReshape::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> input;
  PatternNode<AnfNodePtr> shape;
  ShapeVector shape_vec;
  auto symbol_engine = GetSymbolEngine(node);
  if (symbol_engine == nullptr) {
    return nullptr;
  }

  auto MakeReshape = [&shape_vec, &node, &symbol_engine]() -> AnfNodePtr {
    auto shape_val = MakeValue(shape_vec);
    auto shape = NewValueNode(shape_val);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(INFO) << "For node " << cnode->DebugString()
                 << ", the symbolic value of \"shape\" is static or has only one dynamic dim, "
                 << "replace the \"shape\" to a value node: " << shape_val->ToString();
    shape->set_abstract(shape_val->ToAbstract());
    auto reshape = NewCNode({cnode->input(0), cnode->input(1), shape}, node->func_graph());
    reshape->set_abstract(node->abstract());
    symbol_engine->BuildCNodeSymbol(reshape, false);
    return reshape;
  };
  auto CheckShape = [&shape_vec, &symbol_engine](const AnfNodePtr &shape) {
    if (!shape->isa<CNode>()) {
      return false;
    }
    auto symshape = symbol_engine->QuerySymbolicValue(shape);
    if (symshape == nullptr || !symshape->HasData()) {
      return false;
    }
    shape_vec = graphkernel::symbol::ToShape(symshape);
    return std::count(shape_vec.cbegin(), shape_vec.cend(), abstract::Shape::kShapeDimAny) <= 1;
  };
  MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimReshape, input, shape), MakeReshape,
                          CheckShape(shape.GetNode(node)));
  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
