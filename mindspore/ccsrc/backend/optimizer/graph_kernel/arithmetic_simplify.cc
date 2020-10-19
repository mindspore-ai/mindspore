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
#include "backend/optimizer/graph_kernel/arithmetic_simplify.h"
#include <list>
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/pattern_matcher.h"
#include "frontend/operator/ops.h"
#include "utils/convert_utils.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {

AnfNodePtr NewCNodeWithInfo(const AnfNodePtrList &inputs, const AnfNodePtr &ori_node) {
  auto func_graph = ori_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto new_cnode = func_graph->NewCNode(inputs);
  new_cnode->set_abstract(ori_node->abstract());
  new_cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
  if (func_graph->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
    ResetKernelInfo(new_cnode, AKG_KERNEL);
  } else {
    ResetKernelInfo(new_cnode, UNKNOWN_KERNEL_TYPE);
  }
  func_graph->AddNode(new_cnode);
  return new_cnode;
}

AnfNodePtr SimplifyAdd(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimTensorAdd)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x, y, z;
  PConstant<AnfNodePtr> zero_num(node, false, 0);
  PConstant<AnfNodePtr> zero_scalar(node, false, 0, true);
  PConstant<AnfNodePtr> any_const(node);
  PConstant<AnfNodePtr> any_const_2(node);

  auto add_distri_lambda = [&node, &x, &y, &any_const]() -> AnfNodePtr {
    auto node_tmp = NewCNodeWithInfo({NewValueNode(prim::kPrimTensorAdd), x.GetNode(node), y.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), node_tmp, any_const.GetNode(node)}, node);
    return new_cnode;
  };
  auto add_union_lambda = [&node, &x, &any_const, &any_const_2]() -> AnfNodePtr {
    auto new_rhs = any_const.AddByPatternConst(any_const_2, x.GetNode(node));
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimTensorAdd), x.GetNode(node), new_rhs}, node);
    return new_cnode;
  };
  // A + 0 = A
  MATCH_REPLACE(node, x + zero_num, x);
  // A*C + B*C = (A + B)*C
  MATCH_REPLACE_LAMBDA_IF(node, (x * any_const) + (y * any_const_2), add_distri_lambda,
                          PIsEqual<AnfNodePtr>()(any_const.GetNode(node), any_const_2.GetNode(node)));
  // (A + C1) + C2 = A + (C1 + C2)
  MATCH_REPLACE_LAMBDA(node, (x + any_const) + any_const_2, add_union_lambda);
  // A + (-A) = 0
  MATCH_REPLACE_IF(node, x + PUnaryOperation(prim::kPrimNeg, y), zero_scalar.NewValue(),
                   PIsEqual<AnfNodePtr>()(x.GetNode(node), y.GetNode(node)));
  return nullptr;
}

AnfNodePtr SimplifySub(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimSub)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x;
  PConstant<AnfNodePtr> zero_num(node, false, 0);
  PConstant<AnfNodePtr> any_const(node);
  auto sub_toadd_lambda = [&node, &x, &any_const]() -> AnfNodePtr {
    auto new_rhs = any_const.ValueNodeWithOprations(prim::kPrimNeg);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimTensorAdd), x.GetNode(node), new_rhs}, node);
    return new_cnode;
  };
  // A - 0 = A
  MATCH_REPLACE(node, x - zero_num, x);
  // A - const = A + (-const)
  MATCH_REPLACE_LAMBDA(node, x - any_const, sub_toadd_lambda);
  return nullptr;
}

AnfNodePtr SimplifyNeg(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimNeg)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x;
  MATCH_REPLACE(node, PUnaryOperation(prim::kPrimNeg, PUnaryOperation(prim::kPrimNeg, x)), x);
  return nullptr;
}

AnfNodePtr SimplifyLog(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimLog)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x, y;
  auto ln_front_lambda = [&node, &x, &y]() -> AnfNodePtr {
    auto node_tmp = NewCNodeWithInfo({NewValueNode(prim::kPrimAbs), x.GetNode(node)}, node);
    auto node_tmp_2 = NewCNodeWithInfo({NewValueNode(prim::kPrimLog), node_tmp}, node);
    auto new_cnode =
      NewCNodeWithInfo({NewValueNode(prim::kPrimMul), y.GetNode(node), node_tmp_2}, node->cast<CNodePtr>()->input(1));
    return new_cnode;
  };
  auto sqrt_ln_lambda = [&node, &x]() -> AnfNodePtr {
    auto node_tmp = NewCNodeWithInfo({NewValueNode(prim::kPrimLog), x.GetNode(node)}, node);
    auto value = MakeValue(std::make_shared<FP32Imm>(0.5));
    auto tensor_ptr = mindspore::ScalarToTensor(value->cast<ScalarPtr>());
    auto value_node_ptr = MakeValueNode(std::make_shared<ValueNode>(tensor_ptr));
    value_node_ptr->set_abstract(node->abstract());
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), value_node_ptr, node_tmp}, node);
    return new_cnode;
  };
  auto rsqrt_ln_lambda = [&node, &x]() -> AnfNodePtr {
    auto node_tmp = NewCNodeWithInfo({NewValueNode(prim::kPrimLog), x.GetNode(node)}, node);
    auto value = MakeValue(std::make_shared<FP32Imm>(-0.5));
    auto tensor_ptr = mindspore::ScalarToTensor(value->cast<ScalarPtr>());
    auto value_node_ptr = MakeValueNode(std::make_shared<ValueNode>(tensor_ptr));
    value_node_ptr->set_abstract(node->abstract());
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), value_node_ptr, node_tmp}, node);
    return new_cnode;
  };
  // Ln(Exp(A)) = A
  MATCH_REPLACE(node, PUnaryOperation(prim::kPrimLog, PUnaryOperation(prim::kPrimExp, x)), x);
  // Ln(Pow(A,B)) = B*Ln(Abs(A))
  MATCH_REPLACE_LAMBDA(node, PUnaryOperation(prim::kPrimLog, PBinOperation(prim::kPrimPow, x, y, false)),
                       ln_front_lambda);
  // Ln(Sqrt(A)) = 0.5*Ln(A)
  MATCH_REPLACE_LAMBDA(node, PUnaryOperation(prim::kPrimLog, PUnaryOperation(prim::kPrimSqrt, x)), sqrt_ln_lambda);
  // Ln(Rqrt(A)) = -0.5*Ln(A)
  MATCH_REPLACE_LAMBDA(node, PUnaryOperation(prim::kPrimLog, PUnaryOperation(prim::kPrimRsqrt, x)), rsqrt_ln_lambda);
  return nullptr;
}

AnfNodePtr SimplifyPow(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimPow)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x, y;
  PConstant<AnfNodePtr> zero_num(node, false, 0);
  PConstant<AnfNodePtr> one_const(node, false, 1);
  PConstant<AnfNodePtr> two_const(node, false, 2);
  PConstant<AnfNodePtr> negone_const(node, false, -1);
  auto pow_zero_lambda = [&node]() -> AnfNodePtr {
    auto value = MakeValue(std::make_shared<FP32Imm>(1));
    auto tensor_ptr = mindspore::ScalarToTensor(value->cast<ScalarPtr>());
    auto value_node_ptr = MakeValueNode(std::make_shared<ValueNode>(tensor_ptr));
    value_node_ptr->set_abstract(node->abstract());
    return value_node_ptr;
  };
  auto exp_power_lambda = [&node, &x, &y]() -> AnfNodePtr {
    auto node_tmp = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), y.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimExp), node_tmp}, node->cast<CNodePtr>()->input(1));
    return new_cnode;
  };
  auto squre_power_lambda = [&node, &x]() -> AnfNodePtr {
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), x.GetNode(node)}, node);
    return new_cnode;
  };
  auto r_power_lambda = [&node, &x]() -> AnfNodePtr {
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimReciprocal), x.GetNode(node)}, node);
    return new_cnode;
  };
  // Pow(A, 0) = 1
  MATCH_REPLACE_LAMBDA(node, PBinOperation(prim::kPrimPow, x, zero_num, false), pow_zero_lambda);
  // Pow(A, 1) = A
  MATCH_REPLACE(node, PBinOperation(prim::kPrimPow, x, one_const, false), x);
  // Pow(exp(A),B) = exp(A*B)
  MATCH_REPLACE_LAMBDA(node, PBinOperation(prim::kPrimPow, PUnaryOperation(prim::kPrimExp, x), y, false),
                       exp_power_lambda);
  // Pow(A, 2) = A*A
  MATCH_REPLACE_LAMBDA(node, PBinOperation(prim::kPrimPow, x, two_const, false), squre_power_lambda);
  // Pow(A, -1) = 1/A
  MATCH_REPLACE_LAMBDA(node, PBinOperation(prim::kPrimPow, x, negone_const, false), r_power_lambda);
  return nullptr;
}

AnfNodePtr SimplifySqrt(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimSqrt)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x, y;
  auto mul_sqrt_lambda = [&node, &x]() -> AnfNodePtr {
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimAbs), x.GetNode(node)}, node);
    return new_cnode;
  };
  auto square_sqrt_lambda = [&node, &x]() -> AnfNodePtr {
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimAbs), x.GetNode(node)}, node);
    return new_cnode;
  };
  // pattern matcher cannot distinguish the same PatternNode in CaptureNode, so it needs to add judgment
  // Sqrt(A*A) = |A|
  MATCH_REPLACE_LAMBDA_IF(node, PUnaryOperation(prim::kPrimSqrt, PBinOperation(prim::kPrimMul, x, y)), mul_sqrt_lambda,
                          PIsEqual<AnfNodePtr>()(x.GetNode(node), y.GetNode(node)));
  // Sqrt(Square(A)) = |A|
  MATCH_REPLACE_LAMBDA(node, PUnaryOperation(prim::kPrimSqrt, PUnaryOperation(prim::kPrimSquare, x)),
                       square_sqrt_lambda);
  return nullptr;
}

AnfNodePtr SimplifyRsqrt(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimRsqrt)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x;
  PConstant<AnfNodePtr> num_one(node, false, 1, true);
  PConstant<AnfNodePtr> num_negtwo(node, false, -2, true);
  auto power_rsqrt_lambda = [&node, &x]() -> AnfNodePtr {
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimAbs), x.GetNode(node)}, node);
    return new_cnode;
  };
  auto div_rsqrt_lambda = [&node, &x]() -> AnfNodePtr {
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimSqrt), x.GetNode(node)}, node);
    return new_cnode;
  };
  // Rsqrt(Pow(A, -2)) = |A|
  MATCH_REPLACE_LAMBDA(node, PUnaryOperation(prim::kPrimRsqrt, PBinOperation(prim::kPrimPow, x, num_negtwo, false)),
                       power_rsqrt_lambda);
  // Rsqrt(Divide(1, A)) = Sqrt(A)
  MATCH_REPLACE_LAMBDA(node, PUnaryOperation(prim::kPrimRsqrt, PBinOperation(prim::kPrimRealDiv, num_one, x, false)),
                       div_rsqrt_lambda);
  return nullptr;
}

AnfNodePtr SimplifySelect(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimSelect)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x, y, z;
  // select(x,y,y) = y
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimSelect, x, y, z), y,
                   PIsEqual<AnfNodePtr>()(y.GetNode(node), z.GetNode(node)));
  return nullptr;
}

AnfNodePtr SimplifyMul(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimMul)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x, y;
  PConstant<AnfNodePtr> const_1(node), const_2(node);

  auto const_dup_lambda = [&node, &x, &y, &const_1, &const_2]() -> AnfNodePtr {
    auto new_lhs = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), y.GetNode(node)}, node);
    auto new_rhs = const_1.MulByPatternConst(const_2, x.GetNode(node));
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), new_lhs, new_rhs}, node);
    return new_cnode;
  };
  auto exp_merge_lambda = [&node, &x, &y]() -> AnfNodePtr {
    auto node_tmp = NewCNodeWithInfo({NewValueNode(prim::kPrimTensorAdd), x.GetNode(node), y.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimExp), node_tmp}, node);
    return new_cnode;
  };
  auto sqrt_merge_lambda = [&node, &x, &y]() -> AnfNodePtr {
    auto node_tmp = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), y.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimSqrt), node_tmp}, node);
    return new_cnode;
  };
  auto rsqrt_merge_lambda = [&node, &x]() -> AnfNodePtr {
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimReciprocal), x.GetNode(node)}, node);
    return new_cnode;
  };
  auto rsqrt_merge_lambda_2 = [&node, &x, &y]() -> AnfNodePtr {
    auto node_tmp = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), y.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimRsqrt), node_tmp}, node);
    return new_cnode;
  };
  auto rsqrt_merge_lambda_3 = [&node, &x]() -> AnfNodePtr {
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimSqrt), x.GetNode(node)}, node);
    return new_cnode;
  };
  // (x*C1)*(y*C2) ==> (x*y)*(C1*C2)
  MATCH_REPLACE_LAMBDA(node, (const_1 * x) * (const_2 * y), const_dup_lambda);
  // exp(x)*exp(y) ==> exp(x+y)
  MATCH_REPLACE_LAMBDA(node, PUnaryOperation(prim::kPrimExp, x) * PUnaryOperation(prim::kPrimExp, y), exp_merge_lambda);
  // sqrt(x)*sqrt(x) ==> x
  MATCH_REPLACE_IF(node, PUnaryOperation(prim::kPrimSqrt, x) * PUnaryOperation(prim::kPrimSqrt, y), x,
                   PIsEqual<AnfNodePtr>()(x.GetNode(node), y.GetNode(node)));
  // sqrt(x)*sqrt(y) ==> sqrt(x*y)
  MATCH_REPLACE_LAMBDA_IF(node, PUnaryOperation(prim::kPrimSqrt, x) * PUnaryOperation(prim::kPrimSqrt, y),
                          sqrt_merge_lambda, !PIsEqual<AnfNodePtr>()(x.GetNode(node), y.GetNode(node)));
  // rsqrt(x)*rsqrt(x) ==> 1/x
  MATCH_REPLACE_LAMBDA_IF(node, PUnaryOperation(prim::kPrimRsqrt, x) * PUnaryOperation(prim::kPrimRsqrt, y),
                          rsqrt_merge_lambda, PIsEqual<AnfNodePtr>()(x.GetNode(node), y.GetNode(node)));
  // rsqrt(x)*rsqrt(y) ==> rsqrt(x*y)
  MATCH_REPLACE_LAMBDA_IF(node, PUnaryOperation(prim::kPrimRsqrt, x) * PUnaryOperation(prim::kPrimRsqrt, y),
                          rsqrt_merge_lambda_2, !PIsEqual<AnfNodePtr>()(x.GetNode(node), y.GetNode(node)));
  // x*rsqrt(x) ==> sqrt(x)
  MATCH_REPLACE_LAMBDA_IF(node, x * PUnaryOperation(prim::kPrimRsqrt, y), rsqrt_merge_lambda_3,
                          PIsEqual<AnfNodePtr>()(x.GetNode(node), y.GetNode(node)));
  return nullptr;
}

AnfNodePtr SimplifyDiv(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimRealDiv)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x, y, u, v;
  PConstant<AnfNodePtr> const_1(node), const_2(node);
  PConstant<AnfNodePtr> const_one(node, false, 1);
  PConstant<AnfNodePtr> const_one_scalar(node, false, 1, true);

  auto div_exp_lambda_1 = [&node, &x, &y]() -> AnfNodePtr {
    auto node_tmp = NewCNodeWithInfo({NewValueNode(prim::kPrimSub), x.GetNode(node), y.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimExp), node_tmp}, node);
    return new_cnode;
  };
  auto div_exp_lambda_2 = [&node, &x, &y]() -> AnfNodePtr {
    auto node_neg = NewCNodeWithInfo({NewValueNode(prim::kPrimNeg), y.GetNode(node)}, node);
    auto node_exp = NewCNodeWithInfo({NewValueNode(prim::kPrimExp), node_neg}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), node_exp}, node);
    return new_cnode;
  };
  auto div_pow_const = [&node, &x, &y, &const_1]() -> AnfNodePtr {
    auto new_const = const_1.ValueNodeWithOprations(prim::kPrimNeg);
    auto new_rhs = NewCNodeWithInfo({NewValueNode(prim::kPrimPow), y.GetNode(node), new_const}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), new_rhs}, node);
    return new_cnode;
  };
  auto div_sqrt_lambda_1 = [&node, &x]() -> AnfNodePtr {
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimSqrt), x.GetNode(node)}, node);
    return new_cnode;
  };
  auto div_sqrt_lambda_2 = [&node, &x, &y]() -> AnfNodePtr {
    auto node_rsqrt = NewCNodeWithInfo({NewValueNode(prim::kPrimRsqrt), y.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), node_rsqrt}, node);
    return new_cnode;
  };
  auto div_const = [&node, &x, &const_1]() -> AnfNodePtr {
    auto new_const = const_1.ValueNodeWithOprations(prim::kPrimReciprocal);
    if (new_const == nullptr) {
      return nullptr;
    }
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), new_const}, node);
    return new_cnode;
  };
  auto div_rsqrt_lambda = [&node, &x, &y]() -> AnfNodePtr {
    auto node_rsqrt = NewCNodeWithInfo({NewValueNode(prim::kPrimSqrt), y.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), node_rsqrt}, node);
    return new_cnode;
  };
  auto div_lambda_1 = [&node, &x, &y, &u, &v]() -> AnfNodePtr {
    auto new_lhs = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), v.GetNode(node)}, node);
    auto new_rhs = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), y.GetNode(node), u.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimRealDiv), new_lhs, new_rhs}, node);
    return new_cnode;
  };
  auto div_lambda_2 = [&node, &x, &y, &u]() -> AnfNodePtr {
    auto new_rhs = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), y.GetNode(node), u.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimRealDiv), x.GetNode(node), new_rhs}, node);
    return new_cnode;
  };
  auto div_lambda_3 = [&node, &x, &u, &v]() -> AnfNodePtr {
    auto new_lhs = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), x.GetNode(node), v.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimRealDiv), new_lhs, u.GetNode(node)}, node);
    return new_cnode;
  };
  // x/1 ==> x
  MATCH_REPLACE(node, PBinOperation(prim::kPrimScalarDiv, x, const_one_scalar, false), x);
  MATCH_REPLACE(node, x / const_one, x);
  // e^x/e^y ==> e^(x-y)
  MATCH_REPLACE_LAMBDA(node, PUnaryOperation(prim::kPrimExp, x) / PUnaryOperation(prim::kPrimExp, y), div_exp_lambda_1);
  // x / e^y ==> x * e^(-y)
  MATCH_REPLACE_LAMBDA(node, x / PUnaryOperation(prim::kPrimExp, y), div_exp_lambda_2);
  // x / y^const ==> x * y^(-const)
  MATCH_REPLACE_LAMBDA(node, x / PBinOperation(prim::kPrimPow, y, const_1), div_pow_const);
  // x / sqrt(x) ==> sqrt(x)
  MATCH_REPLACE_LAMBDA_IF(node, x / PUnaryOperation(prim::kPrimSqrt, y), div_sqrt_lambda_1,
                          PIsEqual<AnfNodePtr>()(x.GetNode(node), y.GetNode(node)));
  // x / sqrt(y) ==> x * rsqrt(y)
  MATCH_REPLACE_LAMBDA_IF(node, x / PUnaryOperation(prim::kPrimSqrt, y), div_sqrt_lambda_2,
                          !PIsEqual<AnfNodePtr>()(x.GetNode(node), y.GetNode(node)));
  // x / rsqrt(y) ==> x * sqrt(y)
  MATCH_REPLACE_LAMBDA(node, x / PUnaryOperation(prim::kPrimRsqrt, y), div_rsqrt_lambda);
  // // x / const ==> x * (1/const)
  MATCH_REPLACE_LAMBDA(node, x / const_1, div_const);
  // (x/y) / (u/v) ==> (x*v) / (y*u)
  MATCH_REPLACE_LAMBDA(node, (x / y) / (u / v), div_lambda_1);
  // (x/y) / u ==> x / (y*u)
  MATCH_REPLACE_LAMBDA(node, (x / y) / u, div_lambda_2);
  // x / (u/v) ==> (x*v) / u
  MATCH_REPLACE_LAMBDA(node, x / (u / v), div_lambda_3);
  return nullptr;
}

#define PERFORM_REPLACE(OldNode, NewNode, Graph, FLAG) \
  if ((NewNode) != nullptr) {                          \
    (Graph)->manager()->Replace((OldNode), (NewNode)); \
    (FLAG) = true;                                     \
  }

AnfNodePtr TrySimplify(const AnfNodePtr &node) {
  std::list<std::function<AnfNodePtr(AnfNodePtr)>> SimplifyFuncList = {
    SimplifyAdd, SimplifyDiv,   SimplifyLog,    SimplifyMul,  SimplifyNeg,
    SimplifyPow, SimplifyRsqrt, SimplifySelect, SimplifySqrt, SimplifySub};
  for (auto f : SimplifyFuncList) {
    auto ret = f(node);
    if (ret != nullptr) {
      return ret;
    }
  }
  return nullptr;
}

void InlineSubgraph(const CNodePtr &kernel_node, const FuncGraphPtr &sub_graph, const FuncGraphPtr &main_func_graph) {
  AnfNodePtrList ins;
  ins.insert(ins.end(), kernel_node->inputs().begin() + 1, kernel_node->inputs().end());
  auto out = InlineClone(sub_graph, main_func_graph, ins, kernel_node->input(0)->scope());
  auto mng = main_func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  mng->Replace(kernel_node, out);
}

CNodePtr AddIdentityToEmptyPath(const AnfNodePtr &node, const FuncGraphPtr &sub_graph) {
  if (node->isa<Parameter>() || node->isa<ValueNode>()) {
    auto identity_node = sub_graph->NewCNode({NewValueNode(prim::kPrimIdentity), node});
    identity_node->set_abstract(node->abstract());
    sub_graph->AddNode(identity_node);
    return identity_node;
  }
  return nullptr;
}

// If the return of the subgraph contains input Parameters or a new ValueNode,
// add identity mapping to them to avoid dealing with empty paths in subgraphs,
// then inline the subgraph into the main graph
bool CheckAndInlineEmptyGraph(const AnfNodePtr &node, const FuncGraphPtr &main_func_graph) {
  if (!AnfAlgo::IsGraphKernel(node)) {
    MS_LOG(ERROR) << node->ToString() << "is not a graph kernel\n";
    return false;
  }
  auto kernel_node = node->cast<CNodePtr>();
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(kernel_node);
  auto mng = sub_graph->manager();
  if (mng == nullptr) {
    mng = Manage(sub_graph, false);
    sub_graph->set_manager(mng);
  }
  auto sub_return = sub_graph->get_return();
  auto pred_node_of_return = sub_return->input(1);
  bool ret = false;
  if (!IsPrimitiveCNode(pred_node_of_return, prim::kPrimMakeTuple)) {  // Single output
    auto new_cnode = AddIdentityToEmptyPath(pred_node_of_return, sub_graph);
    if (new_cnode != nullptr) {
      sub_return->set_input(1, new_cnode);
      ret = true;
    }
  } else {  // Multiple output
    auto maketuple_node = pred_node_of_return->cast<CNodePtr>();
    size_t size_ret = maketuple_node->inputs().size();
    size_t empty_path_cnt = 0;
    for (size_t i = 1; i < size_ret; i++) {
      auto tmp_node = maketuple_node->input(i);
      auto new_cnode = AddIdentityToEmptyPath(tmp_node, sub_graph);
      if (new_cnode != nullptr) {
        maketuple_node->set_input(i, new_cnode);
        empty_path_cnt++;
      }
    }
    if (empty_path_cnt == 0) {  // normal subgraph
      return false;
    } else if (empty_path_cnt < size_ret - 1) {
      MS_EXCEPTION(NotSupportError);
      return false;
    } else {  // empty subgraph
      ret = true;
    }
  }
  if (ret) {
    InlineSubgraph(kernel_node, sub_graph, main_func_graph);
  }
  return ret;
}

AnfNodePtr MatchIdentity(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimIdentity)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x;
  MATCH_REPLACE(node, PUnaryOperation(prim::kPrimIdentity, x), x);
  return nullptr;
}

void EliminateEmptyGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, false);
    func_graph->set_manager(mng);
  }
  bool empty_graph = false;
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto cnode : cnodes) {
    if (AnfAlgo::IsGraphKernel(cnode)) {
      empty_graph = empty_graph || CheckAndInlineEmptyGraph(cnode, func_graph);
    }
  }
  if (empty_graph) {
    cnodes = func_graph->GetOrderedCnodes();
    for (auto cnode : cnodes) {
      auto node = cnode->cast<AnfNodePtr>();
      auto new_node = MatchIdentity(node);
      if (new_node != nullptr) {
        mng->Replace(node, new_node);
      }
    }
  }
}

bool ArithmeticSimplify::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  bool replaced = false;
  for (auto node : func_graph->GetOrderedCnodes()) {
    if (AnfAlgo::IsGraphKernel(node)) {
      auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
      auto mng_sub = sub_graph->manager();
      if (mng_sub == nullptr) {
        mng_sub = Manage(sub_graph, false);
        sub_graph->set_manager(mng_sub);
      }
      bool sub_graph_changed = false;
      for (auto node_sub : sub_graph->GetOrderedCnodes()) {
        auto new_node = TrySimplify(node_sub);
        if (new_node != nullptr) {
          sub_graph_changed = true;
          PERFORM_REPLACE(node_sub->cast<AnfNodePtr>(), new_node, sub_graph, replaced);
        }
      }
      if (sub_graph_changed) {
        ResetKernelInfo(node, AKG_KERNEL);
      }
    } else {
      auto new_node = TrySimplify(node);
      PERFORM_REPLACE(node->cast<AnfNodePtr>(), new_node, func_graph, replaced);
    }
  }
  EliminateEmptyGraph(func_graph);
  return replaced;
}
}  // namespace opt
}  // namespace mindspore
