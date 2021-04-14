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

#include <algorithm>
#include <list>
#include <utility>
#include <vector>
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "frontend/operator/ops.h"
#include "ir/pattern_matcher.h"
#include "utils/convert_utils.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
AnfNodePtr NewCNodeWithInfo(const AnfNodePtrList &inputs, const AnfNodePtr &ori_node) {
  auto func_graph = ori_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  TraceManager::DebugTrace(std::make_shared<TraceOpt>(ori_node->debug_info()));
  auto new_cnode = func_graph->NewCNode(inputs);
  TraceManager::EndTrace();
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
  if (!IsPrimitiveCNode(node, prim::kPrimAdd)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x, y, z;
  PConstant<AnfNodePtr> zero_num(node, false, 0);
  PConstant<AnfNodePtr> zero_scalar(node, false, 0, true);
  PConstant<AnfNodePtr> any_const(node);
  PConstant<AnfNodePtr> any_const_2(node);

  auto add_distri_lambda = [&node, &x, &y, &any_const]() -> AnfNodePtr {
    auto node_tmp = NewCNodeWithInfo({NewValueNode(prim::kPrimAdd), x.GetNode(node), y.GetNode(node)}, node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimMul), node_tmp, any_const.GetNode(node)}, node);
    return new_cnode;
  };
  auto add_union_lambda = [&node, &x, &any_const, &any_const_2]() -> AnfNodePtr {
    auto new_rhs = any_const.AddByPatternConst(any_const_2, x.GetNode(node));
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimAdd), x.GetNode(node), new_rhs}, node);
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
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimAdd), x.GetNode(node), new_rhs}, node);
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

#define PERFORM_REPLACE(OldNode, NewNode, Graph, FLAG) \
  if ((NewNode) != nullptr) {                          \
    (Graph)->manager()->Replace((OldNode), (NewNode)); \
    (FLAG) = true;                                     \
  }

bool TryTransposeToReshape(const AnfNodePtr &node) {
  auto perm = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, "perm");
  auto ori_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  std::vector<int64_t> remove_one_perm;
  for (auto idx : perm) {
    if (idx < 0 || IntToSize(idx) >= ori_shape.size()) {
      MS_EXCEPTION(ValueError);
      return false;
    }
    if (ori_shape[idx] != 1) {
      remove_one_perm.emplace_back(idx);
    }
  }
  if (remove_one_perm.size() < 2) {
    return true;
  }
  for (size_t idx = 1; idx < remove_one_perm.size(); idx++) {
    if (remove_one_perm[idx] < remove_one_perm[idx - 1]) {
      return false;
    }
  }
  return true;
}

AnfNodePtr SimplifyTranspose(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimTranspose)) {
    return nullptr;
  }
  if (TryTransposeToReshape(node)) {
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimReshape), node->cast<CNodePtr>()->input(1)}, node);
    return new_cnode;
  }
  return nullptr;
}

AnfNodePtr SimplifyMatMul(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimMatMul)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x, y;
  auto matmul_transpose_lambda = [&node, &x, &y]() -> AnfNodePtr {
    auto new_matmul = NewCNodeWithInfo({NewValueNode(prim::kPrimMatMul), y.GetNode(node), x.GetNode(node)}, node);
    auto new_abstract = node->abstract()->Clone();
    auto ori_shape = node->abstract()->GetShapeTrack()->cast<abstract::ShapePtr>();
    auto shape_value = ori_shape->shape();
    ShapeVector new_shape_value;
    std::copy(shape_value.rbegin(), shape_value.rend(), std::back_inserter(new_shape_value));
    auto new_shape = std::make_shared<abstract::Shape>(new_shape_value);
    new_abstract->set_shape(new_shape);
    new_matmul->set_abstract(new_abstract);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimTranspose), new_matmul}, node);
    auto transpose_a = AnfAlgo::GetNodeAttr<ValuePtr>(node, "transpose_a");
    auto transpose_b = AnfAlgo::GetNodeAttr<ValuePtr>(node, "transpose_b");
    auto transpose_x1 = AnfAlgo::GetNodeAttr<ValuePtr>(node, "transpose_x1");
    auto transpose_x2 = AnfAlgo::GetNodeAttr<ValuePtr>(node, "transpose_x2");
    auto perm = AnfAlgo::GetNodeAttr<ValuePtr>(node->cast<CNodePtr>()->input(1), "perm");
    AnfAlgo::SetNodeAttr("transpose_a", transpose_b, new_matmul);
    AnfAlgo::SetNodeAttr("transpose_b", transpose_a, new_matmul);
    AnfAlgo::SetNodeAttr("transpose_x1", transpose_x2, new_matmul);
    AnfAlgo::SetNodeAttr("transpose_x2", transpose_x1, new_matmul);
    AnfAlgo::SetNodeAttr("perm", perm, new_cnode);
    return new_cnode;
  };
  // MatMul(Transpose(x), Transpose(y)) ==> Transpose(MatMul(y, x))
  MATCH_REPLACE_LAMBDA(node,
                       PBinOperation(prim::kPrimMatMul, PUnaryOperation(prim::kPrimTranspose, x),
                                     PUnaryOperation(prim::kPrimTranspose, y), false),
                       matmul_transpose_lambda);
  return nullptr;
}

ShapeVector TransAxisValueToVector(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  ShapeVector axis_vector;
  if (value->isa<Int32Imm>()) {
    axis_vector.emplace_back(GetValue<int64_t>(value));
  }
  if (value->isa<ValueTuple>() || value->isa<ValueList>()) {
    axis_vector = GetValue<std::vector<int64_t>>(value);
  }
  return axis_vector;
}

ShapeVector GetNodeShape(const AnfNodePtr &node) {
  auto base_shape = node->Shape()->cast<abstract::ShapePtr>();
  std::vector<int64_t> shape;
  std::transform(base_shape->shape().begin(), base_shape->shape().end(), std::back_inserter(shape), IntToSize);
  return shape;
}

std::vector<std::pair<int64_t, int64_t>> GetUnmodifiedDim(const ShapeVector &a, const ShapeVector &b) {
  std::vector<std::pair<int64_t, int64_t>> unmodified;
  for (size_t i = 0, j = 0, patial_a = 1, patial_b = 1;;) {
    if (i >= a.size() && j >= b.size()) {
      break;
    }
    if (i == j || patial_a == patial_b) {
      patial_a *= a[i];
      patial_b *= b[j];
    }
    if (patial_a == patial_b && a[i] == b[j]) {
      unmodified.emplace_back(std::make_pair(i, j));
      ++i;
      ++j;
      continue;
    }
    if (patial_a < patial_b) {
      ++i;
      patial_a *= a[i];
      if (patial_a == patial_b) {
        ++i;
        ++j;
      }
      continue;
    }
    if (patial_a > patial_b) {
      ++j;
      patial_b *= b[j];
      if (patial_a == patial_b) {
        ++i;
        ++j;
      }
      continue;
    }
  }
  return unmodified;
}

AnfNodePtr SimplifyReduce(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimReduceMax) && !IsPrimitiveCNode(node, prim::kPrimReduceMin) &&
      !IsPrimitiveCNode(node, prim::kPrimReduceSum)) {
    return nullptr;
  }
  PatternNode<AnfNodePtr> x;
  auto reduce_reduce_lambda = [&node, &x](PrimitivePtr &operation) -> AnfNodePtr {
    auto tmp_node = node->cast<CNodePtr>();
    auto arg_node = tmp_node->input(1);
    auto arg_dimensions = TransAxisValueToVector(AnfAlgo::GetNodeAttr<ValuePtr>(arg_node, "axis"));
    auto reduce_dimensions = TransAxisValueToVector(AnfAlgo::GetNodeAttr<ValuePtr>(tmp_node, "axis"));
    ShapeVector new_dimensions;
    for (size_t i = 0; i < arg_dimensions.size(); ++i) {
      for (size_t j = 0; j < reduce_dimensions.size(); ++j) {
        if (reduce_dimensions[j] >= arg_dimensions[i]) {
          ++reduce_dimensions[j];
        }
      }
    }
    std::merge(arg_dimensions.begin(), arg_dimensions.end(), reduce_dimensions.begin(), reduce_dimensions.end(),
               std::back_inserter(new_dimensions));
    auto new_cnode = NewCNodeWithInfo({NewValueNode(operation), x.GetNode(node)}, node);
    AnfAlgo::SetNodeAttr("axis", MakeValue(new_dimensions), new_cnode);
    AnfAlgo::CopyNodeAttr("keep_dims", node, new_cnode);
    return new_cnode;
  };
  auto neg_reducesum_lambda = [&node, &x]() -> AnfNodePtr {
    auto arg_node = NewCNodeWithInfo({NewValueNode(prim::kPrimReduceSum), x.GetNode(node)}, node);
    AnfAlgo::CopyNodeAttr("axis", node, arg_node);
    AnfAlgo::CopyNodeAttr("keep_dims", node, arg_node);
    auto new_cnode = NewCNodeWithInfo({NewValueNode(prim::kPrimNeg), arg_node}, node);
    return new_cnode;
  };
  std::list<PrimitivePtr> ReduceOperations = {prim::kPrimReduceSum, prim::kPrimReduceMax, prim::kPrimReduceMin};
  for (auto operation : ReduceOperations) {
    // Reduce(Reduce(A)) = Reduce(A)
    MATCH_REPLACE_LAMBDA_FLAG(node, PPrimitive(operation, PPrimitive(operation, x)), reduce_reduce_lambda, operation);
  }
  // ReduceSum(Neg(x)) = Neg(ReduceSum(x))
  MATCH_REPLACE_LAMBDA(node, PPrimitive(prim::kPrimReduceSum, PUnaryOperation(prim::kPrimNeg, x)),
                       neg_reducesum_lambda);
  return nullptr;
}

AnfNodePtr TrySimplify(const AnfNodePtr &node) {
  std::list<std::function<AnfNodePtr(const AnfNodePtr &)>> SimplifyFuncList = {SimplifyReduce};
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
    TraceGuard guard(std::make_shared<TraceOpt>(node->debug_info()));
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
      bool need_traverse = true;
      while (need_traverse) {
        need_traverse = false;
        for (auto node_sub : sub_graph->GetOrderedCnodes()) {
          auto new_node = TrySimplify(node_sub);
          if (new_node != nullptr) {
            PERFORM_REPLACE(node_sub->cast<AnfNodePtr>(), new_node, sub_graph, replaced);
            need_traverse = true;
            break;
          }
        }
      }
    }
  }
  EliminateEmptyGraph(func_graph);
  return replaced;
}
}  // namespace opt
}  // namespace mindspore
