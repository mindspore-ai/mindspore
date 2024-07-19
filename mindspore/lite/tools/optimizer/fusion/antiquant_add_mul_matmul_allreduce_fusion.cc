/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/antiquant_add_mul_matmul_allreduce_fusion.h"
#include <set>
#include <memory>
#include <vector>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/all_reduce.h"
#include "mindspore/core/ops/lite_ops.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "mindspore/ccsrc/frontend/parallel/ops_info/ops_utils.h"
#include "mindspore/core/ir/anf.h"

namespace mindspore::opt {
std::unordered_map<std::string, VectorRef> AntiquantAddMulMatMulAllReduceFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  patterns[kPatternNameAntiquantAddMulMatMulAllReduce] = DefineAntiquantAddMulMatMulAllReducePattern();
  return patterns;
}

bool GetDataTypeAndCheckIsValid(const AnfNodePtr &anf_node, const std::set<TypeId> &support_dtype) {
  TypeId input_datatype;
  if (GetDataTypeFromAnfNode(anf_node, &input_datatype) != RET_OK) {
    MS_LOG(INFO) << "get input node data type failed." << anf_node->fullname_with_scope();
    return false;
  }
  if (support_dtype.find(input_datatype) == support_dtype.end()) {
    MS_LOG(INFO) << "unsupported data type: " << input_datatype;
    return false;
  }
  return true;
}

bool IsOnlyMatMulNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, prim::kPrimMatMul) || CheckPrimitiveType(anf_node, prim::kPrimMatMulV2);
  }
  return false;
}

VectorRef AntiquantAddMulMatMulAllReduceFusion::DefineAntiquantAddMulMatMulAllReducePattern() const {
  auto weight = std::make_shared<Var>();
  auto scale = std::make_shared<Var>();
  auto offset = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(weight != nullptr, {});
  MS_CHECK_TRUE_RET(scale != nullptr, {});
  MS_CHECK_TRUE_RET(offset != nullptr, {});
  auto is_antiquant = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAntiQuant>);
  MS_CHECK_TRUE_RET(is_antiquant != nullptr, {});
  VectorRef antiquant_ref = VectorRef({is_antiquant, weight, scale, offset});

  auto add_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add_input_2 != nullptr, {});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAdd>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  VectorRef add_ref = VectorRef({is_add, antiquant_ref, add_input_2});

  auto mul_input_2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mul_input_2 != nullptr, {});
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMul>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  VectorRef mul_ref = VectorRef({is_mul, add_ref, mul_input_2});

  auto matmul_input_1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(matmul_input_1 != nullptr, {});
  auto is_matmul = std::make_shared<CondVar>(IsOnlyMatMulNode);
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  VectorRef matmul_ref = VectorRef({is_matmul, matmul_input_1, mul_ref});

  auto is_allreduce = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAllReduce>);
  MS_CHECK_TRUE_RET(is_allreduce != nullptr, {});
  VectorRef pattern_ref = VectorRef({is_allreduce, matmul_ref});
  return pattern_ref;
}

CNodePtr AntiquantAddMulMatMulAllReduceFusion::CreateAntiquantAddMulMatMulAllReduceNode(const FuncGraphPtr &func_graph,
                                                                                        const AnfNodePtr &node) const {
  MS_LOG(INFO) << "start create AntiquantAddMulMatMulAllReduce";
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto allreduce_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(allreduce_cnode);
  auto mm_cnode = allreduce_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mm_cnode);
  auto mul_cnode = mm_cnode->input(kIndex2)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul_cnode);
  auto add_cnode = mul_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(add_cnode);
  auto antiquant_cnode = add_cnode->input(kIndex1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(antiquant_cnode);

  // bias node
  AnfNodePtr bias_node;
  if (mm_cnode->size() >= kInputIndexThree + 1) {
    bias_node = mm_cnode->input(kInputIndexThree);
    MS_EXCEPTION_IF_NULL(bias_node);
  } else {
    auto none = std::make_shared<None>();
    bias_node = NewValueNode(none);
    MS_EXCEPTION_IF_NULL(bias_node);
    bias_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  }

  auto input_x = mm_cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(input_x);
  auto input_w = antiquant_cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(input_w);
  auto input_antiquant_offset = add_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input_antiquant_offset);
  auto input_antiquant_scale = mul_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input_antiquant_scale);

  // check dtype
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16};
  if (!GetDataTypeAndCheckIsValid(input_x, support_dtype)) {
    MS_LOG(INFO) << "CreateAntiquantAddMulMatMulAllReduceNode failed. input_x dtype not float16.";
    return nullptr;
  }
  if (!GetDataTypeAndCheckIsValid(input_antiquant_offset, support_dtype)) {
    MS_LOG(INFO) << "CreateAntiquantAddMulMatMulAllReduceNode failed. antiquant_offset(add input1) dtype not float16.";
    return nullptr;
  }
  if (!GetDataTypeAndCheckIsValid(input_antiquant_scale, support_dtype)) {
    MS_LOG(INFO) << "CreateAntiquantAddMulMatMulAllReduceNode failed. antiquant_scale(mul input1) dtype not float16.";
    return nullptr;
  }

  // create op
  auto matmul_allreduce_prim = std::make_shared<Primitive>("MatMulAllReduce")->Clone();
  MS_CHECK_TRUE_RET(matmul_allreduce_prim, {});

  // add attr
  auto allreduce_prim = GetCNodePrimitive(allreduce_cnode);
  auto mm_prim = GetCNodePrimitive(mm_cnode);
  matmul_allreduce_prim->AddAttr(kAttrNameCommRenuse, allreduce_prim->GetAttr(kAttrNameCommRenuse));
  matmul_allreduce_prim->AddAttr(kAttrNameGroup, allreduce_prim->GetAttr(kAttrNameGroup));
  matmul_allreduce_prim->AddAttr(kAttrNameFusion, allreduce_prim->GetAttr(kAttrNameFusion));
  matmul_allreduce_prim->AddAttr(kAttrNameOp, allreduce_prim->GetAttr(kAttrNameOp));
  matmul_allreduce_prim->AddAttr(kAttrNameTransposeA, mm_prim->GetAttr(kAttrTransposeX1));
  matmul_allreduce_prim->AddAttr(kAttrNameTransposeB, mm_prim->GetAttr(kAttrTransposeX2));

  // prepare None node
  auto none = std::make_shared<None>();
  auto none_node = NewValueNode(none);
  MS_EXCEPTION_IF_NULL(none_node);
  none_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  func_graph->NewCNode({none_node});

  // insert newnode
  auto matmul_allreduce_cnode =
    func_graph->NewCNode({NewValueNode(matmul_allreduce_prim), input_x, input_w, bias_node, none_node,
                          input_antiquant_scale, input_antiquant_offset, none_node});
  if (matmul_allreduce_cnode == nullptr) {
    MS_LOG(ERROR) << "New matmul_allreduce_cnode should not be null, but it is null.";
    return nullptr;
  }
  matmul_allreduce_cnode->set_abstract(allreduce_cnode->abstract()->Clone());
  matmul_allreduce_cnode->set_fullname_with_scope(allreduce_cnode->fullname_with_scope() + "_aamm_allreduce_fusion");
  MS_LOG(INFO) << "create AntiquantAddMulMatMulAllReduceNode success.";
  return matmul_allreduce_cnode;
}

AnfNodePtr AntiquantAddMulMatMulAllReduceFusion::Process(const std::string &pattern_name,
                                                         const mindspore::FuncGraphPtr &func_graph,
                                                         const mindspore::AnfNodePtr &node,
                                                         const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  auto allreduce_cnode = node->cast<CNodePtr>();
  if (allreduce_cnode->size() != kInputSizeTwo) {
    return nullptr;
  }

  CNodePtr matmul_allreduce_cnode = CreateAntiquantAddMulMatMulAllReduceNode(func_graph, node);
  MS_CHECK_TRUE_RET(matmul_allreduce_cnode != nullptr, nullptr);

  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, nullptr);

  // replace allreduce to MatMulAllReduce
  (void)manager->Replace(allreduce_cnode, matmul_allreduce_cnode);
  MS_LOG(INFO) << "MatMulAllReduce replace AntiquantAddMulMatMulAllReduce success";
  return matmul_allreduce_cnode;
}
}  // namespace mindspore::opt
