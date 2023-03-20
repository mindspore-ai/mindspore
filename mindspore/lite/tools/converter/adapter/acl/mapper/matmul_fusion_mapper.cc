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

#include "tools/converter/adapter/acl/mapper/matmul_fusion_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "ops/batch_matmul.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kInputSizeWithoutBias = 3;  // primitive, x1, x2
constexpr size_t kInputSizeWithBias = 4;     // primitive, x1, x2, bias
constexpr size_t kInputX1Idx = 1;
constexpr size_t kInputX2Idx = 2;
constexpr size_t kInputBiasIdx = 3;
}  // namespace
STATUS MatMulFusionMapper::Mapper(const CNodePtr &cnode) {
  auto quant_holder = GetCNodeQuantHolder(cnode);
  if (quant_holder->quant_type() != quant::QUANT_NONE) {
    return QuantMapper(cnode);
  }
  if (cnode->size() < kInputSizeWithoutBias) {
    MS_LOG(ERROR) << "Input size cannot < " << kInputSizeWithoutBias << ", node " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  std::vector<int64_t> shape_vector;
  if (acl::GetShapeVectorFromCNode(cnode, &shape_vector) != RET_OK) {
    MS_LOG(ERROR) << "Get cnode shape failed, cnode " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  PrimitiveCPtr dst_prim = nullptr;
  if (shape_vector.size() == DIMENSION_2D) {
    ops::MatMul mat_mul;
    dst_prim = mat_mul.GetPrim();
    value_node->set_value(dst_prim);
  } else if (cnode->size() == kInputSizeWithoutBias) {
    ops::BatchMatMul mat_mul;
    dst_prim = mat_mul.GetPrim();
    value_node->set_value(dst_prim);
  } else {
    auto func_graph = cnode->func_graph();
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Failed to get func graph from cnode " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    auto graph_manager = func_graph->manager();
    if (graph_manager == nullptr) {
      MS_LOG(ERROR) << "Failed to get func graph manager from cnode " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    auto x1_input = cnode->input(kInputX1Idx);
    auto x2_input = cnode->input(kInputX2Idx);
    auto bias_input = cnode->input(kInputBiasIdx);
    ops::BatchMatMul batch_mat_mul;
    dst_prim = batch_mat_mul.GetPrim();
    auto batch_matmul = NewCNode(cnode, dst_prim, {x1_input, x2_input}, cnode->abstract()->Clone(),
                                 cnode->fullname_with_scope() + "_batch_matmul");
    if (batch_matmul == nullptr) {
      MS_LOG(ERROR) << "Failed to create BatchMatMul node for node " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    auto add_node = NewCNode(cnode, prim::kPrimAdd, {batch_matmul, bias_input}, cnode->abstract()->Clone(),
                             cnode->fullname_with_scope() + "_add_bias");
    if (add_node == nullptr) {
      MS_LOG(ERROR) << "Failed to create Add bias node for node " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (!graph_manager->Replace(cnode, add_node)) {
      MS_LOG(ERROR) << "Failed to replace MatMul with BatchMatMul, cnode " << cnode->fullname_with_scope()
                    << ", input size " << cnode->size();
      return RET_ERROR;
    }
  }
  auto transpose_a = src_prim->GetAttr(mindspore::ops::kTransposeA);
  auto transpose_b = src_prim->GetAttr(mindspore::ops::kTransposeB);
  if (transpose_a != nullptr) {
    dst_prim->AddAttr("transpose_x1", transpose_a);
  }
  if (transpose_b != nullptr) {
    dst_prim->AddAttr("transpose_x2", transpose_b);
  }
  dst_prim->SetAttrs(src_prim->attrs());
  return RET_OK;
}

STATUS MatMulFusionMapper::QuantMapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get shape of cnode failed.";
    return RET_ERROR;
  }
  std::vector<int64_t> shape_vector;
  if (acl::GetShapeVectorFromCNode(cnode, &shape_vector) != RET_OK) {
    MS_LOG(ERROR) << "Get shape of cnode failed.";
    return RET_ERROR;
  }

  ops::MatMul mat_mul;
  auto dst_prim = std::make_shared<acl::BatchMatMulV2>();
  auto transpose_a = src_prim->GetAttr(mindspore::ops::kTransposeA);
  auto transpose_b = src_prim->GetAttr(mindspore::ops::kTransposeB);

  if (transpose_a != nullptr) {
    dst_prim->AddAttr("transpose_x1", transpose_a);
  }
  if (transpose_b != nullptr) {
    dst_prim->AddAttr("transpose_x2", transpose_b);
    if (GetValue<bool>(transpose_b)) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " transpose_b dont support true.";
      return RET_ERROR;
    }
  }
  value_node->set_value(dst_prim);
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameMatMulFusion, MatMulFusionMapper)
}  // namespace lite
}  // namespace mindspore
