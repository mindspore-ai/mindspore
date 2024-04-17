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
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "ops/mat_mul.h"
#include "ops/batch_matmul.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/auto_generate/gen_lite_ops.h"
#include "nnacl/op_base.h"

namespace mindspore {
using mindspore::ops::kNameBatchMatMul;
using mindspore::ops::kNameMatMul;
namespace lite {
namespace {
constexpr size_t kInputSizeWithoutBias = 3;  // primitive, x1, x2
constexpr size_t kInputSizeWithBias = 4;     // primitive, x1, x2, bias
constexpr size_t kInputX1Idx = 1;
constexpr size_t kInputX2Idx = 2;
constexpr size_t kInputBiasIdx = 3;
constexpr size_t kNumIndex0 = 0;
constexpr size_t kNumIndex1 = 1;
constexpr size_t kNumIndex2 = 2;

std::vector<int64_t> GetTensorShape(CNodePtr cnode, size_t input_index) {
  auto abstract = opt::GetCNodeInputAbstract(cnode, input_index);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "GetCNodeInputAbstract in promapt flash attention fusion.";
    return {};
  }
  std::vector<int64_t> shape = {};
  if (opt::FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "FetchShapeFromAbstract failed.";
    return {};
  }
  return shape;
}

}  // namespace
void MatMulFusionMapper::SetMatMulTransposeAttr(const PrimitivePtr &src_prim, const PrimitivePtr &dst_prim) {
  auto transpose_a = src_prim->GetAttr(mindspore::ops::kTransposeA);
  auto transpose_b = src_prim->GetAttr(mindspore::ops::kTransposeB);
  if (transpose_a != nullptr) {
    dst_prim->AddAttr("transpose_x1", transpose_a);
    dst_prim->AddAttr("transpose_a", transpose_a);
  } else {
    dst_prim->AddAttr("transpose_x1", MakeValue(false));
    dst_prim->AddAttr("transpose_a", MakeValue(false));
  }
  if (transpose_b != nullptr) {
    dst_prim->AddAttr("transpose_x2", transpose_b);
    dst_prim->AddAttr("transpose_b", transpose_b);
  } else {
    dst_prim->AddAttr("transpose_x2", MakeValue(false));
    dst_prim->AddAttr("transpose_b", MakeValue(false));
  }
}

// BMM(x1=[a,b,c],x2=[c,d]) -> reshape(x1,[a*b,c]) + output=MM(x1=[a*b,c],x2=[c,d]) + reshape(output,[a,b,d])
PrimitiveCPtr MatMulFusionMapper::BMMToMM(const CNodePtr &cnode, const std::vector<int64_t> &shape_vector) {
  auto input_1_shape = GetTensorShape(cnode, kNumIndex1);
  auto input_2_shape = GetTensorShape(cnode, kNumIndex2);
  if (input_1_shape.size() != DIMENSION_3D || input_2_shape.size() != DIMENSION_2D ||
      shape_vector.size() != DIMENSION_3D) {
    MS_LOG(ERROR) << "This mapper is not BMM(3D,2D)";
    return nullptr;
  }
  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to get func graph from cnode " << cnode->fullname_with_scope();
    return nullptr;
  }
  auto graph_manager = func_graph->manager();
  if (graph_manager == nullptr) {
    MS_LOG(ERROR) << "Failed to get func graph manager from cnode " << cnode->fullname_with_scope();
    return nullptr;
  }
  auto x1_input = cnode->input(kInputX1Idx);
  auto x2_input = cnode->input(kInputX2Idx);
  ops::MatMul mat_mul;
  PrimitiveCPtr dst_prim = mat_mul.GetPrim();

  std::vector<int32_t> MM_shape = {-1, static_cast<int32_t>(input_1_shape[kNumIndex2])};
  auto shape_parm_node =
    opt::BuildIntVecParameterNode(func_graph, MM_shape, cnode->fullname_with_scope() + "_input_shape_perm");
  MS_CHECK_TRUE_MSG(shape_parm_node != nullptr, nullptr, "create shape_parm_node return nullptr");
  std::vector<AnfNodePtr> op_inputs = {x1_input, shape_parm_node};
  auto reshape_prim = std::make_shared<ops::Reshape>();
  MS_CHECK_TRUE_MSG(reshape_prim != nullptr, nullptr, "create reshape_prim return nullptr");
  auto reshape_prim_c = reshape_prim->GetPrim();
  MS_CHECK_TRUE_MSG(reshape_prim_c != nullptr, nullptr, "create prim_c return nullptr");
  auto reshape_node = func_graph->NewCNode(reshape_prim_c, op_inputs);
  MS_CHECK_TRUE_MSG(reshape_node != nullptr, nullptr, "create reshape_node return nullptr");
  reshape_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_input_reshape");
  if (cnode->abstract() != nullptr) {
    reshape_node->set_abstract(cnode->abstract()->Clone());
  }
  auto matmul =
    NewCNode(cnode, dst_prim, {reshape_node, x2_input}, cnode->abstract()->Clone(), cnode->fullname_with_scope());
  MS_CHECK_TRUE_MSG(matmul != nullptr, nullptr, "Failed to create MatMul node");

  std::vector<int32_t> output_shape = {static_cast<int32_t>(shape_vector[kNumIndex0]),
                                       static_cast<int32_t>(shape_vector[kNumIndex1]),
                                       static_cast<int32_t>(shape_vector[kNumIndex2])};
  auto shape_output_parm_node =
    opt::BuildIntVecParameterNode(func_graph, output_shape, cnode->fullname_with_scope() + "_output_shape_perm");
  MS_CHECK_TRUE_MSG(shape_output_parm_node != nullptr, nullptr, "create shape_parm_node return nullptr");
  auto reshape_output_prim = std::make_shared<ops::Reshape>();
  MS_CHECK_TRUE_MSG(reshape_output_prim != nullptr, nullptr, "create reshape_output_prim return nullptr");
  auto reshape_output_prim_c = reshape_output_prim->GetPrim();
  MS_CHECK_TRUE_MSG(reshape_output_prim_c != nullptr, nullptr, "create prim_c return nullptr");
  auto reshape_output_node = func_graph->NewCNode(reshape_output_prim_c, {matmul, shape_output_parm_node});
  MS_CHECK_TRUE_MSG(reshape_output_node != nullptr, nullptr, "create reshape_output_node return nullptr");
  reshape_output_node->set_fullname_with_scope(cnode->fullname_with_scope() + "_output_reshape");
  if (cnode->abstract() != nullptr) {
    reshape_output_node->set_abstract(cnode->abstract()->Clone());
  }
  if (!graph_manager->Replace(cnode, reshape_output_node)) {
    MS_LOG(ERROR) << "Failed to replace MatMul with BatchMatMul, cnode " << cnode->fullname_with_scope()
                  << ", input size " << cnode->size();
    return nullptr;
  }
  MS_LOG(INFO) << "BMM->MM SUCCESS";
  return dst_prim;
}

STATUS MatMulFusionMapper::Mapper(const CNodePtr &cnode) {
  auto quant_holder = GetCNodeQuantHolder(cnode);
  auto cnode_primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(cnode_primitive != nullptr, RET_NULL_PTR, "Primitive is nullptr.");
  if (quant_holder->quant_type() != quant::QUANT_NONE) {
    return QuantMapper(cnode);
  } else if (cnode_primitive->HasAttr(quant::kQuantType)) {
    auto quant_type_attr = cnode_primitive->GetAttr(quant::kQuantType);
    auto quant_type = static_cast<quant::QuantType>(GetValue<int32_t>(quant_type_attr));
    if (quant_type != quant::QUANT_NONE) {
      return QuantMapper(cnode);
    }
  } else if (opt::CheckPrimitiveType(cnode, prim::kPrimBatchMatMul)) {
    ValueNodePtr value_node = nullptr;
    PrimitivePtr src_prim = nullptr;
    if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
      MS_LOG(ERROR) << "Get primitive from cnode failed.";
      return lite::RET_ERROR;
    }
    SetMatMulTransposeAttr(src_prim, src_prim);
    return RET_OK;
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
  auto input_1_shape = GetTensorShape(cnode, 1);
  auto input_2_shape = GetTensorShape(cnode, 2);
  int64_t kNumDynShape = -1;
  bool is_dyn_input_1 =
    std::any_of(input_1_shape.begin(), input_1_shape.end(), [kNumDynShape](int y) { return kNumDynShape == y; });
  bool is_dyn_input_2 =
    std::any_of(input_2_shape.begin(), input_2_shape.end(), [kNumDynShape](int y) { return kNumDynShape == y; });

  if (shape_vector.size() == DIMENSION_2D) {
    dst_prim = std::make_shared<acl::MatMulV2>();
    value_node->set_value(dst_prim);
  } else if (cnode->size() == kInputSizeWithoutBias) {
    if (input_1_shape.size() == DIMENSION_3D && input_2_shape.size() == DIMENSION_2D && !is_dyn_input_1 &&
        !is_dyn_input_2) {
      dst_prim = BMMToMM(cnode, shape_vector);
    } else {
      ops::BatchMatMul mat_mul;
      dst_prim = mat_mul.GetPrim();
      value_node->set_value(dst_prim);
    }
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
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "dst_prim is nullptr.";
    return RET_ERROR;
  }
  SetMatMulTransposeAttr(src_prim, dst_prim);
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

  auto dst_prim = std::make_shared<acl::BatchMatMulV2>();
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return RET_OK;
}

// Graph_ir mapped MatMul to MatlMulV2 in old version, which has now been corrected.
// Lite MatlMul with bias or quantized should be mappered to MatlMulV2 for anf_graph, and mappered to MatlMulV2 for GE.
REGISTER_PRIMITIVE_MAPPER(kNameMatMul, MatMulFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNameMatMulFusion, MatMulFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNameBatchMatMul, MatMulFusionMapper)
}  // namespace lite
}  // namespace mindspore
