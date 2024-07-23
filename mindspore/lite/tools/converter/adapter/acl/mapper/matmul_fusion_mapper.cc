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
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "tools/common/tensor_util.h"
#include "ir/named.h"
#include "ir/func_graph.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/auto_generate/gen_lite_ops.h"
#include "nnacl/op_base.h"
#include "ops/base_operator.h"

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
constexpr size_t kNumIndex3 = 3;
constexpr size_t kSize_0 = 0;
constexpr size_t kSize_1 = 1;
constexpr size_t kSize_2 = 2;
constexpr size_t kSize_3 = 3;
constexpr size_t kSize_4 = 4;

CNodePtr InsertCastFp32(const FuncGraphPtr &func_graph, const CNodePtr &fp16_node) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(fp16_node != nullptr, nullptr);
  auto cast_value = NewValueNode(mindspore::TypeIdToType(TypeId::kNumberTypeFloat32));
  MS_CHECK_TRUE_RET(cast_value != nullptr, nullptr);
  std::vector<AnfNodePtr> cast_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), fp16_node,
                                         cast_value};
  auto cast_fp32_cnode = func_graph->NewCNode(cast_inputs);
  MS_CHECK_TRUE_RET(cast_fp32_cnode != nullptr, nullptr);
  cast_fp32_cnode->set_fullname_with_scope(fp16_node->fullname_with_scope() + "_castfp32");
  MS_CHECK_TRUE_RET(fp16_node->abstract() != nullptr, nullptr);
  cast_fp32_cnode->set_abstract(fp16_node->abstract()->Clone());
  return cast_fp32_cnode;
}

CNodePtr InsertAdd(const FuncGraphPtr &func_graph, const CNodePtr &QBMM_cnode, const AnfNodePtr &bias) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(QBMM_cnode != nullptr, nullptr);
  auto prim = std::make_unique<ops::Add>();
  MS_CHECK_TRUE_MSG(prim != nullptr, nullptr, "Create AddFusion prim failed!");
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_MSG(prim_c != nullptr, nullptr, "Get prim_c is nullptr!");
  auto add_primitive = NewValueNode(prim_c);
  MS_CHECK_TRUE_RET(add_primitive != nullptr, nullptr);
  auto add_fusion = func_graph->NewCNode({add_primitive, QBMM_cnode, bias});
  MS_CHECK_TRUE_MSG(add_fusion != nullptr, nullptr, "Create AddFusion CNode failed!");
  add_fusion->set_fullname_with_scope(QBMM_cnode->fullname_with_scope() + "_add");
  MS_CHECK_TRUE_RET(QBMM_cnode->abstract() != nullptr, nullptr);
  add_fusion->set_abstract(QBMM_cnode->abstract()->Clone());
  return add_fusion;
}

std::shared_ptr<CNode> CreateTransQuantParamV2(const FuncGraphPtr &func_graph, const CNodePtr &mm_cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(mm_cnode != nullptr, nullptr);
  auto quant_param_holder = mindspore::lite::GetCNodeQuantHolder(mm_cnode);
  MS_CHECK_TRUE_RET(quant_param_holder != nullptr, nullptr);
  auto quant_params_vec = quant_param_holder->get_input_quant_params();
  if (quant_params_vec.empty()) {
    return nullptr;
  }
  auto quant_params_x1 = quant_params_vec.at(kNumIndex0);
  if (quant_params_x1.size() != kSize_1) {
    MS_LOG(ERROR) << "For active quantization, only per_tensor mode is currently supported."
                  << " Scale size should be 1, but get scale size is: " << quant_params_x1.size();
    return nullptr;
  }
  auto quant_param_x1 = quant_params_x1.front();
  auto scale_x1 = quant_param_x1.scale;
  auto zero_point_x1 = quant_param_x1.zeroPoint;
  if (zero_point_x1 != 0) {
    MS_LOG(ERROR) << "Only support zero_point = 0! zero_point_x1 is: " << zero_point_x1;
    return nullptr;
  }
  auto quant_params_x2 = quant_params_vec.at(kNumIndex1);
  if (quant_params_x2.empty()) {
    return nullptr;
  }
  auto per_channel_size = quant_params_x2.size();
  std::vector<int64_t> shape_vector = {static_cast<int64_t>(per_channel_size)};
  auto buf = std::make_unique<float[]>(per_channel_size);
  MS_CHECK_TRUE_RET(buf != nullptr, nullptr);
  // QuantBatchMatmul scale = scale1 * scale2
  for (uint64_t i = 0; i < per_channel_size; i++) {
    buf[i] = scale_x1 * quant_params_x2.at(i).scale;
    if (quant_params_x2.at(i).zeroPoint != 0) {
      MS_LOG(ERROR) << "Only support zero_point = 0!";
      return nullptr;
    }
  }
  auto tensor_info = lite::CreateTensorInfo(buf.get(), per_channel_size * sizeof(float), shape_vector,
                                            mindspore::TypeId::kNumberTypeFloat32);
  MS_CHECK_TRUE_RET(tensor_info != nullptr, nullptr);
  auto scale_param_node =
    opt::BuildParameterNode(func_graph, tensor_info, mm_cnode->fullname_with_scope() + "_scale", false);
  MS_CHECK_TRUE_RET(scale_param_node != nullptr, nullptr);
  MS_CHECK_TRUE_RET(mm_cnode->abstract() != nullptr, nullptr);
  scale_param_node->set_abstract(mm_cnode->abstract()->Clone());
  // insert TransQuantParamV2
  auto trans_quant_param_prim = std::make_shared<mindspore::lite::acl::TransQuantParamV2>();
  MS_CHECK_TRUE_RET(trans_quant_param_prim != nullptr, nullptr);
  std::vector<AnfNodePtr> trans_quant_param_inputs = {NewValueNode(trans_quant_param_prim), scale_param_node};
  auto trans_quant_param_cnode = func_graph->NewCNode(trans_quant_param_inputs);
  MS_CHECK_TRUE_RET(trans_quant_param_cnode != nullptr, nullptr);
  trans_quant_param_cnode->set_fullname_with_scope(mm_cnode->fullname_with_scope() + "_TransQuantParamV2");
  trans_quant_param_cnode->set_abstract(mm_cnode->abstract()->Clone());
  return trans_quant_param_cnode;
}

int ReplaceMMToQuantBatchMatmul(const FuncGraphPtr &func_graph, const CNodePtr &mm_cnode) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(mm_cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(mm_cnode->size() >= kSize_3, RET_ERROR);
  auto input_x1 = mm_cnode->input(kNumIndex1);
  MS_CHECK_TRUE_RET(input_x1 != nullptr, RET_ERROR);
  auto input_x2 = mm_cnode->input(kNumIndex2);
  MS_CHECK_TRUE_RET(input_x2 != nullptr, RET_ERROR);
  auto QBMM_prim = std::make_shared<mindspore::lite::acl::QuantBatchMatmul>();
  MS_CHECK_TRUE_RET(QBMM_prim != nullptr, RET_ERROR);
  auto mm_prim = GetValueNode<PrimitivePtr>(mm_cnode->input(kNumIndex0));
  MS_CHECK_TRUE_RET(mm_prim != nullptr, RET_ERROR);
  auto transpose_a = mm_prim->GetAttr(mindspore::ops::kTransposeA);
  auto transpose_b = mm_prim->GetAttr(mindspore::ops::kTransposeB);
  if (transpose_a != nullptr) {
    QBMM_prim->AddAttr(kAttrTransposeX1, transpose_a);
  } else {
    QBMM_prim->AddAttr(kAttrTransposeX1, MakeValue(false));
  }
  if (transpose_b != nullptr) {
    QBMM_prim->AddAttr(kAttrTransposeX2, transpose_b);
  } else {
    QBMM_prim->AddAttr(kAttrTransposeX2, MakeValue(false));
  }
  QBMM_prim->AddAttr(kAttrDType, MakeValue(kFloat16));
  auto trans_quant_param_cnode = CreateTransQuantParamV2(func_graph, mm_cnode);
  MS_CHECK_TRUE_RET(trans_quant_param_cnode != nullptr, RET_ERROR);
  auto none_value_node_offset = NewValueNode(std::make_shared<mindspore::None>());
  MS_CHECK_TRUE_RET(none_value_node_offset != nullptr, RET_ERROR);
  none_value_node_offset->set_abstract(std::make_shared<abstract::AbstractNone>());
  auto none_value_node_bias = NewValueNode(std::make_shared<mindspore::None>());
  MS_CHECK_TRUE_RET(none_value_node_bias != nullptr, RET_ERROR);
  none_value_node_bias->set_abstract(std::make_shared<abstract::AbstractNone>());
  std::vector<AnfNodePtr> quant_op_inputs = {
    NewValueNode(QBMM_prim), input_x1, input_x2, trans_quant_param_cnode, none_value_node_offset, none_value_node_bias};
  auto QBMM_cnode = func_graph->NewCNode(quant_op_inputs);
  MS_CHECK_TRUE_RET(QBMM_cnode != nullptr, RET_ERROR);
  QBMM_cnode->set_fullname_with_scope(mm_cnode->fullname_with_scope() + "_QuantBatchMatmul");
  MS_CHECK_TRUE_RET(mm_cnode->abstract() != nullptr, RET_ERROR);
  QBMM_cnode->set_abstract(mm_cnode->abstract()->Clone());
  MS_LOG(INFO) << "QuantBatchMatmul name: " << QBMM_cnode->fullname_with_scope() << ", prim name: " << QBMM_prim->name()
               << ", input1: " << input_x1->DebugString() << ", input2: " << input_x2->DebugString();
  auto manager = Manage(func_graph);
  MS_CHECK_TRUE_RET(manager != nullptr, RET_ERROR);
  if (mm_cnode->size() == kSize_4) {
    // Gemm(prim, input1, input2, bias) -> QuantBMM(prim, input1, input2, scale) + add(bias)
    auto bias = mm_cnode->input(kNumIndex3);
    MS_CHECK_TRUE_RET(bias != nullptr, RET_ERROR);
    auto add_fusion = InsertAdd(func_graph, QBMM_cnode, bias);
    MS_CHECK_TRUE_RET(add_fusion != nullptr, RET_ERROR);
    auto cast_fp32_node = InsertCastFp32(func_graph, add_fusion);
    MS_CHECK_TRUE_RET(cast_fp32_node != nullptr, RET_ERROR);
    (void)manager->Replace(mm_cnode, cast_fp32_node);
    return RET_OK;
  }
  auto cast_fp32_node = InsertCastFp32(func_graph, QBMM_cnode);
  (void)manager->Replace(mm_cnode, cast_fp32_node);
  return RET_OK;
}

int QuantBatchMatmulMapper(const CNodePtr &mm_cnode) {
  auto func_graph = mm_cnode->func_graph();
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  auto graph_manager = func_graph->manager();
  MS_CHECK_TRUE_RET(graph_manager != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(mm_cnode != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(mm_cnode->abstract() != nullptr, RET_ERROR);
  // size(prim, input1, input2) = 3 or size(prim, input1, input2, bias) = 4
  if (mm_cnode->size() < kSize_3 || mm_cnode->size() > kSize_4) {
    MS_LOG(ERROR) << "The number of inputs of Matmul_node can only be 3 or 4!, but get " << mm_cnode->size();
    return RET_ERROR;
  }
  auto quant_param_holder = mindspore::lite::GetCNodeQuantHolder(mm_cnode);
  MS_CHECK_TRUE_RET(quant_param_holder != nullptr, RET_ERROR);
  if (!quant_param_holder->IsInputQuantParamsInited()) {
    MS_LOG(INFO) << "InputQuantParamsInited is false, this node is " << mm_cnode->fullname_with_scope();
    return RET_OK;
  }
  auto quant_params_vec = quant_param_holder->get_input_quant_params();
  lite::quant::InsertQuantNodeManager insert_node_manager;
  auto input_2_node = mm_cnode->input(kNumIndex2);
  MS_CHECK_TRUE_RET(input_2_node != nullptr, RET_ERROR);
  if (quant_params_vec.size() == kSize_0) {
    MS_LOG(INFO) << "This node has no quantization parameter. Skip it. node name: " << mm_cnode->fullname_with_scope();
    return RET_OK;
  } else if (quant_params_vec.size() == kSize_2 &&
             (input_2_node->isa<mindspore::CNode>() || input_2_node->isa<mindspore::Parameter>())) {
    MS_LOG(INFO) << "Start do double per_tensor(A&W) pass or Start do per_tensor(A) + per_channel(W) pass(The weight "
                    "is already of the int8 type.).";
    return ReplaceMMToQuantBatchMatmul(func_graph, mm_cnode);
  } else if (quant_params_vec.size() == kSize_3) {
    MS_LOG(ERROR) << "quant_params_vec size is 3. The matmul node conversion with quantized bias is not supported now!";
    return RET_ERROR;
  } else {
    MS_LOG(ERROR) << "Dont support! The number of quantization parameters is " << quant_params_vec.size();
    return RET_ERROR;
  }
}

}  // namespace
void MatMulFusionMapper::SetMatMulTransposeAttr(const PrimitivePtr &src_prim, const PrimitivePtr &dst_prim) {
  auto transpose_a = src_prim->GetAttr(mindspore::ops::kTransposeA);
  auto transpose_b = src_prim->GetAttr(mindspore::ops::kTransposeB);
  if (transpose_a != nullptr) {
    dst_prim->AddAttr(kTransposeA, transpose_a);
  } else {
    dst_prim->AddAttr(kTransposeA, MakeValue(false));
  }
  if (transpose_b != nullptr) {
    dst_prim->AddAttr(kTransposeB, transpose_b);
  } else {
    dst_prim->AddAttr(kTransposeB, MakeValue(false));
  }
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
  } else if (quant_holder->IsInputQuantParamsInited()) {
    MS_LOG(INFO) << "quant_holder IsInputQuantParamsInited is true. quant_holder->quant_type(): "
                 << quant_holder->quant_type()
                 << ", cnode_primitive->HasAttr(quant_type): " << cnode_primitive->HasAttr(quant::kQuantType);
    return QuantBatchMatmulMapper(cnode);
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
  if (shape_vector.size() == DIMENSION_2D) {
    dst_prim = std::make_shared<acl::MatMulV2>();
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
