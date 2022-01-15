/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/weight_quantizer.h"
#include <list>
#include <string>
#include <utility>
#include <set>
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_util.h"

namespace mindspore::lite::quant {
WeightQuantizer::~WeightQuantizer() {
  for (const auto &fp32_output_tensor : fp32_output_tensors_) {
    for (const auto &kv : fp32_output_tensor) {
      delete kv.second;
    }
  }
}

int WeightQuantizer::WeightQuant(const FuncGraphPtr &func_graph,
                                 const std::set<PrimitivePtr> &support_weight_quant_types,
                                 const std::set<PrimitivePtr> &per_layer_types,
                                 const std::set<PrimitivePtr> &symmetric_types) {
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto primitive = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(DEBUG) << cnode->fullname_with_scope() << " : primitive is nullptr";
      continue;
    }
    if (!CheckNodeInSet(cnode, support_weight_quant_types)) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " of type: " << primitive->name() << " dont need weight quant.";
      continue;
    }
    WeightQuantType weight_quant_type = WeightQuantType::FIXED_BIT_PER_CHANNEL;
    if (CheckNodeInSet(cnode, per_layer_types)) {
      weight_quant_type = WeightQuantType::FIXED_BIT_PER_LAYER;
    }
    bool symmetric = false;
    int q_min = quant_min_;
    int q_max = quant_max_;
    if (CheckNodeInSet(cnode, symmetric_types)) {
      symmetric = true;
      q_min = symmetric_quant_min_;
      q_max = symmetric_quant_max_;
    }
    std::vector<int> weight_indices;
    if (opt::CheckPrimitiveType(cnode, prim::kPrimAdam)) {
      weight_indices = {2, 3};
    } else if (opt::CheckPrimitiveType(cnode, prim::kPrimSGD)) {
      weight_indices = {4, 6};
    } else if (opt::CheckPrimitiveType(cnode, prim::kPrimApplyMomentum)) {
      weight_indices = {2};
    } else {
      for (size_t i = 1; i < cnode->size(); ++i) {
        weight_indices.push_back(i);
      }
    }
    auto status = DoCNodeWeightQuant(func_graph, cnode, weight_indices, weight_quant_type, q_min, q_max, symmetric);
    if (status != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " do weight quantize error";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int WeightQuantizer::DoCNodeWeightQuant(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                        const std::vector<int> &weight_indices, WeightQuantType weight_quant_type,
                                        int q_min, int q_max, bool symmetric) {
  CHECK_NULL_RETURN(cnode);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  auto manager = api::FuncGraphManager::Manage(func_graph, true);
  CHECK_NULL_RETURN(manager);
  for (auto idx : weight_indices) {
    auto input = cnode->input(idx);
    ParameterPtr parameter;
    tensor::TensorPtr tensor_info;
    GetLiteParameter(input, &parameter, &tensor_info);
    if (parameter == nullptr || tensor_info == nullptr || tensor_info->data_type() != TypeId::kNumberTypeFloat32) {
      MS_LOG(INFO) << "This op " << cnode->fullname_with_scope() << " can not quant weight";
      continue;
    }
    int preferred_dim = GetPreferredDim(primitive, idx - 1, ConvertShapeVectorToInt32(tensor_info->shape()));
    auto quant_strategy = std::make_unique<QuantStrategy>(flags_.commonQuantParam.min_quant_weight_size,
                                                          flags_.commonQuantParam.min_quant_weight_channel,
                                                          flags_.commonQuantParam.skip_quant_node);
    CHECK_NULL_RETURN(quant_strategy);
    if (!quant_strategy->CanTensorQuantized(cnode, input, preferred_dim)) {
      MS_LOG(INFO) << input->fullname_with_scope() << " is not quantizable";
      continue;
    }
    // support for matmul shared weight
    auto node_map = manager->node_users();
    auto node_user = node_map[input];
    auto tmp_weight_quant_type = weight_quant_type;
    if (node_user.size() > 1 && opt::CheckPrimitiveType(cnode, prim::kPrimMatMulFusion)) {
      MS_LOG(INFO) << input->fullname_with_scope() << " is shared weight.";
      tmp_weight_quant_type = WeightQuantType::FIXED_BIT_PER_LAYER;
    }
    auto status = RET_ERROR;
    if (is_mixed_bit_) {
      status = MixedBitQuantFilter(parameter, tensor_info, primitive, QuantType_QUANT_WEIGHT,
                                   WeightQuantType::MIXED_BIT_PER_LAYER, type_id_, mixed_bit_init_scale_, idx - 1);
    } else if (type_id_ == kNumberTypeInt8) {
      status = FixedBitQuantFilter<int8_t>(parameter, tensor_info, primitive, QuantType_QUANT_WEIGHT, q_max, q_min,
                                           bit_num_, tmp_weight_quant_type, type_id_, idx - 1, symmetric);
    } else if (type_id_ == kNumberTypeInt16) {
      status = FixedBitQuantFilter<int16_t>(parameter, tensor_info, primitive, QuantType_QUANT_WEIGHT, q_max, q_min,
                                            bit_num_, tmp_weight_quant_type, type_id_, idx - 1, symmetric);
    }
    if (status == RET_NO_CHANGE) {
      continue;
    } else if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
    weight_quantized_tensors_.insert(tensor_info);
  }
  return RET_OK;
}

int WeightQuantizer::DoMarkWeightQuantizeIfQuantized(const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_NULL_PTR);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_ERROR;
  }

  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  if (quant_param_holder->quant_type() == schema::QuantType_QUANT_WEIGHT ||
      quant_param_holder->quant_type() == schema::QuantType_QUANT_DANAMIC) {
    // already marked with QuantType_QUANT_WEIGHT or QuantType_QUANT_DANAMIC
    return RET_OK;
  }

  for (size_t i = 1; i < cnode->size(); i++) {
    auto inputNode = cnode->input(i);
    if (inputNode->isa<Parameter>()) {
      ParameterPtr param_node;
      tensor::TensorPtr tensor_info;
      GetLiteParameter(inputNode, &param_node, &tensor_info);
      auto param = weight_quantized_tensors_.find(tensor_info);
      if (param != weight_quantized_tensors_.end()) {
        quant_param_holder->set_quant_type(schema::QuantType_QUANT_WEIGHT);
        continue;
      }
    }
  }
  return RET_OK;
}

int WeightQuantizer::MarkWeightQuantizationInNodes(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_NULL_PTR);
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto primitive = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(DEBUG) << cnode->fullname_with_scope() << " : primitive is nullptr";
      continue;
    }
    auto status = DoMarkWeightQuantizeIfQuantized(cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "MarkWeightQuantizationInNodes error marking " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int WeightQuantizer::DoQuantize(const FuncGraphPtr &func_graph, double init_scale) {
  mixed_bit_init_scale_ = init_scale;
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_NULL_PTR);
  weight_quantized_tensors_.clear();
  const std::set<PrimitivePtr> support_primitive_types = {prim::kPrimConv2DFusion, prim::kPrimConv2dTransposeFusion,
                                                          prim::kPrimMatMulFusion, prim::kPrimFullConnection,
                                                          prim::kPrimLstm,         prim::kPrimGather,
                                                          prim::kPrimAdam,         prim::kPrimSGD,
                                                          prim::kPrimApplyMomentum};
  std::set<PrimitivePtr> per_layer_primitive_types = {prim::kPrimAdam, prim::kPrimSGD, prim::kPrimApplyMomentum};
  auto ret = WeightQuant(func_graph, support_primitive_types, per_layer_primitive_types, {});
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Weight Quant failed.";
    return ret;
  }
  return MarkWeightQuantizationInNodes(func_graph);
}

int WeightQuantizer::DoQuantize(FuncGraphPtr func_graph) { return DoQuantize(func_graph, mixed_bit_init_scale_); }
}  // namespace mindspore::lite::quant
