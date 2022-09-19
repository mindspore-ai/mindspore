/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/weight_quantizer.h"
#include <list>
#include <string>
#include <utility>
#include <set>
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_util.h"
#include "tools/converter/quantizer/fse_encoder.h"
#include "tools/converter/quantizer/tensor_compressor.h"
#include "tools/converter/quantizer/cluster_quantization.h"
#include "tools/converter/quantizer/mixed_bit_weight_quantization.h"
#include "tools/converter/quantizer/fixed_bit_weight_quantization.h"
#include "tools/common/node_util.h"
#include "src/common/quant_utils.h"

namespace mindspore::lite::quant {
int WeightQuantizer::WeightQuant(const FuncGraphPtr &func_graph,
                                 const std::set<PrimitivePtr> &support_weight_quant_types,
                                 const std::set<PrimitivePtr> &per_layer_types,
                                 const std::set<PrimitivePtr> &symmetric_types, bool compression,
                                 bool check_quant_conditions, bool update_tensor) {
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (check_quant_conditions) {
      auto quant_param_holder = GetCNodeQuantHolder(cnode);
      if (quant_param_holder == nullptr) {
        continue;
      }
      auto quant_type = quant_param_holder->quant_type();
      if (quant_type != schema::QuantType_QUANT_WEIGHT && quant_type != schema::QuantType_QUANT_ALL) {
        MS_LOG(DEBUG) << "Invalid quant type, dont need weight quant.";
        continue;
      }
      if (CheckWeightQuantExist(cnode)) {
        MS_LOG(INFO) << "Weight quant param exist, cnode name: " << cnode->fullname_with_scope();
        continue;
      }
    }
    auto ret = WeightQuantPerCNode(func_graph, cnode, support_weight_quant_types, per_layer_types, symmetric_types,
                                   compression, update_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " execute weight quantize error.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int WeightQuantizer::WeightQuantPerCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                         const std::set<PrimitivePtr> &support_weight_quant_types,
                                         const std::set<PrimitivePtr> &per_layer_types,
                                         const std::set<PrimitivePtr> &symmetric_types, bool compression,
                                         bool update_tensor) {
  auto primitive = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(DEBUG) << cnode->fullname_with_scope() << " : primitive is nullptr";
    return RET_OK;
  }
  auto op_name = cnode->fullname_with_scope();
  if (skip_quant_node_.find(op_name) != skip_quant_node_.end()) {
    MS_LOG(INFO) << op_name << " is skip dynamic quant.";
    return RET_OK;
  }
  if (!CheckNodeInSet(cnode, support_weight_quant_types)) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << " of type: " << primitive->name() << " dont need weight quant.";
    return RET_OK;
  }

  // Init weight quant index.
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

  if (linear_quant_) {
    auto ret =
      LinearQuant(func_graph, cnode, per_layer_types, symmetric_types, weight_indices, compression, update_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " execute linear weight quantize error.";
      return RET_ERROR;
    }
  } else {
    ClusterQuantization cluster;
    auto ret = cluster.KMeansQuantization(cnode, weight_indices);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " execute k-means weight quantize error.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int WeightQuantizer::LinearQuant(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                 const std::set<PrimitivePtr> &per_layer_types,
                                 const std::set<PrimitivePtr> &symmetric_types, const std::vector<int> &weight_indices,
                                 bool compression, bool update_tensor) {
  CHECK_NULL_RETURN(cnode);
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

  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  auto manager = mindspore::Manage(func_graph, true);
  CHECK_NULL_RETURN(manager);
  for (auto idx : weight_indices) {
    auto input = cnode->input(idx);
    ParameterPtr parameter;
    tensor::TensorPtr tensor_info;
    GetLiteParameter(input, &parameter, &tensor_info);
    if (parameter == nullptr || tensor_info == nullptr || tensor_info->data_type() != TypeId::kNumberTypeFloat32 ||
        tensor_info->compression_type() != kNoCompression) {
      MS_LOG(INFO) << "This op " << cnode->fullname_with_scope() << " dont need quant weight";
      continue;
    }
    int preferred_dim = GetPreferredDim(cnode, idx - 1, ConvertShapeVectorToInt32(tensor_info->shape()));
    if (quant_strategy_ != nullptr && !quant_strategy_->CanTensorQuantized(cnode, input, preferred_dim)) {
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
      status = DoMixBitQuant(cnode, parameter, idx, tensor_info, preferred_dim, tmp_weight_quant_type, symmetric);
    } else {
      FixedBitWeightQuantization fixed_bit_quant;
      if (update_tensor) {
        status = fixed_bit_quant.QuantFilter(parameter, tensor_info, primitive, quant_type_, q_max, q_min, bit_num_,
                                             tmp_weight_quant_type, type_id_, idx - 1, preferred_dim, symmetric);
      } else {
        status =
          fixed_bit_quant.StatisticsFilter(parameter, tensor_info, primitive, quant_type_, q_max, q_min, bit_num_,
                                           tmp_weight_quant_type, type_id_, idx - 1, preferred_dim, symmetric);
      }
    }
    if (status == RET_NO_CHANGE) {
      continue;
    } else if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
    if (compression && !is_mixed_bit_) {
      status = DoCompression(cnode, parameter, idx);
      if (status != RET_OK) {
        MS_LOG(ERROR) << cnode->fullname_with_scope() << " compression failed.";
        return status;
      }
    }
    weight_quantized_tensors_.insert(tensor_info);
  }
  return RET_OK;
}

int WeightQuantizer::DoCompression(const CNodePtr &cnode, const ParameterPtr &parameter, int idx) {
  int ret = RET_OK;
  TensorCompressor compressor;
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  auto tensor_quant_params = quant_param_holder->get_input_quant_params();
  MS_CHECK_GT(static_cast<int>(tensor_quant_params.size()), idx - 1, RET_ERROR);
  auto quant_params = tensor_quant_params.at(idx - 1);
  if (type_id_ == kNumberTypeInt8) {
    ret = compressor.DoSparseCompress<int8_t>(parameter, bit_num_, quant_params);
  } else if (type_id_ == kNumberTypeInt16) {
    ret = compressor.DoSparseCompress<int16_t>(parameter, bit_num_, quant_params);
  }
  if (ret != RET_OK) {
    if (bit_num_ != k8Bit && bit_num_ != k16Bit) {
      auto status = compressor.DoBitPack(parameter, bit_num_);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "do bit pack failed. " << status;
        return RET_ERROR;
      }
    }
  } else {
    auto tensor_info = parameter->default_param()->cast<tensor::TensorPtr>();
    CHECK_NULL_RETURN(tensor_info);
    weight_quantized_tensors_.insert(tensor_info);
    MS_LOG(INFO) << parameter->fullname_with_scope() << " compression success.";
  }
  return RET_OK;
}

int WeightQuantizer::DoMixBitQuant(const CNodePtr &cnode, const ParameterPtr &parameter, int idx,
                                   const tensor::TensorPtr &tensor_info, int preferred_dim,
                                   WeightQuantType weight_quant_type, bool symmetric, bool update_tensor) {
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  auto mixed_bit_quantization = MixedBitWeightQuantization(mixed_bit_init_scale_);
  auto status =
    mixed_bit_quantization.QuantFilter(primitive, parameter, tensor_info, idx - 1, quant_type_, is_auto_tune_);
  if (status == RET_OK) {
    FSEEncoder fse_encoder;
    auto quant_param_holder = GetCNodeQuantHolder(primitive);
    auto tensor_quant_params = quant_param_holder->get_input_quant_params();
    MS_CHECK_GT(static_cast<int>(tensor_quant_params.size()), idx - 1, RET_ERROR);
    auto quant_params = tensor_quant_params.at(idx - 1);
    status = fse_encoder.Compress(parameter, quant_params, kFSE);
    if (status == RET_OK) {
      quant_param_holder->ClearQuantParams();
    }
  }
  // rollback to 8 bit.
  if (status == RET_ERROR || status == RET_NO_CHANGE) {
    const int quant_min = QuantMin(k8Bit, false, false);  // -128
    const int quant_max = QuantMax(k8Bit);                // 127
    MS_LOG(WARNING)
      << parameter->fullname_with_scope()
      << " mixed bit quantization search failed, the current layer rolls back to 8 bit fixed quantization.";
    FixedBitWeightQuantization fixed_bit_quant;
    if (update_tensor) {
      status =
        fixed_bit_quant.QuantFilter(parameter, tensor_info, primitive, quant_type_, quant_max, quant_min, bit_num_,
                                    weight_quant_type, kNumberTypeInt8, idx - 1, preferred_dim, symmetric);
    } else {
      status =
        fixed_bit_quant.StatisticsFilter(parameter, tensor_info, primitive, quant_type_, quant_max, quant_min, bit_num_,
                                         weight_quant_type, kNumberTypeInt8, idx - 1, preferred_dim, symmetric);
    }
  }
  return status;
}

int WeightQuantizer::MarkCnodeWeightQuantType(const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_NULL_PTR);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_ERROR;
  }

  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, RET_NULL_PTR, "quant_param_holder is nullptr.");
  if (quant_param_holder->quant_type() == schema::QuantType_QUANT_WEIGHT) {
    // already marked with QuantType_QUANT_WEIGHT
    return RET_OK;
  }

  // Support Share Weight Quant.
  for (size_t i = kPrimOffset; i < cnode->size(); i++) {
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

int WeightQuantizer::MarkGraphWeightQuantType(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_NULL_PTR);
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto primitive = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(DEBUG) << cnode->fullname_with_scope() << " : primitive is nullptr";
      continue;
    }
    auto status = MarkCnodeWeightQuantType(cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "MarkGraphWeightQuantType error marking " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

bool WeightQuantizer::CheckWeightQuantExist(const CNodePtr &cnode) {
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  MS_CHECK_TRUE_RET(quant_param_holder != nullptr, false);
  for (size_t index = 1; index < cnode->size(); index++) {
    auto input_node = cnode->input(index);
    if (IsGraphInput(input_node)) {
      continue;
    }
    if (input_node->isa<mindspore::Parameter>()) {
      if (index == THIRD_INPUT + 1 && quant::CheckNodeInSet(cnode, quant::kHasBiasOperator)) {
        continue;
      }
      // Constants have quantization parameters
      if (quant_param_holder->CheckInit(index - quant::kPrimOffset, true)) {
        return true;
      }
    }
  }
  return false;
}

int WeightQuantizer::DoQuantize(FuncGraphPtr func_graph) {
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
  return MarkGraphWeightQuantType(func_graph);
}
}  // namespace mindspore::lite::quant
