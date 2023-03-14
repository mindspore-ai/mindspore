/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "tools/common/node_util.h"
#include "src/common/quant_utils.h"

namespace mindspore::lite::quant {
namespace {
tensor::TensorPtr ConvertParameterFp16TensorToFp32(const ParameterPtr &parameter) {
  if (!parameter->has_default()) {
    MS_LOG(WARNING) << parameter->fullname_with_scope() << " not has_default";
    return nullptr;
  }
  auto tensor_info = parameter->default_param()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(WARNING) << "default_param can not cast to tensor::Tensor";
    return nullptr;
  }
  if (tensor_info->data_type() == kNumberTypeFloat16) {
    MS_LOG(INFO) << "convert " << parameter->fullname_with_scope() << " from fp16 to fp32.";
    auto data = static_cast<float16 *>(tensor_info->data_c());
    std::vector<float> fp32_data(tensor_info->DataSize());
    for (size_t j = 0; j < tensor_info->DataSize(); j++) {
      fp32_data[j] = mindspore::Float16::ToFloat32(data[j]);
    }
    mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(
      kNumberTypeFloat32, tensor_info->shape_c(), fp32_data.data(), fp32_data.size() * sizeof(float));
    parameter->set_default_param(tensor_ptr);
    parameter->set_abstract(tensor_ptr->ToAbstract());
    return tensor_ptr;
  }
  return tensor_info;
}
}  // namespace
int WeightQuantizer::WeightQuant(const FuncGraphPtr &func_graph,
                                 const std::set<PrimitivePtr> &support_weight_quant_types,
                                 const std::set<PrimitivePtr> &per_layer_types,
                                 const std::set<PrimitivePtr> &symmetric_types, bool compression) {
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto ret =
      WeightQuantPerCNode(func_graph, cnode, support_weight_quant_types, per_layer_types, symmetric_types, compression);
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
                                         const std::set<PrimitivePtr> &symmetric_types, bool compression) {
  auto primitive = GetValueNode<std::shared_ptr<Primitive>>(cnode->input(0));
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
    auto ret = LinearQuant(func_graph, cnode, per_layer_types, symmetric_types, weight_indices, compression);
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
                                 bool compression) {
  CHECK_NULL_RETURN(cnode);
  // Avoid affecting other operators
  auto tmp_weight_quant_type = weight_quant_type_;
  if (CheckNodeInSet(cnode, per_layer_types)) {
    tmp_weight_quant_type = WeightQuantType::FIXED_BIT_PER_LAYER;
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
    GetParameterAndTensor(input, &parameter, &tensor_info);
    if (parameter == nullptr || tensor_info == nullptr ||
        tensor_info->compression_type() != mindspore::kNoCompression) {
      MS_LOG(INFO) << "This op " << cnode->fullname_with_scope() << " dont need quant weight";
      continue;
    }
    tensor_info = ConvertParameterFp16TensorToFp32(parameter);
    if (tensor_info == nullptr || tensor_info->data_type() != TypeId::kNumberTypeFloat32) {
      MS_LOG(INFO) << "This op " << input->fullname_with_scope() << " is null or dtype is not fp32.";
      continue;
    }
    int preferred_dim = GetPreferredDim(cnode, idx - 1, ConvertShapeVectorToInt32(tensor_info->shape()));
    if (quant_strategy_ != nullptr && !quant_strategy_->CanTensorQuantized(cnode, input, preferred_dim)) {
      MS_LOG(INFO) << input->fullname_with_scope() << " will not quantify";
      continue;
    }
    // support for matmul shared weight
    auto node_map = manager->node_users();
    auto node_user = node_map[input];
    if (node_user.size() > 1 && opt::CheckPrimitiveType(cnode, prim::kPrimMatMulFusion)) {
      MS_LOG(INFO) << input->fullname_with_scope() << " is shared weight.";
      tmp_weight_quant_type = WeightQuantType::FIXED_BIT_PER_LAYER;
    }
    auto status = RET_ERROR;
    if (is_mixed_bit_) {
      status = DoMixBitQuant(cnode, parameter, idx, tensor_info, preferred_dim, tmp_weight_quant_type, symmetric);
    } else {
      FixedBitWeightQuantization fixed_bit_quant;
      status = fixed_bit_quant.QuantFilter(parameter, tensor_info, primitive, quant_type_, q_max, q_min, bit_num_,
                                           tmp_weight_quant_type, type_id_, idx - 1, preferred_dim, symmetric);
    }
    if (status == RET_NO_CHANGE) {
      continue;
    } else if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
    bool is_compression = (compression && !is_mixed_bit_ && enable_encode_);
    if (is_compression) {
      status = DoCompression(cnode, parameter, idx);
      if (status != RET_OK) {
        MS_LOG(ERROR) << cnode->fullname_with_scope() << " compression failed.";
        return status;
      }
    }
    weight_quantized_tensors_.insert(tensor_info);
    if (dequant_strategy_ == ON_THE_FLY) {
      status = InsertDequantNode(func_graph, cnode, parameter, idx, tensor_info);
      if (status == RET_NO_CHANGE) {
        continue;
      } else if (status != RET_OK) {
        MS_LOG(ERROR) << cnode->fullname_with_scope() << " insert dequan node failed.";
        return status;
      }
    }
  }
  return RET_OK;
}

int WeightQuantizer::DoCompression(const CNodePtr &cnode, const ParameterPtr &parameter, int idx) {
  int ret = RET_OK;
  if (dequant_strategy_ == ON_THE_FLY) {
    if (bit_num_ < k8Bit) {
      FSEEncoder fse_encoder;
      auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
      CHECK_NULL_RETURN(primitive);
      auto quant_param_holder = GetCNodeQuantHolder(primitive);
      auto tensor_quant_params = quant_param_holder->get_input_quant_params();
      MS_CHECK_GT(static_cast<int>(tensor_quant_params.size()), idx - 1, RET_ERROR);
      auto quant_params = tensor_quant_params.at(idx - 1);
      mindspore::TensorCompressionType compress_type =
        dequant_strategy_ == ON_THE_FLY ? mindspore::kFSEInfer : mindspore::kFSE;
      ret = fse_encoder.Compress(parameter, quant_params, compress_type, max_segments_);
      auto new_tensor_info = parameter->default_param()->cast<tensor::TensorPtr>();
      CHECK_NULL_RETURN(new_tensor_info);
      weight_quantized_tensors_.insert(new_tensor_info);
      return ret;
    } else {
      return RET_OK;
    }
  }
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
    // compressed tensor is a new tensor.
    auto tensor_info = parameter->default_param()->cast<tensor::TensorPtr>();
    CHECK_NULL_RETURN(tensor_info);
    weight_quantized_tensors_.insert(tensor_info);
    MS_LOG(INFO) << parameter->fullname_with_scope() << " compression success.";
  }
  return RET_OK;
}

int WeightQuantizer::DoMixBitQuant(const CNodePtr &cnode, const ParameterPtr &parameter, int idx,
                                   const tensor::TensorPtr &tensor_info, int preferred_dim,
                                   WeightQuantType weight_quant_type, bool symmetric) {
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
    mindspore::TensorCompressionType compress_type =
      dequant_strategy_ == ON_THE_FLY ? mindspore::kFSEInfer : mindspore::kFSE;
    status = fse_encoder.Compress(parameter, quant_params, compress_type);
    if (status == RET_OK) {
      quant_param_holder->ClearQuantParams();
    }
    auto new_tensor_info = parameter->default_param()->cast<tensor::TensorPtr>();
    CHECK_NULL_RETURN(new_tensor_info);
    weight_quantized_tensors_.insert(new_tensor_info);
  }
  // rollback to 8 bit.
  if (status == RET_ERROR || status == RET_NO_CHANGE) {
    const int quant_min = QuantMin(k8Bit, false, false);  // -128
    const int quant_max = QuantMax(k8Bit);                // 127
    MS_LOG(WARNING)
      << parameter->fullname_with_scope()
      << " mixed bit quantization search failed, the current layer rolls back to 8 bit fixed quantization.";
    FixedBitWeightQuantization fixed_bit_quant;
    status = fixed_bit_quant.QuantFilter(parameter, tensor_info, primitive, quant_type_, quant_max, quant_min, bit_num_,
                                         weight_quant_type, kNumberTypeInt8, idx - 1, preferred_dim, symmetric);
  }
  return status;
}

int WeightQuantizer::InsertDequantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                       const ParameterPtr &parameter, int idx, const tensor::TensorPtr &tensor_info) {
  InsertQuantNodeManager quant_manager;
  CHECK_NULL_RETURN(func_graph);
  TypeId type_id;
  int status;
  auto tensor_name = parameter->fullname_with_scope();
  if (opt::GetDataTypeFromAnfNode(cnode, &type_id) != RET_OK) {
    MS_LOG(WARNING) << cnode->fullname_with_scope() << " Get data type failed.";
    return RET_NO_CHANGE;
  }
  if (parameter->has_default() &&
      parameter->default_param()->cast<tensor::TensorPtr>()->compression_type() == mindspore::kFSEInfer) {
    MS_LOG(INFO) << tensor_name << " insert FSEDecode node";
    if (type_id == kNumberTypeFloat32) {
      status = quant_manager.InsertFSEDecodeNode(func_graph, cnode, idx, kNumberTypeFloat32);
    } else {
      status = quant_manager.InsertFSEDecodeNode(func_graph, cnode, idx, kNumberTypeFloat16);
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << tensor_name << " insert FSEDecode node failed.";
      return status;
    }
  } else {
    MS_LOG(INFO) << tensor_name << " insert WeightQuant node";
    auto axis = GetPreferredDim(cnode, idx - kPrimOffset, ConvertShapeVectorToInt32(tensor_info->shape_c()));
    if (type_id == kNumberTypeFloat32) {
      status =
        quant_manager.InsertQuantDtypeCastFlyNode(func_graph, cnode, idx, kNumberTypeInt8, kNumberTypeFloat32, axis);
    } else {
      status =
        quant_manager.InsertQuantDtypeCastFlyNode(func_graph, cnode, idx, kNumberTypeInt8, kNumberTypeFloat16, axis);
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << tensor_name << " insert weight quant node failed.";
      return status;
    }
  }
  return RET_OK;
}

int WeightQuantizer::MarkCNodeWeightQuantType(const CNodePtr &cnode) {
  CHECK_NULL_RETURN(cnode);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_ERROR;
  }

  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  CHECK_NULL_RETURN(quant_param_holder);
  if (quant_param_holder->quant_type() == quant::QUANT_WEIGHT) {
    // already marked with QuantType_QUANT_WEIGHT
    return RET_OK;
  }

  // Support Share Weight Quant.
  for (size_t i = kPrimOffset; i < cnode->size(); i++) {
    auto input_node = cnode->input(i);
    if (input_node->isa<Parameter>()) {
      ParameterPtr param_node;
      tensor::TensorPtr tensor_info;
      GetParameterAndTensor(input_node, &param_node, &tensor_info);
      auto param = weight_quantized_tensors_.find(tensor_info);
      if (param != weight_quantized_tensors_.end()) {
        quant_param_holder->set_quant_type(quant::QUANT_WEIGHT);
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
      MS_LOG(DEBUG) << cnode->fullname_with_scope() << " primitive is nullptr";
      continue;
    }
    auto status = MarkCNodeWeightQuantType(cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " mark graph QuantType failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int WeightQuantizer::DoQuantize(FuncGraphPtr func_graph) {
  CHECK_NULL_RETURN(func_graph);
  weight_quantized_tensors_.clear();
  const std::set<PrimitivePtr> support_primitive_types = {prim::kPrimConv2DFusion,  prim::kPrimConv2dTransposeFusion,
                                                          prim::kPrimMatMulFusion,  prim::kPrimFullConnection,
                                                          prim::kPrimLstm,          prim::kPrimGather,
                                                          prim::kPrimAdam,          prim::kPrimSGD,
                                                          prim::kPrimApplyMomentum, prim::kPrimConv2D,
                                                          prim::kPrimMatMul};
  std::set<PrimitivePtr> per_layer_primitive_types = {prim::kPrimAdam, prim::kPrimSGD, prim::kPrimApplyMomentum};
  auto ret = WeightQuant(func_graph, support_primitive_types, per_layer_primitive_types, {});
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Weight Quant failed.";
    return ret;
  }
  if (dequant_strategy_ != ON_THE_FLY) {
    return MarkGraphWeightQuantType(func_graph);
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
