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
#include <vector>
#include <utility>
#include <unordered_map>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/preprocess/image_preprocess.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
WeightQuantizer::WeightQuantizer(FuncGraphPtr graph, const converter::Flags &config) : Quantizer(std::move(graph)) {
  auto quant_size = config.commonQuantParam.min_quant_weight_size;
  this->bit_num_ = config.commonQuantParam.bit_num;
  if (this->bit_num_ == 0) {
    type_id_ = kNumberTypeInt16;
    this->is_mixed_bit_ = true;
  }
  quant_strategy_ = std::make_unique<QuantStrategy>(quant_size, config.commonQuantParam.min_quant_weight_channel);
  // parse param for fixed bit quant.
  if (!this->is_mixed_bit_) {
    quant_max_ = (1 << (unsigned int)(this->bit_num_ - 1)) - 1;
    quant_min_ = -(1 << (unsigned int)(this->bit_num_ - 1));
    // parse type_id_
    if (this->bit_num_ > 0 && this->bit_num_ <= kMaxBit) {
      type_id_ = kNumberTypeInt8;
    } else if (this->bit_num_ <= (kMaxBit * 2)) {
      type_id_ = kNumberTypeInt16;
    } else {
      MS_LOG(ERROR) << "invalid input bits";
    }
  }
}

WeightQuantizer::~WeightQuantizer() {
  for (const auto &fp32_output_tensor : fp32_output_tensors_) {
    for (const auto &kv : fp32_output_tensor) {
      delete kv.second;
    }
  }
}

STATUS WeightQuantizer::SetAbstract(const tensor::TensorPtr &tensor_info, const ParameterPtr &param_node,
                                    const PrimitivePtr &primitive) {
  MS_CHECK_TRUE_MSG(tensor_info != nullptr, RET_NULL_PTR, "tensor_info is nullptr.");
  MS_CHECK_TRUE_MSG(param_node != nullptr, RET_NULL_PTR, "param_node is nullptr.");
  MS_CHECK_TRUE_MSG(primitive != nullptr, RET_NULL_PTR, "primitive is nullptr.");

  // set dtype
  tensor_info->set_data_type(type_id_);
  auto abstract_base = param_node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
    return RET_ERROR;
  }
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << param_node->name();
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  MS_ASSERT(abstract_tensor != nullptr);
  abstract_tensor->element()->set_type(TypeIdToType(type_id_));
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  quant_param_holder->set_quant_type(schema::QuantType_QUANT_WEIGHT);

  weight_quantized_tensors_.insert({tensor_info, param_node});
  return RET_OK;
}

STATUS WeightQuantizer::DoConvQuantize(const CNodePtr &cnode) {
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_ERROR;
  }

  auto input_node = cnode->input(2);
  if (!input_node->isa<Parameter>()) {
    return RET_ERROR;
  }

  ParameterPtr param_node;
  tensor::TensorPtr tensor_info;

  GetLiteParameter(input_node, &param_node, &tensor_info);
  if (param_node == nullptr || tensor_info == nullptr) {
    MS_LOG(ERROR) << "GetLiteParameter error";
    return RET_ERROR;
  }

  if (tensor_info->data_type() != mindspore::kNumberTypeFloat32) {
    MS_LOG(WARNING) << cnode->fullname_with_scope() << " weight data type is not fp32 but " << tensor_info->data_type();
    return RET_OK;
  }
  auto status = RET_ERROR;
  if (is_mixed_bit_) {
    type_id_ = kNumberTypeInt16;
    status = MixedBitQuantFilter(tensor_info, primitive, QuantType_QUANT_WEIGHT, WeightQuantType::MIXED_BIT_PER_LAYER,
                                 type_id_, flags.mixedBitWeightQuantParam.init_scale, 1);
  } else if (type_id_ == kNumberTypeInt8) {
    status = FixedBitQuantFilter<int8_t>(tensor_info, primitive, QuantType_QUANT_WEIGHT, quant_max_, quant_min_,
                                         bit_num_, WeightQuantType::FIXED_BIT_PER_CHANNEL, type_id_);
  } else if (type_id_ == kNumberTypeInt16) {
    status = FixedBitQuantFilter<int16_t>(tensor_info, primitive, QuantType_QUANT_WEIGHT, quant_max_, quant_min_,
                                          bit_num_, WeightQuantType::FIXED_BIT_PER_CHANNEL, type_id_);
  }
  if (status == RET_CONTINUE) {
    return RET_OK;
  } else if (status != RET_OK) {
    MS_LOG(ERROR) << "MixedBitQuantFilter failed : " << status;
    return status;
  }
  status = SetAbstract(tensor_info, param_node, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SetAbstract failed : " << status;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS WeightQuantizer::DoMulQuantize(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  for (size_t i = 1; i < cnode->size(); i++) {
    auto inputNode = cnode->input(i);
    if (inputNode->isa<Parameter>()) {
      auto param_node = inputNode->cast<ParameterPtr>();
      if ((param_node != nullptr) && param_node->has_default()) {
        auto tensor_info = std::static_pointer_cast<tensor::Tensor>(param_node->default_param());
        if ((tensor_info != nullptr) && (tensor_info->data_type() == mindspore::kNumberTypeFloat32) &&
            (tensor_info->Size() > 0) && (tensor_info->data_c() != nullptr)) {
          auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
          if (primitive == nullptr) {
            MS_LOG(ERROR) << "primitive is nullptr";
            return RET_ERROR;
          }

          auto status = RET_ERROR;
          auto weight_quant_type = WeightQuantType::FIXED_BIT_PER_CHANNEL;
          if (i == 3) {
            weight_quant_type = WeightQuantType::FIXED_BIT_PER_LAYER;
          }
          if (is_mixed_bit_) {
            status =
              MixedBitQuantFilter(tensor_info, primitive, QuantType_QUANT_WEIGHT, WeightQuantType::MIXED_BIT_PER_LAYER,
                                  type_id_, flags.mixedBitWeightQuantParam.init_scale, i - 1);
          } else if (type_id_ == kNumberTypeInt8) {
            status = FixedBitQuantFilter<int8_t>(tensor_info, primitive, QuantType_QUANT_WEIGHT, quant_max_, quant_min_,
                                                 bit_num_, weight_quant_type, type_id_, i - 1);
          } else if (type_id_ == kNumberTypeInt16) {
            status = FixedBitQuantFilter<int16_t>(tensor_info, primitive, QuantType_QUANT_WEIGHT, quant_max_,
                                                  quant_min_, bit_num_, weight_quant_type, type_id_, i - 1);
          }
          if (status == RET_CONTINUE) {
            continue;
          } else if (status != RET_OK) {
            MS_LOG(ERROR) << cnode->fullname_with_scope() << " input " << i
                          << " MixedBitQuantFilter failed : " << status;
            return status;
          }
          status = SetAbstract(tensor_info, param_node, primitive);
          if (status != RET_OK) {
            MS_LOG(ERROR) << cnode->fullname_with_scope() << " input " << i << " SetAbstract failed : " << status;
            return RET_ERROR;
          }
        }
      }
    }
  }
  return RET_OK;
}

STATUS WeightQuantizer::DoLstmQuantize(const CNodePtr &cnode) {
  MS_CHECK_FALSE(cnode == nullptr, RET_NULL_PTR);
  auto op_name = cnode->fullname_with_scope();

  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_FALSE(primitive == nullptr, RET_NULL_PTR);

  if (cnode->inputs().size() < 4) {
    MS_LOG(ERROR) << op_name << " inputs is " << cnode->inputs().size();
    return RET_ERROR;
  }

  auto status = ProcessLstmWeightByIndex(cnode, primitive, 2);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Process lstm weight i failed.";
    return RET_ERROR;
  }
  status = ProcessLstmWeightByIndex(cnode, primitive, 3);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Process lstm weight h failed.";
    return RET_ERROR;
  }
  if (cnode->inputs().size() > 4) {
    status = ProcessLstmWeightByIndex(cnode, primitive, 4);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Process lstm bias failed.";
      return RET_ERROR;
    }
  }

  return status;
}

STATUS WeightQuantizer::DoGatherQuantize(const CNodePtr &cnode) {
  MS_CHECK_FALSE(cnode == nullptr, RET_NULL_PTR);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_FALSE(primitive == nullptr, RET_NULL_PTR);
  auto first_input = cnode->input(1);
  ParameterPtr param_node;
  tensor::TensorPtr tensor_info;
  GetLiteParameter(first_input, &param_node, &tensor_info);
  if (param_node == nullptr || tensor_info == nullptr || tensor_info->data_type() != TypeId::kNumberTypeFloat32) {
    MS_LOG(INFO) << "This Gather op " << cnode->fullname_with_scope() << " can not quant weight";
    return RET_OK;
  }

  if (tensor_info->Size() / sizeof(float) < quant_strategy_->m_weight_size_) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << " param cnt: " << (tensor_info->Size() / sizeof(float)) << " < "
                 << quant_strategy_->m_weight_size_;
    return RET_OK;
  }

  auto status = RET_ERROR;
  if (is_mixed_bit_) {
    status = MixedBitQuantFilter(tensor_info, primitive, QuantType_QUANT_WEIGHT, WeightQuantType::MIXED_BIT_PER_LAYER,
                                 type_id_, flags.mixedBitWeightQuantParam.init_scale, 0);
  } else if (type_id_ == kNumberTypeInt8) {
    status = FixedBitQuantFilter<int8_t>(tensor_info, primitive, QuantType_QUANT_WEIGHT, quant_max_, quant_min_,
                                         bit_num_, WeightQuantType::FIXED_BIT_PER_LAYER, type_id_, 0);
  } else if (type_id_ == kNumberTypeInt16) {
    status = FixedBitQuantFilter<int16_t>(tensor_info, primitive, QuantType_QUANT_WEIGHT, quant_max_, quant_min_,
                                          bit_num_, WeightQuantType::FIXED_BIT_PER_LAYER, type_id_, 0);
  }
  if (status == RET_CONTINUE) {
    return RET_OK;
  } else if (status != RET_OK) {
    MS_LOG(ERROR) << "MixedBitQuantFilter failed : " << status;
    return status;
  }
  status = SetAbstract(tensor_info, param_node, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SetAbstract failed : " << status;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS WeightQuantizer::DoOptimizerQuantize(const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_NULL_PTR);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_RET(primitive != nullptr, RET_NULL_PTR);

  std::vector<int> weight_indices = {2};
  if (opt::CheckPrimitiveType(cnode, prim::kPrimAdam)) {
    weight_indices = {2, 3};
  }
  if (opt::CheckPrimitiveType(cnode, prim::kPrimSGD)) {
    weight_indices = {4, 6};
  }

  for (int idx : weight_indices) {
    auto input = cnode->input(idx);
    if (!quant_strategy_->CanTensorQuantized(input)) {
      MS_LOG(INFO) << "Input " << idx << "of Optimizer is not quantizable";
      continue;
    }
    ParameterPtr param_node;
    tensor::TensorPtr tensor_info;
    GetLiteParameter(input, &param_node, &tensor_info);
    if (param_node == nullptr || tensor_info == nullptr || tensor_info->data_type() != TypeId::kNumberTypeFloat32) {
      MS_LOG(INFO) << "This Gather op " << cnode->fullname_with_scope() << " can not quant weight";
      return RET_OK;
    }

    auto status = RET_ERROR;
    if (type_id_ == kNumberTypeInt8) {
      status = FixedBitQuantFilter<int8_t>(tensor_info, primitive, QuantType_QUANT_WEIGHT, quant_max_, quant_min_,
                                           bit_num_, WeightQuantType::FIXED_BIT_PER_LAYER, type_id_, idx - 1);
    } else if (type_id_ == kNumberTypeInt16) {
      status = FixedBitQuantFilter<int16_t>(tensor_info, primitive, QuantType_QUANT_WEIGHT, quant_max_, quant_min_,
                                            bit_num_, WeightQuantType::FIXED_BIT_PER_LAYER, type_id_, idx - 1);
    }
    if (status != RET_OK && status != RET_CONTINUE) {
      MS_LOG(ERROR) << "MixedBitQuantFilter failed : " << status;
      return status;
    }
    status = SetAbstract(tensor_info, param_node, primitive);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "SetAbstract failed : " << status;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS WeightQuantizer::DoMarkWeightQuantizeIfQuantized(const CNodePtr &cnode) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_NULL_PTR);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return RET_ERROR;
  }

  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  if (quant_param_holder->quant_type() == schema::QuantType_QUANT_WEIGHT) {
    // already marked with QUANT_WEIGHT
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

STATUS WeightQuantizer::ProcessLstmWeightByIndex(const CNodePtr &cnode, const PrimitivePtr &primitive,
                                                 const int &index) {
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_NULL_PTR);
  MS_CHECK_TRUE_RET(primitive != nullptr, RET_NULL_PTR);
  auto op_name = cnode->fullname_with_scope();
  auto weight_i = cnode->input(index);
  ParameterPtr param_node;

  tensor::TensorPtr tensor_info;
  GetLiteParameter(weight_i, &param_node, &tensor_info);
  if (param_node == nullptr || tensor_info == nullptr) {
    MS_LOG(INFO) << "LSTM input index " << index << " is not weight";
    return RET_OK;
  }
  if (tensor_info->data_type() != TypeId::kNumberTypeFloat32) {
    MS_LOG(WARNING) << "tensor_info tensor type is: " << tensor_info->data_type() << " not quant";
    return RET_OK;
  }
  if (tensor_info->Size() / sizeof(float) < quant_strategy_->m_weight_size_) {
    MS_LOG(INFO) << op_name << " weight_i cnt: " << (tensor_info->Size() / sizeof(float)) << " < "
                 << quant_strategy_->m_weight_size_;
    return RET_OK;
  }
  auto status = RET_ERROR;
  if (is_mixed_bit_) {
    status = MixedBitQuantFilter(tensor_info, primitive, QuantType_QUANT_WEIGHT, WeightQuantType::MIXED_BIT_PER_LAYER,
                                 type_id_, flags.mixedBitWeightQuantParam.init_scale, index - 1);
  } else if (type_id_ == kNumberTypeInt8) {
    status = FixedBitQuantFilter<int8_t>(tensor_info, primitive, QuantType_QUANT_WEIGHT, quant_max_, quant_min_,
                                         bit_num_, WeightQuantType::FIXED_BIT_PER_CHANNEL, type_id_, index - 1);
  } else if (type_id_ == kNumberTypeInt16) {
    status = FixedBitQuantFilter<int16_t>(tensor_info, primitive, QuantType_QUANT_WEIGHT, quant_max_, quant_min_,
                                          bit_num_, WeightQuantType::FIXED_BIT_PER_CHANNEL, type_id_, index - 1);
  }
  if (status == RET_CONTINUE) {
    return RET_OK;
  } else if (status != RET_OK) {
    MS_LOG(ERROR) << "MixedBitQuantFilter failed : " << status;
    return status;
  }
  status = SetAbstract(tensor_info, param_node, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SetAbstract failed : " << status;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS WeightQuantizer::MarkWeightQuantizationInNodes(const FuncGraphPtr &func_graph) {
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

STATUS WeightQuantizer::DoQuantize(FuncGraphPtr func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_NULL_PTR);
  weight_quantized_tensors_.clear();

  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto primitive = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(DEBUG) << cnode->fullname_with_scope() << " : primitive is nullptr";
      continue;
    }
    auto op_name = cnode->fullname_with_scope();

    if (quant_strategy_->CanConvOpQuantized(cnode)) {
      auto status = DoConvQuantize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoConvQuantize error";
        return RET_ERROR;
      }
    } else if (quant_strategy_->CanMulOpQuantized(cnode)) {
      auto status = DoMulQuantize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoMulQuantize error";
        return RET_ERROR;
      }
    } else if (opt::CheckPrimitiveType(cnode, prim::kPrimLstm)) {
      auto status = DoLstmQuantize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoLstmQuantize error";
        return RET_ERROR;
      }
    } else if (opt::CheckPrimitiveType(cnode, prim::kPrimGather)) {
      auto status = DoGatherQuantize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoGatherQuantize error";
        return RET_ERROR;
      }
    } else if ((opt::CheckPrimitiveType(cnode, prim::kPrimAdam)) || (opt::CheckPrimitiveType(cnode, prim::kPrimSGD)) ||
               (opt::CheckPrimitiveType(cnode, prim::kPrimApplyMomentum))) {
      auto status = DoOptimizerQuantize(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "DoOptimizerQuantize error";
        return RET_ERROR;
      }
    } else {
      MS_LOG(DEBUG) << op_name << " of type: " << primitive->name() << " no need quant";
    }
  }
  return MarkWeightQuantizationInNodes(func_graph);
}
}  // namespace mindspore::lite::quant
