/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/full_quant_quantizer.h"
#include <dirent.h>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include "ops/tuple_get_item.h"
#include "src/tensor.h"
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_adapter.h"
#include "tools/common/tensor_util.h"
#include "src/common/utils.h"
#include "tools/common/node_util.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "tools/converter/quantizer/bias_correction_strategy.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
FullQuantQuantizer::~FullQuantQuantizer() {}

int FullQuantQuantizer::SetInOutQuantParam(const AnfNodePtr &input_node, const std::unique_ptr<DataDistribution> &info,
                                           const PrimitivePtr &primitive, size_t index, bool is_input) const {
  auto quant_param_holder = GetCNodeQuantHolder(primitive);
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, RET_NULL_PTR, "quant_param_holder is nullptr.");
  schema::QuantParamT quant_param;
  TypeId type_id = kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(input_node, &type_id) != RET_OK) {
    MS_LOG(ERROR) << "Get data type failed.";
    return RET_ERROR;
  }
  if (type_id == kNumberTypeFloat32 && info != nullptr) {
    quant_param.scale = info->GetScale();
    quant_param.zeroPoint = info->GetZeroPoint();
    quant_param.max = info->GetEncodeMax();
    quant_param.min = info->GetEncodeMin();
    quant_param.numBits = init_param_.bit_num_;
    quant_param.narrowRange = true;
    quant_param.inited = true;
    quant_param.roundType = 1;
    quant_param.multiplier = 1;
  } else {
    quant_param.inited = false;
  }
  std::vector<schema::QuantParamT> quant_params = {quant_param};
  if (is_input) {
    quant_param_holder->set_input_quant_param(index, quant_params);
  } else {
    quant_param_holder->set_output_quant_param(index, quant_params);
  }
  return RET_OK;
}

int FullQuantQuantizer::QuantWeight(const CNodePtr &cnode, const PrimitivePtr &primitive, const AnfNodePtr &weight,
                                    int input_index, const tensor::TensorPtr &tensor_info, bool per_channel) {
  int preferred_dim = GetPreferredDim(cnode, input_index - 1, ConvertShapeVectorToInt32(tensor_info->shape()));
  auto weight_quant_type = per_channel ? WeightQuantType::FIXED_BIT_PER_CHANNEL : WeightQuantType::FIXED_BIT_PER_LAYER;
  auto weight_q_min = per_channel ? init_param_.weight_channel_q_min_ : init_param_.weight_layer_q_min_;
  auto weight_q_max = per_channel ? init_param_.weight_channel_q_max_ : init_param_.weight_layer_q_max_;
  auto symmetric = per_channel ? init_param_.weight_channel_symmetric_ : init_param_.weight_layer_symmetric_;
  return fixed_bit_quant_.QuantFilter(weight, tensor_info, primitive, quant::QUANT_ALL, weight_q_max, weight_q_min,
                                      init_param_.bit_num_, weight_quant_type, kNumberTypeInt8, input_index - 1,
                                      preferred_dim, symmetric);
}

int FullQuantQuantizer::DoParameterWeightQuant(const CNodePtr &cnode, const ParameterPtr &weight,
                                               const PrimitivePtr &primitive, int input_index, bool per_channel) {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(weight);
  CHECK_NULL_RETURN(primitive);
  auto tensor_info = weight->default_param()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " can't get value";
    return RET_NULL_PTR;
  }
  return QuantWeight(cnode, primitive, weight, input_index, tensor_info, per_channel);
}

int FullQuantQuantizer::DoValueNodeWeightQuant(const CNodePtr &cnode, const ValueNodePtr &weight,
                                               const PrimitivePtr &primitive, int input_index, bool per_channel) {
  CHECK_NULL_RETURN(weight);
  CHECK_NULL_RETURN(primitive);
  auto tensor_info = weight->value()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " can't get value";
    return RET_NULL_PTR;
  }
  return QuantWeight(cnode, primitive, weight, input_index, tensor_info, per_channel);
}

int FullQuantQuantizer::IsSupportWeightQuant(const CNodePtr &cnode, const AnfNodePtr &input_node, size_t input_index) {
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    return RET_ERROR;
  }
  auto op_name = cnode->fullname_with_scope();
  TypeId type_id = kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(input_node, &type_id) != RET_OK) {
    MS_LOG(ERROR) << "Get data type failed.";
    return RET_ERROR;
  }
  // support for share weight.
  if (type_id == kNumberTypeInt8) {
    auto iter = weight_quant_params_bak_.find(input_node->fullname_with_scope());
    if (iter == weight_quant_params_bak_.end()) {
      return RET_ERROR;
    } else {
      auto quant_param_holder = GetCNodeQuantHolder(primitive);
      MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, RET_NULL_PTR, "quant_param_holder is nullptr.");
      quant_param_holder->set_input_quant_param(input_index - 1, iter->second);
      return RET_NO_CHANGE;
    }
  }
  // Only data the data type is fp32 can be quant.
  if (type_id != kNumberTypeFloat32) {
    auto ret = SetInOutQuantParam(input_node, nullptr, primitive, input_index - 1, true);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op_name << " Set In/Out quant param failed.";
      return ret;
    }
    return RET_NO_CHANGE;
  }
  return RET_OK;
}

int FullQuantQuantizer::DoParameterNodeQuant(const CNodePtr &cnode, const ParameterPtr &input_node,
                                             size_t input_index) {
  auto ret = IsSupportWeightQuant(cnode, input_node, input_index);
  if (ret != RET_OK) {
    return ret;
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  auto op_name = cnode->fullname_with_scope();
  if (input_index == THIRD_INPUT + 1 && CheckNodeInSet(cnode, kHasBiasOperator)) {
    ret = fixed_bit_quant_.QuantBias(input_node, primitive);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op_name << " Do bias quant failed.";
      return ret;
    }
  } else if (param_->fullQuantParam.per_channel && CheckNodeInSet(cnode, per_channel_ops_)) {
    ret = DoParameterWeightQuant(cnode, input_node, primitive, input_index, true);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op_name << " Do bias quant failed.";
      return ret;
    }
  } else {
    ret = DoParameterWeightQuant(cnode, input_node, primitive, input_index, false);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op_name << " Do bias quant failed.";
      return ret;
    }
  }
  return RET_OK;
}

int FullQuantQuantizer::DoValueNodeQuant(const CNodePtr &cnode, const ValueNodePtr &input_node, size_t input_index) {
  auto ret = IsSupportWeightQuant(cnode, input_node, input_index);
  if (ret != RET_OK) {
    return ret;
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  auto op_name = cnode->fullname_with_scope();
  ret = DoValueNodeWeightQuant(cnode, input_node, primitive, input_index, false);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << op_name << " Do value node weight quant failed.";
    return ret;
  }
  return RET_OK;
}

int FullQuantQuantizer::QuantNodeSimpleOp(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto inputs_diverg_info = calibrator_->GetInputDivergInfo();
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    return RET_ERROR;
  }
  auto op_name = cnode->fullname_with_scope();
  auto primitive_quant_holder = GetCNodeQuantHolder(primitive);
  MS_CHECK_TRUE_MSG(primitive_quant_holder != nullptr, RET_NULL_PTR, "primitive_quant_holder is nullptr.");
  MS_ASSERT(cnode->inputs().size() - 1 <= (*inputs_diverg_info)[op_name].size());
  int ret;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    MS_ASSERT(input_node != nullptr);
    bool is_graph_input = IsGraphInput(input_node);
    if (is_graph_input) {
      // do input quant
      auto &info = (*inputs_diverg_info)[op_name][i - 1];
      ret = SetInOutQuantParam(input_node, info, primitive, i - 1, true);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << input_node->fullname_with_scope() << " Set activation quant failed.";
        return ret;
      }
    } else if (input_node->isa<mindspore::CNode>()) {
      auto input_cnode = input_node->cast<mindspore::CNodePtr>();
      MS_CHECK_TRUE_MSG(input_cnode != nullptr, RET_NULL_PTR, "input_cnode is nullptr.");
      auto input_cnode_primitive = GetValueNode<PrimitivePtr>(input_cnode->input(0));
      if (input_cnode_primitive == nullptr) {
        MS_LOG(DEBUG) << "input: " << i << " " << input_cnode->fullname_with_scope() << ": "
                      << " Primitive is null";
        continue;
      }
      auto input_primitive_quant_holder = GetCNodeQuantHolder(input_cnode_primitive);
      MS_CHECK_TRUE_MSG(input_primitive_quant_holder != nullptr, RET_NULL_PTR,
                        "input_primitive_quant_holder is nullptr.");
      if (input_primitive_quant_holder->IsOutputQuantParamsInited()) {
        auto quant_param = input_primitive_quant_holder->get_output_quant_params().front();
        primitive_quant_holder->set_input_quant_param(i - 1, quant_param);
      } else {
        // do input quant
        auto &info = (*inputs_diverg_info)[op_name][i - 1];
        ret = SetInOutQuantParam(input_node, info, primitive, i - 1, true);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << input_node->fullname_with_scope() << " Set activation quant failed.";
          return ret;
        }
      }
    } else if (input_node->isa<mindspore::Parameter>()) {
      if (init_param_.weight_data_type_ == kTypeUnknown) {
        MS_LOG(INFO) << "weight parameters do not need to be quantified.";
        continue;
      }
      ret = DoParameterNodeQuant(cnode, input_node->cast<ParameterPtr>(), i);
      if (ret == RET_NO_CHANGE) {
        continue;
      } else if (ret != RET_OK) {
        MS_LOG(ERROR) << input_node->fullname_with_scope() << " Do parameter node quant failed.";
        return ret;
      }
      // support shared weight
      weight_quant_params_bak_[input_node->fullname_with_scope()] =
        primitive_quant_holder->get_input_quant_params()[i - 1];
    } else if (input_node->isa<mindspore::ValueNode>()) {
      if (init_param_.weight_data_type_ == kTypeUnknown) {
        MS_LOG(INFO) << "weight parameters do not need to be quantified.";
        continue;
      }
      ret = DoValueNodeQuant(cnode, input_node->cast<ValueNodePtr>(), i);
      if (ret == RET_NO_CHANGE) {
        continue;
      } else if (ret != RET_OK) {
        MS_LOG(ERROR) << input_node->fullname_with_scope() << " Do value node quant failed.";
        return ret;
      }
      // support shared weight
      weight_quant_params_bak_[input_node->fullname_with_scope()] =
        primitive_quant_holder->get_input_quant_params()[i - 1];
    } else {
      MS_LOG(ERROR) << input_node->fullname_with_scope() << ":" << input_node->type_name() << " is not support type";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int FullQuantQuantizer::QuantNode(const FuncGraphPtr &func_graph) {
  auto inputs_diverg_info = calibrator_->GetInputDivergInfo();
  auto outputs_diverg_info = calibrator_->GetOutputDivergInfo();

  auto cnodes = func_graph->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    auto op_name = cnode->fullname_with_scope();
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "primitive is nullptr";
      return RET_ERROR;
    }
    auto primitive_quant_holder = GetCNodeQuantHolder(primitive);
    MS_CHECK_TRUE_MSG(primitive_quant_holder != nullptr, RET_NULL_PTR, "primitive_quant_holder is nullptr.");
    if (inputs_diverg_info->find(op_name) == inputs_diverg_info->end()) {
      MS_LOG(INFO) << op_name << " can not do quant";
      primitive_quant_holder->set_quant_type(quant::QUANT_NONE);
      continue;
    }

    auto op_type = primitive->name();
    MS_LOG(DEBUG) << "OpName: " << op_name;
    if (op_type == mindspore::ops::kNameTupleGetItem) {
      constexpr int tuple_get_item_input_size = 3;
      MS_CHECK_TRUE_MSG(cnode->size() == tuple_get_item_input_size, RET_ERROR, "cnode->size() != 3");
      auto index_node = cnode->input(THIRD_INPUT);
      auto index_value_node = index_node->cast<mindspore::ValueNodePtr>();
      if (index_value_node == nullptr) {
        MS_LOG(WARNING) << "index value node is null";
        continue;
      }
      size_t index = static_cast<size_t>(opt::CastToInt(index_value_node->value()).front());
      auto input_node = cnode->input(SECOND_INPUT);
      MS_CHECK_TRUE_MSG(input_node != nullptr, RET_ERROR, "input_node == nullptr");
      auto input_cnode = input_node->cast<mindspore::CNodePtr>();
      MS_CHECK_TRUE_MSG(input_cnode != nullptr, RET_ERROR, "input_cnode == nullptr");
      auto input_cnode_primitive = GetValueNode<PrimitivePtr>(input_cnode->input(0));
      if (input_cnode_primitive == nullptr) {
        MS_LOG(WARNING) << "input_cnode_primitive is null";
        continue;
      }
      auto input_primitive_quant_holder = GetCNodeQuantHolder(input_cnode_primitive);
      MS_CHECK_TRUE_MSG(input_primitive_quant_holder != nullptr, RET_NULL_PTR,
                        "input_primitive_quant_holder is nullptr.");

      if (input_primitive_quant_holder->get_output_quant_params().size() > index) {
        auto quant_param = input_primitive_quant_holder->get_output_quant_params()[index];
        primitive_quant_holder->set_input_quant_param(0, quant_param);
        primitive_quant_holder->set_output_quant_param(0, quant_param);
      } else {
        MS_LOG(WARNING) << "this TupleGetItem node's input node: " << input_cnode->fullname_with_scope()
                        << "'s output quant_params size: "
                        << input_primitive_quant_holder->get_output_quant_params().size() << ", but index: " << index;
      }
      primitive_quant_holder->set_quant_type(quant::QUANT_ALL);
      continue;
    } else {  // do simple op quant
      auto status = QuantNodeSimpleOp(cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "simple op quant failed.";
        return status;
      }
    }
    // do output quant, there may multi-output
    auto &infos = (*outputs_diverg_info)[op_name];
    for (size_t index = 0; index < infos.size(); index++) {
      auto &info = infos.at(index);
      auto ret = SetInOutQuantParam(cnode, info, primitive, index, false);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Set In/Out quant param failed.";
        return ret;
      }
      primitive_quant_holder->set_quant_type(quant::QUANT_ALL);
    }
  }
  return RET_OK;
}

int FullQuantQuantizer::UpdateDivergeInterval() {
  auto ret = this->calibrator_->UpdateDivergInterval();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Update input diverge interval failed.";
    return ret;
  }
  return RET_OK;
}

int FullQuantQuantizer::QuantWithKL() {
  MS_LOG(INFO) << "start to update divergence's interval";
  auto status = UpdateDivergeInterval();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Update diverge interval failed.";
    return status;
  }
  MS_LOG(INFO) << "start to collect data's distribution";
  status = DoInference(KL_BIN);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Collect data frequency failed.";
    return status;
  }
  MS_LOG(INFO) << "compute the best threshold";
  status = this->calibrator_->ComputeThreshold();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "compute threshold failed.";
    return status;
  }
  return RET_OK;
}

void FullQuantQuantizer::InitCpuConfig() {
  init_param_.activation_quant_data_type_ = kNumberTypeInt8;
  init_param_.activation_target_data_type_ = kNumberTypeInt8;
  init_param_.weight_data_type_ = kNumberTypeInt8;
  init_param_.activation_symmetric_ = false;
  init_param_.weight_channel_symmetric_ = true;
  init_param_.weight_layer_symmetric_ = false;
  support_int8_ops_ = {
    // Compute
    prim::kPrimConv2DFusion,
    prim::kPrimFullConnection,
    prim::kPrimMatMulFusion,
    // Memory
    prim::kPrimReshape,
    prim::kPrimTranspose,
    prim::kPrimShape,
    prim::kPrimUnsqueeze,
  };
  skip_check_dtype_ops_ = {prim::kPrimTupleGetItem, prim::kPrimShape};
  per_channel_ops_ = {prim::kPrimConv2DFusion, prim::kPrimConv2dTransposeFusion, prim::kPrimMatMulFusion,
                      prim::kPrimFullConnection, prim::kPrimLayerNormFusion};
  support_activation_ = {
    RELU, RELU6, HSWISH, SIGMOID, TANH,
    // LEAKY_RELU must be symmetric.
  };
}

void FullQuantQuantizer::InitKirinConfig() {
  // `kTypeUnknown` represents the original data type
  init_param_.activation_quant_data_type_ = kNumberTypeUInt8;
  init_param_.activation_target_data_type_ = kTypeUnknown;
  init_param_.weight_data_type_ = kNumberTypeInt8;
  init_param_.activation_symmetric_ = false;
  init_param_.weight_channel_symmetric_ = true;
  init_param_.weight_layer_symmetric_ = false;
  support_int8_ops_ = {prim::kPrimConv2DFusion, prim::kPrimFullConnection};
  param_->fullQuantParam.bias_correction = false;
  per_channel_ops_ = {prim::kPrimConv2DFusion};
}

void FullQuantQuantizer::InitNvGpuConfig() {
  // `kTypeUnknown` represents the original data type
  init_param_.activation_target_data_type_ = kTypeUnknown;
  init_param_.activation_symmetric_ = true;
  init_param_.weight_data_type_ = kTypeUnknown;
  init_param_.weight_channel_symmetric_ = true;
  init_param_.weight_layer_symmetric_ = false;
  support_int8_ops_ = {prim::kPrimConv2DFusion, prim::kPrimMatMul, prim::kPrimActivation,
                       prim::kPrimConv2dTransposeFusion};
  per_channel_ops_ = {};
  param_->fullQuantParam.bias_correction = false;
}

void FullQuantQuantizer::InitDSPConfig() {
  init_param_.activation_quant_data_type_ = kNumberTypeInt8;
  init_param_.activation_target_data_type_ = kNumberTypeInt8;
  init_param_.weight_data_type_ = kNumberTypeInt8;
  init_param_.activation_symmetric_ = false;
  init_param_.weight_channel_symmetric_ = true;
  init_param_.weight_layer_symmetric_ = false;
  // support 52 operators
  support_int8_ops_ = {prim::kPrimAbs,
                       prim::kPrimActivation,
                       prim::kPrimAddFusion,
                       prim::kPrimArgMaxFusion,
                       prim::kPrimAvgPoolFusion,
                       prim::kPrimBatchNorm,
                       prim::kPrimBatchToSpace,
                       prim::kPrimBatchToSpaceND,
                       prim::kPrimCeil,
                       prim::kPrimConcat,
                       prim::kPrimConv2DFusion,
                       prim::kPrimConv2dTransposeFusion,
                       prim::kPrimDepthToSpace,
                       prim::kPrimDivFusion,
                       prim::kPrimExpFusion,
                       prim::kPrimFloor,
                       prim::kPrimFullConnection,
                       prim::kPrimGather,
                       prim::kPrimInstanceNorm,
                       prim::kPrimLeakyRelu,
                       prim::kPrimMatMulFusion,
                       prim::kPrimMaximum,
                       prim::kPrimMaxPoolFusion,
                       prim::kPrimMinimum,
                       prim::kPrimMulFusion,
                       prim::kPrimNeg,
                       prim::kPrimPadFusion,
                       prim::kPrimPReLUFusion,
                       prim::kPrimReduceFusion,
                       prim::kPrimReduceMax,
                       prim::kPrimReduceMean,
                       prim::kPrimReduceMin,
                       prim::kPrimReduceSum,
                       prim::kPrimReduceMax,
                       prim::kPrimReshape,
                       prim::kPrimResize,
                       prim::kPrimResizeBilinear,
                       prim::kPrimResizeNearestNeighbor,
                       prim::kPrimRound,
                       prim::kPrimScaleFusion,
                       prim::kPrimSoftmax,
                       prim::kPrimSpaceToBatch,
                       prim::kPrimSpaceToBatchND,
                       prim::kPrimSpaceToDepth,
                       prim::kPrimSplit,
                       prim::kPrimSqrt,
                       prim::kPrimSqueeze,
                       prim::kPrimStridedSlice,
                       prim::kPrimStack,
                       prim::kPrimSubFusion,
                       prim::kPrimTileFusion,
                       prim::kPrimTranspose,
                       prim::kPrimUnsqueeze};
  skip_check_dtype_ops_ = {prim::kPrimTupleGetItem, prim::kPrimShape};
  per_channel_ops_ = {prim::kPrimConv2DFusion, prim::kPrimConv2dTransposeFusion};
  support_activation_ = {RELU, RELU6, SIGMOID, TANH};
}

void FullQuantQuantizer::InitAscendConfig() {
  // `kTypeUnknown` represents the original data type
  init_param_.activation_quant_data_type_ = kNumberTypeInt8;
  init_param_.activation_target_data_type_ = kNumberTypeInt8;  // It will update to Int32 in acl pass
  init_param_.weight_data_type_ = kNumberTypeInt8;
  init_param_.activation_symmetric_ = false;
  init_param_.weight_channel_symmetric_ = true;
  init_param_.weight_layer_symmetric_ = false;
  support_int8_ops_ = {prim::kPrimConv2DFusion, prim::kPrimFullConnection};
  per_channel_ops_ = {prim::kPrimConv2DFusion};
}

void FullQuantQuantizer::InitQMinMax() {
  MS_ASSERT(init_param_.activation_quant_data_type_ == kNumberTypeInt8 ||
            init_param_.activation_quant_data_type_ == kNumberTypeUInt8);
  if (init_param_.activation_quant_data_type_ == kNumberTypeInt8) {
    init_param_.activation_q_min_ = QuantMin(this->init_param_.bit_num_, false,
                                             init_param_.activation_symmetric_);  // -128
    init_param_.activation_q_max_ = QuantMax(this->init_param_.bit_num_, false);  // 127
  } else if (init_param_.activation_quant_data_type_ == kNumberTypeUInt8) {
    init_param_.activation_q_min_ = QuantMin(this->init_param_.bit_num_, true, false);  // 0
    init_param_.activation_q_max_ = QuantMax(this->init_param_.bit_num_, true);         // 255
  }
  MS_ASSERT(init_param_.weight_data_type_ == kNumberTypeInt8 || init_param_.weight_data_type_ == kNumberTypeUInt8);
  if (init_param_.weight_data_type_ == kNumberTypeInt8) {
    init_param_.weight_channel_q_min_ = QuantMin(this->init_param_.bit_num_, false,
                                                 init_param_.weight_channel_symmetric_);  // -127
    init_param_.weight_channel_q_max_ = QuantMax(this->init_param_.bit_num_, false);      // 127
  } else if (init_param_.activation_quant_data_type_ == kNumberTypeUInt8) {
    init_param_.weight_channel_q_min_ = QuantMin(this->init_param_.bit_num_, true, false);  // 0
    init_param_.weight_channel_q_max_ = QuantMax(this->init_param_.bit_num_, true);         // 255
  }
  if (init_param_.weight_data_type_ == kNumberTypeInt8) {
    init_param_.weight_layer_q_min_ = QuantMin(this->init_param_.bit_num_, false,
                                               init_param_.weight_layer_symmetric_);  // -128
    init_param_.weight_layer_q_max_ = QuantMax(this->init_param_.bit_num_, false);    // 127
  } else if (init_param_.activation_quant_data_type_ == kNumberTypeUInt8) {
    init_param_.weight_layer_q_min_ = QuantMin(this->init_param_.bit_num_, true, false);  // 0
    init_param_.weight_layer_q_max_ = QuantMax(this->init_param_.bit_num_, true);         // 255
  }
}

int FullQuantQuantizer::MarkQuantNode(const FuncGraphPtr &func_graph) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto is_skip_op = quant_strategy_->IsSkipOp(cnode->fullname_with_scope());
    if (is_skip_op) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " is skip quant.";
      continue;
    }
    //  Mark quantifiable nodes
    auto is_support_op =
      quant_strategy_->CanOpFullQuantized(cnode, support_int8_ops_, skip_check_dtype_ops_, support_activation_);
    if (is_support_op) {
      auto ret = calibrator_->AddQuantizedOp(cnode);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << cnode->fullname_with_scope() << " add quantized op failed.";
        return ret;
      }
    }
  }
  return RET_OK;
}

int FullQuantQuantizer::InitDeviceConfig(const FuncGraphPtr &func_graph) {
  switch (param_->fullQuantParam.target_device) {
    case CPU:
      InitCpuConfig();
      break;
    case KIRIN:
      InitKirinConfig();
      break;
    case NVGPU:
      InitNvGpuConfig();
      break;
    case DSP:
      InitDSPConfig();
      break;
    case ASCEND:
      InitAscendConfig();
      break;
    default:
      MS_LOG(ERROR) << " Unsupported device " << param_->fullQuantParam.target_device;
      return RET_ERROR;
  }
  InitQMinMax();
  calibrator_ =
    std::make_shared<Calibrator>(this->init_param_.bit_num_, init_param_.activation_q_max_,
                                 init_param_.activation_q_min_, this->param_->fullQuantParam.activation_quant_method,
                                 this->param_->dataPreProcessParam, init_param_.activation_symmetric_);
  MSLITE_CHECK_PTR(calibrator_);
  quant_strategy_ = std::make_unique<QuantStrategy>(param_->commonQuantParam.min_quant_weight_size,
                                                    param_->commonQuantParam.min_quant_weight_channel,
                                                    param_->commonQuantParam.skip_quant_node);
  CHECK_NULL_RETURN(quant_strategy_);
  auto ret = MarkQuantNode(func_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Mark quant node failed.";
    return ret;
  }
  return RET_OK;
}

int FullQuantQuantizer::DoInference(CollectType collect_type) {
  // get input tensor
  vector<mindspore::MSTensor> inputs = fp32_ms_model_->GetInputs();
  if (inputs.size() != calibrator_->GetInputNum()) {
    MS_LOG(ERROR) << "model's input tensor count: " << inputs.size() << " != "
                  << " calibrator count:" << calibrator_->GetInputNum();
    return RET_ERROR;
  }

  for (size_t calib_index = 0; calib_index < calibrator_->GetBatchNum(); calib_index++) {
    MS_LOG(INFO) << "Do inference round: " << calib_index;
    // set multi-input data
    for (auto tensor : inputs) {
      int status = calibrator_->GenerateInputData(tensor.Name(), calib_index, &tensor);
      MS_CHECK_TRUE_MSG(status == RET_OK, RET_ERROR, "generate input data from images failed!");
    }
    MSKernelCallBack beforeCallBack = [&](const std::vector<mindspore::MSTensor> &beforeInputs,
                                          const std::vector<mindspore::MSTensor> &beforeOutputs,
                                          const MSCallBackParam &callParam) -> bool {
      auto diverg_info_map = calibrator_->GetInputDivergInfo();
      auto ret = calibrator_->CollectDataDistribution(callParam.node_name, beforeInputs, diverg_info_map, collect_type);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "CollectDataDistribution failed.";
        return false;
      }
      return true;
    };
    // func
    MSKernelCallBack afterCallBack = [&](const std::vector<mindspore::MSTensor> &afterInputs,
                                         const std::vector<mindspore::MSTensor> &afterOutputs,
                                         const MSCallBackParam &callParam) -> bool {
      auto diverg_info_map = calibrator_->GetOutputDivergInfo();
      auto ret = calibrator_->CollectDataDistribution(callParam.node_name, afterOutputs, diverg_info_map, collect_type);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "CollectDataDistribution failed.";
        return false;
      }
      return true;
    };
    auto outputs = fp32_ms_model_->GetOutputs();
    auto status = fp32_ms_model_->Predict(inputs, &outputs, beforeCallBack, afterCallBack);
    if (status != mindspore::kSuccess) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int FullQuantQuantizer::DoQuantize(FuncGraphPtr func_graph) {
  MS_ASSERT(func_graph != nullptr);
  MS_LOG(INFO) << "start to parse config file";
  if (param_->dataPreProcessParam.calibrate_path.empty()) {
    MS_LOG(ERROR) << "calibrate path must pass. The format is input_name_1:input_1_dir,input_name_2:input_2_dir.";
    return RET_INPUT_PARAM_INVALID;
  }

  auto status = InitDeviceConfig(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "do pre process failed!";
    return status;
  }

  // anf -- fb
  MS_LOG(INFO) << "start create session";
  fp32_ms_model_ = std::make_shared<mindspore::Model>();
  if (fp32_ms_model_ == nullptr) {
    MS_LOG(ERROR) << "New model failed.";
    return RET_ERROR;
  }
  size_t size = 0;
  auto ret = BuildModelByFuncGraph(fp32_ms_model_, func_graph, param_, &size);
  if (ret != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Build model failed.";
    return RET_ERROR;
  }
  MS_LOG(INFO) << "start to update divergence's max value";
  status = DoInference(MIN_MAX);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Do inference failed.";
    return status;
  }

  if (param_->fullQuantParam.activation_quant_method == KL) {
    status = QuantWithKL();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Quant with KL failed.";
      return status;
    }
  }

  MS_LOG(INFO) << "start to generate quant param and quantize tensor's data";
  status = QuantNode(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Quant node failed.";
    return status;
  }

  if (init_param_.activation_target_data_type_ == kNumberTypeInt8 ||
      init_param_.activation_target_data_type_ == kNumberTypeUInt8) {  // ASCEND bias correction also need it.
    // add quant_cast
    for (auto &cnode : func_graph->GetOrderedCnodes()) {
      quant::QuantType curr_quant_type;
      if (GetQuantType(cnode, &curr_quant_type) != RET_OK) {
        MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
      quant::InsertQuantNodeManager insert_node_manager;
      status = insert_node_manager.InsertCastNodeForFullQuant(func_graph, cnode, kNumberTypeFloat32, curr_quant_type);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertForwardCastNode failed, cnode name: " << cnode->fullname_with_scope();
        ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
        return status;
      }
    }
  }

  if (this->param_->fullQuantParam.bias_correction) {
    MS_LOG(INFO) << "do bias correction";
    BiasCorrectionStrategy strategy(param_, calibrator_, quant_strategy_, fp32_ms_model_, init_param_.activation_q_min_,
                                    init_param_.activation_q_max_);
    status = strategy.DoBiasCorrection(func_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do bias correction failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
