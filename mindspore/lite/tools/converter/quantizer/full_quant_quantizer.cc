/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/full_quant_quantizer.h"
#include <dirent.h>
#include <set>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include "ops/tuple_get_item.h"
#include "src/tensor.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_adapter.h"
#include "securec/include/securec.h"
#include "tools/common/tensor_util.h"
#include "src/common/utils.h"
#include "tools/common/node_util.h"
#include "nnacl/op_base.h"
#include "src/common/log_util.h"
#include "tools/converter/quantizer/bias_correction_strategy.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
namespace {
static const std::set<PrimitivePtr> has_bias_operator = {prim::kPrimConv2DFusion, prim::kPrimConv2dTransposeFusion,
                                                         prim::kPrimMatMulFusion, prim::kPrimFullConnection,
                                                         prim::kPrimLayerNormFusion};
}  // namespace
FullQuantQuantizer::~FullQuantQuantizer() {
  delete fp32_session_;
  delete fp32_model_;
}

int FullQuantQuantizer::SetInOutQuantParam(const AnfNodePtr &input_node, const std::unique_ptr<DataDistribution> &info,
                                           const PrimitivePtr &primitive, bool is_input, size_t index) const {
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
    quant_param.numBits = bit_num_;
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

int FullQuantQuantizer::DoParameterWeightQuant(const ParameterPtr &weight, const PrimitivePtr &primitive,
                                               bool per_channel, int input_index) const {
  CHECK_NULL_RETURN(weight);
  CHECK_NULL_RETURN(primitive);
  auto tensor_info = weight->default_param()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " can not get value";
    return RET_NULL_PTR;
  }
  auto weight_quant_type = per_channel ? WeightQuantType::FIXED_BIT_PER_CHANNEL : WeightQuantType::FIXED_BIT_PER_LAYER;
  auto status =
    FixedBitQuantFilter<int8_t>(weight, tensor_info, primitive, QuantType_QUANT_ALL, weight_q_max_, weight_q_min_,
                                bit_num_, weight_quant_type, kNumberTypeInt8, input_index - 1, weight_symmetry_, true);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed: " << status;
    return status;
  }
  return RET_OK;
}

int FullQuantQuantizer::DoValueNodeWeightQuant(const ValueNodePtr &weight, const PrimitivePtr &primitive,
                                               bool per_channel, int input_index) const {
  CHECK_NULL_RETURN(weight);
  CHECK_NULL_RETURN(primitive);
  auto tensor_info = weight->value()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << weight->fullname_with_scope() << " can not get value";
    return RET_NULL_PTR;
  }
  auto weight_quant_type = per_channel ? WeightQuantType::FIXED_BIT_PER_CHANNEL : WeightQuantType::FIXED_BIT_PER_LAYER;
  auto status =
    FixedBitQuantFilter<int8_t>(weight, tensor_info, primitive, QuantType_QUANT_ALL, weight_q_max_, weight_q_min_,
                                bit_num_, weight_quant_type, kNumberTypeInt8, input_index - 1, weight_symmetry_, true);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed: " << status;
    return status;
  }
  return RET_OK;
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
    auto iter = weight_quant_params_bak.find(input_node->fullname_with_scope());
    if (iter == weight_quant_params_bak.end()) {
      return RET_ERROR;
    } else {
      auto quant_param_holder = GetCNodeQuantHolder(primitive);
      quant_param_holder->set_input_quant_param(input_index - 1, iter->second);
      return RET_NO_CHANGE;
    }
  }
  // Only data the data type is fp32 can be quant.
  if (type_id != kNumberTypeFloat32) {
    auto ret = SetInOutQuantParam(input_node, nullptr, primitive, true, input_index - 1);
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
  if (input_index == THIRD_INPUT + 1 && CheckNodeInSet(cnode, has_bias_operator)) {
    ret = DoParameterBiasQuant(input_node, primitive);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op_name << " Do bias quant failed.";
      return ret;
    }
  } else if (CheckNodeInSet(cnode, per_channel_ops_)) {
    ret = DoParameterWeightQuant(input_node, primitive, true, input_index);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op_name << " Do bias quant failed.";
      return ret;
    }
  } else {
    ret = DoParameterWeightQuant(input_node, primitive, false, input_index);
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
  ret = DoValueNodeWeightQuant(input_node, primitive, false, input_index);
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
  int ret;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    MS_ASSERT(input_node != nullptr);
    bool is_graph_input = IsGraphInput(input_node);
    if (is_graph_input) {
      // do input quant
      auto &info = (*inputs_diverg_info)[op_name][i - 1];
      ret = SetInOutQuantParam(input_node, info, primitive, true, i - 1);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << input_node->fullname_with_scope() << " Set activation quant failed.";
        return ret;
      }
    } else if (input_node->isa<mindspore::CNode>()) {
      auto input_cnode = input_node->cast<mindspore::CNodePtr>();
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
        ret = SetInOutQuantParam(input_node, info, primitive, true, i - 1);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << input_node->fullname_with_scope() << " Set activation quant failed.";
          return ret;
        }
      }
    } else if (input_node->isa<mindspore::Parameter>()) {
      ret = DoParameterNodeQuant(cnode, input_node->cast<ParameterPtr>(), i);
      if (ret == RET_NO_CHANGE) {
        continue;
      } else if (ret != RET_OK) {
        MS_LOG(ERROR) << input_node->fullname_with_scope() << " Do parameter node quant failed.";
        return ret;
      }
      // support shared weight
      weight_quant_params_bak[input_node->fullname_with_scope()] =
        primitive_quant_holder->get_input_quant_params()[i - 1];
    } else if (input_node->isa<mindspore::ValueNode>()) {
      ret = DoValueNodeQuant(cnode, input_node->cast<ValueNodePtr>(), i);
      if (ret == RET_NO_CHANGE) {
        continue;
      } else if (ret != RET_OK) {
        MS_LOG(ERROR) << input_node->fullname_with_scope() << " Do value node quant failed.";
        return ret;
      }
      // support shared weight
      weight_quant_params_bak[input_node->fullname_with_scope()] =
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
  for (auto &cnode : cnodes) {
    auto op_name = cnode->fullname_with_scope();
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "primitive is nullptr";
      continue;
    }
    auto primitive_quant_holder = GetCNodeQuantHolder(primitive);
    MS_CHECK_TRUE_MSG(primitive_quant_holder != nullptr, RET_NULL_PTR, "primitive_quant_holder is nullptr.");
    if (inputs_diverg_info->find(op_name) == inputs_diverg_info->end()) {
      MS_LOG(INFO) << op_name << " can not do quant";
      primitive_quant_holder->set_quant_type(schema::QuantType_QUANT_NONE);
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
      size_t index = opt::CastToInt(index_value_node->value()).front();
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
      primitive_quant_holder->set_quant_type(schema::QuantType_QUANT_ALL);
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
      auto ret = SetInOutQuantParam(cnode, info, primitive, false, index);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Set In/Out quant param failed.";
        return ret;
      }
      primitive_quant_holder->set_quant_type(schema::QuantType_QUANT_ALL);
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

void FullQuantQuantizer::InitCpuConfig() {
  activation_quant_data_type_ = kNumberTypeInt8;
  activation_target_data_type_ = kNumberTypeInt8;
  weight_data_type_ = kNumberTypeInt8;
  activation_symmetry_ = false;
  weight_symmetry_ = true;
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
    prim::kPrimSplit,
    prim::kPrimTupleGetItem,
    prim::kPrimConcat,
    prim::kPrimCrop,
    prim::kPrimGather,
    prim::kPrimReduceFusion,
    prim::kPrimAffine,
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
  activation_quant_data_type_ = kNumberTypeUInt8;
  activation_target_data_type_ = kTypeUnknown;
  weight_data_type_ = kNumberTypeInt8;
  activation_symmetry_ = false;
  weight_symmetry_ = true;
  support_int8_ops_ = {prim::kPrimConv2DFusion, prim::kPrimFullConnection};
  flags_.fullQuantParam.bias_correction = false;
  per_channel_ops_ = {prim::kPrimConv2DFusion};
}

void FullQuantQuantizer::InitQMinMax() {
  MS_ASSERT(activation_quant_data_type_ == kNumberTypeInt8 || activation_quant_data_type_ == kNumberTypeUInt8);
  if (activation_quant_data_type_ == kNumberTypeInt8) {
    activation_q_min_ = QuantMin(this->bit_num_, false, activation_symmetry_);  // -128
    activation_q_max_ = QuantMax(this->bit_num_, false);                        // 127
  } else if (activation_quant_data_type_ == kNumberTypeUInt8) {
    activation_q_min_ = QuantMin(this->bit_num_, true, false);  // 0
    activation_q_max_ = QuantMax(this->bit_num_, true);         // 255
  }
  MS_ASSERT(weight_data_type_ == kNumberTypeInt8 || weight_data_type_ == kNumberTypeUInt8);
  if (weight_data_type_ == kNumberTypeInt8) {
    weight_q_min_ = QuantMin(this->bit_num_, false, weight_symmetry_);  // -127
    weight_q_max_ = QuantMax(this->bit_num_, false);                    // 127
  } else if (activation_quant_data_type_ == kNumberTypeUInt8) {
    weight_q_min_ = QuantMin(this->bit_num_, true, false);  // 0
    weight_q_max_ = QuantMax(this->bit_num_, true);         // 255
  }
}

int FullQuantQuantizer::MarkQuantNode(const FuncGraphPtr &func_graph) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto anode = cnode->cast<AnfNodePtr>();
    if (anode == nullptr) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " cnode is null";
      return RET_NULL_PTR;
    }
    auto is_skip_op = quant_strategy_->IsSkipOp(anode->fullname_with_scope());
    if (is_skip_op) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " is skip quant.";
      continue;
    }
    //  Mark quantifiable nodes
    auto is_support_op =
      quant_strategy_->CanOpFullQuantized(anode, support_int8_ops_, skip_check_dtype_ops_, support_activation_);
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

int FullQuantQuantizer::PreProcess(const FuncGraphPtr &func_graph) {
  switch (flags_.fullQuantParam.target_device) {
    case CPU:
      InitCpuConfig();
      break;
    case KIRIN:
      InitKirinConfig();
      break;
    default:
      MS_LOG(ERROR) << " Unsupported device " << flags_.fullQuantParam.target_device;
      return RET_ERROR;
      break;
  }
  InitQMinMax();
  calibrator_ = std::make_shared<Calibrator>(this->bit_num_, activation_q_max_, activation_q_min_,
                                             this->flags_.fullQuantParam.activation_quant_method,
                                             this->flags_.dataPreProcessParam, activation_symmetry_);
  MSLITE_CHECK_PTR(calibrator_);
  quant_strategy_ = std::make_unique<QuantStrategy>(flags_.commonQuantParam.min_quant_weight_size,
                                                    flags_.commonQuantParam.min_quant_weight_channel,
                                                    flags_.commonQuantParam.skip_quant_node);
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
  vector<mindspore::tensor::MSTensor *> inputs = fp32_session_->GetInputs();
  if (inputs.size() != calibrator_->GetInputNum()) {
    MS_LOG(ERROR) << "model's input tensor count: " << inputs.size() << " != "
                  << " calibrator count:" << calibrator_->GetInputNum();
    return RET_ERROR;
  }

  for (size_t calib_index = 0; calib_index < calibrator_->GetBatchNum(); calib_index++) {
    // set multi-input data
    for (size_t input_index = 0; input_index < inputs.size(); input_index++) {
      int status = calibrator_->GenerateInputData(inputs[input_index]->tensor_name(), calib_index, inputs[input_index]);
      MS_CHECK_TRUE_MSG(status == RET_OK, RET_ERROR, "generate input data from images failed!");
    }

    KernelCallBack beforeCallBack = [&](const std::vector<mindspore::tensor::MSTensor *> &beforeInputs,
                                        const std::vector<mindspore::tensor::MSTensor *> &beforeOutputs,
                                        const CallBackParam &callParam) -> bool {
      auto diverg_info_map = calibrator_->GetInputDivergInfo();
      auto ret = calibrator_->CollectDataDistribution(callParam.node_name, beforeInputs, diverg_info_map, collect_type);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "CollectDataDistribution failed.";
        return false;
      }
      return true;
    };
    // func
    KernelCallBack afterCallBack = [&](const std::vector<mindspore::tensor::MSTensor *> &afterInputs,
                                       const std::vector<mindspore::tensor::MSTensor *> &afterOutputs,
                                       const CallBackParam &callParam) -> bool {
      auto diverg_info_map = calibrator_->GetOutputDivergInfo();
      auto ret = calibrator_->CollectDataDistribution(callParam.node_name, afterOutputs, diverg_info_map, collect_type);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "CollectDataDistribution failed.";
        return false;
      }
      return true;
    };
    fp32_session_->BindThread(true);
    auto status = fp32_session_->RunGraph(beforeCallBack, afterCallBack);
    fp32_session_->BindThread(false);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "run model failed!";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int FullQuantQuantizer::DoQuantize(FuncGraphPtr func_graph) {
  MS_LOG(INFO) << "start to parse config file";
  if (flags_.dataPreProcessParam.calibrate_path.empty()) {
    MS_LOG(ERROR) << "calibrate path must pass. The format is input_name_1:input_1_dir,input_name_2:input_2_dir.";
    return RET_INPUT_PARAM_INVALID;
  }

  int status = PreProcess(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "do pre process failed!";
    return status;
  }

  // anf -- fb
  flags_.commonQuantParam.quant_type = schema::QuantType_QUANT_NONE;
  MS_LOG(INFO) << "start create session";
  auto sm = CreateSessionByFuncGraph(func_graph, flags_, this->flags_.commonQuantParam.thread_num);
  fp32_session_ = sm.session;
  fp32_model_ = sm.model;
  if (fp32_session_ == nullptr || fp32_model_ == nullptr) {
    MS_LOG(ERROR) << "create session failed!";
    return RET_ERROR;
  }
  MS_LOG(INFO) << "start to update divergence's max value";
  status = DoInference(MIN_MAX);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Do inference failed.";
    return status;
  }

  if (flags_.fullQuantParam.activation_quant_method == KL) {
    MS_LOG(INFO) << "start to update divergence's interval";
    status = UpdateDivergeInterval();
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
  }

  MS_LOG(INFO) << "start to generate quant param and quantize tensor's data";
  status = QuantNode(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Quant node failed.";
    return status;
  }
  if (activation_target_data_type_ == kNumberTypeInt8 || activation_target_data_type_ == kNumberTypeUInt8) {
    // add quant_cast
    quant::QuantCast quant_cast;
    status = quant_cast.Run(func_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "add QuantCast error";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return RET_ERROR;
    }
  }
  if (this->flags_.fullQuantParam.bias_correction) {
    MS_LOG(INFO) << "do bias correction";
    BiasCorrectionStrategy strategy(flags_, calibrator_, quant_strategy_, fp32_session_, fp32_model_, activation_q_min_,
                                    activation_q_max_);
    switch (this->flags_.fullQuantParam.target_device) {
      case CPU:
        status = strategy.DoCPUBiasCorrection(func_graph);
        break;
      case NVGPU:
        status = strategy.DoNVGPUBiasCorrection(func_graph);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported target device " << this->flags_.fullQuantParam.target_device
                      << " for bias correction.";
        return RET_ERROR;
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "bias_correction failed.";
      return status;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
