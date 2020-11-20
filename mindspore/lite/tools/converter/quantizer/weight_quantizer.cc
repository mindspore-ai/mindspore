/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/common/common.h"
#include "ir/dtype/type_id.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
bool WeightQuantizer::IsPosNum(const std::string &str) {
  for (size_t i = 0; i < str.size(); i++) {
    if (str.at(i) < '0' || str.at(i) > '9') {
      return false;
    }
    if (str.at(i) == '0' && i == 0 && str.size() != 1) {
      return false;
    }
  }
  return true;
}
STATUS WeightQuantizer::WeightQuantInputCheck(const converter::Flags *config) {
  MS_ASSERT(config != nullptr);
  if (!WeightQuantizer::IsPosNum(config->quantWeightChannel)) {
    MS_LOG(ERROR) << "quantWeightChannel must be valid pos num.";
    return RET_ERROR;
  }
  if (!WeightQuantizer::IsPosNum(config->quantWeightSize)) {
    MS_LOG(ERROR) << "quantWeightSize must be valid pos num.";
    return RET_ERROR;
  }
  if (!WeightQuantizer::IsPosNum(config->bitNum)) {
    MS_LOG(ERROR) << "bitNum must be valid pos num.";
    return RET_ERROR;
  }
  int bitNum = std::stoi(config->bitNum);
  if (bitNum <= 0 || bitNum > 16) {
    MS_LOG(ERROR) << "bitNum should be more than 0 and less than 16 currently.";
    return RET_ERROR;
  }
  return RET_OK;
}
WeightQuantizer::WeightQuantizer(FuncGraphPtr graph, const string &weightSize,
                                 const std::string &convWeightChannelThreshold, const std::string &bitNum)
    : Quantizer(graph) {
  auto quantSize = static_cast<size_t>(std::stoull(weightSize));
  this->bitNum = static_cast<size_t>(std::stoull(bitNum));
  auto convQuantWeightChannelThreshold = static_cast<size_t>(std::stoull(convWeightChannelThreshold));
  mStrategy = std::make_unique<QuantStrategy>(quantSize, convQuantWeightChannelThreshold);
  quant_max = (1 << (unsigned int)(this->bitNum - 1)) - 1;
  quant_min = -(1 << (unsigned int)(this->bitNum - 1));
  // parse type_id
  if (this->bitNum > 0 && this->bitNum <= 8) {
    type_id = kNumberTypeInt8;
  } else if (this->bitNum <= 16) {
    type_id = kNumberTypeInt16;
  } else {
    MS_LOG(ERROR) << "invalid input bits";
  }
}

STATUS WeightQuantizer::DoConvQuantize(const std::list<CNodePtr> &nodes) {
  for (auto &cnode : nodes) {
    if (!mStrategy->CanConvOpQuantized(cnode)) {
      continue;
    }

    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "primitive_c is nullptr";
      return RET_ERROR;
    }

    auto input_node = cnode->input(2);
    if (!input_node->isa<Parameter>()) {
      return RET_ERROR;
    }

    auto param_node = input_node->cast<ParameterPtr>();
    if (!param_node->has_default()) {
      return RET_ERROR;
    }

    ParamValueLitePtr param_value = std::static_pointer_cast<ParamValueLite>(param_node->default_param());
    if (param_value == nullptr) {
      return RET_ERROR;
    }
    if (param_value->tensor_type() != mindspore::kNumberTypeFloat32) {
      MS_LOG(ERROR) << "model weight data type invalid which is " << param_value->tensor_type();
      return RET_ERROR;
    }
    auto status = RET_ERROR;
    if (type_id == kNumberTypeInt8) {
      status = QuantFilter<int8_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max, quant_min, bitNum, true);
    } else if (type_id == kNumberTypeInt16) {
      status =
        QuantFilter<int16_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max, quant_min, bitNum, true);
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
    // set dtype
    param_value->set_tensor_type(type_id);
    auto abstractBase = param_node->abstract();
    if (abstractBase == nullptr) {
      MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
      return RET_ERROR;
    }
    if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
      MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << param_node->name();
      return RET_ERROR;
    }
    auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
    abstractTensor->element()->set_type(TypeIdToType(type_id));
    primitive_c->set_quant_type(schema::QuantType_WeightQuant);
  }
  return RET_OK;
}

STATUS WeightQuantizer::DoMulQuantize(const std::list<CNodePtr> &nodes) {
  for (auto &node : nodes) {
    if (!mStrategy->CanMulOpQuantized(node)) {
      continue;
    }
    auto already_quant = false;
    ParamValueLitePtr param_value = nullptr;
    ParameterPtr param_node = nullptr;
    for (size_t i = 1; i < node->size(); i++) {
      auto inputNode = node->input(i);
      if (inputNode->isa<Parameter>()) {
        param_node = inputNode->cast<ParameterPtr>();
        if ((param_node != nullptr) && param_node->has_default()) {
          param_value = std::static_pointer_cast<ParamValueLite>(param_node->default_param());
          if ((param_value == nullptr) || (param_value->tensor_size() == 0) ||
              (param_value->tensor_addr() == nullptr)) {
            param_value = nullptr;
            continue;
          } else if (param_value->tensor_type() == mindspore::kNumberTypeInt8 ||
                     param_value->tensor_type() == mindspore::kNumberTypeInt16) {
            MS_LOG(INFO) << "the node: " << node->fullname_with_scope() << " input_i: " << i << "has been "
                         << " quantized";
            already_quant = true;
            break;
          } else if (param_value->tensor_type() != mindspore::kNumberTypeFloat32) {
            param_value = nullptr;
            continue;
          } else {
            break;
          }
        }
      }
    }

    if (already_quant) {
      continue;
    }

    if (param_value == nullptr) {
      MS_LOG(ERROR) << "No valid input param node !";
      return RET_ERROR;
    }

    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(node->input(0));
    if (primitive_c == nullptr) {
      MS_LOG(ERROR) << "primitive_c is nullptr";
      return RET_ERROR;
    }

    auto status = RET_ERROR;
    if (type_id == kNumberTypeInt8) {
      status = QuantFilter<int8_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max, quant_min, bitNum, true);
    } else if (type_id == kNumberTypeInt16) {
      status =
        QuantFilter<int16_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max, quant_min, bitNum, true);
    }
    if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
    param_value->set_tensor_type(type_id);
    // set dtype
    auto abstractBase = param_node->abstract();
    if (abstractBase == nullptr) {
      MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
      return RET_ERROR;
    }
    if (!utils::isa<abstract::AbstractTensorPtr>(abstractBase)) {
      MS_LOG(ERROR) << "Abstract of parameter should be anstract tensor, " << param_node->name();
      return RET_ERROR;
    }
    auto abstractTensor = utils::cast<abstract::AbstractTensorPtr>(abstractBase);
    abstractTensor->element()->set_type(TypeIdToType(type_id));
    primitive_c->set_quant_type(schema::QuantType_WeightQuant);
  }

  return RET_OK;
}

STATUS WeightQuantizer::DoQuantize(FuncGraphPtr funcGraph) {
  MS_ASSERT(funcGraph != nullptr);
  STATUS ret;
  auto cnodes = funcGraph->GetOrderedCnodes();
  ret = DoConvQuantize(cnodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoConvQuantize failed :" << ret;
    return ret;
  }
  ret = DoMulQuantize(cnodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoMulQuantize failed :" << ret;
    return ret;
  }
  return ret;
}
}  // namespace mindspore::lite::quant
