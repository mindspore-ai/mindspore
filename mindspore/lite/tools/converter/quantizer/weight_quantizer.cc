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

namespace mindspore {
namespace lite {
namespace quant {
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
  if (!WeightQuantizer::IsPosNum(config->convWeightQuantChannelThreshold)) {
    MS_LOG(ERROR) << "convWeightQuantChannelThreshold must be valid pos num.";
    return RET_ERROR;
  }
  if (!WeightQuantizer::IsPosNum(config->quantSize)) {
    MS_LOG(ERROR) << "quantSize must be valid pos num.";
    return RET_ERROR;
  }
  if (!WeightQuantizer::IsPosNum(config->bitNum) || config->bitNum != "8") {
    MS_LOG(ERROR) << "bitNum must be valid pos num, current only support 8 bit weight quant.";
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
  mStrategy.reset(new QuantStrategy(quantSize, convQuantWeightChannelThreshold));
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

    std::vector<schema::QuantParamT> quant_params;
    primitive_c->AddInputQuantParam(quant_params);
    auto status =
      QuantFilter<int8_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max, quant_min, bitNum, true);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
    // set dtype
    param_value->set_tensor_type(kNumberTypeInt8);
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
    abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt8));
    primitive_c->SetQuantType(schema::QuantType_WeightQuant);
  }

  return RET_OK;
}

STATUS WeightQuantizer::DoMulQuantize(const std::list<CNodePtr> &nodes) {
  for (auto &node : nodes) {
    if (!mStrategy->CanMulOpQuantized(node)) {
      continue;
    }

    ParamValueLitePtr param_value = nullptr;
    ParameterPtr param_node = nullptr;
    for (size_t i = 1; i < node->size(); i++) {
      auto inputNode = node->input(i);
      if (inputNode->isa<Parameter>() == true) {
        param_node = inputNode->cast<ParameterPtr>();
        if ((param_node != nullptr) && (param_node->has_default() == true)) {
          param_value = std::static_pointer_cast<ParamValueLite>(param_node->default_param());
          if ((param_value == nullptr) || (param_value->tensor_size() == 0) ||
              (param_value->tensor_addr() == nullptr) ||
              (param_value->tensor_type() != mindspore::kNumberTypeFloat32)) {
            param_value = nullptr;
            continue;
          } else {
            break;
          }
        }
      }
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

    std::vector<schema::QuantParamT> quant_params;
    primitive_c->AddInputQuantParam(quant_params);
    auto status =
      QuantFilter<int8_t>(param_value, primitive_c, QuantType_WeightQuant, quant_max, quant_min, bitNum, true);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
    param_value->set_tensor_type(kNumberTypeInt8);
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
    abstractTensor->element()->set_type(TypeIdToType(kNumberTypeInt8));
    primitive_c->SetQuantType(schema::QuantType_WeightQuant);
  }

  return RET_OK;
}

STATUS WeightQuantizer::DoQuantize(FuncGraphPtr funcGraph) {
  auto ret = RET_OK;
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
}  // namespace quant
}  // namespace lite
}  // namespace mindspore
