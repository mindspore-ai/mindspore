/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/converter/quantizer/quant_helper/quant_node_helper.h"
#include <unordered_map>
#include <memory>
#include "mindspore/core/utils/log_adapter.h"
#include "tools/converter/quantizer/quant_helper/bias_add_quant_param_propogator.h"
#include "tools/converter/quantizer/quant_helper/carry_data_quant_param_propogator.h"
#include "tools/converter/quantizer/quant_helper/carry_data_quant_type_determiner.h"
#include "tools/converter/quantizer/quant_helper/concat_quant_param_propogator.h"
#include "tools/converter/quantizer/quant_helper/conv_quant_param_propogator.h"
#include "tools/converter/quantizer/quant_helper/conv_quant_type_determiner.h"
#include "tools/converter/quantizer/quant_helper/default_quant_all_quant_type_determiner.h"
#include "tools/converter/quantizer/quant_helper/only_need_inputs_quant_type_determiner.h"
#include "tools/converter/quantizer/quant_helper/quant_dtype_cast_quant_param_propogator.h"
#include "tools/converter/quantizer/quant_helper/matmul_quant_type_determiner.h"
#include "src/lite_kernel.h"
#include "src/kernel_registry.h"

namespace mindspore::lite {
void QuantNodeBase::UpdateQuantParamsNum(const schema::MetaGraphT &graph, const schema::CNodeT &node) {
  // update input quant params num
  input_inited_quant_params_ = 0;
  for (auto index : node.inputIndex) {
    MS_ASSERT(graph.allTensors.size() > index);
    auto &input_tensor = graph.allTensors.at(index);
    if (!input_tensor->quantParams.empty()) {
      bool is_quant_params_inited =
        !std::any_of(input_tensor->quantParams.begin(), input_tensor->quantParams.end(),
                     [](const std::unique_ptr<schema::QuantParamT> &quant_param) { return !quant_param->inited; });

      if (is_quant_params_inited) {
        input_inited_quant_params_++;
      }
    }
  }

  // update output quant params num
  output_inited_quant_params_ = 0;
  for (auto index : node.outputIndex) {
    MS_ASSERT(graph.allTensors.size() > index);
    auto &output_tensor = graph.allTensors.at(index);
    if (!output_tensor->quantParams.empty()) {
      bool is_quant_params_inited =
        !std::any_of(output_tensor->quantParams.begin(), output_tensor->quantParams.end(),
                     [](const std::unique_ptr<schema::QuantParamT> &quant_param) { return !quant_param->inited; });

      if (is_quant_params_inited) {
        output_inited_quant_params_++;
      }
    }
  }
}

bool QuantTypeDeterminer::DetermineQuantAll(const schema::MetaGraphT &graph, schema::CNodeT *node) {
  MS_ASSERT(node != nullptr);
  kernel::KernelKey desc{kernel::kCPU, kNumberTypeInt8, node->primitive->value.type, ""};
  if (!KernelRegistry::GetInstance()->SupportKernel(desc)) {
    return false;
  }
  if (node->quantType != schema::QuantType_QUANT_NONE) {
    return node->quantType == schema::QuantType_QUANT_ALL;
  }

  UpdateQuantParamsNum(graph, *node);
  if (input_inited_quant_params_ == node->inputIndex.size() &&
      output_inited_quant_params_ == node->outputIndex.size()) {
    node->quantType = schema::QuantType_QUANT_ALL;
    return true;
  }
  return false;
}
bool QuantTypeDeterminer::DetermineQuantWeight(const schema::MetaGraphT &graph, schema::CNodeT *node) {
  return node->quantType == schema::QuantType_QUANT_WEIGHT;
}

int QuantNodeHelper::NodeQuantPreprocess(schema::MetaGraphT *graph, schema::CNodeT *node) {
  MS_ASSERT(node != nullptr);
  if (quant_type_determiner_->DetermineQuantWeight(*graph, node)) {
    return RET_OK;
  }
  auto ret = quant_param_propogator_->PropogateQuantParams(graph, *node);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << node->name << " propagate Quant Params failed.";
    return ret;
  }
  auto bool_ret = quant_type_determiner_->DetermineQuantAll(*graph, node);
  if (!bool_ret) {
    MS_LOG(DEBUG) << node->name << " dont need quant.";
    return RET_OK;
  }
  return RET_OK;
}

QuantHelperRegister *QuantHelperRegister::GetInstance() {
  static QuantHelperRegister instance;
  return &instance;
}

QuantNodeHelper *QuantHelperRegister::GetQuantHelper(schema::PrimitiveType op_type) {
  auto it = register_map_.find(op_type);
  if (it != register_map_.end()) {
    return it->second;
  }
  return register_map_[schema::PrimitiveType_NONE];
}

QuantHelperRegister::QuantHelperRegister() {
  auto base_propogator = std::make_shared<QuantParamPropogator>();
  auto base_determiner = std::make_shared<QuantTypeDeterminer>();
  auto quant_dtype_cast_propogator = std::make_shared<QuantDtypeCastQuantParamPropogator>();
  auto bias_add_propogator = std::make_shared<BiasAddQuantParamPropogator>();
  auto carry_data_propogator = std::make_shared<CarryDataQuantParamPropogator>();
  auto carry_data_determiner = std::make_shared<CarryDataQuantTypeDeterminer>();
  auto concat_propogator = std::make_shared<ConcatQuantParamPropogator>();
  auto conv_propogator = std::make_shared<ConvQuantParamPropogator>();
  auto conv_determiner = std::make_shared<ConvQuantTypeDeterminer>();
  auto default_quant_all_determiner = std::make_shared<DefaultQuantAllQuantTypeDeterminer>();
  auto only_need_inputs_determiner = std::make_shared<OnlyNeedInputsQuantTypeDeterminer>();
  auto matmul_determiner = std::make_shared<MatmulQuantTypeDeterminer>();

  register_map_[schema::PrimitiveType_BiasAdd] = new QuantNodeHelper(bias_add_propogator, base_determiner);

  register_map_[schema::PrimitiveType_MaxPoolFusion] =
    new QuantNodeHelper(carry_data_propogator, carry_data_determiner);
  register_map_[schema::PrimitiveType_Resize] = new QuantNodeHelper(carry_data_propogator, carry_data_determiner);
  register_map_[schema::PrimitiveType_Reshape] = new QuantNodeHelper(carry_data_propogator, carry_data_determiner);
  register_map_[schema::PrimitiveType_StridedSlice] = new QuantNodeHelper(carry_data_propogator, carry_data_determiner);
  register_map_[schema::PrimitiveType_Transpose] = new QuantNodeHelper(carry_data_propogator, carry_data_determiner);
  register_map_[schema::PrimitiveType_PadFusion] = new QuantNodeHelper(carry_data_propogator, carry_data_determiner);
  register_map_[schema::PrimitiveType_ReduceFusion] = new QuantNodeHelper(base_propogator, carry_data_determiner);
  register_map_[schema::PrimitiveType_Gather] = new QuantNodeHelper(carry_data_propogator, carry_data_determiner);

  register_map_[schema::PrimitiveType_Concat] = new QuantNodeHelper(concat_propogator, base_determiner);

  register_map_[schema::PrimitiveType_Conv2DFusion] = new QuantNodeHelper(conv_propogator, conv_determiner);
  register_map_[schema::PrimitiveType_MatMul] = new QuantNodeHelper(conv_propogator, matmul_determiner);
  register_map_[schema::PrimitiveType_FullConnection] = new QuantNodeHelper(conv_propogator, matmul_determiner);

  register_map_[schema::PrimitiveType_QuantDTypeCast] =
    new QuantNodeHelper(quant_dtype_cast_propogator, default_quant_all_determiner);

  register_map_[schema::PrimitiveType_DetectionPostProcess] =
    new QuantNodeHelper(base_propogator, only_need_inputs_determiner);
  register_map_[schema::PrimitiveType_NONE] = new QuantNodeHelper(base_propogator, base_determiner);
}

QuantHelperRegister::~QuantHelperRegister() {
  for (const auto &iter : register_map_) {
    delete iter.second;
  }
  this->register_map_.clear();
}
}  // namespace mindspore::lite
