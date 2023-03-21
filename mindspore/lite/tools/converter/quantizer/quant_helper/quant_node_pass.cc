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

#include "tools/converter/quantizer/quant_helper/quant_node_pass.h"
#include <memory>
#include <functional>
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/quant_strategy.h"
#include "tools/converter/quantizer/fixed_bit_weight_quantization.h"
#include "tools/common/node_util.h"

namespace mindspore::lite::quant {
int QuantNodePass::DoWeightQuant(const CNodePtr &cnode) {
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, RET_NULL_PTR, "quant_param_holder is nullptr.");

  auto manager = mindspore::Manage(func_graph_, true);
  CHECK_NULL_RETURN(manager);

  for (size_t idx = 1; idx < cnode->size(); ++idx) {
    auto input = cnode->input(idx);
    ParameterPtr parameter;
    tensor::TensorPtr weight;
    GetParameterAndTensor(input, &parameter, &weight);
    if (parameter == nullptr || weight == nullptr || weight->data_type() != TypeId::kNumberTypeFloat32) {
      MS_LOG(INFO) << "This op " << cnode->fullname_with_scope() << " can not quant weight";
      continue;
    }
    if (!CanTensorQuantized(cnode, input)) {
      MS_LOG(INFO) << input->fullname_with_scope() << " is not quantized.";
      continue;
    }
    int preferred_dim = GetPreferredDim(cnode, idx - 1, ConvertShapeVectorToInt32(weight->shape()));
    MS_CHECK_GT(static_cast<int>(quant_param_holder->get_input_quant_params().size()), static_cast<int>(idx - 1),
                RET_ERROR);
    auto quant_params = quant_param_holder->get_input_quant_params().at(idx - 1);
    MS_CHECK_FALSE_MSG(quant_params.empty(), RET_ERROR, "quant_params is empty.");

    auto status = QuantFilter(parameter, weight, quant_params, preferred_dim);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
  }
  return RET_OK;
}

int QuantNodePass::QuantFilter(const AnfNodePtr &parameter_node, const tensor::TensorPtr &weight,
                               const std::vector<schema::QuantParamT> &quant_params, int preferred_dim) {
  size_t elem_count = weight->DataSize();
  auto raw_datas = static_cast<float *>(weight->data_c());
  std::vector<int8_t> quant_datas(elem_count);
  auto dims = ConvertShapeVectorToInt32(weight->shape());
  MS_CHECK_FALSE_MSG(raw_datas == nullptr, RET_ERROR, "raw_data is nullptr.");

  size_t bit_num = quant_params.front().numBits;
  int quant_min = QuantMin(bit_num, false, false);
  int quant_max = QuantMax(bit_num, false);
  if (IsPerchannelWeight(quant_params, weight, preferred_dim)) {
    auto count = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<>());
    MS_CHECK_FALSE_MSG(static_cast<size_t>(count) != elem_count, RET_ERROR, "element != count.");
    CHECK_LESS_RETURN(dims.size(), static_cast<size_t>(preferred_dim + 1));
    // Do quant
    for (size_t i = 0; i < elem_count; i++) {
      float raw_data = raw_datas[i];
      auto bucket_index = GetBucketIndex(dims, preferred_dim, i);
      auto quant_param = quant_params.at(bucket_index);
      auto quant_data = QuantizeData<int8_t>(raw_data, &quant_param, quant_max, quant_min);
      quant_datas[i] = quant_data;
    }
  } else {
    // update data
    auto quant_param = quant_params.front();
    for (uint32_t i = 0; i < elem_count; i++) {
      float raw_data = raw_datas[i];
      auto quant_data = QuantizeData<int8_t>(raw_data, &quant_param, quant_max, quant_min);
      quant_datas[i] = quant_data;
    }
  }
  auto status = UpdateTensorDataAndSize(parameter_node, weight, quant_datas.data(), quant_datas.size() * sizeof(int8_t),
                                        kNumberTypeInt8);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateTensorDataAndSize error";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantNodePass::CheckNodeDType(const CNodePtr &cnode, const AnfNodePtr &input_node, size_t input_index) {
  TypeId type_id = kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(input_node, &type_id) != RET_OK) {
    MS_LOG(ERROR) << "Get data type failed.";
    return RET_ERROR;
  }

  if (type_id == kNumberTypeInt8) {
    auto iter = weight_quant_params_bak_.find(input_node->fullname_with_scope());
    if (iter != weight_quant_params_bak_.end()) {
      //  support share weight
      auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
      MS_CHECK_TRUE_RET(primitive != nullptr, RET_NULL_PTR);
      auto quant_param_holder = GetCNodeQuantHolder(primitive);
      MS_CHECK_TRUE_MSG(quant_param_holder != nullptr, RET_NULL_PTR, "quant_param_holder is nullptr.");
      quant_param_holder->set_input_quant_param(input_index - 1, iter->second);
    }
    return RET_NO_CHANGE;
  }

  if (type_id != kNumberTypeFloat32) {
    MS_LOG(INFO) << "Input node data type not fp32, input node name: " << input_node->fullname_with_scope();
    return RET_NO_CHANGE;
  }
  return RET_OK;
}

int QuantNodePass::DoParameterNodeQuant(const CNodePtr &cnode, const ParameterPtr &input_node, size_t input_index) {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(input_node);
  auto ret = CheckNodeDType(cnode, input_node, input_index);
  if (ret != RET_OK) {
    return ret;
  }
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  CHECK_NULL_RETURN(quant_param_holder);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  auto op_name = cnode->fullname_with_scope();
  if (input_index == THIRD_INPUT + 1 && CheckNodeInSet(cnode, kHasBiasOperator)) {
    FixedBitWeightQuantization fixed_bit_quant;
    ret = fixed_bit_quant.QuantBias(input_node, primitive);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op_name << " Do bias quant failed.";
      return ret;
    }
  } else {
    // quant weight
    auto tensor_info = input_node->default_param()->cast<tensor::TensorPtr>();
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << input_node->fullname_with_scope() << " can not get value";
      return RET_NULL_PTR;
    }
    if (tensor_info->data_type() != kNumberTypeFloat32) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " is not float32, data will not quant.";
      return RET_OK;
    }
    int preferred_dim = GetPreferredDim(cnode, input_index - 1, ConvertShapeVectorToInt32(tensor_info->shape()));
    MS_CHECK_GT(static_cast<int>(quant_param_holder->get_input_quant_params().size()),
                static_cast<int>(input_index) - 1, RET_ERROR);
    auto quant_params = quant_param_holder->get_input_quant_params().at(input_index - 1);
    MS_CHECK_FALSE_MSG(quant_params.empty(), RET_ERROR, "quant_params is empty.");

    auto status = QuantFilter(input_node, tensor_info, quant_params, preferred_dim);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantFilter failed : " << status;
      return status;
    }
  }
  return RET_OK;
}

int QuantNodePass::DoValueNodeQuant(const CNodePtr &cnode, const ValueNodePtr &input_node, size_t input_index) {
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  CHECK_NULL_RETURN(quant_param_holder);
  auto ret = CheckNodeDType(cnode, input_node, input_index);
  if (ret != RET_OK) {
    return ret;
  }

  auto tensor_info = input_node->value()->cast<tensor::TensorPtr>();
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << input_node->fullname_with_scope() << " can not get value";
    return RET_NULL_PTR;
  }
  if (tensor_info->data_type() != kNumberTypeFloat32) {
    MS_LOG(INFO) << cnode->fullname_with_scope() << " is not float32, data will not quant.";
    return RET_OK;
  }
  int preferred_dim = GetPreferredDim(cnode, input_index - 1, ConvertShapeVectorToInt32(tensor_info->shape()));
  MS_CHECK_GT(static_cast<int>(quant_param_holder->get_input_quant_params().size()), static_cast<int>(input_index) - 1,
              RET_ERROR);
  auto quant_params = quant_param_holder->get_input_quant_params().at(input_index - 1);
  MS_CHECK_FALSE_MSG(quant_params.empty(), RET_ERROR, "quant_params is empty.");

  auto status = QuantFilter(input_node, tensor_info, quant_params, preferred_dim);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "QuantFilter failed : " << status;
    return status;
  }
  return RET_OK;
}

int QuantNodePass::DoFullQuant(const CNodePtr &cnode) {
  auto op_name = cnode->fullname_with_scope();
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(primitive != nullptr, RET_NULL_PTR, "primitive is nullptr.");
  auto primitive_quant_holder = GetCNodeQuantHolder(primitive);
  MS_CHECK_TRUE_MSG(primitive_quant_holder != nullptr, RET_NULL_PTR, "primitive_quant_holder is nullptr.");

  if (primitive->name() == mindspore::ops::kNameTupleGetItem) {
    return RET_OK;
  }
  if (UpdateDataType(cnode, kNumberTypeInt8) != RET_OK) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " set data_type failed.";
    return RET_ERROR;
  }
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    MS_ASSERT(input_node != nullptr);
    // activation quant
    if (IsGraphInput(input_node) || input_node->isa<mindspore::CNode>()) {
      continue;
    } else if (input_node->isa<mindspore::Parameter>()) {
      auto ret = DoParameterNodeQuant(cnode, input_node->cast<ParameterPtr>(), i);
      if (ret == RET_NO_CHANGE) {
        continue;
      } else if (ret != RET_OK) {
        MS_LOG(ERROR) << input_node->fullname_with_scope() << " Do parameter node quant failed.";
        return ret;
      }
      //  support shared weight
      weight_quant_params_bak_[input_node->fullname_with_scope()] =
        primitive_quant_holder->get_input_quant_params()[i - 1];
    } else if (input_node->isa<mindspore::ValueNode>()) {
      // do valueNode quant to int8
      auto ret = DoValueNodeQuant(cnode, input_node->cast<ValueNodePtr>(), i);
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
  primitive_quant_holder->set_quant_type(quant::QUANT_ALL);
  // do output quant
  return RET_OK;
}

bool QuantNodePass::CanTensorQuantized(const CNodePtr &cnode, const AnfNodePtr &input_node) {
  if (input_node == nullptr) {
    MS_LOG(INFO) << "CanTensorQuantized input is nullptr!";
    return false;
  }
  ParameterPtr param_node = nullptr;
  if (input_node->isa<Parameter>()) {
    param_node = input_node->cast<ParameterPtr>();
  }
  if (param_node == nullptr) {
    MS_LOG(INFO) << "CanTensorQuantized invalid param_node!";
    return false;
  }
  if (!param_node->has_default()) {
    MS_LOG(INFO) << "param_node don't has default.";
    return false;
  }
  auto abstract_base = param_node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(INFO) << "abstract is nullptr";
    return false;
  }
  if (!utils::isa<abstract::ShapePtr>(abstract_base->GetShapeTrack())) {
    MS_LOG(INFO) << "Shape of Abstract of parameter should be ShapePtr " << param_node->name();
    return false;
  }
  auto weight_shape = utils::cast<abstract::ShapePtr>(abstract_base->GetShapeTrack())->shape();
  MS_ASSERT(weight_shape != nullptr);
  if (weight_shape.size() < DIMENSION_2D) {  // do not quant single dim tensors
    return false;
  }
  return true;
}

int QuantNodePass::Quant() {
  CHECK_NULL_RETURN(func_graph_);
  auto cnodes = func_graph_->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    if (opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
      continue;
    }
    auto quant_holder = GetCNodeQuantHolder(cnode);
    if (quant_holder == nullptr) {
      continue;
    }
    auto quant_type = quant_holder->quant_type();
    auto node_name = cnode->fullname_with_scope();
    if (quant_type == quant::QUANT_NONE) {
      if (opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
        continue;
      }
      // Remove unused quant param.
      MS_LOG(INFO) << node_name << " is not quant node.";
      quant_holder->ClearQuantParams();
    } else if (quant_type == quant::QUANT_WEIGHT) {
      // Reference Weight Quant
      if (DoWeightQuant(cnode) != RET_OK) {
        MS_LOG(INFO) << node_name << " quant weight failed.";
      }
    } else if (quant_type == quant::QUANT_ALL) {
      if (DoFullQuant(cnode) != RET_OK) {
        MS_LOG(INFO) << node_name << " full quant failed.";
      }
    } else {
      MS_LOG(WARNING) << "Not supported quant type: " << quant_type << " cnode name: " << node_name;
      continue;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
