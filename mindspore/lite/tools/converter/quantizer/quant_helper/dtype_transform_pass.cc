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

#include "tools/converter/quantizer/quant_helper/dtype_transform_pass.h"
#include <set>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "tools/common/node_util.h"
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::lite::quant {
// only enable for uint8
int DTypeTransformPass::Transform() {
  auto cnodes = func_graph_->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!CheckNeedDTypeTrans(cnode)) {
      MS_LOG(INFO) << "CheckNeedDTypeTrans invalid cnode, cnode name: " << cnode->fullname_with_scope();
      continue;
    }
    auto status = DoNodeDTypeTrans(cnode);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DoNodeDTypeTrans failed, cnode name: " << cnode->fullname_with_scope();
      return status;
    }
    schema::QuantType curr_quant_type;
    if (GetQuantType(cnode, &curr_quant_type) != RET_OK) {
      MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (curr_quant_type != schema::QuantType_QUANT_ALL) {
      MS_LOG(ERROR) << "Invalid cnode quant type, cnode name: " << cnode->fullname_with_scope()
                    << " quant type: " << curr_quant_type;
      continue;
    }
    status = InsertForwardCastNode(cnode, curr_quant_type);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "InsertForwardCastNode failed, cnode name: " << cnode->fullname_with_scope();
      return status;
    }
    // DetectionPostProcess op(Uint8toFp32, not need backward cast node)
    if (!CheckNodeInSet(cnode, kUint8toFP32Operator)) {
      status = InsertBackwardCastNode(cnode, curr_quant_type);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertBackwardCastNode failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
    }
  }  // for
  return RET_OK;
}

int DTypeTransformPass::DoParameterNodeTrans(const CNodePtr &cnode, const ParameterPtr &input_node,
                                             size_t input_index) {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(input_node);
  if (input_index == THIRD_INPUT + 1 && CheckNodeInSet(cnode, kHasBiasOperator)) {
    return RET_OK;
  }
  auto tensor_info = input_node->default_param()->cast<tensor::TensorPtr>();
  CHECK_NULL_RETURN(tensor_info);
  if (tensor_info->data_type() != kNumberTypeUInt8) {
    MS_LOG(INFO) << input_node->fullname_with_scope() << " dtype not uint8.";
    return RET_ERROR;
  }
  size_t elem_count = tensor_info->DataSize();
  auto ret = Uint8toInt8(static_cast<uint8_t *>(tensor_info->data().data()), elem_count);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << input_node->fullname_with_scope() << " transform data uint8 to int8 failed.";
    return ret;
  }
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  if (quant_param_holder->get_input_quant_params().size() < input_index) {
    MS_LOG(ERROR) << "Invalid quant params. input node  name: " << input_node->fullname_with_scope();
    return RET_ERROR;
  }
  auto quant_params = quant_param_holder->get_input_quant_params().at(input_index - 1);
  for (auto &quant_param : quant_params) {
    quant_param.zeroPoint -= kU8ZeroPointOffset;
  }
  quant_param_holder->set_input_quant_param(input_index - 1, quant_params);

  // set dtype
  tensor_info->set_data_type(kNumberTypeInt8);
  ret = UpdateDataType(input_node, kNumberTypeInt8);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << input_node->fullname_with_scope() << " set new dtype failed.";
    return ret;
  }
  return RET_OK;
}

int DTypeTransformPass::Uint8toInt8(uint8_t *data, int size) {
  CHECK_NULL_RETURN(data);

  for (int i = 0; i < size; i++) {
    int temp = static_cast<int>(data[i]) - kU8ZeroPointOffset;
    if (temp > INT8_MAX) {
      data[i] = INT8_MAX;
    } else if (temp < INT8_MIN) {
      data[i] = INT8_MIN;
    } else {
      data[i] = static_cast<int8_t>(temp);
    }
  }
  return RET_OK;
}

int DTypeTransformPass::GetQuantType(const CNodePtr &cnode, schema::QuantType *quant_type) {
  CHECK_NULL_RETURN(cnode);
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  CHECK_NULL_RETURN(quant_param_holder);
  *quant_type = quant_param_holder->quant_type();
  return RET_OK;
}

/**
 * Transform CNode(dtype,uint8toint8,weigh data)
 * */
int DTypeTransformPass::DoNodeDTypeTrans(const CNodePtr &cnode) {
  auto curr_quant_param_holder = GetCNodeQuantHolder(cnode);
  CHECK_NULL_RETURN(curr_quant_param_holder);
  TypeId cnode_dtype = kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(cnode, &cnode_dtype) != RET_OK) {
    MS_LOG(ERROR) << "Get data type failed, cnode name: " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (cnode_dtype == kNumberTypeUInt8) {
    MS_LOG(INFO) << "cnode dtype kNumberTypeUInt8, cnode name: " << cnode->fullname_with_scope();
    if (UpdateDataType(cnode, kNumberTypeInt8) != RET_OK) {
      MS_LOG(ERROR) << "Update data type failed, cnode name: " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
      auto primitive_c = GetValueNode<std::shared_ptr<mindspore::Primitive>>(cnode->input(0));
      auto primc = api::MakeShared<mindspore::ops::QuantDTypeCast>(primitive_c);
      primc->set_dst_t(kNumberTypeInt8);
    }
    // update output quant param zp
    if (curr_quant_param_holder->get_output_quant_params().empty()) {
      MS_LOG(ERROR) << "output quant params empty.";
      return RET_ERROR;
    }
    auto out_quant_params = curr_quant_param_holder->get_output_quant_params()[0];
    for (auto &quant_param : out_quant_params) {
      quant_param.zeroPoint -= kU8ZeroPointOffset;
    }
    curr_quant_param_holder->set_output_quant_param(0, out_quant_params);
  }

  // DTypeCastNode, set quant type
  if (opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
    curr_quant_param_holder->set_quant_type(schema::QuantType_QUANT_NONE);
  }

  for (size_t index = 1; index < cnode->size(); index++) {
    auto input_node = cnode->input(index);
    CHECK_NULL_RETURN(input_node);
    if (IsGraphInput(input_node) || input_node->isa<mindspore::CNode>()) {
      // updata graph input quant params
      if (curr_quant_param_holder->get_input_quant_params().size() < index) {
        MS_LOG(WARNING) << "quant params invalid, input node name: " << input_node->fullname_with_scope();
        continue;
      }
      auto input_quant_params = curr_quant_param_holder->get_input_quant_params()[index - 1];
      if (input_quant_params.empty() || !input_quant_params.front().inited) {
        MS_LOG(WARNING) << "input node not quantizied, input node name: " << input_node->fullname_with_scope();
        continue;
      }
      for (auto &quant_param : input_quant_params) {
        quant_param.zeroPoint -= kU8ZeroPointOffset;
      }
      curr_quant_param_holder->set_input_quant_param(index - 1, input_quant_params);
    } else if (input_node->isa<mindspore::Parameter>()) {  // weight data
      auto ret = DoParameterNodeTrans(cnode, input_node->cast<ParameterPtr>(), index);
      if (ret != RET_OK) {
        MS_LOG(WARNING) << "DoParameterNodeTrans failed, input node name: " << input_node->fullname_with_scope();
      }
    }
  }
  return RET_OK;
}

int DTypeTransformPass::InsertForwardCastNode(const CNodePtr &cnode, schema::QuantType curr_quant_type) {
  // inputs
  quant::InsertQuantNodeManager insert_node_manager;
  for (size_t index = 1; index < cnode->size(); index++) {
    auto input_node = cnode->input(index);
    CHECK_NULL_RETURN(input_node);
    if (!input_node->isa<mindspore::CNode>() && !IsGraphInput(input_node)) {
      MS_LOG(DEBUG) << "Invalid input node, not CNode and graph input.";
      continue;
    }
    schema::QuantType input_quant_type;
    if (GetQuantType(cnode, &input_quant_type) != RET_OK) {
      MS_LOG(WARNING) << "Get quant type failed, input node name: " << input_node->fullname_with_scope();
      return RET_ERROR;
    }
    schema::QuantType pre_quant_type = schema::QuantType_QUANT_NONE;
    if (input_node->isa<mindspore::CNode>()) {
      if (GetQuantType(input_node->cast<mindspore::CNodePtr>(), &pre_quant_type) != RET_OK) {
        MS_LOG(ERROR) << "Get quant type failed, cnode name: " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
    }
    if (pre_quant_type == schema::QuantType_QUANT_NONE && curr_quant_type == schema::QuantType_QUANT_ALL) {
      auto status = insert_node_manager.InserQuantCastNode(this->func_graph_, cnode, FORWARD, kNumberTypeUInt8, kQuant,
                                                           index, nullptr);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InserQuantCastNode failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
      MS_LOG(INFO) << "InserQuantCastNode forward Uint8toInt8, cnode name: " << cnode->fullname_with_scope();
    }
  }
  return RET_OK;
}

int DTypeTransformPass::InsertBackwardCastNode(const CNodePtr &cnode, schema::QuantType curr_quant_type) {
  // outputs
  auto manager = this->func_graph_->manager();
  if (manager == nullptr) {
    manager = Manage(this->func_graph_, true);
  }
  CHECK_NULL_RETURN(manager);
  auto node_users = manager->node_users()[cnode];
  quant::InsertQuantNodeManager insert_node_manager;
  for (auto &node_user : node_users) {
    auto output_cnode = node_user.first->cast<CNodePtr>();
    schema::QuantType post_quant_type;
    if (GetQuantType(output_cnode, &post_quant_type) != RET_OK) {
      MS_LOG(ERROR) << "Get quant type failed, cnode name: " << output_cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (curr_quant_type == schema::QuantType_QUANT_ALL && post_quant_type == schema::QuantType_QUANT_NONE) {
      auto status = insert_node_manager.InserQuantCastNode(this->func_graph_, cnode, BACKWARD, kNumberTypeUInt8,
                                                           kDeQuant, node_user.second, node_user.first);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InserQuantCastNode dequant failed, cnode name: " << cnode->fullname_with_scope();
        return status;
      }
      MS_LOG(INFO) << "InserQuantCastNode backward Int8toUint8, cnode name: " << cnode->fullname_with_scope();
    }
  }  // node_users
  return RET_OK;
}

bool DTypeTransformPass::CheckNeedDTypeTrans(const CNodePtr &cnode) {
  if (opt::IsSpecialType(cnode)) {
    return false;
  }
  if (IsGraphInDTypeCast(cnode) || IsGraphOutDTypeCast(func_graph_, cnode)) {
    return false;
  }
  TypeId cnode_dtype = kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(cnode, &cnode_dtype) != RET_OK) {
    MS_LOG(ERROR) << "Get data type failed, cnode name: " << cnode->fullname_with_scope();
    return false;
  }
  bool is_fp32_output =
    opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast) || CheckNodeInSet(cnode, kUint8toFP32Operator);
  if (cnode_dtype != kNumberTypeUInt8 && !is_fp32_output) {
    MS_LOG(DEBUG) << "dtype not kNumberTypeUInt8, cnode name: " << cnode->fullname_with_scope();
    return false;
  }
  return true;
}
}  // namespace mindspore::lite::quant
