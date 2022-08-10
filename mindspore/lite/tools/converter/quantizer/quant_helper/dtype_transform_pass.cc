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
#include "tools/common/node_util.h"
#include "tools/converter/quantizer/insert_quant_node_manager.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite::quant {
// only enable for uint8
int DTypeTransformPass::Transform() {
  CHECK_NULL_RETURN(func_graph_);
  // insert CastNode Uint8toInt8 & Int8toUint8
  quant::InsertQuantNodeManager insert_quant_node_manager;
  auto ret = insert_quant_node_manager.InsertQuantDtypeCastNode(func_graph_, kNumberTypeUInt8);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Insert Uint8toInt8 CastNode failed.";
    return RET_ERROR;
  }

  // update data type and zp
  auto cnodes = func_graph_->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
      continue;
    }
    TypeId cnode_dtype;
    if (opt::GetDataTypeFromAnfNode(cnode, &cnode_dtype) != RET_OK) {
      MS_LOG(ERROR) << "Get data type failed, cnode type: " << cnode->type_name();
      return RET_ERROR;
    }
    if (cnode_dtype != kNumberTypeUInt8) {
      continue;
    }
    if (UpdateDataType(cnode, kNumberTypeInt8) != RET_OK) {
      MS_LOG(ERROR) << "Update data type failed, cnode name: " << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    auto curr_quant_param_holder = GetCNodeQuantHolder(cnode);
    CHECK_NULL_RETURN(curr_quant_param_holder);
    if (curr_quant_param_holder->get_output_quant_params().empty()) {
      MS_LOG(ERROR) << "output quant params empty.";
      return RET_ERROR;
    }
    auto out_quant_params = curr_quant_param_holder->get_output_quant_params()[0];
    for (auto &quant_param : out_quant_params) {
      quant_param.zeroPoint -= kU8ZeroPointOffset;
    }
    curr_quant_param_holder->set_output_quant_param(0, out_quant_params);
    for (size_t i = 1; i < cnode->size(); i++) {
      auto input_node = cnode->input(i);
      CHECK_NULL_RETURN(input_node);
      if (IsGraphInput(input_node)) {
        continue;
      } else if (input_node->isa<mindspore::CNode>()) {
        // updata input_quant_params
        if (curr_quant_param_holder->get_input_quant_params().size() < i) {
          MS_LOG(ERROR) << "quant params invalid.";
          return RET_ERROR;
        }
        auto input_quant_params = curr_quant_param_holder->get_input_quant_params()[i - 1];
        for (auto &quant_param : input_quant_params) {
          quant_param.zeroPoint -= kU8ZeroPointOffset;
        }
        curr_quant_param_holder->set_input_quant_param(i - 1, input_quant_params);
      } else if (input_node->isa<mindspore::Parameter>()) {
        ret = DoParameterNodeTrans(cnode, input_node->cast<ParameterPtr>(), i);
        if (ret != RET_OK) {
          MS_LOG(WARNING) << "DoParameterNodeTrans failed, input node name: " << input_node->fullname_with_scope();
        }
      }
    }
  }
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
  auto quant_params = quant_param_holder->get_input_quant_params().at(input_index - 1);
  MS_CHECK_FALSE_MSG(quant_params.empty(), RET_ERROR, "Quant params is empty.");
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

  auto abstract_base = input_node->abstract();
  CHECK_NULL_RETURN(abstract_base);
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    MS_LOG(ERROR) << "Abstract of node should be abstract tensor, input node name: "
                  << input_node->fullname_with_scope();
    return RET_ERROR;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  CHECK_NULL_RETURN(abstract_tensor);
  CHECK_NULL_RETURN(abstract_tensor->element());
  abstract_tensor->element()->set_type(TypeIdToType(kNumberTypeInt8));
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
}  // namespace mindspore::lite::quant
