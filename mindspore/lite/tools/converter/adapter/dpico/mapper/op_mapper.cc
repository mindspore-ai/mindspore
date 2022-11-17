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

#include "mapper/op_mapper.h"
#include <functional>
#include <algorithm>
#include "ops/tuple_get_item.h"
#include "common/op_attr.h"
#include "common/op_enum.h"
#include "common/anf_util.h"
#include "common/string_util.h"
#include "third_party/securec/include/securec.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS SetOpInputs(const api::CNodePtr &cnode, mapper::BaseOperator *base_operator) {
  if (base_operator == nullptr) {
    MS_LOG(ERROR) << "base_operator is nullptr.";
    return RET_ERROR;
  }
  std::vector<std::string> input_names;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_anode = cnode->input(i);
    MS_ASSERT(input_anode != nullptr);
    if (api::utils::isa<api::ParameterPtr>(input_anode)) {
      auto param_node = input_anode->cast<api::ParameterPtr>();
      if (param_node != nullptr && !param_node->has_default()) {  // graph input
        (void)input_names.emplace_back(input_anode->fullname_with_scope());
      }
    } else if (api::utils::isa<api::CNodePtr>(input_anode)) {
      auto input_cnode = input_anode->cast<api::CNodePtr>();
      if (input_cnode == nullptr) {
        MS_LOG(ERROR) << "input node must be cnode.";
        return RET_ERROR;
      }
      auto node_name = input_cnode->fullname_with_scope();
      if (input_cnode->GetAttr(kOutputsNames) != nullptr) {
        auto output_names = api::GetValue<std::vector<std::string>>(input_cnode->GetAttr(kOutputsNames));
        if (output_names.size() == 1) {
          node_name = output_names.at(0);
        }
      }
      (void)input_names.emplace_back(node_name);
    }
  }
  base_operator->SetInputNamesVec(input_names);
  return RET_OK;
}

STATUS FillMultiOutOpOutputs(const api::CNodePtr &cnode, mapper::BaseOperator *base_operator,
                             const api::CNodePtrList &output_cnodes) {
  MS_ASSERT(base_operator != nullptr);
  if (std::any_of(output_cnodes.begin(), output_cnodes.end(), [](const api::CNodePtr &cnode) {
        return !CheckPrimitiveType(cnode, api::MakeShared<ops::TupleGetItem>());
      })) {
    MS_LOG(ERROR) << "multi-out op must be connected with tuple-get-item node.";
    return RET_ERROR;
  }
  auto abstract = cnode->abstract();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "each node's abstract must be not a nullptr.";
    return RET_ERROR;
  }
  if (!abstract->isa<api::AbstractTuple>()) {
    MS_LOG(ERROR) << "multi-out op's abstract must be a tuple.";
    return RET_ERROR;
  }
  auto abstract_tuple = abstract->cast<api::AbstractTuplePtr>();
  MS_ASSERT(abstract_tuple != nullptr);
  auto output_num = abstract_tuple->elements().size();
  std::vector<std::string> output_names;
  // pre-fill the output names, because maybe there are unused outputs.
  for (size_t i = 0; i < output_num; ++i) {
    (void)output_names.emplace_back(cnode->fullname_with_scope() + "_unused_" + std::to_string(i));
  }
  for (const auto &output_cnode : output_cnodes) {
    if (output_cnode->size() != kInputIndex3) {
      MS_LOG(ERROR) << "tuple-get_item's inputs size must be 3.";
      return RET_ERROR;
    }
    auto index_node = output_cnode->input(kInputIndex2);
    MS_CHECK_TRUE_MSG(index_node != nullptr, RET_ERROR, "node is nullptr.");
    auto value_ptr = api::GetValueNode(index_node);
    MS_CHECK_TRUE_MSG(value_ptr != nullptr, RET_ERROR, "tuple_get_item's second input must be a value.");
    auto num_str = value_ptr->ToString();
    MS_CHECK_TRUE_MSG(IsValidUnsignedNum(num_str), RET_ERROR, "tuple_get_item's second input must be an unsigned int");
    auto index = stoi(num_str);
    MS_CHECK_TRUE_MSG(index >= 0 && static_cast<size_t>(index) < output_num, RET_ERROR,
                      "tuple_get_item index is invalid.");
    output_names[index] = output_cnode->fullname_with_scope();
  }
  base_operator->SetOutputNamesVec(output_names);
  return RET_OK;
}

STATUS SetOpOutputs(const api::CNodePtr &cnode, mapper::BaseOperator *base_operator,
                    const api::CNodePtrList &output_cnodes) {
  if (cnode == nullptr || base_operator == nullptr ||
      std::any_of(output_cnodes.begin(), output_cnodes.end(),
                  [](const api::CNodePtr &cnode) { return cnode == nullptr; })) {
    MS_LOG(ERROR) << "the function exist that input parameter is a nullptr.";
    return RET_ERROR;
  }
  if (std::all_of(output_cnodes.begin(), output_cnodes.end(), [](const api::CNodePtr &cnode) {
        return !CheckPrimitiveType(cnode, api::MakeShared<ops::TupleGetItem>());
      })) {
    // single output op
    std::vector<std::string> output_names;
    (void)output_names.emplace_back(cnode->fullname_with_scope());
    base_operator->SetOutputNamesVec(output_names);
    return RET_OK;
  }

  // multi output op
  if (FillMultiOutOpOutputs(cnode, base_operator, output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set multi-out op's output names failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

STATUS SetCommonAttr(const api::CNodePtr &cnode, mapper::BaseOperator *base_operator,
                     const api::CNodePtrList &output_cnodes) {
  if (base_operator == nullptr) {
    MS_LOG(ERROR) << "base operator is nullptr.";
    return RET_ERROR;
  }
  base_operator->SetOpName(cnode->fullname_with_scope());
  if (SetOpInputs(cnode, base_operator) != RET_OK) {
    MS_LOG(ERROR) << "set op inputs failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (SetOpOutputs(cnode, base_operator, output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set op outputs failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS SetConvFcDataInfo(const api::CNodePtr &cnode, mapper::BaseOperator *base_operator) {
  if (base_operator == nullptr) {
    MS_LOG(ERROR) << "base_operator is nullptr.";
    return RET_ERROR;
  }
  for (size_t i = 2; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    MS_ASSERT(input_node != nullptr);
    auto param_node = input_node->cast<api::ParameterPtr>();
    if (param_node == nullptr || !param_node->has_default()) {
      continue;
    }
    auto tensor_info = param_node->default_param()->cast<api::TensorPtr>();
    if (tensor_info != nullptr && tensor_info->DataSize() != 0) {
      auto data = reinterpret_cast<float *>(tensor_info->data());
      MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is nullptr.");
      if (i == kInputIndex2) {
        base_operator->SetWeightDataPtr(data);
        base_operator->SetWeightSize(tensor_info->DataSize());
      } else if (i == kInputIndex3) {
        base_operator->SetBiasDataPtr(data);
        base_operator->SetBiasSize(tensor_info->DataSize());
      } else {
        MS_LOG(ERROR) << "conv or fc operator only support 2 offline inputs at most, but "
                      << cnode->fullname_with_scope() << " has " << i << " offline inputs.";
        return RET_ERROR;
      }
    } else {
      MS_LOG(ERROR) << "param node's tensor info is invalid. " << input_node->fullname_with_scope();
      return RET_ERROR;
    }
  }

  return RET_OK;
}
STATUS SetRecurrentDataInfo(const api::CNodePtr &cnode, mapper::RecurrentOperator *recurrent_operator) {
  if (recurrent_operator == nullptr) {
    MS_LOG(ERROR) << "recurrent_operator is nullptr.";
    return RET_ERROR;
  }
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (api::utils::isa<api::CNode>(input_node)) {
      MS_LOG(INFO) << "cnode don't have blobs";
      continue;
    }
    if (api::utils::isa<api::ParameterPtr>(input_node)) {
      auto input_param_node = input_node->cast<api::ParameterPtr>();
      if (!input_param_node->has_default()) {
        MS_LOG(INFO) << "graph input don't have blobs";
        continue;
      }
      auto tensor_info = input_param_node->default_param()->cast<api::TensorPtr>();
      if (tensor_info != nullptr && tensor_info->DataSize() != 0) {
        auto raw_datas = static_cast<float *>(tensor_info->data());
        auto elem_count = tensor_info->DataSize();
        auto weight_data = new (std::nothrow) float[tensor_info->DataSize()];
        if (weight_data == nullptr) {
          MS_LOG(ERROR) << "new float[] failed.";
          return RET_ERROR;
        }
        if (memcpy_s(weight_data, static_cast<size_t>(tensor_info->DataSize()) * sizeof(float), raw_datas,
                     static_cast<size_t>(tensor_info->DataSize()) * sizeof(float)) != EOK) {
          MS_LOG(ERROR) << "memcpy_s failed.";
          delete[] weight_data;
          return RET_ERROR;
        }
        recurrent_operator->AddRecurrentParamVec(weight_data);
        recurrent_operator->AddRecurrentParamLengthVec(elem_count);
      } else {
        MS_LOG(ERROR) << "tensor_info is nullptr, or DataSize equals zero. " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
STATUS PushOfflineArgs(const api::CNodePtr &cnode, mapper::BaseOperator *base_operator, size_t offline_args_size) {
  if (base_operator == nullptr) {
    MS_LOG(ERROR) << "base_operator is nullptr.";
    return RET_ERROR;
  }
  if (offline_args_size > cnode->inputs().size()) {
    MS_LOG(ERROR) << "input offline_args_size:" << offline_args_size
                  << " is greater than cnode input size:" << cnode->inputs().size() << " "
                  << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto inputs_size = std::min(offline_args_size + 1, cnode->inputs().size());
  std::vector<std::pair<std::vector<float>, std::vector<int32_t>>> offline_args;
  bool has_offline_args = false;
  for (size_t i = 1; i < inputs_size; i++) {
    auto input_node = cnode->input(i);
    if (api::utils::isa<api::CNode>(input_node)) {
      MS_LOG(INFO) << "cnode don't have blobs";
      (void)offline_args.emplace_back();
      continue;
    }
    if (api::utils::isa<api::ParameterPtr>(input_node)) {
      auto input_param_node = input_node->cast<api::ParameterPtr>();
      if (!input_param_node->has_default()) {
        MS_LOG(INFO) << "graph input don't have blobs";
        (void)offline_args.emplace_back();
        continue;
      }
      auto tensor_info = input_param_node->default_param()->cast<api::TensorPtr>();
      if (tensor_info != nullptr && tensor_info->DataSize() != 0) {
        has_offline_args = true;
        std::vector<float> offline_data;
        auto elem_count = tensor_info->DataSize();
        if (tensor_info->data_type() == kNumberTypeInt32 || tensor_info->data_type() == kNumberTypeInt) {
          auto raw_datas = static_cast<int32_t *>(tensor_info->data());
          offline_data = std::vector<float>(raw_datas, raw_datas + elem_count);
        } else if (tensor_info->data_type() == kNumberTypeFloat32 || tensor_info->data_type() == kNumberTypeFloat) {
          auto raw_datas = static_cast<float *>(tensor_info->data());
          offline_data = std::vector<float>(raw_datas, raw_datas + elem_count);
        } else {
          MS_LOG(ERROR) << "unsupported param type. " << tensor_info->data_type();
          return RET_ERROR;
        }
        std::vector<int32_t> offline_shape;
        ShapeVector shape_vector;
        if (GetShapeVectorFromParameter(input_param_node, &shape_vector) != RET_OK) {
          MS_LOG(ERROR) << "get shape vector from parameter failed. " << input_param_node->fullname_with_scope();
          return RET_ERROR;
        }
        (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(offline_shape),
                             [](const int64_t dim) { return static_cast<int32_t>(dim); });
        (void)offline_args.emplace_back(std::make_pair(offline_data, offline_shape));
      } else {
        MS_LOG(ERROR) << "tensor_info is nullptr, or DataSize equals zero. " << cnode->fullname_with_scope();
        return RET_ERROR;
      }
    }
  }
  if (has_offline_args) {
    for (auto &offline_arg : offline_args) {
      base_operator->PushOfflineArgs(std::move(offline_arg));
    }
  }
  return RET_OK;
}
}  // namespace dpico
}  // namespace mindspore
