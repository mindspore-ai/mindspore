/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include <memory>
#include <vector>
#include "src/tensorlist.h"
#include "tools/optimizer/common/format_utils.h"
#include "utils/ms_utils_secure.h"
#include "nnacl/op_base.h"
#include "ops/core_ops.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int kElementShapeIndex = 1;
constexpr int kElementNumOffset = 2;
constexpr int kBasicInfoMinSize = 3;
bool CheckTensorListIsValid(const std::vector<uint8_t> &tensorlist_data) {
  if (tensorlist_data.empty()) {
    return true;
  }
  auto basic_data_size = tensorlist_data.size() / sizeof(int);
  auto *data = reinterpret_cast<const int *>(tensorlist_data.data());
  if (basic_data_size < static_cast<size_t>(kBasicInfoMinSize)) {
    MS_LOG(ERROR) << "tensorlist data length illegal, which should be at least 3, now is " << basic_data_size;
    return false;
  }
  if (data[kElementShapeIndex] < 0 || INT_ADD_OVERFLOW(data[kElementShapeIndex], kBasicInfoMinSize)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT add overflow.";
    return false;
  }
  if (static_cast<size_t>((data[kElementShapeIndex] + kBasicInfoMinSize)) > basic_data_size) {
    MS_LOG(ERROR) << "tensorlist data length illegal. current tensorlist data length should be at least "
                  << (data[kElementShapeIndex] + kBasicInfoMinSize) << ", but now is " << basic_data_size;
    return false;
  }
  auto element_num = data[data[kElementShapeIndex] + kElementNumOffset];
  if (element_num > 0 && INT_ADD_OVERFLOW(element_num, 1)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT add overflow.";
    return false;
  }
  auto shape_once = data[kElementShapeIndex] + 1;
  auto shape_group_num = element_num < 0 ? 1 : element_num + 1;
  if (INT_MUL_OVERFLOW(shape_once, shape_group_num)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT mul overflow.";
    return false;
  }
  auto shape_info_size = shape_once * shape_group_num;
  if (INT_ADD_OVERFLOW(shape_info_size, kElementNumOffset)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT add overflow.";
    return false;
  }
  size_t real_data_size = static_cast<size_t>(shape_info_size + kElementNumOffset);
  if (real_data_size != basic_data_size) {
    MS_LOG(ERROR) << "current tensorlist data length should be " << real_data_size << ", but now is "
                  << basic_data_size;
    return false;
  }
  return true;
}

int ConvertToLiteTensor(const std::vector<lite::DataInfo> &data_infos, std::vector<TensorPtr> *tensors) {
  MS_ASSERT(tensors != nullptr);
  for (auto &data_info : data_infos) {
    auto tensor_category = lite::TensorCategory(lite::NodeType(data_info.node_type_), data_info.shape_.size(),
                                                TypeId(data_info.data_type_), data_info.data_.size());
    TensorPtr tensor;
    if (data_info.data_type_ != kObjectTypeTensorType) {
      tensor = std::make_shared<lite::Tensor>(TypeId(data_info.data_type_), data_info.shape_,
                                              (mindspore::Format)data_info.format_, tensor_category);
    } else {
      tensor = std::make_shared<lite::TensorList>(data_info.shape_, std::vector<int>(), tensor_category);
    }
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "new a lite tensor failed.";
      return lite::RET_ERROR;
    }
    auto tensor_size = data_info.data_.size();
    if (tensor_size > 0) {
      if (data_info.data_type_ == kObjectTypeTensorType) {
        auto tensor_list = std::static_pointer_cast<lite::TensorList>(tensor);
        if (!CheckTensorListIsValid(data_info.data_)) {
          MS_LOG(ERROR) << "tensor list is invalid.";
          return lite::RET_ERROR;
        }
        if (tensor_list->Decode(reinterpret_cast<const int *>(data_info.data_.data()), tensor_size) != lite::RET_OK) {
          MS_LOG(ERROR) << "Decode tensorlist data failed.";
          return lite::RET_ERROR;
        }
      } else {
        auto tensor_data = malloc(tensor_size);
        if (tensor_data == nullptr) {
          MS_LOG(ERROR) << "tensor_data is nullptr.";
          return lite::RET_ERROR;
        }
        if (memcpy_s(tensor_data, tensor_size, data_info.data_.data(), tensor_size) != EOK) {
          free(tensor_data);
          MS_LOG(ERROR) << "memcpy data error.";
          return lite::RET_ERROR;
        }
        tensor->set_data(tensor_data);
      }
    }

    if (tensor_size == 0 && data_info.data_ptr_ != nullptr) {
      tensor->set_data(data_info.data_ptr_);
      tensor->set_own_data(false);
    }
    tensors->emplace_back(tensor);
  }
  return lite::RET_OK;
}

TensorPtr GetCNodeTensorListVarInput(const lite::DataInfo &data_info) {
  auto tensor_list = std::make_shared<lite::TensorList>(data_info.shape_, std::vector<int>{});
  if (tensor_list == nullptr) {
    MS_LOG(ERROR) << "new a lite tensor list failed";
    return nullptr;
  }
  if (data_info.data_.empty()) {
    return tensor_list;
  }
  if (!CheckTensorListIsValid(data_info.data_)) {
    MS_LOG(ERROR) << "tensor list is invalid.";
    return nullptr;
  }
  auto status = tensor_list->Decode(reinterpret_cast<const int *>(data_info.data_.data()), data_info.data_.size());
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "decode tensor list failed.";
    return nullptr;
  }
  return tensor_list;
}
}  // namespace

int LiteTensorExtractor::GetCNodeConstInput(const CNodePtr &cnode, std::vector<TensorPtr> *const_ms_inputs,
                                            converter::FmkType fmk_type, bool train_flag, bool copy_data) {
  MS_ASSERT(cnode != nullptr && const_ms_inputs != nullptr);
  std::vector<lite::DataInfo> data_infos;
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      continue;
    }
    STATUS status;
    lite::DataInfo data_info;
    if (utils::isa<ParameterPtr>(cnode->input(i))) {
      status = lite::FetchDataFromParameterNode(cnode, i, fmk_type, &data_info, copy_data);
    } else {
      status = lite::FetchDataFromValueNode(cnode, i, fmk_type, train_flag, &data_info, copy_data);
    }
    if (status == lite::RET_NO_CHANGE) {
      continue;
    }
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch const input data failed.";
      return status;
    }
    data_infos.emplace_back(data_info);
  }
  return ConvertToLiteTensor(data_infos, const_ms_inputs);
}

int LiteTensorExtractor::GetCNodeVarInput(const CNodePtr &cnode, std::vector<TensorPtr> *var_ms_inputs,
                                          converter::FmkType fmk_type) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(var_ms_inputs != nullptr);
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (!utils::isa<CNodePtr>(cnode->input(i))) {
      continue;
    }
    lite::DataInfo data_info;
    if (lite::FetchDataFromCNode(cnode, i, &data_info) != lite::RET_OK) {
      MS_LOG(ERROR) << "parse cnode failed.";
      return lite::RET_ERROR;
    }
    TensorPtr tensor;
    if (data_info.data_type_ == kObjectTypeTensorType) {
      tensor = GetCNodeTensorListVarInput(data_info);
    } else {
      tensor = std::make_shared<lite::Tensor>(TypeId(data_info.data_type_), data_info.shape_);
    }
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "new a lite tensor failed.";
      return lite::RET_ERROR;
    }
    tensor->set_format((Format)(data_info.format_));
    bool has_inferred{false};
    auto ret = DetermineCertainVarInputHasInferred(cnode, i, &has_inferred);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "determine infer flag failed.");
    if (!has_inferred) {
      tensor->set_shape({-1});
    }
    var_ms_inputs->emplace_back(tensor);
  }
  return lite::RET_OK;
}

int LiteTensorExtractor::GetCNodeInputTensors(const CNodePtr &cnode, std::vector<TensorPtr> *inputs,
                                              converter::FmkType fmk_type, bool train_flag, bool copy_data) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(inputs != nullptr);
  auto origin_inputs = cnode->inputs();
  if (lite::RemoveIfDepend(cnode) != RET_OK) {
    MS_LOG(ERROR) << "remove depend failed.";
    return RET_ERROR;
  }
  if (lite::RemoveIfMakeTuple(cnode)) {
    MS_LOG(ERROR) << "remove makeTuple failed.";
    return RET_ERROR;
  }
  RemoveIfMonad(cnode);
  std::vector<TensorPtr> const_inputs;
  if (GetCNodeConstInput(cnode, &const_inputs, fmk_type, train_flag, copy_data) != lite::RET_OK) {
    MS_LOG(ERROR) << "get const inputs failed.";
    cnode->set_inputs(origin_inputs);
    return lite::RET_ERROR;
  }
  std::vector<TensorPtr> var_inputs;
  if (GetCNodeVarInput(cnode, &var_inputs, fmk_type) != lite::RET_OK) {
    MS_LOG(ERROR) << "get var inputs failed.";
    cnode->set_inputs(origin_inputs);
    return lite::RET_ERROR;
  }
  size_t const_index = 0;
  size_t var_index = 0;
  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      if (var_index >= var_inputs.size()) {
        MS_LOG(ERROR) << "var inputs size invalid.";
        cnode->set_inputs(origin_inputs);
        return lite::RET_ERROR;
      }
      inputs->emplace_back(var_inputs[var_index++]);
    } else {
      if (const_index >= const_inputs.size()) {
        MS_LOG(ERROR) << "const inputs size invalid.";
        cnode->set_inputs(origin_inputs);
        return lite::RET_ERROR;
      }
      inputs->emplace_back(const_inputs[const_index++]);
    }
  }
  cnode->set_inputs(origin_inputs);
  return lite::RET_OK;
}

int LiteTensorExtractor::GetCNodeOutputTensors(const CNodePtr &cnode, std::vector<TensorPtr> *outputs,
                                               bool train_flag) {
  MS_ASSERT(cnode != nullptr);
  MS_ASSERT(outputs != nullptr);
  std::vector<lite::DataInfo> data_infos;
  if (utils::isa<abstract::AbstractTuple>(cnode->abstract())) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(cnode->abstract());
    if (tuple == nullptr) {
      MS_LOG(ERROR) << "tuple is nullptr.";
      return lite::RET_ERROR;
    }
    auto elements = tuple->elements();
    for (size_t i = 0; i < elements.size(); i++) {
      lite::DataInfo data_info;
      data_info.node_type_ = lite::NodeType_CNode;
      if (train_flag) {
        data_infos.emplace_back(data_info);
        if (CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) || CheckPrimitiveType(cnode, prim::kPrimAdam)) {
          break;
        }
      } else {
        if (!utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
          MS_LOG(ERROR) << "abstract is not AbstractTensor.";
          return lite::RET_ERROR;
        }
        auto type = kNumberTypeFloat32;
        if (utils::isa<abstract::AbstractTensorPtr>(elements[i])) {
          auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(elements[i]);
          MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
          MS_CHECK_TRUE_RET(abstract_tensor->element() != nullptr, lite::RET_NULL_PTR);
          auto typePtr = abstract_tensor->element()->GetTypeTrack();
          MS_CHECK_TRUE_RET(typePtr != nullptr, lite::RET_NULL_PTR);
          type = typePtr->type_id();
        }
        data_info.data_type_ = type;
        data_infos.emplace_back(data_info);
        if (CheckPrimitiveType(cnode, prim::kPrimConv2DFusion) ||
            CheckPrimitiveType(cnode, prim::kPrimFusedBatchNorm)) {
          break;
        }
      }
    }
  } else {
    lite::DataInfo data_info;
    auto type = kNumberTypeFloat32;
    if (utils::isa<abstract::AbstractTensorPtr>(cnode->abstract())) {
      auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(cnode->abstract());
      MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
      MS_CHECK_TRUE_RET(abstract_tensor->element() != nullptr, lite::RET_NULL_PTR);
      auto typePtr = abstract_tensor->element()->GetTypeTrack();
      MS_CHECK_TRUE_RET(typePtr != nullptr, lite::RET_NULL_PTR);
      type = typePtr->type_id();
    }
    data_info.data_type_ = type;
    data_info.node_type_ = lite::NodeType_CNode;
    data_infos.emplace_back(data_info);
  }
  return ConvertToLiteTensor(data_infos, outputs);
}
}  // namespace opt
}  // namespace mindspore
