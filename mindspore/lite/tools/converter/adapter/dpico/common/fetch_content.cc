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

#include "common/fetch_content.h"
#include <string>
#include <vector>
#include "common/anf_util.h"
#include "common/check_base.h"
#include "third_party/securec/include/securec.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr size_t kTensorListMinSize = 3 * sizeof(int32_t);
}  // namespace

int FetchFromDefaultParam(const api::ParameterPtr &param_node, DataInfo *data_info) {
  MS_ASSERT(param_node != nullptr && data_info != nullptr);
  ShapeVector shape_vector;
  TypeId data_type;
  auto status = GetDataTypeAndShape(param_node, &data_type, &shape_vector);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get data type and shape from param node failed.";
    return RET_ERROR;
  }
  data_info->data_type_ = data_type;
  auto tensor_info = param_node->default_param()->cast<api::TensorPtr>();
  size_t offset = 0;
  if (!shape_vector.empty() && data_type == kObjectTypeString) {
    status = GetShapeVectorFromStringTensor(tensor_info, &shape_vector, &offset);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "get shape vector from string tensor failed.";
      return RET_ERROR;
    }
  }
  std::vector<int32_t> dims(shape_vector.begin(), shape_vector.end());
  data_info->shape_ = dims;
  if (tensor_info != nullptr && tensor_info->Size() != 0) {
    if (data_type != kObjectTypeTensorType || tensor_info->Size() >= kTensorListMinSize) {
      if (tensor_info->Size() < offset) {
        MS_LOG(ERROR) << "tensor_info elem size should be greater than offset.";
        return RET_ERROR;
      }
      data_info->data_.resize(tensor_info->Size() - offset);
      if (EOK != memcpy_s(data_info->data_.data(), data_info->data_.size(),
                          static_cast<uint8_t *>(tensor_info->data()) + offset, tensor_info->Size() - offset)) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return RET_ERROR;
      }
    }
  }

  return RET_OK;
}

int FetchDataFromParameterNode(const api::CNodePtr &cnode, size_t index, DataInfo *data_info) {
  MS_ASSERT(cnode != nullptr && data_info != nullptr);
  if (index >= cnode->inputs().size()) {
    MS_LOG(ERROR) << "input index: " << index << " is greater than cnode inputs size " << cnode->inputs().size();
    return RET_ERROR;
  }
  auto param_node = cnode->input(index)->cast<api::ParameterPtr>();
  if (param_node == nullptr) {
    MS_LOG(ERROR) << "input node is not parameter node.";
    return RET_ERROR;
  }

  if (FetchFromDefaultParam(param_node, data_info) != RET_OK) {
    MS_LOG(ERROR) << "fetch information from default param failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

int GetDataSizeFromTensor(DataInfo *data_info, int *data_size) {
  if (data_info == nullptr) {
    MS_LOG(ERROR) << "data info is nullptr. ";
    return RET_ERROR;
  }
  if (data_size == nullptr) {
    MS_LOG(ERROR) << "elem size is nullptr";
    return RET_ERROR;
  }
  *data_size = 1;
  for (size_t i = 0; i < data_info->shape_.size(); i++) {
    if (INT_MUL_OVERFLOW(*data_size, data_info->shape_.at(i))) {
      MS_LOG(ERROR) << "int mul overflow.";
      return RET_ERROR;
    }
    *data_size *= data_info->shape_.at(i);
  }
  return RET_OK;
}
}  // namespace dpico
}  // namespace mindspore
