/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/common/tensor_util.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "abstract/utils.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
std::unique_ptr<QuantParamT> GetTensorQuantParam(const std::unique_ptr<TensorT> &tensor) {
  MS_ASSERT(tensor != nullptr);
  auto &quantParams = tensor->quantParams;
  if (!quantParams.empty()) {
    return CopyQuantParamT(quantParams.front());
  } else {
    return nullptr;
  }
}
std::unique_ptr<schema::QuantParamT> CopyQuantParamT(const std::unique_ptr<schema::QuantParamT> &srcQuantParam) {
  MS_ASSERT(srcQuantParam != nullptr);
  std::unique_ptr<schema::QuantParamT> dstQuantParam = std::make_unique<schema::QuantParamT>();
  dstQuantParam->inited = srcQuantParam->inited;
  dstQuantParam->scale = srcQuantParam->scale;
  dstQuantParam->zeroPoint = srcQuantParam->zeroPoint;
  dstQuantParam->min = srcQuantParam->min;
  dstQuantParam->max = srcQuantParam->max;
  dstQuantParam->narrowRange = srcQuantParam->narrowRange;
  dstQuantParam->numBits = srcQuantParam->numBits;
  dstQuantParam->dstDtype = srcQuantParam->dstDtype;
  dstQuantParam->multiplier = srcQuantParam->multiplier;
  return dstQuantParam;
}

tensor::TensorPtr CreateTensorInfo(const void *data, size_t data_size, const std::vector<int64_t> &shape,
                                   TypeId data_type) {
  if (data_type == kTypeUnknown) {
    MS_LOG(ERROR) << "data type of tensor is unknown";
    return nullptr;
  }
  tensor::TensorPtr tensor_info = nullptr;
  if (shape.empty() && data_size == mindspore::abstract::TypeIdSize(data_type)) {
    ShapeVector scalar_shape = {1};
    tensor_info = std::make_shared<tensor::Tensor>(data_type, scalar_shape);
    if (tensor_info == nullptr) {
      MS_LOG(ERROR) << "new tensor init failed";
      return nullptr;
    }
    tensor_info->set_shape({});
  } else {
    tensor_info = std::make_shared<tensor::Tensor>(data_type, shape);
  }
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "new tensor init failed";
    return nullptr;
  }
  if (data_size == 0) {
    return tensor_info;
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "input tensor data is nullptr";
    return nullptr;
  }
  auto ret = memcpy_s(tensor_info->data_c(), tensor_info->data().nbytes(), data, data_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s error : " << ret;
    return nullptr;
  }
  return tensor_info;
}

AbstractBasePtr CreateTensorAbstract(const std::vector<int64_t> &shape, TypeId data_type) {
  auto tensor_info = CreateTensorInfo(nullptr, 0, shape, data_type);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return nullptr;
  }
  auto abstract = tensor_info->ToAbstract();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return nullptr;
  }
  return abstract;
}

int SetParameterAbstractAndParam(const ParameterPtr &parameter, const void *data, size_t data_size,
                                 const std::vector<int64_t> &shape, TypeId data_type) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "Input parameter is nullptr";
    return RET_INPUT_PARAM_INVALID;
  }
  auto tensor_info = CreateTensorInfo(data, data_size, shape, data_type);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed";
    return RET_ERROR;
  }
  auto abstract = tensor_info->ToAbstract();
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return RET_ERROR;
  }
  parameter->set_abstract(abstract);
  return RET_OK;
}

int SetTensorData(const tensor::TensorPtr &tensor_info, const void *data, size_t data_size) {
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "tensor info is nullptr.";
    return RET_ERROR;
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "data is nullptr.";
    return RET_ERROR;
  }
  auto ret = memcpy_s(tensor_info->data_c(), tensor_info->data().nbytes(), data, data_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s error : " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

std::unique_ptr<schema::TensorT> CreateTensorTFromTensorInfo(const tensor::TensorPtr &tensor_info,
                                                             const std::string &tensor_name) {
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Input tensor is nullptr";
    return nullptr;
  }
  auto schema_tensor = std::make_unique<schema::TensorT>();
  MS_CHECK_TRUE_MSG(schema_tensor != nullptr, nullptr, "schema_tensor is nullptr");
  schema_tensor->name = tensor_name;
  auto ret = UpdateTensorTFromTensorInfo(tensor_info, &schema_tensor);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init schema tensor failed";
    return nullptr;
  }
  return schema_tensor;
}

int UpdateTensorTFromTensorInfo(const tensor::TensorPtr &src_tensor, std::unique_ptr<schema::TensorT> *dst_tensor) {
  if (src_tensor == nullptr) {
    MS_LOG(ERROR) << "Input tensor info is nullptr";
    return RET_INPUT_PARAM_INVALID;
  }
  if (dst_tensor == nullptr || *dst_tensor == nullptr) {
    MS_LOG(ERROR) << "Input schema tensor is nullptr";
    return RET_INPUT_PARAM_INVALID;
  }
  auto &schema_tensor = *dst_tensor;
  schema_tensor->format = schema::Format_NHWC;
  schema_tensor->dataType = src_tensor->data_type();
  auto &shape_vector = src_tensor->shape();
  std::vector<int32_t> dims;
  (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(dims),
                       [](const int64_t &value) { return static_cast<int32_t>(value); });
  schema_tensor->dims = dims;
  if (src_tensor->data().data() != nullptr) {
    schema_tensor->data.resize(src_tensor->data().nbytes());
    if (EOK != memcpy_s(schema_tensor->data.data(), schema_tensor->data.size(), src_tensor->data().data(),
                        src_tensor->data().nbytes())) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int InitParameterFromTensorInfo(const ParameterPtr &param_node, const tensor::TensorPtr &tensor_info) {
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "tensor info is nullptr.";
    return RET_ERROR;
  }
  auto abstract_tensor = tensor_info->ToAbstract();
  if (abstract_tensor == nullptr) {
    MS_LOG(ERROR) << "Create abstract tensor failed.";
    return RET_ERROR;
  }
  param_node->set_abstract(abstract_tensor);
  param_node->set_default_param(tensor_info);
  return RET_OK;
}

size_t GetElementSize(const TensorT &tensor) { return GetElementSize(TypeId(tensor.dataType)); }

size_t GetElementSize(const TypeId &dataType) {
  switch (dataType) {
    case kNumberTypeUInt8:
      return sizeof(uint8_t);
    case kNumberTypeInt32:
      return sizeof(int32_t);
    case kNumberTypeFloat:
      return sizeof(float);
    case kNumberTypeInt16:
      return sizeof(int16_t);
    case kNumberTypeInt8:
      return sizeof(int8_t);
    case kNumberTypeUInt32:
      return sizeof(uint32_t);
    default:
      return sizeof(float);
  }
}

size_t GetShapeSize(const TensorT &tensor) {
  auto shape = tensor.dims;
  size_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

std::unique_ptr<TensorT> CopyTensorDefT(const std::unique_ptr<TensorT> &oldTensor) {
  auto newTensor = std::unique_ptr<TensorT>(new (std::nothrow) TensorT);
  if (newTensor == nullptr) {
    MS_LOG(ERROR) << "new TensorT failed";
    return nullptr;
  }
  newTensor->dims = oldTensor->dims;
  newTensor->format = oldTensor->format;
  newTensor->dataType = oldTensor->dataType;
  newTensor->refCount = oldTensor->refCount;
  newTensor->nodeType = oldTensor->nodeType;
  newTensor->data = oldTensor->data;
  if (!oldTensor->quantParams.empty()) {
    newTensor->quantParams.emplace_back(GetTensorQuantParam(oldTensor));
  }
  return newTensor;
}

size_t GetRefCount(MetaGraphT *graphT, uint32_t tensorIdx) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(graphT->allTensors.size() > tensorIdx);
  size_t refCount = 0;
  for (auto &node : graphT->nodes) {
    MS_ASSERT(node != nullptr);
    if (IsContain(node->inputIndex, tensorIdx)) {
      refCount++;
    }
  }
  return refCount;
}
size_t GetShapeSize(const std::vector<int32_t> &shape) {
  size_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}
}  // namespace mindspore::lite
