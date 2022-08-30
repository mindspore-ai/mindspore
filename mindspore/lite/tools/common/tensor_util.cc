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

#include "src/tensor.h"
#include "tools/common/tensor_util.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "abstract/utils.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
namespace {
constexpr float kInputDataFloatMin = 0.1f;
constexpr float kInputDataFloatMax = 1.0f;
constexpr double kInputDataDoubleMin = 0.1;
constexpr double kInputDataDoubleMax = 1.0;
constexpr int64_t kInputDataInt64Min = 0;
constexpr int64_t kInputDataInt64Max = 1;
constexpr int32_t kInputDataInt32Min = 0;
constexpr int32_t kInputDataInt32Max = 1;
constexpr int16_t kInputDataInt16Min = 0;
constexpr int16_t kInputDataInt16Max = 1;
constexpr int16_t kInputDataInt8Min = -127;
constexpr int16_t kInputDataInt8Max = 127;
constexpr int16_t kInputDataUint8Min = 0;
constexpr int16_t kInputDataUint8Max = 254;
}  // namespace
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
  int size = 1;
  for (auto dim : shape) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(size, dim, nullptr);
    size *= dim;
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
  MS_CHECK_TRUE_MSG(tensor_info->Size() == data_size, nullptr, "invalid const tensor");
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
  MS_CHECK_TRUE_MSG(tensor_info->Size() == data_size, RET_ERROR, "invalid const tensor");
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
    if (dim <= 0) {
      MS_LOG(WARNING) << "Dim value less than or equal to 0 found in tensor's shape.";
      return 0;
    }
    shapeSize *= static_cast<size_t>(dim);
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
  MS_CHECK_TRUE_MSG(graphT != nullptr, 0, "graphT is nullptr");
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
    if (dim <= 0) {
      MS_LOG(WARNING) << "Dim value: " << dim << " is less than or equal to 0 found in tensor's shape.";
      return 0;
    }
    shapeSize *= static_cast<size_t>(dim);
  }
  return shapeSize;
}

int GenerateRandomData(size_t size, void *data, int data_type) {
  MS_ASSERT(data != nullptr);
  switch (data_type) {
    case kNumberTypeFloat32:
    case kNumberTypeFloat:
      FillInputData<float>(size, data, std::uniform_real_distribution<float>(kInputDataFloatMin, kInputDataFloatMax));
      break;
    case kNumberTypeFloat64:
      FillInputData<double>(size, data,
                            std::uniform_real_distribution<double>(kInputDataDoubleMin, kInputDataDoubleMax));
      break;
    case kNumberTypeInt64:
      FillInputData<int64_t>(size, data,
                             std::uniform_int_distribution<int64_t>(kInputDataInt64Min, kInputDataInt64Max));
      break;
    case kNumberTypeInt:
    case kNumberTypeInt32:
      FillInputData<int32_t>(size, data,
                             std::uniform_int_distribution<int32_t>(kInputDataInt32Min, kInputDataInt32Max));
      break;
    case kNumberTypeInt16:
      FillInputData<int16_t>(size, data,
                             std::uniform_int_distribution<int16_t>(kInputDataInt16Min, kInputDataInt16Max));
      break;
    case kNumberTypeInt8:
      FillInputData<int8_t>(size, data, std::uniform_int_distribution<int16_t>(kInputDataInt8Min, kInputDataInt8Max));
      break;
    case kNumberTypeUInt8:
      FillInputData<uint8_t>(size, data,
                             std::uniform_int_distribution<uint16_t>(kInputDataUint8Min, kInputDataUint8Max));
      break;
    default:
      char *casted_data = static_cast<char *>(data);
      for (size_t i = 0; i < size; i++) {
        casted_data[i] = static_cast<char>(i);
      }
  }
  return RET_OK;
}

int GenerateRandomData(mindspore::MSTensor *tensor) {
  MS_CHECK_TRUE_MSG(tensor != nullptr, RET_NULL_PTR, "tensor is nullptr");
  auto input_data = tensor->MutableData();
  if (input_data == nullptr) {
    MS_LOG(ERROR) << "MallocData for inTensor failed";
    return RET_ERROR;
  }
  int status = RET_ERROR;
  if (static_cast<TypeId>(tensor->DataType()) == kObjectTypeString) {
    MSTensor *input = MSTensor::StringsToTensor(tensor->Name(), {"you're the best."});
    if (input == nullptr) {
      std::cerr << "StringsToTensor failed" << std::endl;
      MS_LOG(ERROR) << "StringsToTensor failed";
      return RET_ERROR;
    }
    *tensor = *input;
    delete input;
  } else {
    status = GenerateRandomData(tensor->DataSize(), input_data, static_cast<int>(tensor->DataType()));
  }
  if (status != RET_OK) {
    std::cerr << "GenerateRandomData for inTensor failed: " << status << std::endl;
    MS_LOG(ERROR) << "GenerateRandomData for inTensor failed:" << status;
    return status;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
