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
#include <map>
#include <functional>
#include <numeric>
#include "common/kernel_base.h"
#include "common/kernel_errcode.h"
#include "common/tensor.h"

namespace aicpu {
namespace {
// max param len limit 10k.
constexpr uint32_t MAX_PARAM_LEN = 10240;
// max io address num limit 1024
constexpr uint32_t MAX_IO_ADDR_NUMPARAM_LEN = 1024;
}  // namespace

static const std::map<const ::aicpuops::DataType, size_t> kKernelBaseDataTypeSize = {
  {aicpuops::MS_BOOL, sizeof(bool)},           {aicpuops::MS_INT8, sizeof(int8_t)},
  {aicpuops::MS_UINT8, sizeof(uint8_t)},       {aicpuops::MS_INT16, sizeof(int16_t)},
  {aicpuops::MS_UINT16, sizeof(uint16_t)},     {aicpuops::MS_INT32, sizeof(int32_t)},
  {aicpuops::MS_UINT32, sizeof(uint32_t)},     {aicpuops::MS_INT64, sizeof(int64_t)},
  {aicpuops::MS_UINT64, sizeof(uint64_t)},     {aicpuops::MS_FLOAT16, sizeof(float) / 2},
  {aicpuops::MS_FLOAT32, sizeof(float)},       {aicpuops::MS_FLOAT64, sizeof(double)},
  {aicpuops::MS_COMPLEX64, sizeof(float) * 2}, {aicpuops::MS_COMPLEX128, sizeof(double) * 2}};

KernelBase::KernelBase(const std::string &kernel_name)
    : kernel_name_(kernel_name),
      extend_param_len_(0),
      extend_param_base_(nullptr),
      param_head_(nullptr),
      unknow_shape_(false) {}

uint32_t KernelBase::ParseParam(void *param) {
  if (param == nullptr) {
    AICPU_LOGE("Kernel:%s ParseParam param is null.", kernel_name_.c_str());
    return kAicpuKernelStateInvalid;
  }

  // parse param_len
  param_head_ = static_cast<AicpuParamHead *>(param);
  if (param_head_->length < sizeof(AicpuParamHead) || param_head_->length > MAX_PARAM_LEN) {
    AICPU_LOGE("Kernel:%s param length=%u not in [%zu, %u].", kernel_name_.c_str(), param_head_->length,
               sizeof(AicpuParamHead), MAX_PARAM_LEN);
    return kAicpuKernelStateInvalid;
  }

  auto param_base = static_cast<uint8_t *>(param);
  extend_param_base_ = param_base + sizeof(AicpuParamHead);
  extend_param_len_ = param_head_->length - sizeof(AicpuParamHead);

  if (param_head_->ioAddrNum > 0) {
    if (param_head_->ioAddrNum > MAX_IO_ADDR_NUMPARAM_LEN) {
      AICPU_LOGE("Kernel:%s param ioAddrNum=%u is over %u.", kernel_name_.c_str(), param_head_->ioAddrNum,
                 MAX_IO_ADDR_NUMPARAM_LEN);
      return kAicpuKernelStateInvalid;
    }
    uint32_t addr_len = static_cast<uint32_t>(param_head_->ioAddrNum * sizeof(uint64_t));
    if (extend_param_len_ < addr_len) {
      AICPU_LOGE("Kernel:%s extend param is not enough for io addr, ioAddrNum=%u, extendParamLen=%u.",
                 kernel_name_.c_str(), param_head_->ioAddrNum, extend_param_len_);
      return kAicpuKernelStateInvalid;
    }
    auto io_addr_base = reinterpret_cast<uint64_t *>(extend_param_base_);
    for (uint32_t i = 0; i < param_head_->ioAddrNum; ++i) {
      io_addrs_.push_back(static_cast<uintptr_t>(io_addr_base[i]));
    }
    extend_param_base_ = extend_param_base_ + addr_len;
    extend_param_len_ -= addr_len;
  }
  AICPU_CHK_STATUS_RET(ParseNodeDef())
  AICPU_CHK_STATUS_RET(ParseExtInfo())
  if (unknow_shape_) {
    AICPU_LOGI("Unknown shape op: %s", kernel_name_.c_str());
    UpdateInputShape();
    UpdateOutputShape();
  }
  return ParseKernelParam();
}

uint32_t KernelBase::Compute(void *param) {
  uint32_t ret = ParseParam(param);
  if (ret != kAicpuKernelStateSucess) {
    AICPU_LOGE("Kernel:%s ParseParam failed, ret=%u.", kernel_name_.c_str(), ret);
    return ret;
  }
  return DoCompute();
}

size_t KernelBase::GetDataTypeSize(::aicpuops::DataType data_type) {
  auto it = kKernelBaseDataTypeSize.find(data_type);
  if (it == kKernelBaseDataTypeSize.end()) {
    AICPU_LOGE("don't support input tensor types");
    return 0;
  }
  return it->second;
}

size_t KernelBase::GetTensorMemSizeByShape(::aicpuops::Tensor tensor) {
  std::vector<int64_t> shape;
  auto tensor_shape = tensor.tensor_shape();
  for (int i = 0; i < tensor_shape.dim_size(); ++i) {
    shape.push_back(tensor_shape.dim(i).size());
  }
  auto data_type = static_cast<aicpuops::DataType>(tensor.tensor_type());
  int64_t element_num = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  return LongToSize(element_num) * GetDataTypeSize(data_type);
}

template <typename T>
uint32_t KernelBase::ParseExtendParam(T *param_var, const std::string &param_name) {
  if (extend_param_len_ < sizeof(T)) {
    AICPU_LOGE("Kernel:%s extend param is not enough for [%s] addr, need_len=%u, extendParamLen=%u.",
               kernel_name_.c_str(), param_name.c_str(), sizeof(T), extend_param_len_);
    return kAicpuKernelStateInvalid;
  }
  T *param = reinterpret_cast<T *>(extend_param_base_);
  if (param != nullptr) {
    *param_var = *param;
    extend_param_base_ += sizeof(T);
    extend_param_len_ -= sizeof(T);
    return kAicpuKernelStateSucess;
  }
  AICPU_LOGE("Kernel:%s extend param for [%s] addr is invalid.", kernel_name_.c_str(), param_name.c_str());
  return kAicpuKernelStateInvalid;
}

uint32_t KernelBase::ParseNodeDef() {
  uint32_t node_def_len;
  AICPU_CHK_STATUS_RET(ParseExtendParam(&node_def_len, "node_def_len"))

  if (extend_param_len_ < node_def_len) {
    AICPU_LOGE("Kernel:%s extend param is not enough for customizeAttr addr, node_def_len=%u, extendParamLen=%u.",
               kernel_name_.c_str(), node_def_len, extend_param_len_);
    return kAicpuKernelStateInvalid;
  }
  std::string std_data(reinterpret_cast<char *>(extend_param_base_), node_def_len);
  if (!node_def_.ParseFromString(std_data)) {
    AICPU_LOGE("parse %s KernelBase proto failed, nodeDef=%s.", kernel_name_.c_str(), std_data.c_str());
    return kAicpuKernelStateInvalid;
  }
  extend_param_base_ += node_def_len;
  extend_param_len_ -= node_def_len;
  return kAicpuKernelStateSucess;
}

uint32_t KernelBase::ParseExtShapeType(FWKAdapter::ExtInfo *ext_info) {
  if (ext_info->infoLen != sizeof(int32_t)) {
    AICPU_LOGE("Kernel:%s parse ext shape type failed as infoLen must be %zu but %u.", kernel_name_.c_str(),
               sizeof(int32_t), ext_info->infoLen);
    return kAicpuKernelStateInvalid;
  }
  unknow_shape_ = true;
  return kAicpuKernelStateSucess;
}

uint32_t KernelBase::ParseExtInputShape(FWKAdapter::ExtInfo *ext_info) {
  // no overflow
  auto need_len = node_def_.inputs_size() * sizeof(FWKAdapter::ShapeAndType);
  if (ext_info->infoLen != need_len) {
    AICPU_LOGE(
      "Kernel:%s parse ext input shape failed as infoLen must be "
      "input_num[%d]*sizeof(ShapeAndType)[%zu], but %u.",
      kernel_name_.c_str(), node_def_.inputs_size(), sizeof(FWKAdapter::ShapeAndType), ext_info->infoLen);
    return kAicpuKernelStateInvalid;
  }
  input_shape_and_type_.clear();
  auto input = reinterpret_cast<FWKAdapter::ShapeAndType *>(ext_info->infoMsg);
  for (int index = 0; index < node_def_.inputs_size(); ++index) {
    (void)input_shape_and_type_.emplace_back(&input[index]);
  }
  return kAicpuKernelStateSucess;
}

uint32_t KernelBase::ParseExtOutputShape(FWKAdapter::ExtInfo *ext_info) {
  // no overflow
  auto need_len = node_def_.outputs_size() * sizeof(FWKAdapter::ShapeAndType);
  if (ext_info->infoLen != need_len) {
    AICPU_LOGE(
      "Kernel:%s parse ext output shape failed as infoLen must be "
      "output_num[%d]*sizeof(ShapeAndType)[%zu], but %u.",
      kernel_name_.c_str(), node_def_.outputs_size(), sizeof(FWKAdapter::ShapeAndType), ext_info->infoLen);
    return kAicpuKernelStateInvalid;
  }
  output_shape_and_type_.clear();
  auto output = reinterpret_cast<FWKAdapter::ShapeAndType *>(ext_info->infoMsg);
  for (int index = 0; index < node_def_.outputs_size(); ++index) {
    (void)output_shape_and_type_.emplace_back(&output[index]);
  }
  return kAicpuKernelStateSucess;
}

uint32_t KernelBase::ParseExtInfo() {
  uint32_t offset = 0;
  FWKAdapter::ExtInfo *ext_info_ptr = nullptr;
  char *ext_info_buf = reinterpret_cast<char *>(static_cast<uintptr_t>(param_head_->extInfoAddr));
  while (offset + sizeof(FWKAdapter::ExtInfo) <= param_head_->extInfoLength) {
    ext_info_ptr = reinterpret_cast<FWKAdapter::ExtInfo *>(ext_info_buf + offset);
    if (ext_info_ptr == nullptr) {
      AICPU_LOGE("Kernel:%s ext_info is nullptr, extInfoLength=%u, extInfoAddr=%p, offset=%zu.", kernel_name_.c_str(),
                 param_head_->extInfoLength, param_head_->extInfoAddr, offset);
      return kAicpuKernelStateInvalid;
    }
    switch (ext_info_ptr->infoType) {
      case FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE:
        AICPU_CHK_STATUS_RET(ParseExtShapeType(ext_info_ptr))
        break;
      case FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE:
        AICPU_CHK_STATUS_RET(ParseExtInputShape(ext_info_ptr))
        break;
      case FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:
        AICPU_CHK_STATUS_RET(ParseExtOutputShape(ext_info_ptr))
        break;
      default:
        AICPU_LOGI("Kernel:%s ignore infoType=%d, infoLen=%u.", kernel_name_.c_str(), ext_info_ptr->infoType,
                   ext_info_ptr->infoLen);
        break;
    }
    // not overflow
    offset += FWKAdapter::kExtInfoHeadSize;
    offset += ext_info_ptr->infoLen;
  }
  return kAicpuKernelStateSucess;
}

void KernelBase::UpdateInputShape() {
  for (int i = 0; i < node_def_.inputs_size(); ++i) {
    aicpuops::Tensor *input_tensor = node_def_.mutable_inputs(i);
    aicpuops::TensorShape *input_tensor_shape = input_tensor->mutable_tensor_shape();
    input_tensor_shape->clear_dim();
    for (uint32_t index = 0; index < FWKAdapter::kMaxShapeDims; ++index) {
      // LLONG_MIN for dim end flag
      if (input_shape_and_type_[i]->dims[index] == LLONG_MIN) {
        break;
      }
      input_tensor_shape->add_dim()->set_size(input_shape_and_type_[IntToSize(i)]->dims[index]);
    }
  }
}

void KernelBase::UpdateOutputShape() {
  for (int i = 0; i < node_def_.outputs_size(); ++i) {
    aicpuops::Tensor *output_tensor = node_def_.mutable_outputs(i);
    aicpuops::TensorShape *output_tensor_shape = output_tensor->mutable_tensor_shape();
    output_tensor_shape->clear_dim();
    for (uint32_t index = 0; index < FWKAdapter::kMaxShapeDims; ++index) {
      // LLONG_MIN for dim end flag
      if (output_shape_and_type_[i]->dims[index] == LLONG_MIN) {
        break;
      }
      output_tensor_shape->add_dim()->set_size(output_shape_and_type_[i]->dims[index]);
    }
  }
}
}  // namespace aicpu
