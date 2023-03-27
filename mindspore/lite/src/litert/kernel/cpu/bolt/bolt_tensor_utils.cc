/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "bolt/bolt_tensor_utils.h"
#include <vector>
#include <memory>
#include <map>

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_OK;

namespace mindspore::kernel::bolt {
static const std::map<TypeId, BoltDataType> type_map = {
  {kNumberTypeInt, BoltDataType::DT_I32},     {kNumberTypeInt32, BoltDataType::DT_I32},
  {kNumberTypeInt8, BoltDataType::DT_I8},     {kNumberTypeInt64, BoltDataType::DT_I64},
  {kNumberTypeUInt, BoltDataType::DT_U32},    {kNumberTypeUInt32, BoltDataType::DT_U32},
  {kNumberTypeUInt32, BoltDataType::DT_U32},  {kNumberTypeUInt8, BoltDataType::DT_U8},
  {kNumberTypeUInt64, BoltDataType::DT_U64},  {kNumberTypeFloat, BoltDataType::DT_F32},
  {kNumberTypeFloat32, BoltDataType::DT_F32}, {kNumberTypeFloat16, BoltDataType::DT_F16},
  {kNumberTypeFloat64, BoltDataType::DT_F64},
};

static const std::map<Format, DataFormat> format_map = {
  {NCHW, DataFormat::DF_NCHW},
  {NHWC, DataFormat::DF_NHWC},
  {NC4HW4, DataFormat::DF_NCHWC4},
  {NC8HW8, DataFormat::DF_NCHWC8},
};

int ConvertTensorDatatype(const TypeId &lite_dtype, BoltDataType *bolt_dtype) {
  auto itr = type_map.find(lite_dtype);
  if (itr == type_map.end()) {
    MS_LOG(ERROR) << "Unsupported data type: " << lite_dtype << " for bolt.";
    return RET_NOT_SUPPORT;
  }
  *bolt_dtype = itr->second;
  return RET_OK;
}

int ConvertTensorFormat(const Format &lite_format, DataFormat *bolt_format) {
  auto itr = format_map.find(lite_format);
  if (itr == format_map.end()) {
    MS_LOG(ERROR) << "Unsupported format: " << lite_format << " for bolt.";
    return RET_NOT_SUPPORT;
  }
  *bolt_format = itr->second;
  return RET_OK;
}

int ConvertTensorShape(const std::vector<int> &lite_shape, U32 *bolt_shape) {
  if (lite_shape.size() > DIM_LEN) {
    MS_LOG(ERROR) << "The bolt tensor max dim is 6.";
    return RET_ERROR;
  }
  auto dim = lite_shape.size();
  for (size_t i = 0; i < dim; i++) {
    bolt_shape[i] = lite_shape[dim - i - 1];
  }
  return RET_OK;
}

void AdjustBoltTensorDesc(TensorDesc *desc) {
  // For bolt tensor, the format is NCHW, the shape is (W, H, C, N)
  if (desc->df == DataFormat::DF_NCHWC4) {
    desc->dims[DIMENSION_2D] = UP_ROUND(desc->dims[DIMENSION_2D], C4NUM);
  }
  if (desc->df == DataFormat::DF_NCHWC8) {
    desc->dims[DIMENSION_2D] = UP_ROUND(desc->dims[DIMENSION_2D], C8NUM);
  }
}

int LiteTensor2BoltTensor(const lite::Tensor *src_tensor, BoltTensor *dst_tensor) {
  TensorDesc bolt_tensor_desc;
  auto ret = ConvertTensorDatatype(src_tensor->data_type(), &bolt_tensor_desc.dt);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Convert lite tensor datatype to bolt tensor datatype failed.";
    return ret;
  }
  ret = ConvertTensorFormat(src_tensor->format(), &bolt_tensor_desc.df);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Convert lite tensor format to bolt tensor format failed.";
    return ret;
  }
  bolt_tensor_desc.nDims = src_tensor->shape().size();
  ret = ConvertTensorShape(src_tensor->shape(), bolt_tensor_desc.dims);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Convert lite tensor shape to bolt tensor dims failed.";
    return ret;
  }
  AdjustBoltTensorDesc(&bolt_tensor_desc);
  // set TensorDesc
  dst_tensor->resize(bolt_tensor_desc);
  // set data
  if (src_tensor->data() != nullptr) {
    std::shared_ptr<U8> data(reinterpret_cast<U8 *>(src_tensor->data()), [](U8 *ptr) {});
    reinterpret_cast<CpuMemory *>(dst_tensor->get_memory())->set_shared_ptr(data);
  }
  return RET_OK;
}
}  // namespace mindspore::kernel::bolt
