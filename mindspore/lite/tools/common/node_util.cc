/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "tools/common/node_util.h"
#include <memory>
#include <vector>
#include "src/common/common.h"
#include "utils/log_adapter.h"
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"

namespace mindspore {
namespace lite {
static const std::vector<schema::PrimitiveType> nhwcOpList = {
  schema::PrimitiveType_Conv2D,          schema::PrimitiveType_DeConv2D,
  schema::PrimitiveType_DepthwiseConv2D, schema::PrimitiveType_DeDepthwiseConv2D,
  schema::PrimitiveType_Pooling,         schema::PrimitiveType_Resize,
  schema::PrimitiveType_BatchNorm,       schema::PrimitiveType_FusedBatchNorm,
  schema::PrimitiveType_CaffePReLU};

static const std::vector<schema::PrimitiveType> fp32FullOpList = {
  schema::PrimitiveType_Concat, schema::PrimitiveType_Add,
  schema::PrimitiveType_Floor};  // fp32 ops support C4 and nhwc in fp32

static const std::vector<schema::PrimitiveType> uint8NeedNhwcOpList = {};

static const std::vector<schema::PrimitiveType> uint8OpList = {
  schema::PrimitiveType_Nchw2Nhwc,       schema::PrimitiveType_Nhwc2Nchw, schema::PrimitiveType_Conv2D,
  schema::PrimitiveType_DepthwiseConv2D, schema::PrimitiveType_Add,       schema::PrimitiveType_Pooling,
  schema::PrimitiveType_Concat,          schema::PrimitiveType_SoftMax,   schema::PrimitiveType_Reshape,
  schema::PrimitiveType_Activation};

std::vector<schema::PrimitiveType> Getfp32FullOpList() { return fp32FullOpList; }

std::vector<schema::PrimitiveType> GetNhwcOpList() { return nhwcOpList; }

std::vector<schema::PrimitiveType> GetUint8NhwcOpList() { return uint8NeedNhwcOpList; }

std::vector<schema::PrimitiveType> GetUint8OpList() { return uint8OpList; }

STATUS NodeUtils::ConvertDims(mindspore::lite::Format src_format, const std::vector<int32_t> &src_dims,
                              mindspore::lite::Format dst_format, std::vector<int32_t> *dst_dims) {
  if ((src_dims.size() != DIM_DEFAULT_SIZE && src_dims.size() != 3) || src_format == dst_format) {
    MS_LOG(ERROR) << "Convert format , src size " << src_dims.size()
                  << " <3 or src format is equal to dst format,not need convert";
    *dst_dims = src_dims;
    return RET_PARAM_INVALID;
  }

  std::vector<int32_t> nchw_dim;
  switch (src_format) {
    case Format_NCHW:
      nchw_dim = src_dims;
      break;
    case Format_NHWC:
      if (src_dims.size() == DIM_DEFAULT_SIZE) {
        nchw_dim.push_back(src_dims[NHWC_N]);
        nchw_dim.push_back(src_dims[NHWC_C]);
        nchw_dim.push_back(src_dims[NHWC_H]);
        nchw_dim.push_back(src_dims[NHWC_W]);
      } else {
        nchw_dim.push_back(src_dims[HWC_C]);
        nchw_dim.push_back(src_dims[HWC_H]);
        nchw_dim.push_back(src_dims[HWC_W]);
      }
      break;
    default:
      MS_LOG(ERROR) << "Not support src format: " << schema::EnumNameFormat(src_format);
      return RET_ERROR;
  }

  if (nchw_dim.size() == 0) {
    MS_LOG(ERROR) << "Param nchw_dim is empty!";
    return RET_ERROR;
  }

  switch (dst_format) {
    case Format_NCHW:
      *dst_dims = nchw_dim;
      break;
    case Format_NHWC:
      if (src_dims.size() == DIM_DEFAULT_SIZE) {
        dst_dims->push_back(nchw_dim[NCHW_N]);
        dst_dims->push_back(nchw_dim[NCHW_H]);
        dst_dims->push_back(nchw_dim[NCHW_W]);
        dst_dims->push_back(nchw_dim[NCHW_C]);
      }
      break;
    default:
      // MS_LOG(ERROR)("Not support dst format: %d", dst_format);
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS TransFilterFormat(schema::TensorT *tensor, schema::Format dstFormat) {
  if (tensor == nullptr) {
    return RET_NULL_PTR;
  }
  std::vector<int32_t> oriDims = tensor->dims;
  if (oriDims.size() != (size_t)DIM_DEFAULT_SIZE) {
    MS_LOG(ERROR) << "Filter dim-num is not supported, dim-num: " << oriDims.size();
    return RET_ERROR;
  }
  auto srcFormat = tensor->format;
  auto dataType = tensor->dataType;
  STATUS status;
  switch (dstFormat) {
    case schema::Format_KHWC: {
      switch (srcFormat) {
        case schema::Format_KCHW:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKCHW2KHWC);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKCHW2KHWC);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKCHW2KHWC);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_CKHW:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCKHW2KHWC);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCKHW2KHWC);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCKHW2KHWC);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_CHWK:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCHWK2KHWC);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCHWK2KHWC);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCHWK2KHWC);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_KHWC:
          return RET_OK;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << schema::EnumNameFormat(srcFormat) << " to "
                        << schema::EnumNameFormat(dstFormat);
          return RET_ERROR;
      }
    } break;
    case schema::Format_HWCK: {
      switch (srcFormat) {
        case schema::Format_KCHW:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKCHW2HWCK);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKCHW2HWCK);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKCHW2HWCK);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_KHWC:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKHWC2HWCK);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKHWC2HWCK);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKHWC2HWCK);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_CKHW:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCKHW2HWCK);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCKHW2HWCK);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCKHW2HWCK);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_CHWK:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCHWK2HWCK);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCHWK2HWCK);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCHWK2HWCK);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_HWCK:
          return RET_OK;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << schema::EnumNameFormat(srcFormat) << " to "
                        << schema::EnumNameFormat(dstFormat);
          return RET_ERROR;
      }
    } break;
    case schema::Format_KCHW: {
      switch (srcFormat) {
        case schema::Format_KCHW:
          return RET_OK;
        case schema::Format_HWCK:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kHWCK2KCHW);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kHWCK2KCHW);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kHWCK2KCHW);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_HWKC:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kHWKC2KCHW);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kHWKC2KCHW);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kHWKC2KCHW);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_KHWC:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKHWC2KCHW);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKHWC2KCHW);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKHWC2KCHW);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_CKHW:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCKHW2KCHW);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCKHW2KCHW);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCKHW2KCHW);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_CHWK:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kCHWK2KCHW);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kCHWK2KCHW);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kCHWK2KCHW);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << schema::EnumNameFormat(srcFormat) << " to "
                        << schema::EnumNameFormat(dstFormat);
          return RET_ERROR;
      }
    } break;
    case schema::Format_CKHW: {
      switch (srcFormat) {
        case schema::Format_HWCK:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kHWCK2CKHW);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kHWCK2CKHW);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kHWCK2CKHW);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_HWKC:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kHWKC2CKHW);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kHWKC2CKHW);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kHWKC2CKHW);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_KCHW:
          if (dataType == kNumberTypeFloat32) {
            status = TransFilterFormat<float>(tensor, kKCHW2CKHW);
          } else if (dataType == kNumberTypeUInt8) {
            status = TransFilterFormat<uint8_t>(tensor, kKCHW2CKHW);
          } else if (dataType == kNumberTypeInt8) {
            status = TransFilterFormat<int8_t>(tensor, kKCHW2CKHW);
          } else {
            MS_LOG(ERROR) << "Unsupported dataType: " << dataType;
            return RET_ERROR;
          }
          break;
        case schema::Format_CKHW:
          return RET_OK;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << schema::EnumNameFormat(srcFormat) << " to "
                        << schema::EnumNameFormat(dstFormat);
          return RET_ERROR;
      }
    } break;
    default:
      MS_LOG(ERROR) << "Unsupported transform from " << schema::EnumNameFormat(srcFormat) << " to "
                    << schema::EnumNameFormat(dstFormat);
      return RET_ERROR;
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "TransFilterData failed: " << status;
    return status;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
