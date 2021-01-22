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
#include "src/common/log_adapter.h"
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"

namespace mindspore {
namespace lite {
static const std::vector<schema::PrimitiveType> nhwcOpList = {
#ifdef SUPPORT_TRAIN
  schema::PrimitiveType_Conv2DGradFilter,
  schema::PrimitiveType_Conv2DGradInput,
  schema::PrimitiveType_GroupConv2DGradInput,
  schema::PrimitiveType_PoolingGrad,
  schema::PrimitiveType_BiasGrad,
  schema::PrimitiveType_BNGrad,
  schema::PrimitiveType_ActivationGrad,
  schema::PrimitiveType_ApplyMomentum,
  schema::PrimitiveType_Sgd,
  schema::PrimitiveType_Adam,
#endif
  schema::PrimitiveType_Conv2D,
  schema::PrimitiveType_DeConv2D,
  schema::PrimitiveType_DepthwiseConv2D,
  schema::PrimitiveType_DeDepthwiseConv2D,
  schema::PrimitiveType_Pooling,
  schema::PrimitiveType_LocalResponseNormalization,
  schema::PrimitiveType_Resize,
  schema::PrimitiveType_BatchNorm,
  schema::PrimitiveType_FusedBatchNorm,
  schema::PrimitiveType_PReLU,
  schema::PrimitiveType_BiasAdd,
  schema::PrimitiveType_SpaceToDepth,
  schema::PrimitiveType_DepthToSpace,
  schema::PrimitiveType_TopK};

static const std::vector<schema::PrimitiveType> nhwcOpAllInputList = {
#ifdef SUPPORT_TRAIN
  schema::PrimitiveType_PoolingGrad, schema::PrimitiveType_ActivationGrad, schema::PrimitiveType_Conv2DGradFilter,
  schema::PrimitiveType_BNGrad
#endif
};

static const std::vector<schema::PrimitiveType> fp32FullOpList = {
  schema::PrimitiveType_Concat, schema::PrimitiveType_Add,
  schema::PrimitiveType_Floor};  // fp32 ops support C4 and nhwc in fp32

static const std::vector<schema::PrimitiveType> int8NeedNhwcOpList = {};

static const std::vector<schema::PrimitiveType> int8OpList = {schema::PrimitiveType_Conv2D,
                                                              schema::PrimitiveType_DepthwiseConv2D,
                                                              schema::PrimitiveType_Add,
                                                              schema::PrimitiveType_Transpose,
                                                              schema::PrimitiveType_Pooling,
                                                              schema::PrimitiveType_Concat,
                                                              schema::PrimitiveType_SoftMax,
                                                              schema::PrimitiveType_Reshape,
                                                              schema::PrimitiveType_Activation,
                                                              schema::PrimitiveType_Resize,
                                                              schema::PrimitiveType_FullConnection,
                                                              schema::PrimitiveType_ArgMax,
                                                              schema::PrimitiveType_ArgMin,
                                                              schema::PrimitiveType_BatchNorm,
                                                              schema::PrimitiveType_FusedBatchNorm,
                                                              schema::PrimitiveType_BiasAdd,
                                                              schema::PrimitiveType_Div,
                                                              schema::PrimitiveType_Mul,
                                                              schema::PrimitiveType_Slice,
                                                              schema::PrimitiveType_SoftMax,
                                                              schema::PrimitiveType_Split,
                                                              schema::PrimitiveType_Squeeze,
                                                              schema::PrimitiveType_Sub,
                                                              schema::PrimitiveType_StridedSlice,
                                                              schema::PrimitiveType_TopK,
                                                              schema::PrimitiveType_Unsqueeze,
                                                              schema::PrimitiveType_MatMul,
                                                              schema::PrimitiveType_Pad,
                                                              schema::PrimitiveType_DeConv2D,
                                                              schema::PrimitiveType_Scale,
                                                              schema::PrimitiveType_Cast,
                                                              schema::PrimitiveType_Shape,
                                                              schema::PrimitiveType_ExpandDims,
                                                              schema::PrimitiveType_BatchToSpace,
                                                              schema::PrimitiveType_BatchToSpaceND,
                                                              schema::PrimitiveType_Reduce,
                                                              schema::PrimitiveType_Round,
                                                              schema::PrimitiveType_Floor,
                                                              schema::PrimitiveType_Ceil,
                                                              schema::PrimitiveType_Abs,
                                                              schema::PrimitiveType_Sin,
                                                              schema::PrimitiveType_Cos,
                                                              schema::PrimitiveType_Log,
                                                              schema::PrimitiveType_Sqrt,
                                                              schema::PrimitiveType_Rsqrt,
                                                              schema::PrimitiveType_Square,
                                                              schema::PrimitiveType_LogicalNot,
                                                              schema::PrimitiveType_SpaceToBatch,
                                                              schema::PrimitiveType_SpaceToBatchND,
                                                              schema::PrimitiveType_DepthToSpace,
                                                              schema::PrimitiveType_Power,
                                                              schema::PrimitiveType_GatherNd,
                                                              schema::PrimitiveType_LeakyReLU,
                                                              schema::PrimitiveType_Gather,
                                                              schema::PrimitiveType_Equal,
                                                              schema::PrimitiveType_NotEqual,
                                                              schema::PrimitiveType_LessEqual,
                                                              schema::PrimitiveType_Greater,
                                                              schema::PrimitiveType_GreaterEqual,
                                                              schema::PrimitiveType_Eltwise,
                                                              schema::PrimitiveType_DeDepthwiseConv2D,
                                                              schema::PrimitiveType_DetectionPostProcess,
                                                              schema::PrimitiveType_Crop,
                                                              schema::PrimitiveType_PriorBox,
                                                              schema::PrimitiveType_QuantDTypeCast,
                                                              schema::PrimitiveType_LayerNorm,
                                                              schema::PrimitiveType_L2Norm};

static const std::vector<schema::PrimitiveType> needInsertOpList = {
#ifdef SUPPORT_TRAIN
  schema::PrimitiveType_Eltwise,       schema::PrimitiveType_Activation,   schema::PrimitiveType_Concat,
  schema::PrimitiveType_Power,         schema::PrimitiveType_StridedSlice, schema::PrimitiveType_Split,
  schema::PrimitiveType_Crop,          schema::PrimitiveType_Mul,          schema::PrimitiveType_Add,
  schema::PrimitiveType_ActivationGrad
#else
  schema::PrimitiveType_Eltwise, schema::PrimitiveType_Activation,   schema::PrimitiveType_Concat,
  schema::PrimitiveType_Power,   schema::PrimitiveType_StridedSlice, schema::PrimitiveType_Add,
  schema::PrimitiveType_Split,   schema::PrimitiveType_Slice,        schema::PrimitiveType_Crop,
  schema::PrimitiveType_Mul,     schema::PrimitiveType_Maximum
#endif
};

static const std::unordered_map<int, int> nc2NhAxisMap = {{0, 0}, {1, -1}, {2, 1}, {3, 2}};

std::unordered_map<int, int> GetNc2NhAxisMap() { return nc2NhAxisMap; }

std::vector<schema::PrimitiveType> GetInsertOpList() { return needInsertOpList; }

std::vector<schema::PrimitiveType> Getfp32FullOpList() { return fp32FullOpList; }

std::vector<schema::PrimitiveType> GetNhwcOpList() { return nhwcOpList; }

std::vector<schema::PrimitiveType> GetNhwcAllInputOpList() { return nhwcOpAllInputList; }

std::vector<schema::PrimitiveType> GetUint8NhwcOpList() { return int8NeedNhwcOpList; }

std::vector<schema::PrimitiveType> GetInt8OpList() { return int8OpList; }

STATUS NodeUtils::ConvertDims(mindspore::schema::Format src_format, const std::vector<int32_t> &src_dims,
                              mindspore::schema::Format dst_format, std::vector<int32_t> *dst_dims) {
  MS_ASSERT(nullptr != dst_dims);
  if ((src_dims.size() != DIM_DEFAULT_SIZE && src_dims.size() != 3) || src_format == dst_format) {
    MS_LOG(ERROR) << "Convert format , src size " << src_dims.size()
                  << " <3 or src format is equal to dst format,not need convert";
    *dst_dims = src_dims;
    return RET_PARAM_INVALID;
  }

  std::vector<int32_t> nchw_dim;
  switch (src_format) {
    case schema::Format::Format_NCHW:
      nchw_dim = src_dims;
      break;
    case schema::Format::Format_NHWC:
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
      MS_LOG(ERROR) << "Not support src format: " << EnumNameFormat(src_format);
      return RET_ERROR;
  }

  if (nchw_dim.empty()) {
    MS_LOG(ERROR) << "Param nchw_dim is empty!";
    return RET_ERROR;
  }

  switch (dst_format) {
    case schema::Format::Format_NCHW:
      *dst_dims = nchw_dim;
      break;
    case schema::Format::Format_NHWC:
      if (src_dims.size() == DIM_DEFAULT_SIZE) {
        dst_dims->push_back(nchw_dim[NCHW_N]);
        dst_dims->push_back(nchw_dim[NCHW_H]);
        dst_dims->push_back(nchw_dim[NCHW_W]);
        dst_dims->push_back(nchw_dim[NCHW_C]);
      }
      break;
    default:
      MS_LOG(ERROR) << "Not support dst format: " << dst_format;
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS GetFilterDim(const std::vector<int32_t> &oriDims, kTransFilterType type, int32_t *filterK, int32_t *filterC,
                    int32_t *filterH, int32_t *filterW) {
  if (filterK == nullptr || filterC == nullptr || filterH == nullptr || filterW == nullptr) {
    MS_LOG(ERROR) << "null input";
    return RET_NULL_PTR;
  }
  MS_ASSERT(oriDims.size() == 4);
  if (type == kKCHW2HWCK || type == kKCHW2HWKC || type == kKCHW2KHWC || type == kKCHW2CKHW) {
    *filterK = oriDims.at(KCHW_K);
    *filterC = oriDims.at(KCHW_C);
    *filterH = oriDims.at(KCHW_H);
    *filterW = oriDims.at(KCHW_W);
  } else if (type == kCKHW2HWCK || type == kCKHW2HWKC || type == kCKHW2KHWC) {
    *filterC = oriDims.at(CKHW_C);
    *filterK = oriDims.at(CKHW_K);
    *filterH = oriDims.at(CKHW_H);
    *filterW = oriDims.at(CKHW_W);
  } else if (type == kHWCK2KCHW || type == kHWCK2CKHW) {
    *filterH = oriDims.at(HWCK_H);
    *filterW = oriDims.at(HWCK_W);
    *filterC = oriDims.at(HWCK_C);
    *filterK = oriDims.at(HWCK_K);
  } else if (type == kHWKC2KCHW || type == kHWKC2CKHW) {
    *filterH = oriDims.at(HWKC_H);
    *filterW = oriDims.at(HWKC_W);
    *filterK = oriDims.at(HWKC_K);
    *filterC = oriDims.at(HWKC_C);
  } else if (type == kNHWC2KCHW || type == kNHWC2HWCK || type == kNHWC2CKHW) {
    *filterK = oriDims.at(NHWC_N);
    *filterH = oriDims.at(NHWC_H);
    *filterW = oriDims.at(NHWC_W);
    *filterC = oriDims.at(NHWC_C);
  } else if (type == kCHWK2HWCK || type == kCHWK2KHWC) {
    *filterC = oriDims.at(CHWK_C);
    *filterH = oriDims.at(CHWK_H);
    *filterW = oriDims.at(CHWK_W);
    *filterK = oriDims.at(CHWK_K);
  } else if (type == kKHWC2HWCK || type == kKHWC2CHWK) {
    *filterK = oriDims.at(KHWC_K);
    *filterH = oriDims.at(KHWC_H);
    *filterW = oriDims.at(KHWC_W);
    *filterC = oriDims.at(KHWC_C);
  } else {
    MS_LOG(ERROR) << "Unsupported transFilterType: " << type;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS SetFilterDim(schema::TensorT *tensor, kTransFilterType type, int32_t filterK, int32_t filterC, int32_t filterH,
                    int32_t filterW) {
  MS_ASSERT(tensor != nullptr);
  if (type == kKCHW2HWCK || type == kCKHW2HWCK || type == kNHWC2HWCK || type == kKHWC2HWCK || type == kCHWK2HWCK) {
    tensor->dims = {filterH, filterW, filterC, filterK};
  } else if (type == kKCHW2HWKC || type == kCKHW2HWKC) {
    tensor->dims = {filterH, filterW, filterK, filterC};
  } else if (type == kHWCK2KCHW || type == kHWKC2KCHW || type == kNHWC2KCHW) {
    tensor->dims = {filterK, filterC, filterH, filterW};
  } else if (type == kHWCK2CKHW || type == kHWKC2CKHW || type == kNHWC2CKHW || type == kKCHW2CKHW) {
    tensor->dims = {filterC, filterK, filterH, filterW};
  } else if (type == kKHWC2CHWK) {
    tensor->dims = {filterC, filterH, filterW, filterK};
  } else if (type == kKCHW2KHWC || type == kCKHW2KHWC || type == kCHWK2KHWC) {
    tensor->dims = {filterK, filterH, filterW, filterC};
  } else {
    MS_LOG(ERROR) << "Unsupported transFilterType: " << type;
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS TransFilterFormat(schema::TensorT *tensor, schema::Format dstFormat) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "tensor is null";
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
    case schema::Format::Format_KHWC: {
      switch (srcFormat) {
        case schema::Format::Format_KCHW:
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
        case schema::Format::Format_CKHW:
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
        case schema::Format::Format_CHWK:
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
        case schema::Format::Format_KHWC:
          return RET_OK;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << EnumNameFormat(srcFormat) << " to "
                        << EnumNameFormat(dstFormat);
          return RET_ERROR;
      }
    } break;
    case schema::Format::Format_HWCK: {
      switch (srcFormat) {
        case schema::Format::Format_KCHW:
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
        case schema::Format::Format_KHWC:
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
        case schema::Format::Format_CKHW:
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
        case schema::Format::Format_CHWK:
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
        case schema::Format::Format_HWCK:
          return RET_OK;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << EnumNameFormat(srcFormat) << " to "
                        << EnumNameFormat(dstFormat);
          return RET_ERROR;
      }
    } break;
    case schema::Format::Format_KCHW: {
      switch (srcFormat) {
        case schema::Format::Format_KCHW:
          return RET_OK;
        case schema::Format::Format_HWCK:
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
        case schema::Format::Format_HWKC:
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
        case schema::Format::Format_KHWC:
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
        case schema::Format::Format_CKHW:
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
        case schema::Format::Format_CHWK:
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
          MS_LOG(ERROR) << "Unsupported transform from " << EnumNameFormat(srcFormat) << " to "
                        << EnumNameFormat(dstFormat);
          return RET_ERROR;
      }
    } break;
    case schema::Format::Format_CKHW: {
      switch (srcFormat) {
        case schema::Format::Format_HWCK:
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
        case schema::Format::Format_HWKC:
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
        case schema::Format::Format_KCHW:
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
        case schema::Format::Format_CKHW:
          return RET_OK;
        default:
          MS_LOG(ERROR) << "Unsupported transform from " << EnumNameFormat(srcFormat) << " to "
                        << EnumNameFormat(dstFormat);
          return RET_ERROR;
      }
    } break;
    default:
      MS_LOG(ERROR) << "Unsupported transform from " << EnumNameFormat(srcFormat) << " to "
                    << EnumNameFormat(dstFormat);
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
