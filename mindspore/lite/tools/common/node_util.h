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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_NODE_UTIL_H
#define MINDSPORE_LITE_TOOLS_COMMON_NODE_UTIL_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "schema/inner/model_generated.h"
#include "src/common/common.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace lite {
template <typename T>
int CreateOperator(const std::unique_ptr<schema::PrimitiveT> &primitive, schema::PrimitiveType type) {
  auto attr = std::make_unique<T>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }
  primitive->value.type = type;
  primitive->value.value = attr.release();
  return RET_OK;
}

using STATUS = int;
STATUS BroadCastQuantParam(schema::MetaGraphT *graphT, const std::unique_ptr<schema::CNodeT> &node);

inline schema::PrimitiveType GetCNodeTType(const schema::CNodeT &cNodeT) { return cNodeT.primitive->value.type; }

inline std::string GetCNodeTTypeName(const schema::CNodeT &cNodeT) {
  return schema::EnumNamePrimitiveType(GetCNodeTType(cNodeT));
}

inline schema::PrimitiveType GetOpType(const schema::CNode &opDef) { return opDef.primitive()->value_type(); }

inline std::string GetOpTypeName(const schema::CNode &opDef) { return schema::EnumNamePrimitiveType(GetOpType(opDef)); }

std::unordered_map<int, int> GetNc2NhAxisMap();

std::vector<schema::PrimitiveType> GetInsertOpList();

std::vector<schema::PrimitiveType> GetNhwcOpList();

std::vector<schema::PrimitiveType> GetNchwOpList();

std::vector<schema::PrimitiveType> GetNhwcAllInputOpList();

std::unordered_map<schema::PrimitiveType, std::vector<int>> GetExtNhwcIndexes();

std::vector<schema::PrimitiveType> Getfp32FullOpList();

std::vector<schema::PrimitiveType> GetUint8NhwcOpList();

std::vector<schema::PrimitiveType> GetInt8OpList();

const schema::Primitive *ConvertToPrimitive(schema::PrimitiveT *primitive_t, flatbuffers::FlatBufferBuilder *fbb);

class NodeUtils {
 public:
  static STATUS ConvertDims(schema::Format src_format, const std::vector<int32_t> &src_dims, schema::Format dst_format,
                            std::vector<int32_t> *dst_dims);
};

enum kTransFilterType {
  kKCHW2HWCK,  // 0
  kKCHW2KHWC,
  kCKHW2KHWC,
  kCKHW2HWCK,
  kKCHW2HWKC,
  kCKHW2HWKC,
  kHWCK2KCHW,
  kHWCK2CKHW,
  kHWKC2KCHW,
  kHWKC2CKHW,
  kNHWC2KCHW,  // 10
  kNHWC2CKHW,
  kNHWC2HWCK,
  kKHWC2HWCK,
  kCHWK2HWCK,
  kKHWC2CHWK,
  kCHWK2KHWC,
  kKHWC2KCHW,
  kCKHW2KCHW,
  kCHWK2KCHW,
  kKCHW2CKHW  // 20
};

STATUS GetFilterDim(const std::vector<int32_t> &oriDims, kTransFilterType type, int32_t *filterK, int32_t *filterC,
                    int32_t *filterH, int32_t *filterW);
STATUS SetFilterDim(schema::TensorT *tensor, kTransFilterType type, int32_t filterK, int32_t filterC, int32_t filterH,
                    int32_t filterW);

template <typename T>
static void TransKHWC2CHWK(int32_t filterK, int32_t filterC, int32_t filterH, int32_t filterW, T *srcData, T *dstData) {
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  for (int k = 0; k < filterK; ++k) {
    for (int h = 0; h < filterH; ++h) {
      for (int w = 0; w < filterW; ++w) {
        for (int c = 0; c < filterC; ++c) {
          p1Buff = srcData + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
          p2Buff = dstData + ((c * filterK * filterH * filterW) + (h * filterK * filterW) + (w * filterK) + (k));
          *p2Buff = *p1Buff;
        }
      }
    }
  }
}

template <typename T>
static void TransKHWC2HWCK(int32_t filterK, int32_t filterC, int32_t filterH, int32_t filterW, T *srcData, T *dstData) {
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  for (int k = 0; k < filterK; ++k) {
    for (int h = 0; h < filterH; ++h) {
      for (int w = 0; w < filterW; ++w) {
        for (int c = 0; c < filterC; ++c) {
          p1Buff = srcData + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
          p2Buff = dstData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
          *p2Buff = *p1Buff;
        }
      }
    }
  }
}

template <typename T>
static void TransCKHW(kTransFilterType type, int32_t filterK, int32_t filterC, int32_t filterH, int32_t filterW,
                      T *srcData, T *dstData) {
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  for (int c = 0; c < filterC; ++c) {
    for (int k = 0; k < filterK; ++k) {
      for (int h = 0; h < filterH; ++h) {
        for (int w = 0; w < filterW; ++w) {
          p1Buff = srcData + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
          if (type == kCKHW2HWCK) {
            p2Buff = dstData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
          } else if (type == kCKHW2KHWC) {
            p2Buff = dstData + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
          } else {
            p2Buff = dstData + ((h * filterW * filterK * filterC) + (w * filterK * filterC) + (k * filterC) + (c));
          }
          *p2Buff = *p1Buff;
        }
      }
    }
  }
}

template <typename T>
static void TransKCHW(kTransFilterType type, int32_t filterK, int32_t filterC, int32_t filterH, int32_t filterW,
                      T *srcData, T *dstData) {
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  for (int k = 0; k < filterK; ++k) {
    for (int c = 0; c < filterC; ++c) {
      for (int h = 0; h < filterH; ++h) {
        for (int w = 0; w < filterW; ++w) {
          p1Buff = srcData + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
          if (type == kKCHW2HWCK) {
            p2Buff = dstData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
          } else if (type == kKCHW2KHWC) {
            p2Buff = dstData + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
          } else if (type == kKCHW2CKHW) {
            p2Buff = dstData + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
          } else {
            p2Buff = dstData + ((h * filterW * filterK * filterC) + (w * filterK * filterC) + (k * filterC) + (c));
          }
          *p2Buff = *p1Buff;
        }
      }
    }
  }
}

template <typename T>
static void TransCHWK(kTransFilterType type, int32_t filterK, int32_t filterC, int32_t filterH, int32_t filterW,
                      T *srcData, T *dstData) {
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  for (int c = 0; c < filterC; ++c) {
    for (int h = 0; h < filterH; ++h) {
      for (int w = 0; w < filterW; ++w) {
        for (int k = 0; k < filterK; ++k) {
          p1Buff = srcData + ((c * filterH * filterW * filterK) + (h * filterW * filterK) + (w * filterK) + (k));
          if (type == kCHWK2HWCK) {
            p2Buff = dstData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
          } else {
            p2Buff = dstData + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
          }
          *p2Buff = *p1Buff;
        }
      }
    }
  }
}

template <typename T>
static void TransHWCK(kTransFilterType type, int32_t filterK, int32_t filterC, int32_t filterH, int32_t filterW,
                      T *srcData, T *dstData) {
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  for (int h = 0; h < filterH; ++h) {
    for (int w = 0; w < filterW; ++w) {
      for (int c = 0; c < filterC; ++c) {
        for (int k = 0; k < filterK; ++k) {
          p1Buff = srcData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
          if (type == kHWCK2KCHW) {
            p2Buff = dstData + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
          } else {
            p2Buff = dstData + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
          }
          *p2Buff = *p1Buff;
        }
      }
    }
  }
}

template <typename T>
static void TransHWKC(kTransFilterType type, int32_t filterK, int32_t filterC, int32_t filterH, int32_t filterW,
                      T *srcData, T *dstData) {
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  for (int h = 0; h < filterH; ++h) {
    for (int w = 0; w < filterW; ++w) {
      for (int c = 0; c < filterC; ++c) {
        for (int k = 0; k < filterK; ++k) {
          p1Buff = srcData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (k * filterC) + (c));
          if (type == kHWKC2KCHW) {
            p2Buff = dstData + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
          } else {
            p2Buff = dstData + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
          }
          *p2Buff = *p1Buff;
        }
      }
    }
  }
}

template <typename T>
static void TransNHWC(kTransFilterType type, int32_t filterK, int32_t filterC, int32_t filterH, int32_t filterW,
                      T *srcData, T *dstData) {
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  for (int k = 0; k < filterK; ++k) {
    for (int h = 0; h < filterH; ++h) {
      for (int w = 0; w < filterW; ++w) {
        for (int c = 0; c < filterC; ++c) {
          p1Buff = srcData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (k * filterC) + (c));
          if (type == kNHWC2HWCK) {
            p2Buff = dstData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
          } else if (type == kNHWC2CKHW) {
            p2Buff = dstData + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
          } else {
            p2Buff = dstData + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
          }
          *p2Buff = *p1Buff;
        }
      }
    }
  }
}

template <typename T>
static STATUS TransFilterData(kTransFilterType type, int32_t filterK, int32_t filterC, int32_t filterH, int32_t filterW,
                              T *srcData, T *dstData) {
  switch (type) {
    case kCHWK2HWCK:
    case kCHWK2KHWC: {
      TransCHWK(type, filterK, filterC, filterH, filterW, srcData, dstData);
    } break;
    case kKHWC2HWCK: {
      TransKHWC2HWCK(filterK, filterC, filterH, filterW, srcData, dstData);
    } break;
    case kKCHW2HWCK:
    case kKCHW2CKHW:
    case kKCHW2KHWC:
    case kKCHW2HWKC: {
      TransKCHW(type, filterK, filterC, filterH, filterW, srcData, dstData);
    } break;
    case kCKHW2HWCK:
    case kCKHW2KHWC:
    case kCKHW2HWKC: {
      TransCKHW(type, filterK, filterC, filterH, filterW, srcData, dstData);
    } break;
    case kHWCK2KCHW:
    case kHWCK2CKHW: {
      TransHWCK(type, filterK, filterC, filterH, filterW, srcData, dstData);
    } break;
    case kHWKC2KCHW:
    case kHWKC2CKHW: {
      TransHWKC(type, filterK, filterC, filterH, filterW, srcData, dstData);
    } break;
    case kNHWC2HWCK:
    case kNHWC2KCHW:
    case kNHWC2CKHW: {
      TransNHWC(type, filterK, filterC, filterH, filterW, srcData, dstData);
    } break;
    case kKHWC2CHWK: {
      TransKHWC2CHWK(filterK, filterC, filterH, filterW, srcData, dstData);
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported transFilterType: " << type;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

template <typename T>
static STATUS TransFilterData(schema::TensorT *tensor, kTransFilterType type, int32_t filterK, int32_t filterC,
                              int32_t filterH, int32_t filterW) {
  MS_ASSERT(tensor != nullptr);
  int count = filterH * filterW * filterC * filterK;
  if (count <= 0) {
    MS_LOG(ERROR) << "Dim size invalid";
    return RET_ERROR;
  }
  std::unique_ptr<T[]> buf(new (std::nothrow) T[count]);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "new buf failed";
    return RET_ERROR;
  }

  void *originWeightDate = tensor->data.data();
  T *weightData = static_cast<T *>(originWeightDate);

  if (weightData == nullptr) {
    MS_LOG(ERROR) << "weightData is nullptr";
    return RET_ERROR;
  }

  if (TransFilterData(type, filterK, filterC, filterH, filterW, weightData, buf.get()) != RET_OK) {
    MS_LOG(ERROR) << "TransFilterData failed";
    return RET_ERROR;
  }

  auto ret = ::memcpy_s(tensor->data.data(), count * sizeof(T), buf.get(), count * sizeof(T));
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

template <typename T>
static STATUS TransFilterFormat(schema::TensorT *tensor, kTransFilterType type) {
  MS_ASSERT(tensor != nullptr);
  std::vector<int32_t> oriDims = tensor->dims;
  if (oriDims.size() != (size_t)DIM_DEFAULT_SIZE) {
    MS_LOG(ERROR) << "Filter dim-num is not supported, dim-num: " << oriDims.size();
    return RET_ERROR;
  }

  int32_t filterH;
  int32_t filterW;
  int32_t filterC;
  int32_t filterK;
  auto status = GetFilterDim(oriDims, type, &filterK, &filterC, &filterH, &filterW);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetFilterDim failed: " << status;
    return status;
  }
  status = SetFilterDim(tensor, type, filterK, filterC, filterH, filterW);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "SetFilterDim failed: " << status;
    return status;
  }
  status = TransFilterData<T>(tensor, type, filterK, filterC, filterH, filterW);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "TransFilterData failed: " << status;
    return status;
  }

  return RET_OK;
}

STATUS TransFilterFormat(schema::TensorT *tensor, schema::Format dstFormat);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_COMMON_NODE_UTIL_H
