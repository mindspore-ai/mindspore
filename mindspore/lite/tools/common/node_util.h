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

#ifndef MINDSPORE_PREDICT_NODE_UTIL_H
#define MINDSPORE_PREDICT_NODE_UTIL_H

#include <memory>
#include <vector>
#include "schema/inner/model_generated.h"
#include "src/common/common.h"
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace lite {
using STATUS = int;
STATUS BroadCastQuantParam(schema::MetaGraphT *graphT, const std::unique_ptr<schema::CNodeT> &node);

std::vector<schema::PrimitiveType> GetNhwcOpList();

std::vector<schema::PrimitiveType> Getfp32FullOpList();

std::vector<schema::PrimitiveType> GetUint8NhwcOpList();

std::vector<schema::PrimitiveType> GetUint8OpList();

class NodeUtils {
 public:
  static STATUS ConvertDims(schema::Format src_format, const std::vector<int32_t> &src_dims, schema::Format dst_format,
                            std::vector<int32_t> *dst_dims);

  static void SliceData(std::vector<char *> &input, int64_t chunk_size, std::vector<char *> &output, int64_t begin,
                        int64_t out_dim, int64_t stride);

  static STATUS SetOutputSliceData(void *data, int64_t data_size, int32_t data_type, std::vector<int32_t> &input_dims,
                                   std::vector<int32_t> &begin, std::vector<int32_t> &output_dims,
                                   schema::TensorT *output, std::vector<int32_t> &stride);
};

// todo check this
enum kTransFilterType {
  kKCHW2HWCK,
  kKCHW2KHWC,
  kCKHW2KHWC,
  kCKHW2HWCK,
  kKCHW2HWKC,
  kCKHW2HWKC,
  kHWCK2KCHW,
  kHWCK2CKHW,
  kHWKC2KCHW,
  kHWKC2CKHW,
  kNHWC2KCHW,
  kNHWC2CKHW,
  kNHWC2HWCK,
  kKHWC2HWCK,
  kCHWK2HWCK,
  kKHWC2CHWK,
  kCHWK2KHWC
};

static STATUS GetFilterDim(std::vector<int32_t> &oriDims, kTransFilterType type, int32_t &filterK, int32_t &filterC,
                           int32_t &filterH, int32_t &filterW) {
  MS_ASSERT(oriDims.size() == 4);
  if (type == kKCHW2HWCK || type == kKCHW2HWKC || type == kKCHW2KHWC) {
    filterK = oriDims.at(KCHW_K);
    filterC = oriDims.at(KCHW_C);
    filterH = oriDims.at(KCHW_H);
    filterW = oriDims.at(KCHW_W);
  } else if (type == kCKHW2HWCK || type == kCKHW2HWKC || type == kCKHW2KHWC) {
    filterC = oriDims.at(CKHW_C);
    filterK = oriDims.at(CKHW_K);
    filterH = oriDims.at(CKHW_H);
    filterW = oriDims.at(CKHW_W);
  } else if (type == kHWCK2KCHW || type == kHWCK2CKHW) {
    filterH = oriDims.at(HWCK_H);
    filterW = oriDims.at(HWCK_W);
    filterC = oriDims.at(HWCK_C);
    filterK = oriDims.at(HWCK_K);
  } else if (type == kHWKC2KCHW || type == kHWKC2CKHW) {
    filterH = oriDims.at(HWKC_H);
    filterW = oriDims.at(HWKC_W);
    filterK = oriDims.at(HWKC_K);
    filterC = oriDims.at(HWKC_C);
  } else if (type == kNHWC2KCHW || type == kNHWC2HWCK || type == kNHWC2CKHW) {
    filterK = oriDims.at(NHWC_N);
    filterH = oriDims.at(NHWC_H);
    filterW = oriDims.at(NHWC_W);
    filterC = oriDims.at(NHWC_C);
  } else if (type == kCHWK2HWCK || type == kCHWK2KHWC) {
    filterC = oriDims.at(CHWK_C);
    filterH = oriDims.at(CHWK_H);
    filterW = oriDims.at(CHWK_W);
    filterK = oriDims.at(CHWK_K);
  } else if (type == kKHWC2HWCK || type == kKHWC2CHWK) {
    filterK = oriDims.at(KHWC_K);
    filterH = oriDims.at(KHWC_H);
    filterW = oriDims.at(KHWC_W);
    filterC = oriDims.at(KHWC_C);
  } else {
    MS_LOG(ERROR) << "Unsupported transFilterType: " << type;
    return RET_ERROR;
  }
  return RET_OK;
}

static STATUS SetFilterDim(schema::TensorT *tensor, kTransFilterType type, int32_t filterK, int32_t filterC,
                           int32_t filterH, int32_t filterW) {
  MS_ASSERT(tensor != nullptr);
  if (type == kKCHW2HWCK || type == kCKHW2HWCK || type == kNHWC2HWCK || type == kKHWC2HWCK || type == kCHWK2HWCK) {
    tensor->dims = {filterH, filterW, filterC, filterK};
  } else if (type == kKCHW2HWKC || type == kCKHW2HWKC) {
    tensor->dims = {filterH, filterW, filterK, filterC};
  } else if (type == kHWCK2KCHW || type == kHWKC2KCHW || type == kNHWC2KCHW) {
    tensor->dims = {filterK, filterC, filterH, filterW};
  } else if (type == kHWCK2CKHW || type == kHWKC2CKHW || type == kNHWC2CKHW) {
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

template <typename T>
static STATUS TransFilterData(schema::TensorT *tensor, kTransFilterType type, int32_t filterK, int32_t filterC,
                              int32_t filterH, int32_t filterW) {
  MS_ASSERT(tensor != nullptr);
  int count = filterH * filterW * filterC * filterK;
  if (count <= 0) {
    MS_LOG(ERROR) << "Dim size invalid";
    return RET_ERROR;
  }
  std::unique_ptr<T> buf(new (std::nothrow) T[count]);
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
  T *p1Buff = nullptr;
  T *p2Buff = nullptr;
  switch (type) {
    case kCHWK2HWCK:
    case kCHWK2KHWC: {
      for (int c = 0; c < filterC; ++c) {
        for (int h = 0; h < filterH; ++h) {
          for (int w = 0; w < filterW; ++w) {
            for (int k = 0; k < filterK; ++k) {
              p1Buff = weightData + ((c * filterH * filterW * filterK) + (h * filterW * filterK) + (w * filterK) + (k));
              if (type == kCHWK2HWCK) {
                p2Buff =
                  buf.get() + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              } else if (type == kCHWK2KHWC) {
                p2Buff =
                  buf.get() + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kKHWC2HWCK: {
      for (int k = 0; k < filterK; ++k) {
        for (int h = 0; h < filterH; ++h) {
          for (int w = 0; w < filterW; ++w) {
            for (int c = 0; c < filterC; ++c) {
              p1Buff = weightData + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              p2Buff = buf.get() + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kKCHW2HWCK:
    case kKCHW2KHWC:
    case kKCHW2HWKC: {
      for (int k = 0; k < filterK; ++k) {
        for (int c = 0; c < filterC; ++c) {
          for (int h = 0; h < filterH; ++h) {
            for (int w = 0; w < filterW; ++w) {
              p1Buff = weightData + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
              if (type == kKCHW2HWCK) {
                p2Buff =
                  buf.get() + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              } else if (type == kKCHW2KHWC) {
                p2Buff =
                  buf.get() + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              } else {
                p2Buff =
                  buf.get() + ((h * filterW * filterK * filterC) + (w * filterK * filterC) + (k * filterC) + (c));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kCKHW2HWCK:
    case kCKHW2KHWC:
    case kCKHW2HWKC: {
      for (int c = 0; c < filterC; ++c) {
        for (int k = 0; k < filterK; ++k) {
          for (int h = 0; h < filterH; ++h) {
            for (int w = 0; w < filterW; ++w) {
              p1Buff = weightData + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
              if (type == kCKHW2HWCK) {
                p2Buff =
                  buf.get() + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              } else if (type == kCKHW2KHWC) {
                p2Buff =
                  buf.get() + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              } else {
                p2Buff =
                  buf.get() + ((h * filterW * filterK * filterC) + (w * filterK * filterC) + (k * filterC) + (c));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kHWCK2KCHW:
    case kHWCK2CKHW: {
      for (int h = 0; h < filterH; ++h) {
        for (int w = 0; w < filterW; ++w) {
          for (int c = 0; c < filterC; ++c) {
            for (int k = 0; k < filterK; ++k) {
              p1Buff = weightData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              if (type == kHWCK2KCHW) {
                p2Buff =
                  buf.get() + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
              } else {
                p2Buff =
                  buf.get() + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kHWKC2KCHW:
    case kHWKC2CKHW: {
      for (int h = 0; h < filterH; ++h) {
        for (int w = 0; w < filterW; ++w) {
          for (int c = 0; c < filterC; ++c) {
            for (int k = 0; k < filterK; ++k) {
              p1Buff = weightData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (k * filterC) + (c));
              if (type == kHWKC2KCHW) {
                p2Buff =
                  buf.get() + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
              } else {
                p2Buff =
                  buf.get() + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kNHWC2HWCK:
    case kNHWC2KCHW:
    case kNHWC2CKHW: {
      for (int k = 0; k < filterK; ++k) {
        for (int h = 0; h < filterH; ++h) {
          for (int w = 0; w < filterW; ++w) {
            for (int c = 0; c < filterC; ++c) {
              p1Buff = weightData + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (k * filterC) + (c));
              if (type == kNHWC2HWCK) {
                p2Buff =
                  buf.get() + ((h * filterW * filterC * filterK) + (w * filterC * filterK) + (c * filterK) + (k));
              } else if (type == kNHWC2CKHW) {
                p2Buff =
                  buf.get() + ((c * filterK * filterH * filterW) + (k * filterH * filterW) + (h * filterW) + (w));
              } else {
                p2Buff =
                  buf.get() + ((k * filterC * filterH * filterW) + (c * filterH * filterW) + (h * filterW) + (w));
              }
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    case kKHWC2CHWK: {
      for (int k = 0; k < filterK; ++k) {
        for (int h = 0; h < filterH; ++h) {
          for (int w = 0; w < filterW; ++w) {
            for (int c = 0; c < filterC; ++c) {
              p1Buff = weightData + ((k * filterH * filterW * filterC) + (h * filterW * filterC) + (w * filterC) + (c));
              p2Buff = buf.get() + ((c * filterK * filterH * filterW) + (h * filterK * filterW) + (w * filterK) + (k));
              *p2Buff = *p1Buff;
            }
          }
        }
      }
    } break;
    default: {
      MS_LOG(ERROR) << "Unsupported transFilterType: " << type;
      return RET_ERROR;
    }
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
  auto status = GetFilterDim(oriDims, type, filterK, filterC, filterH, filterW);
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
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_NODE_UTIL_H

