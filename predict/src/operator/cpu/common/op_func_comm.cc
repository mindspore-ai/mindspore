/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/operator/cpu/include/op_func_comm.h"
#include "include/errorcode.h"
#include "include/tensor.h"
#include "common/mslog.h"
#include "securec/include/securec.h"

namespace mindspore {
namespace predict {
#ifndef MS_USE_NEON
#ifndef MS_USE_SSE

void MSAddBias(float *srcPtr, const float *bias, size_t unitSize, size_t count) {
  if (srcPtr == nullptr || bias == nullptr) {
    MS_LOGW("srcPtr or bias is nullptr");
    return;
  }
  for (size_t stepU = 0; stepU < count; stepU++) {
    float *tmpPtr = srcPtr + unitSize * CAL_STEP * stepU;
    const float *biasPtr = bias + CAL_STEP * stepU;
    for (size_t step = 0; step < unitSize; step++) {
      float *dstPtr = tmpPtr + CAL_STEP * step;
      for (int i = 0; i < CAL_STEP; i++) {
        dstPtr[i] += biasPtr[i];
      }
    }
  }
}

void MSAddBiasRelu(float *srcPtr, const float *bias, size_t unitSize, size_t count) {
  if (srcPtr == nullptr || bias == nullptr) {
    MS_LOGW("srcPtr or bias is nullptr");
    return;
  }
  for (size_t stepU = 0; stepU < count; stepU++) {
    float *tmpPtr = srcPtr + unitSize * CAL_STEP * stepU;
    const float *biasPtr = bias + CAL_STEP * stepU;
    for (size_t step = 0; step < unitSize; step++) {
      float *dstPtr = tmpPtr + CAL_STEP * step;
      for (int i = 0; i < CAL_STEP; i++) {
        dstPtr[i] += biasPtr[i];
        dstPtr[i] = (dstPtr[i] < 0) ? 0 : dstPtr[i];
      }
    }
  }
}

void MSAddBiasRelu6(float *srcPtr, const float *bias, size_t unitSize, size_t count) {
  if (srcPtr == nullptr || bias == nullptr) {
    MS_LOGW("srcPtr or bias is nullptr");
    return;
  }
  for (size_t stepU = 0; stepU < count; stepU++) {
    float *tmpPtr = srcPtr + unitSize * CAL_STEP * stepU;
    const float *biasPtr = bias + CAL_STEP * stepU;
    for (size_t step = 0; step < unitSize; step++) {
      float *dstPtr = tmpPtr + CAL_STEP * step;
      for (int i = 0; i < CAL_STEP; i++) {
        dstPtr[i] += biasPtr[i];
        dstPtr[i] = (dstPtr[i] < 0) ? 0 : dstPtr[i];
        dstPtr[i] = (dstPtr[i] > 6.0f) ? 6.0f : dstPtr[i];
      }
    }
  }
}

void MSCopyC4WithStride(const float *srcPtr, float *dstPtr, size_t srcStride, size_t dstStride, size_t count) {
  if (srcPtr == nullptr || dstPtr == nullptr) {
    MS_LOGW("srcPtr or dstPtr is nullptr");
    return;
  }
  for (size_t stepU = 0; stepU < count; stepU++) {
    auto sPtr = srcPtr + stepU * srcStride;
    auto dPtr = dstPtr + stepU * dstStride;
    int tmpC = 0;
    while (tmpC < CAL_STEP) {
      dPtr[tmpC] = sPtr[tmpC];
      tmpC++;
    }
  }
}
#endif  // MS_USE_SSE

int MSPackC4(float *dstPtr, const float *srcPtr, size_t area, size_t depth) {
  if (dstPtr == nullptr || srcPtr == nullptr) {
    MS_LOGE("srcPtr or dstPtr is nullptr");
    return RET_ERROR;
  }
  int cur = 0;
  size_t size = area * UP_DIV(depth, CAL_STEP) * CAL_STEP * sizeof(float);
  auto ret = memset_s(dstPtr, size, 0, size);
  if (ret != EOK) {
    MS_LOGE("memset_s failed!");
    return RET_ERROR;
  }
  for (size_t step = 0; step < depth; step++) {
    auto plane = step / CAL_STEP;
    auto offset = step % CAL_STEP;
    auto dstPlane = plane * area * CAL_STEP + dstPtr;
    for (size_t i = 0; i < area; i++) {
      dstPlane[CAL_STEP * i + offset] = srcPtr[cur++];
    }
  }
  return RET_OK;
}

void MSUnpackC4(float *dstPtr, const float *srcPtr, size_t area, size_t depth) {
  if (dstPtr == nullptr || srcPtr == nullptr) {
    MS_LOGW("srcPtr or dstPtr is nullptr");
    return;
  }
  int cur = 0;
  for (size_t step = 0; step < depth; step++) {
    auto plane = step / CAL_STEP;
    auto offset = step % CAL_STEP;
    auto srcPlane = plane * area * CAL_STEP + srcPtr;
    for (size_t i = 0; i < area; i++) {
      dstPtr[cur++] = srcPlane[CAL_STEP * i + offset];
    }
  }
}

void MSUInt8ToInt16WithOffsetC4Common(int16_t *dstPtr, const uint8_t *srcPtr, size_t zeroPoint, size_t sizeQuad,
                                      size_t dstStride, size_t srcStride) {
  if (dstPtr == nullptr || srcPtr == nullptr) {
    MS_LOGW("srcPtr or dstPtr is nullptr");
    return;
  }
  for (size_t step = 0; step < sizeQuad; step++) {
    auto dstZ = dstPtr + (dstStride / sizeof(int16_t)) * step;
    auto srcZ = srcPtr + (srcStride / sizeof(uint8_t)) * step;
    for (int i = 0; i < CAL_STEP; i++) {
      dstZ[i] = (int16_t)((int32_t)srcZ[i] - (int32_t)zeroPoint);
    }
  }
}

void MSUInt8ToInt16WithOffsetC4Fast(int16_t *colAddr, const uint8_t *srcStart, size_t zeroPoint, size_t sizeQuad,
                                    size_t depthQuad, size_t dstZStep, size_t srcZStep) {
  if (colAddr == nullptr || srcStart == nullptr) {
    MS_LOGW("colAddr or srcStart is nullptr");
    return;
  }
  for (size_t step = 0; step < depthQuad; step++) {
    auto dstZ = colAddr + (dstZStep / sizeof(int16_t)) * step;
    auto srcZ = srcStart + (srcZStep / sizeof(uint8_t)) * step;
    MSUInt8ToInt16WithOffsetC4Common(dstZ, srcZ, zeroPoint, sizeQuad, CAL_STEP * sizeof(int16_t),
                                     CAL_STEP * sizeof(uint8_t));
  }
}
#endif

void MSPackC4Uint8(uint8_t *dstPtr, const uint8_t *srcPtr, size_t area, size_t depth) {
  if (dstPtr == nullptr || srcPtr == nullptr) {
    MS_LOGW("srcPtr or dstPtr is nullptr");
    return;
  }
  int cur = 0;
  size_t size = area * UP_DIV(depth, CAL_STEP) * CAL_STEP * sizeof(uint8_t);
  auto ret = memset_s(dstPtr, size, 0, size);
  if (ret != EOK) {
    MS_LOGE("memset_s failed!");
    return;
  }
  for (size_t step = 0; step < depth; step++) {
    auto plane = step / CAL_STEP;
    auto offset = step % CAL_STEP;
    auto dstPlane = plane * area * CAL_STEP + dstPtr;
    for (size_t x = 0; x < area; ++x) {
      dstPlane[CAL_STEP * x + offset] = srcPtr[cur++];
    }
  }
}

void MSUnpackC4Uint8(uint8_t *dstPtr, const uint8_t *srcPtr, size_t area, size_t depth) {
  if (dstPtr == nullptr || srcPtr == nullptr) {
    MS_LOGW("srcPtr or dstPtr is nullptr");
    return;
  }
  int cur = 0;
  for (size_t step = 0; step < depth; step++) {
    auto srcPlane = (step / CAL_STEP) * area * CAL_STEP + srcPtr;
    for (size_t i = 0; i < area; i++) {
      dstPtr[cur++] = srcPlane[CAL_STEP * i + (step % CAL_STEP)];
    }
  }
}

#ifdef MS_USE_NEON
static void MSTensorConvertNCHWToNC4HW4Depth(float *dst, const float *src, size_t area, size_t depth) {
  if (dstPtr == nullptr || srcPtr == nullptr) {
    MS_LOGW("srcPtr or dstPtr is nullptr");
    return;
  }
  if (1 == depth) {
    auto zeroValue = vmovq_n_f32(0.0f);
    int areaC4 = static_cast<int>(area / CAL_STEP);
    int remain = areaC4 * CAL_STEP;
    for (int i = 0; i < areaC4; ++i) {
      auto srcCur = src + CAL_STEP * i;
      auto dstCur = dst + CAL_STEP * CAL_STEP * i;
      auto srcValue = vld1q_f32(srcCur);
      float32x4x4_t dstValue;
      dstValue.val[0] = srcValue;
      dstValue.val[1] = zeroValue;
      dstValue.val[2] = zeroValue;
      dstValue.val[3] = zeroValue;
      vst4q_f32(dstCur, dstValue);
    }
    for (int i = remain; i < area; ++i) {
      dst[CAL_STEP * i + 0] = src[i];
      dst[CAL_STEP * i + 1] = 0.0f;
      dst[CAL_STEP * i + 2] = 0.0f;
      dst[CAL_STEP * i + 3] = 0.0f;
    }
  } else if (3 == depth) {
    auto zeroValue = vmovq_n_f32(0.0f);
    int areaC4 = static_cast<int>(area / CAL_STEP);
    int remain = areaC4 * CAL_STEP;
    for (int i = 0; i < areaC4; ++i) {
      auto srcCur = src + 12 * i;
      auto dstCur = dst + 16 * i;
      auto srcValue = vld3q_f32(srcCur);
      float32x4x4_t dstValue;
      dstValue.val[0] = srcValue.val[0];
      dstValue.val[1] = srcValue.val[1];
      dstValue.val[2] = srcValue.val[2];
      dstValue.val[3] = zeroValue;
      vst4q_f32(dstCur, dstValue);
    }
    for (int i = remain; i < area; ++i) {
      dst[CAL_STEP * i + 0] = src[3 * i + 0];
      dst[CAL_STEP * i + 1] = src[3 * i + 1];
      dst[CAL_STEP * i + 2] = src[3 * i + 2];
      dst[CAL_STEP * i + 3] = 0.0f;
    }
  }
}
#endif

void MSTensorConvertNHWCToNC4HW4(float *dst, const float *src, size_t area, size_t depth) {
  if (dst == nullptr || src == nullptr) {
    MS_LOGW("srcPtr or dstPtr is nullptr");
    return;
  }
#ifdef MS_USE_NEON
  MSTensorConvertNCHWToNC4HW4Depth(dst, src, area, depth);
  return;
#endif
  int c = static_cast<int>(depth);
  int cDiv4 = c / CAL_STEP;
  int cMod4 = c % CAL_STEP;
  int cAlign = cDiv4 * CAL_STEP;
  for (int hi = 0; hi < area; ++hi) {
    auto srcHeight = src + hi * c;
    auto dstHeight = dst + hi * CAL_STEP;
    for (int ci = 0; ci < cDiv4; ++ci) {
#ifdef MS_USE_NEON
      vst1q_f32(dstHeight + CAL_STEP * ci * area, vld1q_f32(srcHeight + CAL_STEP * ci));
#else
      for (int i = 0; i < CAL_STEP; ++i) {
        dstHeight[ci * area * CAL_STEP + i] = srcHeight[CAL_STEP * ci + i];
      }
#endif
    }
  }

  if (cMod4 == 0) {
    MS_LOGW("depth should be multiple of four");
    return;
  }

  auto srcAlign = src + cAlign;
  auto dstAlign = dst + area * cAlign;

#ifdef MS_USE_NEON
  auto zeroVector = vdupq_n_f32(0.0f);
#endif

  for (int hi = 0; hi < area; ++hi) {
    auto srcHeight = srcAlign + hi * c;
    auto dstHeight = dstAlign + hi * CAL_STEP;
#ifdef MS_USE_NEON
    vst1q_f32(dstHeight, zeroVector);
#else
    for (int i = 0; i < CAL_STEP; ++i) {
      dstHeight[i] = 0;
    }
#endif
    for (int ci = 0; ci < cMod4; ++ci) {
      dstHeight[ci] = srcHeight[ci];
    }
  }
}

void MSTensorConvertNC4HW4ToNHWC(float *dst, const float *src, size_t area, size_t depth) {
  if (dst == nullptr || src == nullptr) {
    MS_LOGW("srcPtr or dstPtr is nullptr");
    return;
  }
  int c = static_cast<int>(depth);
  int cDiv4 = c / CAL_STEP;
  int cMod4 = c % CAL_STEP;
  int cAlign = cDiv4 * CAL_STEP;
  for (int hi = 0; hi < area; ++hi) {
    auto srcHeight = src + hi * CAL_STEP;
    auto dstHeight = dst + hi * c;
    for (int ci = 0; ci < cDiv4; ++ci) {
#ifdef MS_USE_NEON
      vst1q_f32(dstHeight + CAL_STEP * ci, vld1q_f32(srcHeight + CAL_STEP * ci * area));
#else
      for (int i = 0; i < CAL_STEP; ++i) {
        dstHeight[ci * CAL_STEP + i] = srcHeight[CAL_STEP * ci * area + i];
      }
#endif
    }
  }

  if (cMod4 == 0) {
    MS_LOGW("depth should be multiple of four");
    return;
  }

  auto srcAlign = src + area * cAlign;
  auto dstAlign = dst + cAlign;

  for (int hi = 0; hi < area; ++hi) {
    auto srcHeight = srcAlign + hi * CAL_STEP;
    auto dstHeight = dstAlign + hi * c;

    for (int ci = 0; ci < cMod4; ++ci) {
      dstHeight[ci] = srcHeight[ci];
    }
  }
}

int NchwToNc4hw4(const Tensor *input, Tensor *output) {
  if (input == nullptr || output == nullptr) {
    MS_LOGE("input or output is nullptr");
    return RET_ERROR;
  }
  int batch = static_cast<int>(input->Batch());
  int channel = static_cast<int>(input->Channel());
  MS_ASSERT(batch > 0);
  MS_ASSERT(channel > 0);
  int area = static_cast<int>(input->Width()) * static_cast<int>(input->Height());
  int inputStride = input->GetElementSize() / batch;
  int outputStride = output->GetElementSize() / batch;
  DataType dt = input->GetDataType();

  MS_ASSERT(input->GetData());
  MS_ASSERT(output->GetData());

  if (dt == DataType_DT_FLOAT) {
    for (int i = 0; i < batch; ++i) {
      auto ret = MSPackC4(reinterpret_cast<float *>(output->GetData()) + outputStride * i,
                          (const float *)input->GetData() + inputStride * i, area, channel);
      if (ret != RET_OK) {
        MS_LOGE("MSPackC4 failed: %d", ret);
        return RET_ERROR;
      }
    }
  } else if (dt == DataType_DT_UINT8) {
    for (int i = 0; i < batch; ++i) {
      MSPackC4Uint8(reinterpret_cast<uint8_t *>(output->GetData()) + outputStride * i,
                    (const uint8_t *)input->GetData() + inputStride * i, area, channel);
    }
  } else {
    MS_LOGE("Unsupported dataType: %d", dt);
    return RET_ERROR;
  }
  return RET_OK;
}

int Nc4hw4ToNchw(const Tensor *input, Tensor *output) {
  if (input == nullptr || output == nullptr) {
    MS_LOGE("input tensor or output tensor is nullptr");
    return RET_ERROR;
  }

  int batch = static_cast<int>(input->Batch());
  int channel = static_cast<int>(input->Channel());
  MS_ASSERT(batch > 0);
  MS_ASSERT(channel > 0);
  int area = static_cast<int>(input->Width()) * static_cast<int>(input->Height());
  int inputStride = input->GetElementSize() / batch;
  int outputStride = output->GetElementSize() / batch;
  DataType dt = input->GetDataType();
  if (dt == DataType_DT_FLOAT) {
    for (int i = 0; i < batch; ++i) {
      MSUnpackC4(reinterpret_cast<float *>(output->GetData()) + outputStride * i,
                 (const float *)input->GetData() + inputStride * i, area, channel);
    }
  } else if (dt == DataType_DT_UINT8) {
    for (int i = 0; i < batch; ++i) {
      MSUnpackC4Uint8(reinterpret_cast<uint8_t *>(output->GetData()) + outputStride * i,
                      (const uint8_t *)input->GetData() + inputStride * i, area, channel);
    }
  } else {
    MS_LOGE("Unsupported dataType: %d", dt);
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace predict
}  // namespace mindspore
