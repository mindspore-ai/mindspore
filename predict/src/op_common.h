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

#ifndef PREDICT_SRC_OP_COMMON_H_
#define PREDICT_SRC_OP_COMMON_H_
#include <assert.h>

namespace mindspore {
namespace predict {
static inline size_t AlignSize(size_t size, size_t align) { return (size + align - 1) & -align; }

template <typename Tsrc, typename Tdst>
inline void Nchw2Nhwc(const Tsrc *in, Tdst *out, size_t h, size_t w, size_t c) {
  MS_ASSERT(in != nullptr && out != nullptr);
  const size_t sz = w * h;

  for (size_t cc = 0; cc < c; ++cc) {
    auto pi = in + sz * cc;

    for (size_t el = 0; el < sz; ++el) {
      out[cc + el * c] = (Tdst)pi[el];
    }
  }
}

template <typename Tsrc, typename Tdst>
inline void Nhwc2Nchw(const Tsrc *in, Tdst *out, size_t h, size_t w, size_t c) {
  MS_ASSERT(in != nullptr && out != nullptr);
  const size_t sz = w * h;

  for (auto cc = 0; cc < c; ++cc) {
    auto po = out + sz * cc;

    for (size_t el = 0; el < sz; ++el) {
      po[el] = (Tdst)in[cc + el * c];
    }
  }
}

template <typename Tsrc, typename Tdst>
inline void InverseQuantization(const Tsrc *srcdata, Tdst *dstdata, size_t datanum, float *parms) {
  MS_ASSERT(srcdata != nullptr && dstdata != nullptr);
  float scale = parms[2];
  float zeroPoint = parms[3];
  for (size_t i = 0; i < datanum; ++i) {
    dstdata = (scale == 0) ? (0) : (Tdst)((srcdata[i] - zeroPoint) * scale);
  }
}

template <typename Tsrc, typename Tdst>
inline void Astype(const Tsrc *srcdata, Tdst *dstdata, size_t datanum) {
  MS_ASSERT(srcdata != nullptr && dstdata != nullptr);
  for (size_t i = 0; i < datanum; ++i) {
    dstdata[i] = (Tdst)srcdata[i];
  }
}
#define MSMIN(x, y) ((x) < (y) ? (x) : (y))
#define MSMAX(x, y) ((x) > (y) ? (x) : (y))

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define DOWN_DIV(x, y) (((x) - (y) + (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#define ALIGN_UP8(x) ROUND_UP((x), 8)

#define MAX_MALLOC_SIZE 100 * 1024 * 1024
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_SRC_OP_COMMON_H_
