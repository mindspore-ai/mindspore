/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "ms_extension/api.h"

using RopeParam = atb::infer::RopeParam;

namespace atb {
template <>
struct HashOpParam<RopeParam> {
  void operator()(const RopeParam &param) const { add_param_to_buf("rotaryCoeff", param.rotaryCoeff); }
};
}  // namespace atb

static ms::Tensor sequenceLength;
static int64_t previousTokenCount = -1;

// Global caches for cos and sin values
static ms::Tensor cosCacheNeox;
static ms::Tensor sinCacheNeox;
static ms::Tensor cosCache;
static ms::Tensor sinCache;

void InitializeCosSinCache(const ms::Tensor &cos_sin_cache) {
  // Split the input cache tensor into two halves
  auto cosSinChunks = cos_sin_cache.chunk(2, -1);

  // Prepare cosine caches with different repeat strategies
  cosCache = cosSinChunks[0].repeat_interleave(2, 1);
  sinCache = cosSinChunks[1].repeat_interleave(2, 1);
  cosCacheNeox = cosSinChunks[0].repeat({1, 2});
  sinCacheNeox = cosSinChunks[1].repeat({1, 2});
}

void npu_rope(const ms::Tensor &positions, ms::Tensor query, ms::Tensor key, int64_t head_size,
              const ms::Tensor &cos_sin_cache, bool is_neox_style) {
  if (!cosCache.is_defined() || !sinCache.is_defined()) {
    InitializeCosSinCache(cos_sin_cache);
  }

  ms::Tensor flatPositions = positions.flatten();
  int32_t currentTokenCount = flatPositions.shape()[0];
  ms::Tensor cos =
    is_neox_style ? cosCacheNeox.index_select(0, flatPositions) : cosCache.index_select(0, flatPositions);
  ms::Tensor sin =
    is_neox_style ? sinCacheNeox.index_select(0, flatPositions) : sinCache.index_select(0, flatPositions);

  if (!sequenceLength.is_defined() || currentTokenCount != previousTokenCount) {
    previousTokenCount = currentTokenCount;
    sequenceLength = ms::tensor(std::vector<int64_t>{1}, ms::TypeId::kNumberTypeInt32);
  }

  RopeParam param;
  param.rotaryCoeff = is_neox_style ? 2 : head_size;
  std::vector<ms::Tensor> inputs = {query, key, cos, sin, sequenceLength};
  std::vector<ms::Tensor> outputs = {query, key};

  ms::pynative::RunAtbOp("Rope", param, inputs, outputs);
}

auto pyboost_npu_rope(const ms::Tensor &positions, ms::Tensor query, ms::Tensor key, int64_t head_size,
                      const ms::Tensor &cos_sin_cache, bool is_neox_style) {
  return ms::pynative::PyboostRunner::Call<0>(npu_rope, positions, query, key, head_size, cos_sin_cache, is_neox_style);
}

void rope_native_atb(ms::Tensor query, ms::Tensor key, ms::Tensor cos, ms::Tensor sin, ms::Tensor seqLen,
                     int32_t rotaryCoeff) {
  RopeParam param;
  param.rotaryCoeff = rotaryCoeff;
  std::vector<ms::Tensor> inputs = {query, key, cos, sin, sequenceLength};
  std::vector<ms::Tensor> outputs = {query, key};
  ms::pynative::RunAtbOp("Rope", param, inputs, outputs);
}

auto pyboost_rope_native_atb(ms::Tensor query, ms::Tensor key, ms::Tensor cos, ms::Tensor sin, ms::Tensor seqLen,
                             int32_t rotaryCoeff) {
  return ms::pynative::PyboostRunner::Call<0>(rope_native_atb, query, key, cos, sin, seqLen, rotaryCoeff);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_rope", &pyboost_npu_rope, "Rope", pybind11::arg("positions"), pybind11::arg("query"), pybind11::arg("key"),
        pybind11::arg("head_size"), pybind11::arg("cos_sin_cache"), pybind11::arg("is_neox_style"));
  m.def("rope_native_atb", &pyboost_rope_native_atb, "Rope without preprocess", pybind11::arg("query"),
        pybind11::arg("key"), pybind11::arg("cos"), pybind11::arg("sin"), pybind11::arg("seqLen"),
        pybind11::arg("rotaryCoeff"));
}
