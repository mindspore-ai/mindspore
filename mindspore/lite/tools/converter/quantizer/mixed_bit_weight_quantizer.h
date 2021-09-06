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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_MIXED_BIT_WEIGHT_QUANTIZER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_MIXED_BIT_WEIGHT_QUANTIZER_H
#include <cstdint>
#include <vector>
#include <cmath>
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore::lite::quant {
typedef struct {
  int status;
  float scale;
} BinarySearchResult;

typedef struct {
  float min;
  float max;
} MinMax;

class MixedBitWeightQuantizer {
 public:
  explicit MixedBitWeightQuantizer(float target_relative_err = 0.01, float target_search_tolerance = 0.01,
                                   int max_search_iters = 100)
      : target_relative_err_(target_relative_err),
        target_search_tolerance_(target_search_tolerance),
        max_search_iters_(max_search_iters) {}
  ~MixedBitWeightQuantizer() = default;

  int DoQuantization(float *weights, std::vector<int64_t> shape, int preferred_dim,
                     std::vector<schema::QuantParamT> *quant_params, std::vector<int16_t> *quant_datas);

 private:
  float MeasureQuantizationError(float *weights, const int *shape, int dims, int preferred_dim, float scale);

  MinMax GetMinMax(const float *arr, int arrc);

  int QuantizeByScale(const float *weights, int weightsc, float scale, schema::QuantParamT *quant_params,
                      std::vector<int16_t> *quant_datas);

  BinarySearchResult BinarySearchForQuantizationScale(float *weights, int *shape, int dims, int preferred_dim,
                                                      int max_iters, float target_err, float rel_tol);

 private:
  float var_corr_{1};
  float mean_corr_{0};
  float target_relative_err_;
  float target_search_tolerance_;
  int max_search_iters_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_MIXED_BIT_WEIGHT_QUANTIZER_H
