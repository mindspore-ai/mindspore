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

#include "transform/graph_ir/op_declare/math_ops_declare.h"
#include <vector>
#include <string>

namespace mindspore::transform {
// ActsULQ
INPUT_MAP(ActsULQ) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(clamp_min)}, {3, INPUT_DESC(clamp_max)}};
ATTR_MAP(ActsULQ) = {{"fixed_min", ATTR_DESC(fixed_min, AnyTraits<bool>())},
                     {"num_bits", ATTR_DESC(num_bits, AnyTraits<int64_t>())}};
OUTPUT_MAP(ActsULQ) = {{0, OUTPUT_DESC(y)},
                       {1, OUTPUT_DESC(clamp_min_mask)},
                       {2, OUTPUT_DESC(clamp_max_mask)},
                       {3, OUTPUT_DESC(x_clamped_loss)}};
REG_ADPT_DESC(ActsULQ, kNameActsULQ, ADPT_DESC(ActsULQ))

// ActsULQInputGrad
INPUT_MAP(ActsULQInputGrad) = {
  {1, INPUT_DESC(y_grad)}, {2, INPUT_DESC(clamp_min_mask)}, {3, INPUT_DESC(clamp_max_mask)}};
ATTR_MAP(ActsULQInputGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ActsULQInputGrad) = {{0, OUTPUT_DESC(x_grad)}};
REG_ADPT_DESC(ActsULQInputGrad, kNameActsULQInputGrad, ADPT_DESC(ActsULQInputGrad))

// ActULQClampMaxGrad
INPUT_MAP(ActULQClampMaxGrad) = {
  {1, INPUT_DESC(y_grad)}, {2, INPUT_DESC(clamp_max_mask)}, {3, INPUT_DESC(x_clamped_loss)}};
ATTR_MAP(ActULQClampMaxGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ActULQClampMaxGrad) = {{0, OUTPUT_DESC(clamp_max_grad)}};
REG_ADPT_DESC(ActULQClampMaxGrad, kNameActULQClampMaxGrad, ADPT_DESC(ActULQClampMaxGrad))

// ActULQClampMinGrad
INPUT_MAP(ActULQClampMinGrad) = {
  {1, INPUT_DESC(y_grad)}, {2, INPUT_DESC(clamp_min_mask)}, {3, INPUT_DESC(x_clamped_loss)}};
ATTR_MAP(ActULQClampMinGrad) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ActULQClampMinGrad) = {{0, OUTPUT_DESC(clamp_min_grad)}};
REG_ADPT_DESC(ActULQClampMinGrad, kNameActULQClampMinGrad, ADPT_DESC(ActULQClampMinGrad))

// HistogramFixedWidthD
INPUT_MAP(HistogramFixedWidthD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(range)}};
ATTR_MAP(HistogramFixedWidthD) = {{"nbins", ATTR_DESC(nbins, AnyTraits<int64_t>())},
                                  {"dtype", ATTR_DESC(dtype, AnyTraits<int64_t>())}};
OUTPUT_MAP(HistogramFixedWidthD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(HistogramFixedWidthD, kNameHistogramFixedWidthD, ADPT_DESC(HistogramFixedWidthD))

// IFMR
INPUT_MAP(IFMR) = {
  {1, INPUT_DESC(data)}, {2, INPUT_DESC(data_min)}, {3, INPUT_DESC(data_max)}, {4, INPUT_DESC(cumsum)}};
ATTR_MAP(IFMR) = {{"min_percentile", ATTR_DESC(min_percentile, AnyTraits<float>())},
                  {"max_percentile", ATTR_DESC(max_percentile, AnyTraits<float>())},
                  {"search_range", ATTR_DESC(search_range, AnyTraits<std::vector<float>>())},
                  {"search_step", ATTR_DESC(search_step, AnyTraits<float>())}};
OUTPUT_MAP(IFMR) = {{0, OUTPUT_DESC(scale)}, {1, OUTPUT_DESC(offset)}};
REG_ADPT_DESC(IFMR, kNameIFMR, ADPT_DESC(IFMR))

// NLLLoss
INPUT_MAP(NLLLoss) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(target)}, {3, INPUT_DESC(weight)}};
ATTR_MAP(NLLLoss) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(NLLLoss) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(total_weight)}};
REG_ADPT_DESC(NLLLoss, kNameNLLLoss, ADPT_DESC(NLLLoss))

// NLLLossGrad
INPUT_MAP(NLLLossGrad) = {{1, INPUT_DESC(x)},
                          {2, INPUT_DESC(y_grad)},
                          {3, INPUT_DESC(target)},
                          {4, INPUT_DESC(weight)},
                          {5, INPUT_DESC(total_weight)}};
ATTR_MAP(NLLLossGrad) = {{"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())}};
OUTPUT_MAP(NLLLossGrad) = {{0, OUTPUT_DESC(x_grad)}};
REG_ADPT_DESC(NLLLossGrad, kNameNLLLossGrad, ADPT_DESC(NLLLossGrad))

// Erf
INPUT_MAP(Erf) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Erf) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Erf) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Erf, kNameErf, ADPT_DESC(Erf))

// Erfc
INPUT_MAP(Erfc) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Erfc) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Erfc) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Erfc, kNameErfc, ADPT_DESC(Erfc))

// WtsARQ
INPUT_MAP(WtsARQ) = {{1, INPUT_DESC(w)}, {2, INPUT_DESC(w_min)}, {3, INPUT_DESC(w_max)}};
ATTR_MAP(WtsARQ) = {{"num_bits", ATTR_DESC(num_bits, AnyTraits<int64_t>())},
                    {"offset_flag", ATTR_DESC(offset_flag, AnyTraits<bool>())}};
OUTPUT_MAP(WtsARQ) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(WtsARQ, kNameWtsARQ, ADPT_DESC(WtsARQ))

// IsFinite
INPUT_MAP(IsFinite) = {{1, INPUT_DESC(x)}};
ATTR_MAP(IsFinite) = EMPTY_ATTR_MAP;
OUTPUT_MAP(IsFinite) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IsFinite, kNameIsFinite, ADPT_DESC(IsFinite))

// IsNan
INPUT_MAP(IsNan) = {{1, INPUT_DESC(x)}};
ATTR_MAP(IsNan) = EMPTY_ATTR_MAP;
OUTPUT_MAP(IsNan) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IsNan, kNameIsNan, ADPT_DESC(IsNan))

// LpNorm
INPUT_MAP(LpNorm) = {{1, INPUT_DESC(x)}};
ATTR_MAP(LpNorm) = {{"p", ATTR_DESC(p, AnyTraits<int64_t>())},
                    {"axis", ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>())},
                    {"keep_dims", ATTR_DESC(keepdim, AnyTraits<bool>())},
                    {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};
OUTPUT_MAP(LpNorm) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LpNorm, prim::kPrimLpNorm->name(), ADPT_DESC(LpNorm))
}  // namespace mindspore::transform
