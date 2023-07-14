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
#include <string>
#include <vector>
#include "ops/math_ops.h"
#include "ops/nn_ops.h"
#include "ops/other_ops.h"
#include "ops/structure_ops.h"

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

// HistogramFixedWidth
INPUT_MAP(HistogramFixedWidth) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(range)}, {3, INPUT_DESC(nbins)}};
ATTR_INPUT_MAP(HistogramFixedWidth) = {{"nbins", "nbins"}};
ATTR_MAP(HistogramFixedWidth) = {{"dtype", ATTR_DESC(dtype, AnyTraits<int64_t>())}};
OUTPUT_MAP(HistogramFixedWidth) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(HistogramFixedWidth, kHistogramFixedWidthOpName, ADPT_DESC(HistogramFixedWidth))
REG_ADPT_DESC(HistogramFixedWidthD, kHistogramFixedWidthDOpName, ADPT_DESC(HistogramFixedWidth))

// IFMR
INPUT_MAP(IFMR) = {
  {1, INPUT_DESC(data)}, {2, INPUT_DESC(data_min)}, {3, INPUT_DESC(data_max)}, {4, INPUT_DESC(cumsum)}};
ATTR_MAP(IFMR) = {{"min_percentile", ATTR_DESC(min_percentile, AnyTraits<float>())},
                  {"max_percentile", ATTR_DESC(max_percentile, AnyTraits<float>())},
                  {"search_range", ATTR_DESC(search_range, AnyTraits<std::vector<float>>())},
                  {"search_step", ATTR_DESC(search_step, AnyTraits<float>())},
                  {"with_offset", ATTR_DESC(with_offset, AnyTraits<bool>())}};
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

// IsInf
INPUT_MAP(IsInf) = {{1, INPUT_DESC(x)}};
ATTR_MAP(IsInf) = EMPTY_ATTR_MAP;
OUTPUT_MAP(IsInf) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IsInf, prim::kPrimIsInf->name(), ADPT_DESC(IsInf))

// LpNorm
INPUT_MAP(LpNorm) = {{1, INPUT_DESC(x)}};
ATTR_MAP(LpNorm) = {{"p", ATTR_DESC(p, AnyTraits<int64_t>())},
                    {"axis", ATTR_DESC(axes, AnyTraits<std::vector<int64_t>>())},
                    {"keep_dims", ATTR_DESC(keepdim, AnyTraits<bool>())},
                    {"epsilon", ATTR_DESC(epsilon, AnyTraits<float>())}};
OUTPUT_MAP(LpNorm) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LpNorm, prim::kPrimLpNorm->name(), ADPT_DESC(LpNorm))

// Trunc
INPUT_MAP(Trunc) = {{1, INPUT_DESC(input_x)}};
ATTR_MAP(Trunc) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Trunc) = {{0, OUTPUT_DESC(output_y)}};
REG_ADPT_DESC(Trunc, prim::kPrimTrunc->name(), ADPT_DESC(Trunc))

// Histogram
INPUT_MAP(Histogram) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Histogram) = {{"min", ATTR_DESC(min, AnyTraits<float>())},
                       {"max", ATTR_DESC(max, AnyTraits<float>())},
                       {"bins", ATTR_DESC(bins, AnyTraits<int64_t>())}};
OUTPUT_MAP(Histogram) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Histogram, kHistogramOpName, ADPT_DESC(Histogram))

// Pdist
INPUT_MAP(Pdist) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Pdist) = {
  {"p", ATTR_DESC(p, AnyTraits<float>())},
};
OUTPUT_MAP(Pdist) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Pdist, prim::kPrimPdist->name(), ADPT_DESC(Pdist))

// SoftMarginLossGrad
INPUT_MAP(SoftMarginLossGrad) = {{1, INPUT_DESC(predict)}, {2, INPUT_DESC(label)}, {3, INPUT_DESC(dout)}};
ATTR_MAP(SoftMarginLossGrad) = {
  {"reduction", ATTR_DESC(reduction, AnyTraits<std::string>())},
};
OUTPUT_MAP(SoftMarginLossGrad) = {{0, OUTPUT_DESC(gradient)}};
REG_ADPT_DESC(SoftMarginLossGrad, prim::kPrimSoftMarginLossGrad->name(), ADPT_DESC(SoftMarginLossGrad))

// Cdist
INPUT_MAP(Cdist) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Cdist) = {
  {"p", ATTR_DESC(p, AnyTraits<float>())},
};
OUTPUT_MAP(Cdist) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Cdist, prim::kPrimCdist->name(), ADPT_DESC(Cdist))

// CdistGrad
INPUT_MAP(CdistGrad) = {{1, INPUT_DESC(grad)}, {2, INPUT_DESC(x1)}, {3, INPUT_DESC(x2)}, {4, INPUT_DESC(cdist)}};
ATTR_MAP(CdistGrad) = {
  {"p", ATTR_DESC(p, AnyTraits<float>())},
};
OUTPUT_MAP(CdistGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CdistGrad, prim::kPrimCdistGrad->name(), ADPT_DESC(CdistGrad))

// Conj
INPUT_MAP(Conj) = {{1, INPUT_DESC(input)}};
ATTR_MAP(Conj) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Conj) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Conj, prim::kPrimConj->name(), ADPT_DESC(Conj))

// NextAfter
INPUT_MAP(NextAfter) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(NextAfter) = EMPTY_ATTR_MAP;
OUTPUT_MAP(NextAfter) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(NextAfter, prim::kPrimNextAfter->name(), ADPT_DESC(NextAfter))

// InitDataSetQueue
INPUT_MAP(InitData) = EMPTY_INPUT_MAP;
OUTPUT_MAP(InitData) = EMPTY_OUTPUT_MAP;
ATTR_MAP(InitData) = {{"queue_name", ATTR_DESC(channel_name, AnyTraits<string>())}};
REG_ADPT_DESC(InitData, prim::kPrimInitDataSetQueue->name(), ADPT_DESC(InitData))

// GetNext
INPUT_MAP(GetNext) = EMPTY_INPUT_MAP;
DYN_OUTPUT_MAP(GetNext) = {{0, DYN_OUTPUT_DESC(y)}};
ATTR_MAP(GetNext) = {{"types", ATTR_DESC(output_types, AnyTraits<std::vector<GEType>>())},
                     {"shapes", ATTR_DESC(output_shapes, AnyTraits<std::vector<std::vector<int64_t>>>())},
                     {"output_num", ATTR_DESC(output_num, AnyTraits<int64_t>())},
                     {"shared_name", ATTR_DESC(channel_name, AnyTraits<string>())}};
REG_ADPT_DESC(GetNext, prim::kPrimGetNext->name(), ADPT_DESC(GetNext))

INPUT_MAP(STFT) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(window)}};
OUTPUT_MAP(STFT) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(STFT) = {{"hop_length", ATTR_DESC(hop_length, AnyTraits<int64_t>())},
                  {"win_length", ATTR_DESC(win_length, AnyTraits<int64_t>())},
                  {"normalized", ATTR_DESC(normalized, AnyTraits<bool>())},
                  {"onesided", ATTR_DESC(onesided, AnyTraits<bool>())},
                  {"return_complex", ATTR_DESC(return_complex, AnyTraits<bool>())},
                  {"n_fft", ATTR_DESC(n_fft, AnyTraits<int64_t>())}};
REG_ADPT_DESC(STFT, prim::kPrimSTFT->name(), ADPT_DESC(STFT))

// Complex
INPUT_MAP(Complex) = {{1, INPUT_DESC(real)}, {2, INPUT_DESC(imag)}};
ATTR_MAP(Complex) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Complex) = {{0, OUTPUT_DESC(out)}};
REG_ADPT_DESC(Complex, prim::kPrimComplex->name(), ADPT_DESC(Complex));

// Betainc
INPUT_MAP(Betainc) = {{1, INPUT_DESC(a)}, {2, INPUT_DESC(b)}, {3, INPUT_DESC(x)}};
ATTR_MAP(Betainc) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Betainc) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(Betainc, prim::kPrimBetainc->name(), ADPT_DESC(Betainc));

// CholeskySolve
CUST_INPUT_MAP(CholeskySolve) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
CUST_ATTR_MAP(CholeskySolve) = {{"upper", ATTR_DESC(upper, AnyTraits<bool>())}};
CUST_OUTPUT_MAP(CholeskySolve) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CholeskySolve, prim::kPrimCholeskySolve->name(), CUST_ADPT_DESC(CholeskySolve));

// ComplexAbs
INPUT_MAP(ComplexAbs) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ComplexAbs) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ComplexAbs) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ComplexAbs, prim::kPrimComplexAbs->name(), ADPT_DESC(ComplexAbs));

// Bucketize
INPUT_MAP(Bucketize) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Bucketize) = {{"boundaries", ATTR_DESC(boundaries, AnyTraits<std::vector<float>>())}};
OUTPUT_MAP(Bucketize) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Bucketize, prim::kPrimBucketize->name(), ADPT_DESC(Bucketize));

// Cauchy
CUST_INPUT_MAP(Cauchy) = EMPTY_INPUT_MAP;
CUST_ATTR_MAP(Cauchy) = {{"size", ATTR_DESC(size, AnyTraits<std::vector<int64_t>>())},
                         {"sigma", ATTR_DESC(sigma, AnyTraits<float>())},
                         {"median", ATTR_DESC(median, AnyTraits<float>())}};
CUST_OUTPUT_MAP(Cauchy) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Cauchy, prim::kPrimCauchy->name(), CUST_ADPT_DESC(Cauchy));

// Bincount
INPUT_MAP(Bincount) = {{1, INPUT_DESC(array)}, {2, INPUT_DESC(size)}, {3, INPUT_DESC(weights)}};
ATTR_MAP(Bincount) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Bincount) = {{0, OUTPUT_DESC(bins)}};
REG_ADPT_DESC(Bincount, kNameBincount, ADPT_DESC(Bincount));

// CholeskyInverse
CUST_INPUT_MAP(CholeskyInverse) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(CholeskyInverse) = {{"upper", ATTR_DESC(upper, AnyTraits<bool>())}};
CUST_OUTPUT_MAP(CholeskyInverse) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CholeskyInverse, prim::kPrimCholeskyInverse->name(), CUST_ADPT_DESC(CholeskyInverse));

// Eig
CUST_INPUT_MAP(Eig) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(Eig) = {{"compute_v", ATTR_DESC(compute_v, AnyTraits<bool>())}};
CUST_OUTPUT_MAP(Eig) = {{0, OUTPUT_DESC(eigen_values)}, {1, OUTPUT_DESC(eigen_vectors)}};
REG_ADPT_DESC(Eig, prim::kPrimEig->name(), CUST_ADPT_DESC(Eig));

// Hypot
CUST_INPUT_MAP(Hypot) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
CUST_ATTR_MAP(Hypot) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Hypot) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Hypot, prim::kPrimHypot->name(), CUST_ADPT_DESC(Hypot));

// MatrixLogarithm
CUST_INPUT_MAP(MatrixLogarithm) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(MatrixLogarithm) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(MatrixLogarithm) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixLogarithm, prim::kPrimMatrixLogarithm->name(), CUST_ADPT_DESC(MatrixLogarithm));

// Lcm
CUST_INPUT_MAP(Lcm) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
CUST_ATTR_MAP(Lcm) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Lcm) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Lcm, prim::kPrimLcm->name(), CUST_ADPT_DESC(Lcm));

// MatrixExp
CUST_INPUT_MAP(MatrixExp) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(MatrixExp) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(MatrixExp) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixExp, prim::kPrimMatrixExp->name(), CUST_ADPT_DESC(MatrixExp));

// Heaviside
CUST_INPUT_MAP(Heaviside) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(values)}};
CUST_ATTR_MAP(Heaviside) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Heaviside) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Heaviside, prim::kPrimHeaviside->name(), CUST_ADPT_DESC(Heaviside));

// Gcd
CUST_INPUT_MAP(Gcd) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
CUST_ATTR_MAP(Gcd) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Gcd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Gcd, prim::kPrimGcd->name(), CUST_ADPT_DESC(Gcd));

// Orgqr
CUST_INPUT_MAP(Orgqr) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(tau)}};
CUST_ATTR_MAP(Orgqr) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Orgqr) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Orgqr, prim::kPrimOrgqr->name(), CUST_ADPT_DESC(Orgqr));

// RaggedRange
INPUT_MAP(RaggedRange) = {{1, INPUT_DESC(starts)}, {2, INPUT_DESC(limits)}, {3, INPUT_DESC(deltas)}};
ATTR_MAP(RaggedRange) = {{"Tsplits", ATTR_DESC(Tsplits, AnyTraits<GEType>())}};
OUTPUT_MAP(RaggedRange) = {{0, OUTPUT_DESC(rt_nested_splits)}, {1, OUTPUT_DESC(rt_dense_values)}};
REG_ADPT_DESC(RaggedRange, prim::kPrimRaggedRange->name(), ADPT_DESC(RaggedRange));

// Imag
INPUT_MAP(Imag) = {{1, INPUT_DESC(input)}};
ATTR_MAP(Imag) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Imag) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Imag, prim::kPrimImag->name(), ADPT_DESC(Imag));

// Lgamma
CUST_INPUT_MAP(Lgamma) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(Lgamma) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Lgamma) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Lgamma, prim::kPrimLgamma->name(), CUST_ADPT_DESC(Lgamma));

// Real
INPUT_MAP(Real) = {{1, INPUT_DESC(input)}};
ATTR_MAP(Real) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Real) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Real, prim::kPrimReal->name(), ADPT_DESC(Real));
}  // namespace mindspore::transform
