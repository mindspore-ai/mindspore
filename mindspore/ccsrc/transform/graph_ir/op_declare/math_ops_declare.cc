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
ATTR_MAP(NLLLoss) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(NLLLoss) = {{4, ATTR_DESC(reduction, AnyTraits<GEReduction>())},
                           {5, ATTR_DESC(ignore_index, AnyTraits<int64_t>())}};
OUTPUT_MAP(NLLLoss) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(total_weight)}};
REG_ADPT_DESC(NLLLoss, kNameNLLLoss, ADPT_DESC(NLLLoss))

// NLLLossGrad
INPUT_MAP(NLLLossGrad) = {{1, INPUT_DESC(x)},
                          {2, INPUT_DESC(y_grad)},
                          {3, INPUT_DESC(target)},
                          {4, INPUT_DESC(weight)},
                          {5, INPUT_DESC(total_weight)}};
ATTR_MAP(NLLLossGrad) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(NLLLossGrad) = {{6, ATTR_DESC(reduction, AnyTraits<GEReduction>())},
                               {7, ATTR_DESC(ignore_index, AnyTraits<int64_t>())}};
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

// PdistGrad
CUST_INPUT_MAP(PdistGrad) = {{1, INPUT_DESC(y_grad)}, {2, INPUT_DESC(x)}, {3, INPUT_DESC(pdist)}};
CUST_ATTR_MAP(PdistGrad) = {{"p", ATTR_DESC(p, AnyTraits<float>())}};
CUST_OUTPUT_MAP(PdistGrad) = {{0, OUTPUT_DESC(x_grad)}};
REG_ADPT_DESC(PdistGrad, prim::kPrimPdistGrad->name(), CUST_ADPT_DESC(PdistGrad));

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
REG_ADPT_DESC(DynamicGetNextAscend, prim::kPrimDynamicGetNextAscend->name(), ADPT_DESC(GetNext))

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
CUST_INPUT_ATTR_MAP(CholeskyInverse) = {{2, ATTR_DESC(upper, AnyTraits<bool>())}};
CUST_ATTR_MAP(CholeskyInverse) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(CholeskyInverse) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CholeskyInverse, prim::kPrimCholeskyInverse->name(), CUST_ADPT_DESC(CholeskyInverse));

// Eig
CUST_INPUT_MAP(Eig) = {{1, INPUT_DESC(x)}};
CUST_INPUT_ATTR_MAP(Eig) = {{2, ATTR_DESC(compute_v, AnyTraits<bool>())}};
CUST_ATTR_MAP(Eig) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Eig) = {{0, OUTPUT_DESC(eigen_values)}, {1, OUTPUT_DESC(eigen_vectors)}};
REG_ADPT_DESC(Eig, prim::kPrimEig->name(), CUST_ADPT_DESC(Eig));

// Eps
CUST_INPUT_MAP(Eps) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(Eps) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Eps) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Eps, prim::kPrimEps->name(), CUST_ADPT_DESC(Eps));

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
ATTR_MAP(Imag) = {{"Tout", ATTR_DESC(Tout, AnyTraits<GEType>())}};
OUTPUT_MAP(Imag) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Imag, prim::kPrimImag->name(), ADPT_DESC(Imag));

// Lgamma
CUST_INPUT_MAP(Lgamma) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(Lgamma) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Lgamma) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Lgamma, prim::kPrimLgamma->name(), CUST_ADPT_DESC(Lgamma));

// Real
CUST_INPUT_MAP(Real) = {{1, INPUT_DESC(input)}};
CUST_ATTR_MAP(Real) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Real) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Real, prim::kPrimReal->name(), CUST_ADPT_DESC(Real));

// Diagonal
CUST_INPUT_MAP(Diagonal) = {{1, INPUT_DESC(x)}};
CUST_INPUT_ATTR_MAP(Diagonal) = {{2, ATTR_DESC(offset, AnyTraits<int64_t>())},
                                 {3, ATTR_DESC(dim1, AnyTraits<int64_t>())},
                                 {4, ATTR_DESC(dim2, AnyTraits<int64_t>())}};
CUST_ATTR_MAP(Diagonal) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Diagonal) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Diagonal, prim::kPrimDiagonal->name(), CUST_ADPT_DESC(Diagonal));

// FFT
CUST_INPUT_MAP(FFT) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(n)}, {3, INPUT_DESC(dim)}, {4, INPUT_DESC(norm)}};
CUST_ATTR_MAP(FFT) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(FFT) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FFT, prim::kPrimFFT->name(), CUST_ADPT_DESC(FFT));

// FFT2
CUST_INPUT_MAP(FFT2) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(s)}, {3, INPUT_DESC(dim)}, {4, INPUT_DESC(norm)}};
CUST_ATTR_MAP(FFT2) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(FFT2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FFT2, prim::kPrimFFT2->name(), CUST_ADPT_DESC(FFT2));

// FFTN
CUST_INPUT_MAP(FFTN) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(s)}, {3, INPUT_DESC(dim)}, {4, INPUT_DESC(norm)}};
CUST_ATTR_MAP(FFTN) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(FFTN) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FFTN, prim::kPrimFFTN->name(), CUST_ADPT_DESC(FFTN));

// IFFT
CUST_INPUT_MAP(IFFT) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(n)}, {3, INPUT_DESC(dim)}, {4, INPUT_DESC(norm)}};
CUST_ATTR_MAP(IFFT) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(IFFT) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IFFT, prim::kPrimIFFT->name(), CUST_ADPT_DESC(IFFT));

// IFFT2
CUST_INPUT_MAP(IFFT2) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(s)}, {3, INPUT_DESC(dim)}, {4, INPUT_DESC(norm)}};
CUST_ATTR_MAP(IFFT2) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(IFFT2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IFFT2, prim::kPrimIFFT2->name(), CUST_ADPT_DESC(IFFT2));

// IFFTN
CUST_INPUT_MAP(IFFTN) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(s)}, {3, INPUT_DESC(dim)}, {4, INPUT_DESC(norm)}};
CUST_ATTR_MAP(IFFTN) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(IFFTN) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IFFTN, prim::kPrimIFFTN->name(), CUST_ADPT_DESC(IFFTN));

// FFTShapeCopy
CUST_INPUT_MAP(FFTShapeCopy) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(shape)}};
CUST_ATTR_MAP(FFTShapeCopy) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(FFTShapeCopy) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FFTShapeCopy, prim::kPrimFFTShapeCopy->name(), CUST_ADPT_DESC(FFTShapeCopy));

// FFTShift
CUST_INPUT_MAP(FFTShift) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(dim)}};
CUST_ATTR_MAP(FFTShift) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(FFTShift) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FFTShift, prim::kPrimFFTShift->name(), CUST_ADPT_DESC(FFTShift));

// IFFTShift
CUST_INPUT_MAP(IFFTShift) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(dim)}};
CUST_ATTR_MAP(IFFTShift) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(IFFTShift) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IFFTShift, prim::kPrimIFFTShift->name(), CUST_ADPT_DESC(IFFTShift));

// RFFT
CUST_INPUT_MAP(RFFT) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(n)}, {3, INPUT_DESC(dim)}, {4, INPUT_DESC(norm)}};
CUST_ATTR_MAP(RFFT) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(RFFT) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RFFT, prim::kPrimRFFT->name(), CUST_ADPT_DESC(RFFT));

// IRFFT
CUST_INPUT_MAP(IRFFT) = {{1, INPUT_DESC(input)}, {2, INPUT_DESC(n)}, {3, INPUT_DESC(dim)}, {4, INPUT_DESC(norm)}};
CUST_ATTR_MAP(IRFFT) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(IRFFT) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IRFFT, prim::kPrimIRFFT->name(), CUST_ADPT_DESC(IRFFT));

// IRFFTGrad
CUST_INPUT_MAP(IRFFTGrad) = {
  {1, INPUT_DESC(input1)}, {2, INPUT_DESC(input2)}, {3, INPUT_DESC(n)}, {4, INPUT_DESC(dim)}, {5, INPUT_DESC(norm)}};
CUST_ATTR_MAP(IRFFTGrad) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(IRFFTGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(IRFFTGrad, prim::kPrimIRFFTGrad->name(), CUST_ADPT_DESC(IRFFTGrad));

// RFFTGrad
CUST_INPUT_MAP(RFFTGrad) = {
  {1, INPUT_DESC(input1)}, {2, INPUT_DESC(input2)}, {3, INPUT_DESC(n)}, {4, INPUT_DESC(dim)}, {5, INPUT_DESC(norm)}};
CUST_ATTR_MAP(RFFTGrad) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(RFFTGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RFFTGrad, prim::kPrimRFFTGrad->name(), CUST_ADPT_DESC(RFFTGrad));

std::vector<std::string> mode_strings = {"pad", "same", "valid", "full"};
// Correlate
CUST_INPUT_MAP(Correlate) = {{1, INPUT_DESC(a)}, {2, INPUT_DESC(v)}};
CUST_ATTR_MAP(Correlate) = EMPTY_ATTR_MAP;
CUST_INPUT_ATTR_MAP(Correlate) = {{3, ATTR_DESC(mode, AnyTraits<GEEnumToStr>(), mode_strings)}};
CUST_OUTPUT_MAP(Correlate) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Correlate, prim::kPrimCorrelate->name(), CUST_ADPT_DESC(Correlate));

// DCT
CUST_INPUT_MAP(DCT) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(DCT) = EMPTY_ATTR_MAP;
CUST_INPUT_ATTR_MAP(DCT) = {{2, ATTR_DESC(type, AnyTraits<int64_t>())}, {3, ATTR_DESC(n, AnyTraits<int64_t>())},
                            {4, ATTR_DESC(axis, AnyTraits<int64_t>())}, {5, ATTR_DESC(norm, AnyTraits<int64_t>())},
                            {6, ATTR_DESC(forward, AnyTraits<bool>())}, {7, ATTR_DESC(grad, AnyTraits<bool>())}};
CUST_OUTPUT_MAP(DCT) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DCT, prim::kPrimDCT->name(), CUST_ADPT_DESC(DCT));

// Polar
CUST_INPUT_MAP(Polar) = {{1, INPUT_DESC(abs)}, {2, INPUT_DESC(angle)}};
CUST_ATTR_MAP(Polar) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Polar) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Polar, prim::kPrimPolar->name(), CUST_ADPT_DESC(Polar));

// TriuIndices
CUST_INPUT_MAP(TriuIndices) = EMPTY_INPUT_MAP;
CUST_ATTR_MAP(TriuIndices) = {{"row", ATTR_DESC(row, AnyTraits<int64_t>())},
                              {"col", ATTR_DESC(col, AnyTraits<int64_t>())},
                              {"offset", ATTR_DESC(offset, AnyTraits<int64_t>())},
                              {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};
CUST_OUTPUT_MAP(TriuIndices) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(TriuIndices, prim::kPrimTriuIndices->name(), CUST_ADPT_DESC(TriuIndices));

// Digamma
INPUT_MAP(Digamma) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Digamma) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Digamma) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Digamma, prim::kPrimDigamma->name(), ADPT_DESC(Digamma));

// TrilIndices
CUST_INPUT_MAP(TrilIndices) = EMPTY_INPUT_MAP;
CUST_ATTR_MAP(TrilIndices) = {{"row", ATTR_DESC(row, AnyTraits<int64_t>())},
                              {"col", ATTR_DESC(col, AnyTraits<int64_t>())},
                              {"offset", ATTR_DESC(offset, AnyTraits<int64_t>())},
                              {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};
CUST_OUTPUT_MAP(TrilIndices) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(TrilIndices, prim::kPrimTrilIndices->name(), CUST_ADPT_DESC(TrilIndices));

// Angle
INPUT_MAP(Angle) = {{1, INPUT_DESC(input)}};
ATTR_MAP(Angle) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Angle) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Angle, prim::kPrimAngle->name(), ADPT_DESC(Angle));

// Polygamma
CUST_INPUT_MAP(Polygamma) = {{1, INPUT_DESC(a)}, {2, INPUT_DESC(x)}};
CUST_ATTR_MAP(Polygamma) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(Polygamma) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Polygamma, prim::kPrimPolygamma->name(), CUST_ADPT_DESC(Polygamma));

// Igammac
INPUT_MAP(Igammac) = {{1, INPUT_DESC(a)}, {2, INPUT_DESC(x)}};
ATTR_MAP(Igammac) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Igammac) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(Igammac, prim::kPrimIgammac->name(), ADPT_DESC(Igammac));

// FFTWithSize
static const std::vector<std::string> norm_strings = {"backward", "forward", "ortho"};
CUST_INPUT_MAP(FFTWithSize) = {{1, INPUT_DESC(x)}};
CUST_INPUT_ATTR_MAP(FFTWithSize) = {
  {2, ATTR_DESC(signal_ndim, AnyTraits<int64_t>())}, {3, ATTR_DESC(inverse, AnyTraits<bool>())},
  {4, ATTR_DESC(real, AnyTraits<bool>())},           {5, ATTR_DESC(norm, AnyTraits<GEEnumToStr>(), norm_strings)},
  {6, ATTR_DESC(onesided, AnyTraits<bool>())},       {7, ATTR_DESC(signal_sizes, AnyTraits<std::vector<int64_t>>())}};
CUST_ATTR_MAP(FFTWithSize) = EMPTY_ATTR_MAP;
CUST_OUTPUT_MAP(FFTWithSize) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(FFTWithSize, kNameFFTWithSize, CUST_ADPT_DESC(FFTWithSize));

// IgammaGradA
INPUT_MAP(IgammaGradA) = {{1, INPUT_DESC(a)}, {2, INPUT_DESC(x)}};
ATTR_MAP(IgammaGradA) = EMPTY_ATTR_MAP;
OUTPUT_MAP(IgammaGradA) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(IgammaGradA, prim::kPrimIgammaGradA->name(), ADPT_DESC(IgammaGradA));

// Zeta
INPUT_MAP(Zeta) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(q)}};
ATTR_MAP(Zeta) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Zeta) = {{0, OUTPUT_DESC(z)}};
REG_ADPT_DESC(Zeta, prim::kPrimZeta->name(), ADPT_DESC(Zeta));

// Cross
INPUT_MAP(Cross) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Cross) = {{"dim", ATTR_DESC(dim, AnyTraits<int64_t>())}};
OUTPUT_MAP(Cross) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Cross, prim::kPrimCross->name(), ADPT_DESC(Cross))

// Logit
CUST_INPUT_MAP(Logit) = {{1, INPUT_DESC(x)}};
CUST_ATTR_MAP(Logit) = {{"eps", ATTR_DESC(eps, AnyTraits<float>())}};
CUST_OUTPUT_MAP(Logit) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(Logit, prim::kPrimLogit->name(), CUST_ADPT_DESC(Logit))
}  // namespace mindspore::transform
