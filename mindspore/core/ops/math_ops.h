/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_MATH_OPS_H_
#define MINDSPORE_CORE_BASE_MATH_OPS_H_

#include <memory>
#include "ir/anf.h"
#include "mindspore/core/ir/primitive.h"

namespace mindspore {
namespace prim {
// Math
constexpr auto kAbs = "Abs";
constexpr auto kReduceStd = "ReduceStd";
constexpr auto kLog = "Log";
constexpr auto kLogit = "Logit";
constexpr auto kLogitGrad = "LogitGrad";
constexpr auto kAdd = "Add";
constexpr auto kAddV2 = "AddV2";
constexpr auto kAddcdiv = "Addcdiv";
constexpr auto kAddcmul = "Addcmul";
constexpr auto kSub = "Sub";
constexpr auto kMedian = "Median";
constexpr auto kMedianGrad = "MedianGrad";
constexpr auto kMul = "Mul";
constexpr auto kMulNoNan = "MulNoNan";
constexpr auto kACos = "ACos";
constexpr auto kACosGrad = "ACosGrad";
constexpr auto kRealDiv = "RealDiv";
constexpr auto kDivNoNan = "DivNoNan";
constexpr auto kCauchy = "Cauchy";
constexpr auto kCross = "Cross";
constexpr auto kDiagonal = "Diagonal";
constexpr auto kEditDistance = "EditDistance";
constexpr auto kNextAfter = "NextAfter";
constexpr auto kMaximumGradGrad = "MaximumGradGrad";
constexpr auto kMatrixSolveLs = "MatrixSolveLs";
constexpr auto kTridiagonalMatMul = "TridiagonalMatMul";
constexpr auto kTridiagonalSolve = "TridiagonalSolve";
constexpr auto kFFTWithSize = "FFTWithSize";
constexpr auto kTriuIndices = "TriuIndices";
constexpr auto kTrilIndices = "TrilIndices";
constexpr auto kFmax = "Fmax";
constexpr auto kTrace = "Trace";
constexpr auto kTraceGrad = "TraceGrad";
constexpr auto kMatrixLogarithm = "MatrixLogarithm";
constexpr auto kMatrixTriangularSolve = "MatrixTriangularSolve";
constexpr auto kSelfAdjointEig = "SelfAdjointEig";
constexpr auto kBernoulli = "Bernoulli";
constexpr auto kNeg = "Neg";
constexpr auto kSinc = "Sinc";
constexpr auto kCos = "Cos";
constexpr auto kSquare = "Square";
constexpr auto kCumulativeLogsumexp = "CumulativeLogsumexp";
constexpr auto kLpNorm = "LpNorm";
constexpr auto kReciprocal = "Reciprocal";
constexpr auto kInv = "Inv";
constexpr auto kExp = "Exp";
constexpr auto kAsin = "Asin";
constexpr auto kAsinGrad = "AsinGrad";
constexpr auto kAsinh = "Asinh";
constexpr auto kAsinhGrad = "AsinhGrad";
constexpr auto kCdist = "Cdist";
constexpr auto kCdistGrad = "CdistGrad";
constexpr auto kMatrixInverse = "MatrixInverse";
constexpr auto kMatrixSolve = "MatrixSolve";
constexpr auto kMatrixPower = "MatrixPower";
constexpr auto kMatrixDeterminant = "MatrixDeterminant";
constexpr auto kLogMatrixDeterminant = "LogMatrixDeterminant";
constexpr auto kIndexFill = "IndexFill";
constexpr auto kIndexPut = "IndexPut";
constexpr auto kComplex = "Complex";
constexpr auto kAngle = "Angle";
constexpr auto kComplexAbs = "ComplexAbs";
constexpr auto kReal = "Real";
constexpr auto kConj = "Conj";
constexpr auto kImag = "Imag";
constexpr auto kTanh = "Tanh";
constexpr auto kAcosh = "Acosh";
constexpr auto kRoll = "Roll";

// linalg
GVAR_DEF(PrimitivePtr, kPrimGeqrf, std::make_shared<Primitive>("Geqrf"));
GVAR_DEF(PrimitivePtr, kPrimLU, std::make_shared<Primitive>("LU"));
GVAR_DEF(PrimitivePtr, kPrimSolveTriangular, std::make_shared<Primitive>("SolveTriangular"));
GVAR_DEF(PrimitivePtr, kPrimSvd, std::make_shared<Primitive>("Svd"));

// Maths
GVAR_DEF(PrimitivePtr, kPrimCholeskyGrad, std::make_shared<Primitive>("CholeskyGrad"));
GVAR_DEF(PrimitivePtr, kPrimBesselI0e, std::make_shared<Primitive>("BesselI0e"));
GVAR_DEF(PrimitivePtr, kPrimBesselI1e, std::make_shared<Primitive>("BesselI1e"));
GVAR_DEF(PrimitivePtr, kPrimBesselJ0, std::make_shared<Primitive>("BesselJ0"));
GVAR_DEF(PrimitivePtr, kPrimBesselJ1, std::make_shared<Primitive>("BesselJ1"));
GVAR_DEF(PrimitivePtr, kPrimBesselY0, std::make_shared<Primitive>("BesselY0"));
GVAR_DEF(PrimitivePtr, kPrimBesselY1, std::make_shared<Primitive>("BesselY1"));
GVAR_DEF(PrimitivePtr, kPrimTanhGrad, std::make_shared<Primitive>("TanhGrad"));
GVAR_DEF(PrimitivePtr, kPrimTan, std::make_shared<Primitive>("Tan"));
GVAR_DEF(PrimitivePtr, kPrimAtan2, std::make_shared<Primitive>("Atan2"));
GVAR_DEF(PrimitivePtr, kPrimAtan, std::make_shared<Primitive>("Atan"));
GVAR_DEF(PrimitivePtr, kPrimAsin, std::make_shared<Primitive>(kAsin));
GVAR_DEF(PrimitivePtr, kPrimSinc, std::make_shared<Primitive>("Sinc"));
GVAR_DEF(PrimitivePtr, kPrimSinh, std::make_shared<Primitive>("Sinh"));
GVAR_DEF(PrimitivePtr, kPrimCosh, std::make_shared<Primitive>("Cosh"));
GVAR_DEF(PrimitivePtr, kPrimTanh, std::make_shared<Primitive>(kTanh));
GVAR_DEF(PrimitivePtr, kPrimAsinh, std::make_shared<Primitive>(kAsinh));
GVAR_DEF(PrimitivePtr, kPrimAcosh, std::make_shared<Primitive>(kAcosh));
GVAR_DEF(PrimitivePtr, kPrimAtanh, std::make_shared<Primitive>("Atanh"));
GVAR_DEF(PrimitivePtr, kPrimReal, std::make_shared<Primitive>(kReal));
GVAR_DEF(PrimitivePtr, kPrimImag, std::make_shared<Primitive>(kImag));
GVAR_DEF(PrimitivePtr, kPrimConj, std::make_shared<Primitive>(kConj));
GVAR_DEF(PrimitivePtr, kPrimCauchy, std::make_shared<Primitive>(kCauchy));
GVAR_DEF(PrimitivePtr, kPrimNextAfter, std::make_shared<Primitive>(kNextAfter));
GVAR_DEF(PrimitivePtr, kPrimCross, std::make_shared<Primitive>(kCross));
GVAR_DEF(PrimitivePtr, kPrimEditDistance, std::make_shared<Primitive>(kEditDistance));
GVAR_DEF(PrimitivePtr, kPrimBesselI0, std::make_shared<Primitive>("BesselI0"));
GVAR_DEF(PrimitivePtr, kPrimBesselI1, std::make_shared<Primitive>("BesselI1"));
GVAR_DEF(PrimitivePtr, kPrimBesselK0, std::make_shared<Primitive>("BesselK0"));
GVAR_DEF(PrimitivePtr, kPrimBesselK1, std::make_shared<Primitive>("BesselK1"));
GVAR_DEF(PrimitivePtr, kPrimBesselK0e, std::make_shared<Primitive>("BesselK0e"));
GVAR_DEF(PrimitivePtr, kPrimBesselK1e, std::make_shared<Primitive>("BesselK1e"));
GVAR_DEF(PrimitivePtr, kPrimBetainc, std::make_shared<Primitive>("Betainc"));
GVAR_DEF(PrimitivePtr, kPrimGer, std::make_shared<Primitive>("Ger"));
GVAR_DEF(PrimitivePtr, kPrimCeil, std::make_shared<Primitive>("Ceil"));
GVAR_DEF(PrimitivePtr, kPrimDiagonal, std::make_shared<Primitive>(kDiagonal));
GVAR_DEF(PrimitivePtr, kPrimTrunc, std::make_shared<Primitive>("Trunc"));
GVAR_DEF(PrimitivePtr, kPrimLu, std::make_shared<Primitive>("Lu"));
GVAR_DEF(PrimitivePtr, kPrimLuSolve, std::make_shared<Primitive>("LuSolve"));
GVAR_DEF(PrimitivePtr, kPrimMatrixSolve, std::make_shared<Primitive>("MatrixSolve"));
GVAR_DEF(PrimitivePtr, kPrimTridiagonalSolve, std::make_shared<Primitive>(kTridiagonalSolve));
GVAR_DEF(PrimitivePtr, kPrimLuUnpack, std::make_shared<Primitive>("LuUnpack"));
GVAR_DEF(PrimitivePtr, kPrimLuUnpackGrad, std::make_shared<Primitive>("LuUnpackGrad"));
GVAR_DEF(PrimitivePtr, kPrimCholeskyInverse, std::make_shared<Primitive>("CholeskyInverse"));
GVAR_DEF(PrimitivePtr, kPrimTensorAdd, std::make_shared<Primitive>("TensorAdd"));
GVAR_DEF(PrimitivePtr, kPrimAdd, std::make_shared<Primitive>(kAdd, true, kPrimTypeBuiltIn, true));
GVAR_DEF(PrimitivePtr, kPrimAddV2, std::make_shared<Primitive>(kAddV2));
GVAR_DEF(PrimitivePtr, kPrimAddcdiv, std::make_shared<Primitive>(kAddcdiv));
GVAR_DEF(PrimitivePtr, kPrimAddcmul, std::make_shared<Primitive>(kAddcmul));
GVAR_DEF(PrimitivePtr, kPrimMatMul, std::make_shared<Primitive>("MatMul"));
GVAR_DEF(PrimitivePtr, kPrimMatMulV2, std::make_shared<Primitive>("MatMulV2"));
GVAR_DEF(PrimitivePtr, kPrimMatrixDiag, std::make_shared<Primitive>("MatrixDiag"));
GVAR_DEF(PrimitivePtr, kPrimBatchMatMul, std::make_shared<Primitive>("BatchMatMul"));
GVAR_DEF(PrimitivePtr, kPrimBatchMatMulV2, std::make_shared<Primitive>("BatchMatMulV2"));
GVAR_DEF(PrimitivePtr, kPrimFusedMatMulBiasAdd, std::make_shared<Primitive>("FusedMatMulBiasAdd"));
GVAR_DEF(PrimitivePtr, kPrimMaximumGrad, std::make_shared<Primitive>("MaximumGrad"));
GVAR_DEF(PrimitivePtr, kPrimMinimumGrad, std::make_shared<Primitive>("MinimumGrad"));
GVAR_DEF(PrimitivePtr, kPrimMinimumGradGrad, std::make_shared<Primitive>("MinimumGradGrad"));
GVAR_DEF(PrimitivePtr, kPrimMaximumGradGrad, std::make_shared<Primitive>("MaximumGradGrad"));
GVAR_DEF(PrimitivePtr, kPrimMedian, std::make_shared<Primitive>(kMedian));
GVAR_DEF(PrimitivePtr, kPrimMedianGrad, std::make_shared<Primitive>(kMedianGrad));
GVAR_DEF(PrimitivePtr, kPrimReduce, std::make_shared<Primitive>("Reduce"));
GVAR_DEF(PrimitivePtr, kPrimReduceMean, std::make_shared<Primitive>("ReduceMean"));
GVAR_DEF(PrimitivePtr, kPrimReduceMeanD, std::make_shared<Primitive>("ReduceMeanD"));
GVAR_DEF(PrimitivePtr, kPrimReduceSum, std::make_shared<Primitive>("ReduceSum"));
GVAR_DEF(PrimitivePtr, kPrimReduceAll, std::make_shared<Primitive>("ReduceAll"));
GVAR_DEF(PrimitivePtr, kPrimReduceAny, std::make_shared<Primitive>("ReduceAny"));
GVAR_DEF(PrimitivePtr, kPrimReduceAllD, std::make_shared<Primitive>("ReduceAllD"));
GVAR_DEF(PrimitivePtr, kPrimReduceAnyD, std::make_shared<Primitive>("ReduceAnyD"));
GVAR_DEF(PrimitivePtr, kPrimReduceMax, std::make_shared<Primitive>("ReduceMax"));
GVAR_DEF(PrimitivePtr, kPrimReduceMaxD, std::make_shared<Primitive>("ReduceMaxD"));
GVAR_DEF(PrimitivePtr, kPrimReduceMin, std::make_shared<Primitive>("ReduceMin"));
GVAR_DEF(PrimitivePtr, kPrimReduceMinD, std::make_shared<Primitive>("ReduceMinD"));
GVAR_DEF(PrimitivePtr, kPrimReduceProd, std::make_shared<Primitive>("ReduceProd"));
GVAR_DEF(PrimitivePtr, kPrimReduceProdD, std::make_shared<Primitive>("ReduceProdD"));
GVAR_DEF(PrimitivePtr, kPrimReduceStd, std::make_shared<Primitive>(kReduceStd));
GVAR_DEF(PrimitivePtr, kPrimCentralization, std::make_shared<Primitive>("Centralization"));
GVAR_DEF(PrimitivePtr, kPrimNeg, std::make_shared<Primitive>(kNeg));
GVAR_DEF(PrimitivePtr, kPrimLcm, std::make_shared<Primitive>("Lcm"));
GVAR_DEF(PrimitivePtr, kPrimSin, std::make_shared<Primitive>("Sin"));
GVAR_DEF(PrimitivePtr, kPrimCos, std::make_shared<Primitive>(kCos));
GVAR_DEF(PrimitivePtr, kPrimGcd, std::make_shared<Primitive>("Gcd"));
GVAR_DEF(PrimitivePtr, kPrimSub, std::make_shared<Primitive>(kSub));
GVAR_DEF(PrimitivePtr, kPrimHypot, std::make_shared<Primitive>("Hypot"));
GVAR_DEF(PrimitivePtr, kPrimHeaviside, std::make_shared<Primitive>("Heaviside"));
GVAR_DEF(PrimitivePtr, kPrimMul, std::make_shared<Primitive>(kMul));
GVAR_DEF(PrimitivePtr, kPrimMulNoNan, std::make_shared<Primitive>(kMulNoNan));
GVAR_DEF(PrimitivePtr, kPrimDiv, std::make_shared<Primitive>("Div"));
GVAR_DEF(PrimitivePtr, kPrimMod, std::make_shared<Primitive>("Mod"));
GVAR_DEF(PrimitivePtr, kPrimFloor, std::make_shared<Primitive>("Floor"));
GVAR_DEF(PrimitivePtr, kPrimInvert, std::make_shared<Primitive>("Invert"));
GVAR_DEF(PrimitivePtr, kPrimDivNoNan, std::make_shared<Primitive>("DivNoNan"));
GVAR_DEF(PrimitivePtr, kPrimNanToNum, std::make_shared<Primitive>("NanToNum"));
GVAR_DEF(PrimitivePtr, kPrimMinimum, std::make_shared<Primitive>("Minimum"));
GVAR_DEF(PrimitivePtr, kPrimHistogram, std::make_shared<Primitive>("Histogram"));
GVAR_DEF(PrimitivePtr, kPrimMaximum, std::make_shared<Primitive>("Maximum"));
GVAR_DEF(PrimitivePtr, kPrimSquare, std::make_shared<Primitive>(kSquare));
GVAR_DEF(PrimitivePtr, kPrimCumSum, std::make_shared<Primitive>("CumSum"));
GVAR_DEF(PrimitivePtr, kPrimCumulativeLogsumexp, std::make_shared<Primitive>(kCumulativeLogsumexp));
GVAR_DEF(PrimitivePtr, kPrimCumsum, std::make_shared<Primitive>("Cumsum"));
GVAR_DEF(PrimitivePtr, kPrimCumProd, std::make_shared<Primitive>("CumProd"));
GVAR_DEF(PrimitivePtr, kPrimSubscalar, std::make_shared<Primitive>("Subscalar"));
GVAR_DEF(PrimitivePtr, kPrimInplaceAdd, std::make_shared<Primitive>("InplaceAdd"));
GVAR_DEF(PrimitivePtr, kPrimInplaceAddD, std::make_shared<Primitive>("InplaceAddD"));
GVAR_DEF(PrimitivePtr, kPrimInplaceIndexAdd, std::make_shared<Primitive>("InplaceIndexAdd"));
GVAR_DEF(PrimitivePtr, kPrimInplaceUpdate, std::make_shared<Primitive>("InplaceUpdate"));
GVAR_DEF(PrimitivePtr, kPrimInplaceUpdateD, std::make_shared<Primitive>("InplaceUpdateD"));
GVAR_DEF(PrimitivePtr, kPrimInplaceUpdateV2, std::make_shared<Primitive>("InplaceUpdateV2"));
GVAR_DEF(PrimitivePtr, kPrimLpNorm, std::make_shared<Primitive>(kLpNorm));
GVAR_DEF(PrimitivePtr, kPrimInplaceSub, std::make_shared<Primitive>("InplaceSub"));
GVAR_DEF(PrimitivePtr, kPrimPow, std::make_shared<Primitive>("Pow"));
GVAR_DEF(PrimitivePtr, kPrimPower, std::make_shared<Primitive>("Power"));
GVAR_DEF(PrimitivePtr, kPrimRealDiv, std::make_shared<Primitive>(kRealDiv));
GVAR_DEF(PrimitivePtr, kPrimFloorDiv, std::make_shared<Primitive>("FloorDiv"));
GVAR_DEF(PrimitivePtr, kPrimTruncateDiv, std::make_shared<Primitive>("TruncateDiv"));
GVAR_DEF(PrimitivePtr, kPrimSqrt, std::make_shared<Primitive>("Sqrt"));
GVAR_DEF(PrimitivePtr, kPrimTruncateMod, std::make_shared<Primitive>("TruncateMod"));
GVAR_DEF(PrimitivePtr, kPrimSqrtGrad, std::make_shared<Primitive>("SqrtGrad"));
GVAR_DEF(PrimitivePtr, kPrimReciprocal, std::make_shared<Primitive>(kReciprocal));
GVAR_DEF(PrimitivePtr, kPrimReciprocalGrad, std::make_shared<Primitive>("ReciprocalGrad"));
GVAR_DEF(PrimitivePtr, kPrimInv, std::make_shared<Primitive>(kInv));
GVAR_DEF(PrimitivePtr, kPrimAbs, std::make_shared<Primitive>(kAbs));
GVAR_DEF(PrimitivePtr, kPrimAbsGrad, std::make_shared<Primitive>("AbsGrad"));
GVAR_DEF(PrimitivePtr, kPrimRint, std::make_shared<Primitive>("Rint"));
GVAR_DEF(PrimitivePtr, kPrimRound, std::make_shared<Primitive>("Round"));
GVAR_DEF(PrimitivePtr, kPrimExp, std::make_shared<Primitive>(kExp));
GVAR_DEF(PrimitivePtr, kPrimExpm1, std::make_shared<Primitive>("Expm1"));
GVAR_DEF(PrimitivePtr, kPrimLog, std::make_shared<Primitive>(kLog));
GVAR_DEF(PrimitivePtr, kPrimLogit, std::make_shared<Primitive>(kLogit));
GVAR_DEF(PrimitivePtr, kPrimLogitGrad, std::make_shared<Primitive>(kLogitGrad));
GVAR_DEF(PrimitivePtr, kPrimRsqrt, std::make_shared<Primitive>("Rsqrt"));
GVAR_DEF(PrimitivePtr, kPrimRsqrtGrad, std::make_shared<Primitive>("RsqrtGrad"));
GVAR_DEF(PrimitivePtr, kPrimLinSpace, std::make_shared<Primitive>("LinSpace"));
GVAR_DEF(PrimitivePtr, kPrimNonMaxSuppression, std::make_shared<Primitive>("NonMaxSuppression"));
GVAR_DEF(PrimitivePtr, kPrimSign, std::make_shared<Primitive>("Sign"));
GVAR_DEF(PrimitivePtr, kPrimACos, std::make_shared<Primitive>(kACos));
GVAR_DEF(PrimitivePtr, kPrimMatrixSolveLs, std::make_shared<Primitive>(kMatrixSolveLs));
GVAR_DEF(PrimitivePtr, kPrimAsinGrad, std::make_shared<Primitive>(kAsinGrad));
GVAR_DEF(PrimitivePtr, kPrimACosGrad, std::make_shared<Primitive>(kACosGrad));
GVAR_DEF(PrimitivePtr, kPrimAcosGrad, std::make_shared<Primitive>("AcosGrad"));
GVAR_DEF(PrimitivePtr, kPrimAtanGrad, std::make_shared<Primitive>("AtanGrad"));
GVAR_DEF(PrimitivePtr, kPrimAsinhGrad, std::make_shared<Primitive>(kAsinhGrad));
GVAR_DEF(PrimitivePtr, kPrimAcoshGrad, std::make_shared<Primitive>("AcoshGrad"));
GVAR_DEF(PrimitivePtr, kPrimFloorMod, std::make_shared<Primitive>("FloorMod"));
GVAR_DEF(PrimitivePtr, kPrimCdist, std::make_shared<Primitive>(kCdist));
GVAR_DEF(PrimitivePtr, kPrimCdistGrad, std::make_shared<Primitive>(kCdistGrad));
GVAR_DEF(PrimitivePtr, kPrimWhere, std::make_shared<Primitive>("Where"));
GVAR_DEF(PrimitivePtr, kPrimMatrixInverse, std::make_shared<Primitive>(kMatrixInverse));
GVAR_DEF(PrimitivePtr, kPrimMatrixPower, std::make_shared<Primitive>(kMatrixPower));
GVAR_DEF(PrimitivePtr, kPrimMatrixDeterminant, std::make_shared<Primitive>(kMatrixDeterminant));
GVAR_DEF(PrimitivePtr, kPrimLogMatrixDeterminant, std::make_shared<Primitive>(kLogMatrixDeterminant));
GVAR_DEF(PrimitivePtr, kPrimIndexAdd, std::make_shared<Primitive>("IndexAdd"));
GVAR_DEF(PrimitivePtr, kPrimIndexFill, std::make_shared<Primitive>(kIndexFill));
GVAR_DEF(PrimitivePtr, kPrimIndexPut, std::make_shared<Primitive>(kIndexPut));
GVAR_DEF(PrimitivePtr, kPrimInvGrad, std::make_shared<Primitive>("InvGrad"));
GVAR_DEF(PrimitivePtr, kPrimErfinv, std::make_shared<Primitive>("Erfinv"));
GVAR_DEF(PrimitivePtr, kPrimFloatStatus, std::make_shared<Primitive>("FloatStatus"));
GVAR_DEF(PrimitivePtr, kPrimIsNan, std::make_shared<Primitive>("IsNan"));
GVAR_DEF(PrimitivePtr, kPrimIsInf, std::make_shared<Primitive>("IsInf"));
GVAR_DEF(PrimitivePtr, kPrimIsFinite, std::make_shared<Primitive>("IsFinite"));
GVAR_DEF(PrimitivePtr, kPrimComplexAbs, std::make_shared<Primitive>("ComplexAbs"));
GVAR_DEF(PrimitivePtr, kPrimIsClose, std::make_shared<Primitive>("IsClose"));
GVAR_DEF(PrimitivePtr, kPrimLerp, std::make_shared<Primitive>("Lerp"));
GVAR_DEF(PrimitivePtr, kPrimEuclideanNorm, std::make_shared<Primitive>("EuclideanNorm"));
GVAR_DEF(PrimitivePtr, kPrimSquareSumAll, std::make_shared<Primitive>("SquareSumAll"));
GVAR_DEF(PrimitivePtr, kPrimSquareSumV1, std::make_shared<Primitive>("SquareSumV1"));
GVAR_DEF(PrimitivePtr, kPrimComplex, std::make_shared<Primitive>(kComplex));
GVAR_DEF(PrimitivePtr, kPrimPolar, std::make_shared<Primitive>("Polar"));
GVAR_DEF(PrimitivePtr, kPrimAngle, std::make_shared<Primitive>(kAngle));
GVAR_DEF(PrimitivePtr, kPrimXdivy, std::make_shared<Primitive>("Xdivy"));
GVAR_DEF(PrimitivePtr, kPrimXlogy, std::make_shared<Primitive>("Xlogy"));
GVAR_DEF(PrimitivePtr, kPrimRaggedRange, std::make_shared<Primitive>("RaggedRange"));
GVAR_DEF(PrimitivePtr, kPrimBitwiseOr, std::make_shared<Primitive>("BitwiseOr"));
GVAR_DEF(PrimitivePtr, kPrimBitwiseAnd, std::make_shared<Primitive>("BitwiseAnd"));
GVAR_DEF(PrimitivePtr, kPrimBitwiseXor, std::make_shared<Primitive>("BitwiseXor"));
GVAR_DEF(PrimitivePtr, kPrimClipByValue, std::make_shared<Primitive>("ClipByValue"));
GVAR_DEF(PrimitivePtr, kPrimSTFT, std::make_shared<Primitive>("STFT"));
GVAR_DEF(PrimitivePtr, kPrimBucketize, std::make_shared<Primitive>("Bucketize"));
GVAR_DEF(PrimitivePtr, kPrimEinsum, std::make_shared<Primitive>("Einsum"));
GVAR_DEF(PrimitivePtr, kPrimEinsumGrad, std::make_shared<Primitive>("EinsumGrad"));
GVAR_DEF(PrimitivePtr, kPrimTrace, std::make_shared<Primitive>("Trace"));
GVAR_DEF(PrimitivePtr, kPrimTraceGrad, std::make_shared<Primitive>("TraceGrad"));
GVAR_DEF(PrimitivePtr, kPrimTridiagonalMatMul, std::make_shared<Primitive>(kTridiagonalMatMul));
GVAR_DEF(PrimitivePtr, kPrimZeta, std::make_shared<Primitive>("Zeta"));
GVAR_DEF(PrimitivePtr, kPrimFmax, std::make_shared<Primitive>(kFmax));
GVAR_DEF(PrimitivePtr, kPrimIgamma, std::make_shared<Primitive>("Igamma"));
GVAR_DEF(PrimitivePtr, kPrimIgammac, std::make_shared<Primitive>("Igammac"));
GVAR_DEF(PrimitivePtr, kPrimIgammaGradA, std::make_shared<Primitive>("IgammaGradA"));
GVAR_DEF(PrimitivePtr, kPrimLgamma, std::make_shared<Primitive>("Lgamma"));
GVAR_DEF(PrimitivePtr, kPrimDigamma, std::make_shared<Primitive>("Digamma"));
GVAR_DEF(PrimitivePtr, kPrimPolygamma, std::make_shared<Primitive>("Polygamma"));
GVAR_DEF(PrimitivePtr, kPrimBernoulli, std::make_shared<Primitive>(kBernoulli));
GVAR_DEF(PrimitivePtr, kPrimKLDivLoss, std::make_shared<Primitive>("KLDivLoss"));
GVAR_DEF(PrimitivePtr, kPrimCholesky, std::make_shared<Primitive>("Cholesky"));
GVAR_DEF(PrimitivePtr, kPrimCholeskySolve, std::make_shared<Primitive>("CholeskySolve"));
GVAR_DEF(PrimitivePtr, kPrimKLDivLossGrad, std::make_shared<Primitive>("KLDivLossGrad"));
GVAR_DEF(PrimitivePtr, kPrimFFTWithSize, std::make_shared<Primitive>(kFFTWithSize));
GVAR_DEF(PrimitivePtr, kPrimOrgqr, std::make_shared<Primitive>("Orgqr"));
GVAR_DEF(PrimitivePtr, kPrimFmin, std::make_shared<Primitive>("Fmin"));
GVAR_DEF(PrimitivePtr, kPrimTriuIndices, std::make_shared<Primitive>("TriuIndices"));
GVAR_DEF(PrimitivePtr, kPrimTrilIndices, std::make_shared<Primitive>("TrilIndices"));
GVAR_DEF(PrimitivePtr, kPrimEig, std::make_shared<Primitive>("Eig"));
GVAR_DEF(PrimitivePtr, kPrimEigh, std::make_shared<Primitive>("Eigh"));
GVAR_DEF(PrimitivePtr, kPrimQr, std::make_shared<Primitive>("Qr"));
GVAR_DEF(PrimitivePtr, kPrimMatrixLogarithm, std::make_shared<Primitive>(kMatrixLogarithm));
GVAR_DEF(PrimitivePtr, kPrimMatrixTriangularSolve, std::make_shared<Primitive>(kMatrixTriangularSolve));
GVAR_DEF(PrimitivePtr, kPrimSelfAdjointEig, std::make_shared<Primitive>("SelfAdjointEig"));
GVAR_DEF(PrimitivePtr, kPrimOrmqr, std::make_shared<Primitive>("Ormqr"));
GVAR_DEF(PrimitivePtr, kPrimRoll, std::make_shared<Primitive>(kRoll));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_MATH_OPS_H_
