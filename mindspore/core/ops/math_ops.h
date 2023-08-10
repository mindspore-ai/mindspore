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
#include "ops/math_op_name.h"
#include "ir/primitive.h"

namespace mindspore {
namespace prim {

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
GVAR_DEF(PrimitivePtr, kPrimAsin, std::make_shared<Primitive>(kAsinOpName));
GVAR_DEF(PrimitivePtr, kPrimSinc, std::make_shared<Primitive>("Sinc"));
GVAR_DEF(PrimitivePtr, kPrimSinh, std::make_shared<Primitive>("Sinh"));
GVAR_DEF(PrimitivePtr, kPrimCosh, std::make_shared<Primitive>("Cosh"));
GVAR_DEF(PrimitivePtr, kPrimTanh, std::make_shared<Primitive>(kTanhOpName));
GVAR_DEF(PrimitivePtr, kPrimAsinh, std::make_shared<Primitive>(kAsinhOpName));
GVAR_DEF(PrimitivePtr, kPrimAcosh, std::make_shared<Primitive>(kAcoshOpName));
GVAR_DEF(PrimitivePtr, kPrimAtanh, std::make_shared<Primitive>("Atanh"));
GVAR_DEF(PrimitivePtr, kPrimReal, std::make_shared<Primitive>(kRealOpName));
GVAR_DEF(PrimitivePtr, kPrimImag, std::make_shared<Primitive>(kImagOpName));
GVAR_DEF(PrimitivePtr, kPrimConj, std::make_shared<Primitive>(kConjOpName));
GVAR_DEF(PrimitivePtr, kPrimCauchy, std::make_shared<Primitive>(kCauchyOpName));
GVAR_DEF(PrimitivePtr, kPrimNextAfter, std::make_shared<Primitive>(kNextAfterOpName));
GVAR_DEF(PrimitivePtr, kPrimCross, std::make_shared<Primitive>(kCrossOpName));
GVAR_DEF(PrimitivePtr, kPrimEditDistance, std::make_shared<Primitive>(kEditDistanceOpName));
GVAR_DEF(PrimitivePtr, kPrimBesselI0, std::make_shared<Primitive>("BesselI0"));
GVAR_DEF(PrimitivePtr, kPrimBesselI1, std::make_shared<Primitive>("BesselI1"));
GVAR_DEF(PrimitivePtr, kPrimBesselK0, std::make_shared<Primitive>("BesselK0"));
GVAR_DEF(PrimitivePtr, kPrimBesselK1, std::make_shared<Primitive>("BesselK1"));
GVAR_DEF(PrimitivePtr, kPrimBesselK0e, std::make_shared<Primitive>("BesselK0e"));
GVAR_DEF(PrimitivePtr, kPrimBesselK1e, std::make_shared<Primitive>("BesselK1e"));
GVAR_DEF(PrimitivePtr, kPrimBetainc, std::make_shared<Primitive>("Betainc"));
GVAR_DEF(PrimitivePtr, kPrimGer, std::make_shared<Primitive>("Ger"));
GVAR_DEF(PrimitivePtr, kPrimCeil, std::make_shared<Primitive>("Ceil"));
GVAR_DEF(PrimitivePtr, kPrimDiagonal, std::make_shared<Primitive>(kDiagonalOpName));
GVAR_DEF(PrimitivePtr, kPrimTrunc, std::make_shared<Primitive>("Trunc"));
GVAR_DEF(PrimitivePtr, kPrimLu, std::make_shared<Primitive>("Lu"));
GVAR_DEF(PrimitivePtr, kPrimLuSolve, std::make_shared<Primitive>("LuSolve"));
GVAR_DEF(PrimitivePtr, kPrimMatrixSolve, std::make_shared<Primitive>("MatrixSolve"));
GVAR_DEF(PrimitivePtr, kPrimTridiagonalSolve, std::make_shared<Primitive>(kTridiagonalSolveOpName));
GVAR_DEF(PrimitivePtr, kPrimLuUnpack, std::make_shared<Primitive>("LuUnpack"));
GVAR_DEF(PrimitivePtr, kPrimLuUnpackGrad, std::make_shared<Primitive>("LuUnpackGrad"));
GVAR_DEF(PrimitivePtr, kPrimCholeskyInverse, std::make_shared<Primitive>("CholeskyInverse"));
GVAR_DEF(PrimitivePtr, kPrimTensorAdd, std::make_shared<Primitive>("TensorAdd"));
GVAR_DEF(PrimitivePtr, kPrimAdd, std::make_shared<Primitive>(kAddOpName, true, kPrimTypeBuiltIn, true));
GVAR_DEF(PrimitivePtr, kPrimAddV2, std::make_shared<Primitive>(kAddV2OpName));
GVAR_DEF(PrimitivePtr, kPrimAddcdiv, std::make_shared<Primitive>(kAddcdivOpName));
GVAR_DEF(PrimitivePtr, kPrimAddcmul, std::make_shared<Primitive>(kAddcmulOpName));
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
GVAR_DEF(PrimitivePtr, kPrimMedian, std::make_shared<Primitive>(kMedianOpName));
GVAR_DEF(PrimitivePtr, kPrimMedianGrad, std::make_shared<Primitive>(kMedianGradOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceMean, std::make_shared<Primitive>(kReduceMeanOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceMeanD, std::make_shared<Primitive>(kReduceMeanDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceSum, std::make_shared<Primitive>(kReduceSumOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceAll, std::make_shared<Primitive>(kReduceAllOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceAny, std::make_shared<Primitive>(kReduceAnyOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceAllD, std::make_shared<Primitive>(kReduceAllDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceAnyD, std::make_shared<Primitive>(kReduceAnyDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceMax, std::make_shared<Primitive>(kReduceMaxOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceMaxD, std::make_shared<Primitive>(kReduceMaxDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceMin, std::make_shared<Primitive>(kReduceMinOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceMinD, std::make_shared<Primitive>(kReduceMinDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceProd, std::make_shared<Primitive>(kReduceProdOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceProdD, std::make_shared<Primitive>(kReduceProdDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceStd, std::make_shared<Primitive>(kReduceStdOpName));
GVAR_DEF(PrimitivePtr, kPrimCentralization, std::make_shared<Primitive>("Centralization"));
GVAR_DEF(PrimitivePtr, kPrimNeg, std::make_shared<Primitive>(kNegOpName));
GVAR_DEF(PrimitivePtr, kPrimNegGrad, std::make_shared<Primitive>(kNegOpName));
GVAR_DEF(PrimitivePtr, kPrimLcm, std::make_shared<Primitive>("Lcm"));
GVAR_DEF(PrimitivePtr, kPrimSin, std::make_shared<Primitive>("Sin"));
GVAR_DEF(PrimitivePtr, kPrimCos, std::make_shared<Primitive>(kCosOpName));
GVAR_DEF(PrimitivePtr, kPrimGcd, std::make_shared<Primitive>("Gcd"));
GVAR_DEF(PrimitivePtr, kPrimSub, std::make_shared<Primitive>(kSubOpName));
GVAR_DEF(PrimitivePtr, kPrimHypot, std::make_shared<Primitive>("Hypot"));
GVAR_DEF(PrimitivePtr, kPrimHeaviside, std::make_shared<Primitive>("Heaviside"));
GVAR_DEF(PrimitivePtr, kPrimMul, std::make_shared<Primitive>(kMulOpName));
GVAR_DEF(PrimitivePtr, kPrimMulNoNan, std::make_shared<Primitive>(kMulNoNanOpName));
GVAR_DEF(PrimitivePtr, kPrimDiv, std::make_shared<Primitive>("Div"));
GVAR_DEF(PrimitivePtr, kPrimMod, std::make_shared<Primitive>("Mod"));
GVAR_DEF(PrimitivePtr, kPrimFloor, std::make_shared<Primitive>("Floor"));
GVAR_DEF(PrimitivePtr, kPrimInvert, std::make_shared<Primitive>("Invert"));
GVAR_DEF(PrimitivePtr, kPrimDivNoNan, std::make_shared<Primitive>("DivNoNan"));
GVAR_DEF(PrimitivePtr, kPrimNanToNum, std::make_shared<Primitive>("NanToNum"));
GVAR_DEF(PrimitivePtr, kPrimMinimum, std::make_shared<Primitive>("Minimum"));
GVAR_DEF(PrimitivePtr, kPrimHistogram, std::make_shared<Primitive>("Histogram"));
GVAR_DEF(PrimitivePtr, kPrimMaximum, std::make_shared<Primitive>("Maximum"));
GVAR_DEF(PrimitivePtr, kPrimSquare, std::make_shared<Primitive>(kSquareOpName));
GVAR_DEF(PrimitivePtr, kPrimCumSum, std::make_shared<Primitive>("CumSum"));
GVAR_DEF(PrimitivePtr, kPrimCumulativeLogsumexp, std::make_shared<Primitive>(kCumulativeLogsumexpOpName));
GVAR_DEF(PrimitivePtr, kPrimCumsum, std::make_shared<Primitive>("Cumsum"));
GVAR_DEF(PrimitivePtr, kPrimCumProd, std::make_shared<Primitive>("CumProd"));
GVAR_DEF(PrimitivePtr, kPrimSubscalar, std::make_shared<Primitive>("Subscalar"));
GVAR_DEF(PrimitivePtr, kPrimInplaceAdd, std::make_shared<Primitive>("InplaceAdd"));
GVAR_DEF(PrimitivePtr, kPrimInplaceAddD, std::make_shared<Primitive>("InplaceAddD"));
GVAR_DEF(PrimitivePtr, kPrimInplaceIndexAdd, std::make_shared<Primitive>("InplaceIndexAdd"));
GVAR_DEF(PrimitivePtr, kPrimInplaceUpdate, std::make_shared<Primitive>("InplaceUpdate"));
GVAR_DEF(PrimitivePtr, kPrimInplaceUpdateD, std::make_shared<Primitive>("InplaceUpdateD"));
GVAR_DEF(PrimitivePtr, kPrimInplaceUpdateV2, std::make_shared<Primitive>("InplaceUpdateV2"));
GVAR_DEF(PrimitivePtr, kPrimLpNorm, std::make_shared<Primitive>(kLpNormOpName));
GVAR_DEF(PrimitivePtr, kPrimInplaceSub, std::make_shared<Primitive>("InplaceSub"));
GVAR_DEF(PrimitivePtr, kPrimPow, std::make_shared<Primitive>("Pow"));
GVAR_DEF(PrimitivePtr, kPrimPower, std::make_shared<Primitive>("Power"));
GVAR_DEF(PrimitivePtr, kPrimRealDiv, std::make_shared<Primitive>(kRealDivOpName));
GVAR_DEF(PrimitivePtr, kPrimFloorDiv, std::make_shared<Primitive>("FloorDiv"));
GVAR_DEF(PrimitivePtr, kPrimTruncateDiv, std::make_shared<Primitive>("TruncateDiv"));
GVAR_DEF(PrimitivePtr, kPrimSqrt, std::make_shared<Primitive>("Sqrt"));
GVAR_DEF(PrimitivePtr, kPrimTruncateMod, std::make_shared<Primitive>("TruncateMod"));
GVAR_DEF(PrimitivePtr, kPrimSqrtGrad, std::make_shared<Primitive>(kSqrtGradOpName));
GVAR_DEF(PrimitivePtr, kPrimReciprocal, std::make_shared<Primitive>(kReciprocalOpName));
GVAR_DEF(PrimitivePtr, kPrimReciprocalGrad, std::make_shared<Primitive>(kReciprocalGradOpName));
GVAR_DEF(PrimitivePtr, kPrimInv, std::make_shared<Primitive>(kInvOpName));
GVAR_DEF(PrimitivePtr, kPrimAbs, std::make_shared<Primitive>(kAbsOpName));
GVAR_DEF(PrimitivePtr, kPrimAbsGrad, std::make_shared<Primitive>("AbsGrad"));
GVAR_DEF(PrimitivePtr, kPrimRint, std::make_shared<Primitive>("Rint"));
GVAR_DEF(PrimitivePtr, kPrimRound, std::make_shared<Primitive>("Round"));
GVAR_DEF(PrimitivePtr, kPrimExp, std::make_shared<Primitive>(kExpOpName));
GVAR_DEF(PrimitivePtr, kPrimExpm1, std::make_shared<Primitive>("Expm1"));
GVAR_DEF(PrimitivePtr, kPrimLog, std::make_shared<Primitive>(kLogOpName));
GVAR_DEF(PrimitivePtr, kPrimLogGrad, std::make_shared<Primitive>(kLogGradOpName));
GVAR_DEF(PrimitivePtr, kPrimLogit, std::make_shared<Primitive>(kLogitOpName));
GVAR_DEF(PrimitivePtr, kPrimLogitGrad, std::make_shared<Primitive>(kLogitGradOpName));
GVAR_DEF(PrimitivePtr, kPrimRsqrt, std::make_shared<Primitive>(kRsqrtOpName));
GVAR_DEF(PrimitivePtr, kPrimRsqrtGrad, std::make_shared<Primitive>(kRsqrtGradOpName));
GVAR_DEF(PrimitivePtr, kPrimLinSpace, std::make_shared<Primitive>("LinSpace"));
GVAR_DEF(PrimitivePtr, kPrimNonMaxSuppression, std::make_shared<Primitive>("NonMaxSuppression"));
GVAR_DEF(PrimitivePtr, kPrimSign, std::make_shared<Primitive>(kSignOpName));
GVAR_DEF(PrimitivePtr, kPrimACos, std::make_shared<Primitive>(kACosOpName));
GVAR_DEF(PrimitivePtr, kPrimMatrixSolveLs, std::make_shared<Primitive>(kMatrixSolveLsOpName));
GVAR_DEF(PrimitivePtr, kPrimAsinGrad, std::make_shared<Primitive>(kAsinGradOpName));
GVAR_DEF(PrimitivePtr, kPrimACosGrad, std::make_shared<Primitive>(kACosGradOpName));
GVAR_DEF(PrimitivePtr, kPrimAcosGrad, std::make_shared<Primitive>("AcosGrad"));
GVAR_DEF(PrimitivePtr, kPrimAtanGrad, std::make_shared<Primitive>("AtanGrad"));
GVAR_DEF(PrimitivePtr, kPrimAsinhGrad, std::make_shared<Primitive>(kAsinhGradOpName));
GVAR_DEF(PrimitivePtr, kPrimAcoshGrad, std::make_shared<Primitive>("AcoshGrad"));
GVAR_DEF(PrimitivePtr, kPrimFloorMod, std::make_shared<Primitive>("FloorMod"));
GVAR_DEF(PrimitivePtr, kPrimCdist, std::make_shared<Primitive>(kCdistOpName));
GVAR_DEF(PrimitivePtr, kPrimCdistGrad, std::make_shared<Primitive>(kCdistGradOpName));
GVAR_DEF(PrimitivePtr, kPrimWhere, std::make_shared<Primitive>("Where"));
GVAR_DEF(PrimitivePtr, kPrimMatrixInverse, std::make_shared<Primitive>(kMatrixInverseOpName));
GVAR_DEF(PrimitivePtr, kPrimMatrixPower, std::make_shared<Primitive>(kMatrixPowerOpName));
GVAR_DEF(PrimitivePtr, kPrimMatrixDeterminant, std::make_shared<Primitive>(kMatrixDeterminantOpName));
GVAR_DEF(PrimitivePtr, kPrimLogMatrixDeterminant, std::make_shared<Primitive>(kLogMatrixDeterminantOpName));
GVAR_DEF(PrimitivePtr, kPrimIndexAdd, std::make_shared<Primitive>("IndexAdd"));
GVAR_DEF(PrimitivePtr, kPrimIndexFill, std::make_shared<Primitive>(kIndexFillOpName));
GVAR_DEF(PrimitivePtr, kPrimIndexPut, std::make_shared<Primitive>(kIndexPutOpName));
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
GVAR_DEF(PrimitivePtr, kPrimComplex, std::make_shared<Primitive>(kComplexOpName));
GVAR_DEF(PrimitivePtr, kPrimPolar, std::make_shared<Primitive>("Polar"));
GVAR_DEF(PrimitivePtr, kPrimAngle, std::make_shared<Primitive>(kAngleOpName));
GVAR_DEF(PrimitivePtr, kPrimXdivy, std::make_shared<Primitive>("Xdivy"));
GVAR_DEF(PrimitivePtr, kPrimXlogy, std::make_shared<Primitive>("Xlogy"));
GVAR_DEF(PrimitivePtr, kPrimRaggedRange, std::make_shared<Primitive>(kRaggedRangeOpName));
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
GVAR_DEF(PrimitivePtr, kPrimTridiagonalMatMul, std::make_shared<Primitive>(kTridiagonalMatMulOpName));
GVAR_DEF(PrimitivePtr, kPrimZeta, std::make_shared<Primitive>("Zeta"));
GVAR_DEF(PrimitivePtr, kPrimFmax, std::make_shared<Primitive>(kFmaxOpName));
GVAR_DEF(PrimitivePtr, kPrimIgamma, std::make_shared<Primitive>("Igamma"));
GVAR_DEF(PrimitivePtr, kPrimIgammac, std::make_shared<Primitive>("Igammac"));
GVAR_DEF(PrimitivePtr, kPrimIgammaGradA, std::make_shared<Primitive>("IgammaGradA"));
GVAR_DEF(PrimitivePtr, kPrimLgamma, std::make_shared<Primitive>("Lgamma"));
GVAR_DEF(PrimitivePtr, kPrimDigamma, std::make_shared<Primitive>("Digamma"));
GVAR_DEF(PrimitivePtr, kPrimPolygamma, std::make_shared<Primitive>("Polygamma"));
GVAR_DEF(PrimitivePtr, kPrimBernoulli, std::make_shared<Primitive>(kBernoulliOpName));
GVAR_DEF(PrimitivePtr, kPrimKLDivLoss, std::make_shared<Primitive>("KLDivLoss"));
GVAR_DEF(PrimitivePtr, kPrimCholesky, std::make_shared<Primitive>("Cholesky"));
GVAR_DEF(PrimitivePtr, kPrimCholeskySolve, std::make_shared<Primitive>("CholeskySolve"));
GVAR_DEF(PrimitivePtr, kPrimKLDivLossGrad, std::make_shared<Primitive>("KLDivLossGrad"));
GVAR_DEF(PrimitivePtr, kPrimFFTWithSize, std::make_shared<Primitive>(kFFTWithSizeOpName));
GVAR_DEF(PrimitivePtr, kPrimOrgqr, std::make_shared<Primitive>("Orgqr"));
GVAR_DEF(PrimitivePtr, kPrimFmin, std::make_shared<Primitive>("Fmin"));
GVAR_DEF(PrimitivePtr, kPrimTriuIndices, std::make_shared<Primitive>("TriuIndices"));
GVAR_DEF(PrimitivePtr, kPrimTrilIndices, std::make_shared<Primitive>("TrilIndices"));
GVAR_DEF(PrimitivePtr, kPrimEig, std::make_shared<Primitive>("Eig"));
GVAR_DEF(PrimitivePtr, kPrimEigh, std::make_shared<Primitive>("Eigh"));
GVAR_DEF(PrimitivePtr, kPrimQr, std::make_shared<Primitive>("Qr"));
GVAR_DEF(PrimitivePtr, kPrimMatrixLogarithm, std::make_shared<Primitive>(kMatrixLogarithmOpName));
GVAR_DEF(PrimitivePtr, kPrimMatrixTriangularSolve, std::make_shared<Primitive>(kMatrixTriangularSolveOpName));
GVAR_DEF(PrimitivePtr, kPrimSelfAdjointEig, std::make_shared<Primitive>("SelfAdjointEig"));
GVAR_DEF(PrimitivePtr, kPrimOrmqr, std::make_shared<Primitive>("Ormqr"));
GVAR_DEF(PrimitivePtr, kPrimRoll, std::make_shared<Primitive>(kRollOpName));
GVAR_DEF(PrimitivePtr, kPrimEps, std::make_shared<Primitive>(kEpsOpName));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_MATH_OPS_H_
