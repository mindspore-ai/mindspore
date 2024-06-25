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
#include "ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace prim {

// linalg
GVAR_DEF(PrimitivePtr, kPrimLU, std::make_shared<Primitive>("LU"));
GVAR_DEF(PrimitivePtr, kPrimSvd, std::make_shared<Primitive>("Svd"));

// Maths
GVAR_DEF(PrimitivePtr, kPrimBesselI0e, std::make_shared<Primitive>("BesselI0e"));
GVAR_DEF(PrimitivePtr, kPrimBesselI1e, std::make_shared<Primitive>("BesselI1e"));
GVAR_DEF(PrimitivePtr, kPrimBesselJ0, std::make_shared<Primitive>("BesselJ0"));
GVAR_DEF(PrimitivePtr, kPrimBesselJ1, std::make_shared<Primitive>("BesselJ1"));
GVAR_DEF(PrimitivePtr, kPrimBesselY0, std::make_shared<Primitive>("BesselY0"));
GVAR_DEF(PrimitivePtr, kPrimBesselY1, std::make_shared<Primitive>("BesselY1"));
GVAR_DEF(PrimitivePtr, kPrimTan, std::make_shared<Primitive>("Tan"));
GVAR_DEF(PrimitivePtr, kPrimImag, std::make_shared<Primitive>(kImagOpName));
GVAR_DEF(PrimitivePtr, kPrimCauchy, std::make_shared<Primitive>(kCauchyOpName));
GVAR_DEF(PrimitivePtr, kPrimCross, std::make_shared<Primitive>(kCrossOpName));
GVAR_DEF(PrimitivePtr, kPrimEditDistance, std::make_shared<Primitive>(kEditDistanceOpName));
GVAR_DEF(PrimitivePtr, kPrimBesselI0, std::make_shared<Primitive>("BesselI0"));
GVAR_DEF(PrimitivePtr, kPrimBesselI1, std::make_shared<Primitive>("BesselI1"));
GVAR_DEF(PrimitivePtr, kPrimBesselK0, std::make_shared<Primitive>("BesselK0"));
GVAR_DEF(PrimitivePtr, kPrimBesselK1, std::make_shared<Primitive>("BesselK1"));
GVAR_DEF(PrimitivePtr, kPrimBesselK0e, std::make_shared<Primitive>("BesselK0e"));
GVAR_DEF(PrimitivePtr, kPrimBesselK1e, std::make_shared<Primitive>("BesselK1e"));
GVAR_DEF(PrimitivePtr, kPrimGer, std::make_shared<Primitive>("Ger"));
GVAR_DEF(PrimitivePtr, kPrimTrunc, std::make_shared<Primitive>("Trunc"));
GVAR_DEF(PrimitivePtr, kPrimLu, std::make_shared<Primitive>("Lu"));
GVAR_DEF(PrimitivePtr, kPrimLuSolve, std::make_shared<Primitive>("LuSolve"));
GVAR_DEF(PrimitivePtr, kPrimMatrixSolve, std::make_shared<Primitive>("MatrixSolve"));
GVAR_DEF(PrimitivePtr, kPrimTridiagonalSolve, std::make_shared<Primitive>(kTridiagonalSolveOpName));
GVAR_DEF(PrimitivePtr, kPrimLuUnpack, std::make_shared<Primitive>("LuUnpack"));
GVAR_DEF(PrimitivePtr, kPrimLuUnpackGrad, std::make_shared<Primitive>("LuUnpackGrad"));
GVAR_DEF(PrimitivePtr, kPrimTensorAdd, std::make_shared<Primitive>("TensorAdd"));
GVAR_DEF(PrimitivePtr, kPrimAddV2, std::make_shared<Primitive>(kAddV2OpName));
GVAR_DEF(PrimitivePtr, kPrimAddLayerNorm, std::make_shared<Primitive>("AddLayerNorm"));
GVAR_DEF(PrimitivePtr, kPrimAddRmsNorm, std::make_shared<Primitive>("AddRmsNorm"));
GVAR_DEF(PrimitivePtr, kPrimMatMulV2, std::make_shared<Primitive>("MatMulV2"));
GVAR_DEF(PrimitivePtr, kPrimMatrixDiag, std::make_shared<Primitive>("MatrixDiag"));
GVAR_DEF(PrimitivePtr, kPrimBatchMatMulV2, std::make_shared<Primitive>("BatchMatMulV2"));
GVAR_DEF(PrimitivePtr, kPrimFusedMatMulBiasAdd, std::make_shared<Primitive>("FusedMatMulBiasAdd"));
GVAR_DEF(PrimitivePtr, kPrimMinimumGradGrad, std::make_shared<Primitive>("MinimumGradGrad"));
GVAR_DEF(PrimitivePtr, kPrimMedian, std::make_shared<Primitive>(kMedianOpName));
GVAR_DEF(PrimitivePtr, kPrimMedianGrad, std::make_shared<Primitive>(kMedianGradOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceMeanD, std::make_shared<Primitive>(kReduceMeanDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceAllD, std::make_shared<Primitive>(kReduceAllDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceAnyD, std::make_shared<Primitive>(kReduceAnyDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceMaxD, std::make_shared<Primitive>(kReduceMaxDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceMinD, std::make_shared<Primitive>(kReduceMinDOpName));
GVAR_DEF(PrimitivePtr, kPrimReduceProdD, std::make_shared<Primitive>(kReduceProdDOpName));
GVAR_DEF(PrimitivePtr, kPrimCentralization, std::make_shared<Primitive>("Centralization"));
GVAR_DEF(PrimitivePtr, kPrimNegGrad, std::make_shared<Primitive>(kNegOpName));
GVAR_DEF(PrimitivePtr, kPrimLcm, std::make_shared<Primitive>("Lcm"));
GVAR_DEF(PrimitivePtr, kPrimHypot, std::make_shared<Primitive>("Hypot"));
GVAR_DEF(PrimitivePtr, kPrimHeaviside, std::make_shared<Primitive>("Heaviside"));
GVAR_DEF(PrimitivePtr, kPrimMulNoNan, std::make_shared<Primitive>(kMulNoNanOpName));
GVAR_DEF(PrimitivePtr, kPrimMod, std::make_shared<Primitive>("Mod"));
GVAR_DEF(PrimitivePtr, kPrimInvert, std::make_shared<Primitive>("Invert"));
GVAR_DEF(PrimitivePtr, kPrimDivNoNan, std::make_shared<Primitive>("DivNoNan"));
GVAR_DEF(PrimitivePtr, kPrimHistogram, std::make_shared<Primitive>("Histogram"));
GVAR_DEF(PrimitivePtr, kPrimCumulativeLogsumexp, std::make_shared<Primitive>(kCumulativeLogsumexpOpName));
GVAR_DEF(PrimitivePtr, kPrimSubscalar, std::make_shared<Primitive>("Subscalar"));
GVAR_DEF(PrimitivePtr, kPrimInplaceAdd, std::make_shared<Primitive>("InplaceAdd"));
GVAR_DEF(PrimitivePtr, kPrimInplaceAddD, std::make_shared<Primitive>("InplaceAddD"));
GVAR_DEF(PrimitivePtr, kPrimInplaceIndexAdd, std::make_shared<Primitive>("InplaceIndexAdd"));
GVAR_DEF(PrimitivePtr, kPrimInplaceUpdate, std::make_shared<Primitive>("InplaceUpdate"));
GVAR_DEF(PrimitivePtr, kPrimInplaceUpdateD, std::make_shared<Primitive>("InplaceUpdateD"));
GVAR_DEF(PrimitivePtr, kPrimInplaceUpdateV2, std::make_shared<Primitive>("InplaceUpdateV2"));
GVAR_DEF(PrimitivePtr, kPrimLpNorm, std::make_shared<Primitive>(kLpNormOpName));
GVAR_DEF(PrimitivePtr, kPrimInplaceSub, std::make_shared<Primitive>("InplaceSub"));
GVAR_DEF(PrimitivePtr, kPrimPower, std::make_shared<Primitive>("Power"));
GVAR_DEF(PrimitivePtr, kPrimTruncateDiv, std::make_shared<Primitive>("TruncateDiv"));
GVAR_DEF(PrimitivePtr, kPrimTruncateMod, std::make_shared<Primitive>("TruncateMod"));
GVAR_DEF(PrimitivePtr, kPrimInv, std::make_shared<Primitive>(kInvOpName));
GVAR_DEF(PrimitivePtr, kPrimRint, std::make_shared<Primitive>("Rint"));
GVAR_DEF(PrimitivePtr, kPrimLogGrad, std::make_shared<Primitive>(kLogGradOpName));
GVAR_DEF(PrimitivePtr, kPrimNonMaxSuppression, std::make_shared<Primitive>("NonMaxSuppression"));
GVAR_DEF(PrimitivePtr, kPrimMatrixSolveLs, std::make_shared<Primitive>(kMatrixSolveLsOpName));
GVAR_DEF(PrimitivePtr, kPrimCdist, std::make_shared<Primitive>(kCdistOpName));
GVAR_DEF(PrimitivePtr, kPrimCdistGrad, std::make_shared<Primitive>(kCdistGradOpName));
GVAR_DEF(PrimitivePtr, kPrimWhere, std::make_shared<Primitive>("Where"));
GVAR_DEF(PrimitivePtr, kPrimMatrixInverse, std::make_shared<Primitive>(kMatrixInverseOpName));
GVAR_DEF(PrimitivePtr, kPrimMatrixPower, std::make_shared<Primitive>(kMatrixPowerOpName));
GVAR_DEF(PrimitivePtr, kPrimIndexAdd, std::make_shared<Primitive>("IndexAdd"));
GVAR_DEF(PrimitivePtr, kPrimIndexFill, std::make_shared<Primitive>(kIndexFillOpName));
GVAR_DEF(PrimitivePtr, kPrimIndexPut, std::make_shared<Primitive>(kIndexPutOpName));
GVAR_DEF(PrimitivePtr, kPrimInvGrad, std::make_shared<Primitive>("InvGrad"));
GVAR_DEF(PrimitivePtr, kPrimFloatStatus, std::make_shared<Primitive>("FloatStatus"));
GVAR_DEF(PrimitivePtr, kPrimIsNan, std::make_shared<Primitive>("IsNan"));
GVAR_DEF(PrimitivePtr, kPrimIsInf, std::make_shared<Primitive>("IsInf"));
GVAR_DEF(PrimitivePtr, kPrimComplexAbs, std::make_shared<Primitive>("ComplexAbs"));
GVAR_DEF(PrimitivePtr, kPrimLerp, std::make_shared<Primitive>("Lerp"));
GVAR_DEF(PrimitivePtr, kPrimEuclideanNorm, std::make_shared<Primitive>("EuclideanNorm"));
GVAR_DEF(PrimitivePtr, kPrimSquareSumAll, std::make_shared<Primitive>("SquareSumAll"));
GVAR_DEF(PrimitivePtr, kPrimSquareSumV1, std::make_shared<Primitive>("SquareSumV1"));
GVAR_DEF(PrimitivePtr, kPrimPolar, std::make_shared<Primitive>("Polar"));
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
GVAR_DEF(PrimitivePtr, kPrimTraceGrad, std::make_shared<Primitive>("TraceGrad"));
GVAR_DEF(PrimitivePtr, kPrimSolveTriangularGrad, std::make_shared<Primitive>("SolveTriangularGrad"));
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
GVAR_DEF(PrimitivePtr, kPrimCholeskySolve, std::make_shared<Primitive>("CholeskySolve"));
GVAR_DEF(PrimitivePtr, kPrimKLDivLossGrad, std::make_shared<Primitive>("KLDivLossGrad"));
GVAR_DEF(PrimitivePtr, kPrimOrgqr, std::make_shared<Primitive>("Orgqr"));
GVAR_DEF(PrimitivePtr, kPrimFmin, std::make_shared<Primitive>("Fmin"));
GVAR_DEF(PrimitivePtr, kPrimTriuIndices, std::make_shared<Primitive>("TriuIndices"));
GVAR_DEF(PrimitivePtr, kPrimTrilIndices, std::make_shared<Primitive>("TrilIndices"));
GVAR_DEF(PrimitivePtr, kPrimEigh, std::make_shared<Primitive>("Eigh"));
GVAR_DEF(PrimitivePtr, kPrimMatrixLogarithm, std::make_shared<Primitive>(kMatrixLogarithmOpName));
GVAR_DEF(PrimitivePtr, kPrimMatrixTriangularSolve, std::make_shared<Primitive>(kMatrixTriangularSolveOpName));
GVAR_DEF(PrimitivePtr, kPrimSelfAdjointEig, std::make_shared<Primitive>("SelfAdjointEig"));
GVAR_DEF(PrimitivePtr, kPrimOrmqr, std::make_shared<Primitive>("Ormqr"));
GVAR_DEF(PrimitivePtr, kPrimEps, std::make_shared<Primitive>(kEpsOpName));
GVAR_DEF(PrimitivePtr, kPrimMatmulReduceScatter, std::make_shared<Primitive>(kMatmulReduceScatterOpName));
GVAR_DEF(PrimitivePtr, kPrimAllGatherMatmul, std::make_shared<Primitive>(kAllGatherMatmulOpName));
GVAR_DEF(PrimitivePtr, kPrimSilentCheck, std::make_shared<Primitive>("SilentCheck"));
GVAR_DEF(PrimitivePtr, kPrimFusedMatMulElemUnary, std::make_shared<Primitive>("FusedMatMulElemUnary"));
GVAR_DEF(PrimitivePtr, kPrimFusedMatMulElemBinary, std::make_shared<Primitive>("FusedMatMulElemBinary"));

}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_MATH_OPS_H_
