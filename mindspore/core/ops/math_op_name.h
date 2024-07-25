/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_MATH_OP_NAME_H_
#define MINDSPORE_CORE_BASE_MATH_OP_NAME_H_

namespace mindspore {
// Math
constexpr auto kAbsOpName = "Abs";
constexpr auto kReduceStdOpName = "ReduceStd";
constexpr auto kLogOpName = "Log";
constexpr auto kLogGradOpName = "LogGrad";
constexpr auto kLogitOpName = "Logit";
constexpr auto kLogitGradOpName = "LogitGrad";
constexpr auto kAddOpName = "Add";
constexpr auto kAddExtOpName = "AddExt";
constexpr auto kTensorAddOpName = "Add";
constexpr auto kAddV2OpName = "AddV2";
constexpr auto kAddcdivOpName = "Addcdiv";
constexpr auto kAddcmulOpName = "Addcmul";
constexpr auto kSubOpName = "Sub";
constexpr auto kSubExtOpName = "SubExt";
constexpr auto kMedianOpName = "Median";
constexpr auto kMedianGradOpName = "MedianGrad";
constexpr auto kMulOpName = "Mul";
constexpr auto kMulNoNanOpName = "MulNoNan";
constexpr auto kACosOpName = "ACos";
constexpr auto kACosGradOpName = "ACosGrad";
constexpr auto kRealDivOpName = "RealDiv";
constexpr auto kDivNoNanOpName = "DivNoNan";
constexpr auto kCauchyOpName = "Cauchy";
constexpr auto kCrossOpName = "Cross";
constexpr auto kDiagonalOpName = "Diagonal";
constexpr auto kEditDistanceOpName = "EditDistance";
constexpr auto kNextAfterOpName = "NextAfter";
constexpr auto kMaximumGradGradOpName = "MaximumGradGrad";
constexpr auto kMatrixSolveLsOpName = "MatrixSolveLs";
constexpr auto kTridiagonalMatMulOpName = "TridiagonalMatMul";
constexpr auto kTridiagonalSolveOpName = "TridiagonalSolve";
constexpr auto kFFTWithSizeOpName = "FFTWithSize";
constexpr auto kTriuIndicesOpName = "TriuIndices";
constexpr auto kTrilIndicesOpName = "TrilIndices";
constexpr auto kFmaxOpName = "Fmax";
constexpr auto kTraceOpName = "Trace";
constexpr auto kTraceGradOpName = "TraceGrad";
constexpr auto kSolveTriangularGradOpName = "SolveTriangularGrad";
constexpr auto kMatrixLogarithmOpName = "MatrixLogarithm";
constexpr auto kMatrixTriangularSolveOpName = "MatrixTriangularSolve";
constexpr auto kSelfAdjointEigOpName = "SelfAdjointEig";
constexpr auto kBernoulliOpName = "Bernoulli";
constexpr auto kNegOpName = "Neg";
constexpr auto kNegGradOpName = "NegGrad";
constexpr auto kSincOpName = "Sinc";
constexpr auto kCosOpName = "Cos";
constexpr auto kSquareOpName = "Square";
constexpr auto kCumulativeLogsumexpOpName = "CumulativeLogsumexp";
constexpr auto kLpNormOpName = "LpNorm";
constexpr auto kReciprocalOpName = "Reciprocal";
constexpr auto kInvOpName = "Inv";
constexpr auto kExpOpName = "Exp";
constexpr auto kAsinOpName = "Asin";
constexpr auto kAsinGradOpName = "AsinGrad";
constexpr auto kAsinhOpName = "Asinh";
constexpr auto kAsinhGradOpName = "AsinhGrad";
constexpr auto kCdistOpName = "Cdist";
constexpr auto kCdistGradOpName = "CdistGrad";
constexpr auto kMatrixInverseOpName = "MatrixInverse";
constexpr auto kMatrixSolveOpName = "MatrixSolve";
constexpr auto kMatrixPowerOpName = "MatrixPower";
constexpr auto kMatrixDeterminantOpName = "MatrixDeterminant";
constexpr auto kLogMatrixDeterminantOpName = "LogMatrixDeterminant";
constexpr auto kIndexFillOpName = "IndexFill";
constexpr auto kIndexPutOpName = "IndexPut";
constexpr auto kComplexOpName = "Complex";
constexpr auto kAngleOpName = "Angle";
constexpr auto kComplexAbsOpName = "ComplexAbs";
constexpr auto kRealOpName = "Real";
constexpr auto kConjOpName = "Conj";
constexpr auto kImagOpName = "Imag";
constexpr auto kTanhOpName = "Tanh";
constexpr auto kAcoshOpName = "Acosh";
constexpr auto kRollOpName = "Roll";
constexpr auto kAcosGradOpName = "AcosGrad";
constexpr auto kBatchMatMulOpName = "BatchMatMul";
constexpr auto kBatchMatMulExtOpName = "BatchMatMulExt";
constexpr auto kBatchMatMulV2OpName = "BatchMatMulV2";
constexpr auto kBetaincOpName = "Betainc";
constexpr auto kBesselI0OpName = "BesselI0";
constexpr auto kIndexAddOpName = "IndexAdd";
constexpr auto kBitwiseOrOpName = "BitwiseOr";
constexpr auto kBucketizeOpName = "Bucketize";
constexpr auto kCentralizationOpName = "Centralization";
constexpr auto kCholeskyOpName = "Cholesky";
constexpr auto kCholeskyGradOpName = "CholeskyGrad";
constexpr auto kCholeskyInverseOpName = "CholeskyInverse";
constexpr auto kCholeskySolveOpName = "CholeskySolve";
constexpr auto kClipByValueOpName = "ClipByValue";
constexpr auto kCumProdOpName = "CumProd";
constexpr auto kCumsumOpName = "Cumsum";
constexpr auto kCumsumDOpName = "CumsumD";
constexpr auto kCumSumOpName = "CumSum";
constexpr auto kDigammaOpName = "Digamma";
constexpr auto kDivOpName = "Div";
constexpr auto kDivModOpName = "DivMod";
constexpr auto kEigOpName = "Eig";
constexpr auto kEuclideanNormDOpName = "EuclideanNormD";
constexpr auto kExpm1OpName = "Expm1";
constexpr auto kFloorDivOpName = "FloorDiv";
constexpr auto kFminOpName = "Fmin";
constexpr auto kFusedMatMulBiasAddOpName = "FusedMatMulBiasAdd";
constexpr auto kGcdOpName = "Gcd";
constexpr auto kGeqrfOpName = "Geqrf";
constexpr auto kHistogramOpName = "Histogram";
constexpr auto kHeavisideOpName = "Heaviside";
constexpr auto kHypotOpName = "Hypot";
constexpr auto kIgammaOpName = "Igamma";
constexpr auto kIgammacOpName = "Igammac";
constexpr auto kIgammaGradAOpName = "IgammaGradA";
constexpr auto kInplaceIndexAddOpName = "InplaceIndexAdd";
constexpr auto kInplaceAddOpName = "InplaceAdd";
constexpr auto kInplaceAddDOpName = "InplaceAddD";
constexpr auto kInplaceSubOpName = "InplaceSub";
constexpr auto kInplaceSubDOpName = "InplaceSubD";
constexpr auto kInplaceUpdateOpName = "InplaceUpdate";
constexpr auto kInplaceUpdateDOpName = "InplaceUpdateD";
constexpr auto kInplaceUpdateV2OpName = "InplaceUpdateV2";
constexpr auto kIsNanOpName = "IsNan";
constexpr auto kIsInfOpName = "IsInf";
constexpr auto kKLDivLossOpName = "KLDivLoss";
constexpr auto kKLDivLossGradOpName = "KLDivLossGrad";
constexpr auto kLcmOpName = "Lcm";
constexpr auto kLinSpaceOpName = "LinSpace";
constexpr auto kLinSpaceDOpName = "LinSpaceD";
constexpr auto kLgammaOpName = "Lgamma";
constexpr auto kLuOpName = "Lu";
constexpr auto kLuSolveOpName = "LuSolve";
constexpr auto kLuUnpackOpName = "LuUnpack";
constexpr auto kLuUnpackGradOpName = "LuUnpackGrad";
constexpr auto kMatMulOpName = "MatMul";
constexpr auto kMatMulExtOpName = "MatMulExt";
constexpr auto kMatMulV2OpName = "MatMulV2";
constexpr auto kMatrixDiagOpName = "MatrixDiag";
constexpr auto kMatrixDiagDOpName = "MatrixDiagD";
constexpr auto kMaximumGradOpName = "MaximumGrad";
constexpr auto kMaximumOpName = "Maximum";
constexpr auto kMinimumGradGradOpName = "MinimumGradGrad";
constexpr auto kMinimumGradOpName = "MinimumGrad";
constexpr auto kMinimumOpName = "Minimum";
constexpr auto kNanToNumOpName = "NanToNum";
constexpr auto kOrgqrOpName = "Orgqr";
constexpr auto kPolarOpName = "Polar";
constexpr auto kPolygammaOpName = "Polygamma";
constexpr auto kPowOpName = "Pow";
constexpr auto kQrOpName = "Qr";
constexpr auto kRaggedRangeOpName = "RaggedRange";
constexpr auto kReciprocalGradOpName = "ReciprocalGrad";
constexpr auto kReduceAllOpName = "ReduceAll";
constexpr auto kReduceAllDOpName = "ReduceAllD";
constexpr auto kReduceAnyOpName = "ReduceAny";
constexpr auto kReduceAnyDOpName = "ReduceAnyD";
constexpr auto kReduceMaxOpName = "ReduceMax";
constexpr auto kReduceMaxDOpName = "ReduceMaxD";
constexpr auto kReduceMeanOpName = "ReduceMean";
constexpr auto kReduceMeanDOpName = "ReduceMeanD";
constexpr auto kReduceMinOpName = "ReduceMin";
constexpr auto kReduceMinDOpName = "ReduceMinD";
constexpr auto kReduceProdOpName = "ReduceProd";
constexpr auto kReduceProdDOpName = "ReduceProdD";
constexpr auto kReduceSumOpName = "ReduceSum";
constexpr auto kReverseSequenceOpName = "ReverseSequence";
constexpr auto kRsqrtGradOpName = "RsqrtGrad";
constexpr auto kSqrtGradOpName = "SqrtGrad";
constexpr auto kRsqrtOpName = "Rsqrt";
constexpr auto kSignOpName = "Sign";
constexpr auto kSinOpName = "Sin";
constexpr auto kSinhOpName = "Sinh";
constexpr auto kSolveTriangularOpName = "SolveTriangular";
constexpr auto kSqrtOpName = "Sqrt";
constexpr auto kSquareSumAllOpName = "SquareSumAll";
constexpr auto kSquareSumV1OpName = "SquareSumV1";
constexpr auto kSTFTOpName = "STFT";
constexpr auto kSubscalarOpName = "Subscalar";
constexpr auto kTruncateDivOpName = "TruncateDiv";
constexpr auto kXlogyOpName = "Xlogy";
constexpr auto kXdivyOpName = "Xdivy";
constexpr auto kEpsOpName = "Eps";
constexpr auto kMatmulReduceScatterOpName = "MatmulReduceScatter";
constexpr auto kAllGatherMatmulOpName = "AllGatherMatmul";
constexpr auto kFusedMatMulElemUnaryOpName = "FusedMatMulElemUnary";
constexpr auto kFusedMatMulElemBinaryOpName = "FusedMatMulElemBinary";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_MATH_OP_NAME_H_
