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
#ifndef MINDSPORE_CORE_OP_NAME_H_
#define MINDSPORE_CORE_OP_NAME_H_

namespace mindspore::ops {
constexpr auto kNameAbsGrad = "AbsGrad";
constexpr auto kNameAbs = "Abs";
constexpr auto kNameACosGrad = "ACosGrad";
constexpr auto kNameAcos = "Acos";
constexpr auto kNameAcoshGrad = "AcoshGrad";
constexpr auto kNameAcosh = "Acosh";
constexpr auto kNameAdd = "Add";
constexpr auto kNameAddcdiv = "Addcdiv";
constexpr auto kNameAddcmul = "Addcmul";
constexpr auto kNameAngle = "Angle";
constexpr auto kNameArgmax = "Argmax";
constexpr auto kNameAsinGrad = "AsinGrad";
constexpr auto kNameAsin = "Asin";
constexpr auto kNameAsinhGrad = "AsinhGrad";
constexpr auto kNameAsinh = "Asinh";
constexpr auto kNameAssign = "Assign";
constexpr auto kNameAtan2 = "Atan2";
constexpr auto kNameAtanGrad = "AtanGrad";
constexpr auto kNameAtan = "Atan";
constexpr auto kNameAtanh = "Atanh";
constexpr auto kNameAvgPoolGrad = "AvgPoolGrad";
constexpr auto kNameAvgPool = "AvgPool";
constexpr auto kNameBaddbmm = "Baddbmm";
constexpr auto kNameBatchNormGradGrad = "BatchNormGradGrad";
constexpr auto kNameBatchNormGrad = "BatchNormGrad";
constexpr auto kNameBetainc = "Betainc";
constexpr auto kNameBiasAddGrad = "BiasAddGrad";
constexpr auto kNameBiasAdd = "BiasAdd";
constexpr auto kNameBoolNot = "BoolNot";
constexpr auto kNameCeil = "Ceil";
constexpr auto kNameCeLU = "CeLU";
constexpr auto kNameCholeskyGrad = "CholeskyGrad";
constexpr auto kNameCholeskyInverse = "CholeskyInverse";
constexpr auto kNameCholesky = "Cholesky";
constexpr auto kNameComplex = "Complex";
constexpr auto kNameConj = "Conj";
constexpr auto kNameCos = "Cos";
constexpr auto kNameCosh = "Cosh";
constexpr auto kNameDiag = "Diag";
constexpr auto kNameEig = "Eig";
constexpr auto kNameEluGrad = "EluGrad";
constexpr auto kNameElu = "Elu";
constexpr auto kNameEqual = "Equal";
constexpr auto kNameErf = "Erf";
constexpr auto kNameErfc = "Erfc";
constexpr auto kNameErfinv = "Erfinv";
constexpr auto kNameExp = "Exp";
constexpr auto kNameExpandDims = "ExpandDims";
constexpr auto kNameExpm1 = "Expm1";
constexpr auto kNameEye = "Eye";
constexpr auto kNameFastGeLUGrad = "FastGeLUGrad";
constexpr auto kNameFastGeLU = "FastGeLU";
constexpr auto kNameFFTWithSize = "FFTWithSize";
constexpr auto kNameFlatten = "Flatten";
constexpr auto kNameFloorDiv = "FloorDiv";
constexpr auto kNameFloorMod = "FloorMod";
constexpr auto kNameFloor = "Floor";
constexpr auto kNameGatherDGradV2 = "GatherDGradV2";
constexpr auto kNameGatherD = "GatherD";
constexpr auto kNameGatherNd = "GatherNd";
constexpr auto kNameGather = "Gather";
constexpr auto kNameGcd = "Gcd";
constexpr auto kNameGeLUGrad = "GeLUGrad";
constexpr auto kNameGeLU = "GeLU";
constexpr auto kNameGeqrf = "Geqrf";
constexpr auto kNameGreaterEqual = "GreaterEqual";
constexpr auto kNameGreater = "Greater";
constexpr auto kNameGridSampler2DGrad = "GridSampler2DGrad";
constexpr auto kNameGridSampler2D = "GridSampler2D";
constexpr auto kNameGridSampler3DGrad = "GridSampler3DGrad";
constexpr auto kNameGridSampler3D = "GridSampler3D";
constexpr auto kNameLinSpace = "LinSpace";
constexpr auto kNameLog1p = "Log1p";
constexpr auto kNameLogMatrixDeterminant = "LogMatrixDeterminant";
constexpr auto kNameLog = "Log";
constexpr auto kNameLogSoftmaxGrad = "LogSoftmaxGrad";
constexpr auto kNameLogSoftmax = "LogSoftmax";
constexpr auto kNameLogicalAnd = "LogicalAnd";
constexpr auto kNameLogicalNot = "LogicalNot";
constexpr auto kNameLogicalOr = "LogicalOr";
constexpr auto kNameLogicalXor = "LogicalXor";
constexpr auto kNameLogitGrad = "LogitGrad";
constexpr auto kNameLogit = "Logit";
constexpr auto kNameMatrixDeterminant = "MatrixDeterminant";
constexpr auto kNameMatrixExp = "MatrixExp";
constexpr auto kNameNanToNum = "NanToNum";
constexpr auto kNameNeg = "Neg";
constexpr auto kNameNextAfter = "NextAfter";
constexpr auto kNameNLLLossGrad = "NLLLossGrad";
constexpr auto kNameNLLLoss = "NLLLoss";
constexpr auto kNameNonZero = "NonZero";
constexpr auto kNameNotEqual = "NotEqual";
constexpr auto kNameOneHot = "OneHot";
constexpr auto kNameOnesLike = "OnesLike";
constexpr auto kNamePow = "Pow";
constexpr auto kNamePReLUGrad = "PReLUGrad";
constexpr auto kNamePReLU = "PReLU";
constexpr auto kNameQr = "Qr";
constexpr auto kNameRandpermV2 = "RandpermV2";
constexpr auto kNameRange = "Range";
constexpr auto kNameRank = "Rank";
constexpr auto kNameRealDiv = "RealDiv";
constexpr auto kNameReal = "Real";
constexpr auto kNameReciprocalGrad = "ReciprocalGrad";
constexpr auto kNameReciprocal = "Reciprocal";
constexpr auto kNameReduceAll = "ReduceAll";
constexpr auto kNameReduceAny = "ReduceAny";
constexpr auto kNameReduceMax = "ReduceMax";
constexpr auto kNameReduceMean = "ReduceMean";
constexpr auto kNameReduceMin = "ReduceMin";
constexpr auto kNameReduceProd = "ReduceProd";
constexpr auto kNameReduceStd = "ReduceStd";
constexpr auto kNameReduceSum = "ReduceSum";
constexpr auto kNameReLU6Grad = "ReLU6Grad";
constexpr auto kNameReLU6 = "ReLU6";
constexpr auto kNameReluGrad = "ReluGrad";
constexpr auto kNameReLU = "ReLU";
constexpr auto kNameReshape = "Reshape";
constexpr auto kNameResizeBicubicGrad = "ResizeBicubicGrad";
constexpr auto kNameResizeBicubic = "ResizeBicubic";
constexpr auto kNameResizeBilinearGrad = "ResizeBilinearGrad";
constexpr auto kNameResizeBilinearV2 = "ResizeBilinearV2";
constexpr auto kNameResizeLinear1DGrad = "ResizeLinear1DGrad";
constexpr auto kNameResizeLinear1D = "ResizeLinear1D";
constexpr auto kNameResizeNearestNeighborV2Grad = "ResizeNearestNeighborV2Grad";
constexpr auto kNameResizeNearestNeighborV2 = "ResizeNearestNeighborV2";
constexpr auto kNameReverseV2 = "ReverseV2";
constexpr auto kNameRightShift = "RightShift";
constexpr auto kNameRoll = "Roll";
constexpr auto kNameRound = "Round";
constexpr auto kNameRsqrtGrad = "RsqrtGrad";
constexpr auto kNameRsqrt = "Rsqrt";
constexpr auto kNameScatterNd = "ScatterNd";
constexpr auto kNameSigmoidGrad = "SigmoidGrad";
constexpr auto kNameSigmoid = "Sigmoid";
constexpr auto kNameSin = "Sin";
constexpr auto kNameSinc = "Sinc";
constexpr auto kNameSinh = "Sinh";
constexpr auto kNameSqrtGrad = "SqrtGrad";
constexpr auto kNameSqrt = "Sqrt";
constexpr auto kNameSquare = "Square";
constexpr auto kNameTensorCopySlices = "TensorCopySlices";
constexpr auto kNameTensorShape = "TensorShape";
constexpr auto kNameTrace = "Trace";
constexpr auto kNameTranspose = "Transpose";
constexpr auto kNameZerosLike = "ZerosLike";
constexpr auto kNameExtractImagePatches = "ExtractImagePatches";
}  // namespace mindspore::ops

#endif  // MINDSPORE_CORE_OP_NAME_H_
