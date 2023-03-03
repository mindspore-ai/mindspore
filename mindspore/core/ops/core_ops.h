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

#ifndef MINDSPORE_CORE_BASE_CORE_OPS_H_
#define MINDSPORE_CORE_BASE_CORE_OPS_H_

#include <iostream>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/flags.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
GVAR_DEF(ValuePtr, kValueOne, std::make_shared<Int64Imm>(1));
#define COMMA ,
GVAR_DEF(mindspore::HashMap<std::string COMMA ValuePtr>, kSideEffectPropagate,
         {{mindspore::GRAPH_FLAG_SIDE_EFFECT_PROPAGATE COMMA kValueOne}});
#undef COMMA
constexpr auto kAdjustHue = "AdjustHue";
constexpr auto kAdjustContrastv2 = "AdjustContrastv2";
constexpr auto kAdjustSaturation = "AdjustSaturation";
constexpr auto kExtractGlimpse = "ExtractGlimpse";
constexpr auto kGetNext = "GetNext";
constexpr auto kGather = "Gather";
constexpr auto kAddcdiv = "Addcdiv";
constexpr auto kAddcmul = "Addcmul";
constexpr auto kCdist = "Cdist";
constexpr auto kCdistGrad = "CdistGrad";
constexpr auto kCompareAndBitpack = "CompareAndBitpack";
// image
constexpr auto kCropAndResizeGradBoxes = "CropAndResizeGradBoxes";
constexpr auto kResizeBilinearV2 = "ResizeBilinearV2";
constexpr auto kResizeBilinearGrad = "ResizeBilinearGrad";
constexpr auto kCropAndResize = "CropAndResize";
constexpr auto kCropAndResizeGradImage = "CropAndResizeGradImage";
constexpr auto kScaleAndTranslate = "ScaleAndTranslate";
constexpr auto kScaleAndTranslateGrad = "ScaleAndTranslateGrad";
constexpr auto kResizeV2 = "ResizeV2";
constexpr auto kResizeV2Grad = "ResizeV2Grad";

// String
constexpr auto kStringEq = "string_eq";
constexpr auto kStringLt = "string_lt";
constexpr auto kStringGt = "string_gt";
constexpr auto kStringLe = "string_le";
constexpr auto kStringGe = "string_ge";
constexpr auto kStringConcat = "string_concat";
constexpr auto kStringNot = "string_not";
constexpr auto kStringIn = "string_in";
constexpr auto kStringMul = "string_mul";
constexpr auto kStringGetItem = "string_getitem";

// Arithmetic
constexpr auto kScalarAdd = "ScalarAdd";
constexpr auto kScalarSub = "ScalarSub";
constexpr auto kScalarMul = "ScalarMul";
constexpr auto kScalarDiv = "ScalarDiv";
constexpr auto kScalarFloordiv = "ScalarFloordiv";
constexpr auto kScalarMod = "ScalarMod";
constexpr auto kScalarPow = "ScalarPow";
constexpr auto kScalarTrunc = "ScalarTrunc";
constexpr auto kScalarFloor = "ScalarFloor";
constexpr auto kScalarUadd = "ScalarUadd";
constexpr auto kScalarUsub = "ScalarUsub";
constexpr auto kScalarEq = "scalar_eq";
constexpr auto kScalarLt = "scalar_lt";
constexpr auto kScalarGt = "scalar_gt";
constexpr auto kScalarLe = "scalar_le";
constexpr auto kScalarGe = "scalar_ge";
constexpr auto kScalarBool = "ScalarBool";
constexpr auto kBoolNot = "bool_not";
constexpr auto kTupleEqual = "tuple_equal";
constexpr auto kListEqual = "list_equal";
constexpr auto kScalarBitwiseAnd = "bit_and";
constexpr auto kScalarBitwiseOr = "bit_or";
constexpr auto kTupleLt = "tuple_lt";
constexpr auto kListLt = "list_lt";
constexpr auto kTupleLe = "tuple_le";
constexpr auto kListLe = "list_le";
constexpr auto kExp = "Exp";
constexpr auto kEqual = "Equal";
constexpr auto kNotEqual = "NotEqual";
constexpr auto kNeg = "Neg";
constexpr auto kCumulativeLogsumexp = "CumulativeLogsumexp";
constexpr auto kSub = "Sub";
constexpr auto kMedian = "Median";
constexpr auto kMedianGrad = "MedianGrad";
constexpr auto kMul = "Mul";
constexpr auto kMulNoNan = "MulNoNan";
constexpr auto kACos = "ACos";
constexpr auto kACosGrad = "ACosGrad";
constexpr auto kRealDiv = "RealDiv";
constexpr auto kDivNoNan = "DivNoNan";
constexpr auto kReciprocal = "Reciprocal";
constexpr auto kInv = "Inv";
constexpr auto kReduceStd = "ReduceStd";
constexpr auto kLog = "Log";
constexpr auto kLogit = "Logit";
constexpr auto kLogitGrad = "LogitGrad";
constexpr auto kLogicalXor = "LogicalXor";
constexpr auto kSelect = "Select";
constexpr auto kAdd = "Add";
constexpr auto kAddV2 = "AddV2";
constexpr auto kBiasAdd = "BiasAdd";
constexpr auto kTile = "Tile";
constexpr auto kAcosh = "Acosh";
constexpr auto kAcoshGrad = "AcoshGrad";
constexpr auto kBiasAddGrad = "BiasAddGrad";
constexpr auto kMatrixInverse = "MatrixInverse";
constexpr auto kMatrixSolve = "MatrixSolve";
constexpr auto kMatrixPower = "MatrixPower";
constexpr auto kMatrixDeterminant = "MatrixDeterminant";
constexpr auto kLogMatrixDeterminant = "LogMatrixDeterminant";
constexpr auto kSinc = "Sinc";
constexpr auto kCos = "Cos";
constexpr auto kAsinh = "Asinh";
constexpr auto kAsinhGrad = "AsinhGrad";
constexpr auto kAbs = "Abs";
constexpr auto kAsin = "Asin";
constexpr auto kAsinGrad = "AsinGrad";
constexpr auto kTrunc = "Trunc";
constexpr auto kLpNorm = "LpNorm";
constexpr auto kEuclideanNorm = "EuclideanNorm";
constexpr auto kRenorm = "Renorm";
constexpr auto kSquare = "Square";
constexpr auto kRealInner = "RealInner";
constexpr auto kReal = "Real";
constexpr auto kImag = "Imag";
constexpr auto kComplex = "Complex";
constexpr auto kAngle = "Angle";
constexpr auto kConj = "Conj";
constexpr auto kComplexAbs = "ComplexAbs";
constexpr auto kGer = "Ger";
constexpr auto kZeta = "Zeta";
constexpr auto kBernoulli = "Bernoulli";
constexpr auto kLinearSumAssignment = "LinearSumAssignment";

// Math
constexpr auto kCauchy = "Cauchy";
constexpr auto kCross = "Cross";
constexpr auto kDiagonal = "Diagonal";
constexpr auto kEditDistance = "EditDistance";
constexpr auto kNextAfter = "NextAfter";
constexpr auto kMaximumGradGrad = "MaximumGradGrad";
constexpr auto kMatrixSolveLs = "MatrixSolveLs";
constexpr auto kSparseSegmentMean = "SparseSegmentMean";
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

// Arrays
constexpr auto kLeftShift = "LeftShift";
constexpr auto kCountNonZero = "CountNonZero";
constexpr auto kFillDiagonal = "FillDiagonal";
constexpr auto kSegmentMax = "SegmentMax";
constexpr auto kSegmentSum = "SegmentSum";
constexpr auto kSegmentMin = "SegmentMin";
constexpr auto kSegmentMean = "SegmentMean";
constexpr auto kSegmentProd = "SegmentProd";
constexpr auto kDynamicShape = "DynamicShape";
constexpr auto kTensorShape = "TensorShape";
constexpr auto kCheckNumerics = "CheckNumerics";
constexpr auto kStack = "Stack";
constexpr auto kPack = "Pack";
constexpr auto kLogNormalReverse = "LogNormalReverse";
constexpr auto kUnstack = "Unstack";
constexpr auto kUnpack = "Unpack";
// Tuple
constexpr auto kMakeTuple = "MakeTuple";
constexpr auto kTupleGetItem = "TupleGetItem";
constexpr auto kTupleSetItem = "tuple_setitem";
// List
constexpr auto kMakeList = "MakeList";
constexpr auto kMakeListNew = "make_list";
constexpr auto kListGetItem = "list_getitem";
constexpr auto kListSetItem = "list_setitem";

constexpr auto kSliceGetItem = "SliceGetItem";
constexpr auto kGeLU = "GeLU";
constexpr auto kUnravelIndex = "UnravelIndex";
constexpr auto kGLU = "GLU";
constexpr auto kGluGrad = "GluGrad";
constexpr auto kReLU = "ReLU";
constexpr auto kReLU6 = "ReLU6";
constexpr auto kReLUV2 = "ReLUV2";
constexpr auto kReLUV3 = "ReLUV3";
constexpr auto kReLUGrad = "ReluGrad";
constexpr auto kReLUGradV2 = "ReluGradV2";
constexpr auto kRint = "Rint";
constexpr auto kGeLUGrad = "GeLUGrad";
constexpr auto kFillV2 = "FillV2";
constexpr auto kFills = "Fills";
constexpr auto kFastGeLU = "FastGeLU";
constexpr auto kFastGeLUGrad = "FastGeLUGrad";
constexpr auto kStridedSlice = "StridedSlice";
constexpr auto kStridedSliceGrad = "StridedSliceGrad";
constexpr auto kCoalesce = "Coalesce";
constexpr auto kZerosLike = "ZerosLike";
constexpr auto kOnes = "Ones";
constexpr auto kOnesLike = "OnesLike";
constexpr auto kIdentity = "Identity";
constexpr auto kConcat = "Concat";
constexpr auto kParallelConcat = "ParallelConcat";
constexpr auto kFlattenConcat = "FlattenConcat";
constexpr auto kRightShift = "RightShift";
constexpr auto kDiag = "Diag";
constexpr auto kDiagPart = "DiagPart";
constexpr auto kMatrixDiagV3 = "MatrixDiagV3";
constexpr auto kMatrixDiagPartV3 = "MatrixDiagPartV3";
constexpr auto kMatrixSetDiagV3 = "MatrixSetDiagV3";
constexpr auto kMatrixBandPart = "MatrixBandPart";
constexpr auto kDynamicBroadcastGradientArgs = "DynamicBroadcastGradientArgs";
constexpr auto kTranspose = "Transpose";
constexpr auto kConjugateTranspose = "ConjugateTranspose";
constexpr auto kSplitV = "SplitV";
constexpr auto kListDiff = "ListDiff";
constexpr auto kDynamicBroadcastTo = "DynamicBroadcastTo";
constexpr auto kReshape = "Reshape";
constexpr auto kLstsq = "Lstsq";
constexpr auto kLowerBound = "LowerBound";
constexpr auto kUpperBound = "UpperBound";
constexpr auto kNonZero = "NonZero";
constexpr auto kCummax = "Cummax";
constexpr auto kLogSpace = "LogSpace";
constexpr auto kMvlgamma = "Mvlgamma";
constexpr auto kMvlgammaGrad = "MvlgammaGrad";
constexpr auto kTril = "Tril";
constexpr auto kEye = "Eye";
constexpr auto kTriu = "Triu";
constexpr auto kIndexFill = "IndexFill";
constexpr auto kIndexPut = "IndexPut";
constexpr auto kMeshgrid = "Meshgrid";
constexpr auto kScatterNdMax = "ScatterNdMax";
constexpr auto kScatterNdMin = "ScatterNdMin";
constexpr auto kScatterAddWithAxis = "ScatterAddWithAxis";
constexpr auto kCSRSparseMatrixToSparseTensor = "CSRSparseMatrixToSparseTensor";
constexpr auto kSlice = "Slice";
constexpr auto kAffineGrid = "AffineGrid";
constexpr auto kAffineGridGrad = "AffineGridGrad";
constexpr auto kGatherDGrad = "GatherDGrad";
constexpr auto kGatherDGradV2 = "GatherDGradV2";
constexpr auto kSparseTensorToCSRSparseMatrix = "SparseTensorToCSRSparseMatrix";
constexpr auto kSparseSplit = "SparseSplit";
constexpr auto kReverseV2 = "ReverseV2";
constexpr auto kSparseSparseMinimum = "SparseSparseMinimum";
constexpr auto kBroadcastTo = "BroadcastTo";
constexpr auto kSparseReshape = "SparseReshape";
constexpr auto kUnsortedSegmentSum = "UnsortedSegmentSum";
constexpr auto kUnsortedSegmentSumD = "UnsortedSegmentSumD";
constexpr auto kUnsortedSegmentProd = "UnsortedSegmentProd";
constexpr auto kBincount = "Bincount";
constexpr auto kNoRepeatNGram = "NoRepeatNGram";
constexpr auto kSearchSorted = "SearchSorted";

// Sequence operation
constexpr auto kListAppend = "ListAppend";
constexpr auto kSequenceAdd = "SequenceAdd";
constexpr auto kSequenceCount = "SequenceCount";
constexpr auto kSequenceIndex = "SequenceIndex";
constexpr auto kSequenceMul = "SequenceMul";
constexpr auto kSequenceSlice = "SequenceSlice";
constexpr auto kSequenceLen = "sequence_len";
constexpr auto kSequenceZerosLike = "SequenceZerosLike";
constexpr auto kMakeRange = "make_range";
constexpr auto kSequenceAddOffset = "SequenceAddOffset";
constexpr auto kSequenceSliceGrad = "SequenceSliceGrad";
constexpr auto kSequenceSliceSetItem = "SequenceSliceSetItem";
constexpr auto kSequenceMax = "SequenceMax";
constexpr auto kSequenceMin = "SequenceMin";
constexpr auto kSequenceAddN = "SequenceAddN";
constexpr auto kSequenceConcat = "SequenceConcat";

// NN
constexpr auto kFractionalMaxPoolWithFixedKsize = "FractionalMaxPoolWithFixedKsize";
constexpr auto kFractionalMaxPoolGradWithFixedKsize = "FractionalMaxPoolGradWithFixedKsize";
constexpr auto kApplyAddSign = "ApplyAddSign";
constexpr auto kAdaptiveMaxPool3D = "AdaptiveMaxPool3D";
constexpr auto kFractionalMaxPool3DWithFixedKsize = "FractionalMaxPool3DWithFixedKsize";
constexpr auto kFractionalMaxPool3DGradWithFixedKsize = "FractionalMaxPool3DGradWithFixedKsize";
constexpr auto kFractionalMaxPool = "FractionalMaxPool";
constexpr auto kFractionalMaxPoolGrad = "FractionalMaxPoolGrad";
constexpr auto kFractionalAvgPool = "FractionalAvgPool";
constexpr auto kFractionalAvgPoolGrad = "FractionalAvgPoolGrad";
constexpr auto kMaxUnpool2D = "MaxUnpool2D";
constexpr auto kMaxUnpool2DGrad = "MaxUnpool2DGrad";
constexpr auto kMaxUnpool3D = "MaxUnpool3D";
constexpr auto kMaxUnpool3DGrad = "MaxUnpool3DGrad";
constexpr auto kCTCLoss = "CTCLoss";
constexpr auto kNLLLoss = "NLLLoss";
constexpr auto kNLLLossGrad = "NLLLossGrad";
constexpr auto kMultiMarginLoss = "MultiMarginLoss";
constexpr auto kMultiMarginLossGrad = "MultiMarginLossGrad";
constexpr auto kLayerNorm = "LayerNorm";
constexpr auto kLayerNormGrad = "LayerNormGrad";
constexpr auto kMultilabelMarginLoss = "MultilabelMarginLoss";
constexpr auto kMultilabelMarginLossGrad = "MultilabelMarginLossGrad";
constexpr auto kPadV3 = "PadV3";
constexpr auto kPadV3Grad = "PadV3Grad";
constexpr auto kMirrorPadGrad = "MirrorPadGrad";
constexpr auto kTripletMarginLoss = "TripletMarginLoss";
constexpr auto kDataFormatVecPermute = "DataFormatVecPermute";
constexpr auto kDropoutGenMask = "DropoutGenMask";
constexpr auto kDropoutGenMaskV3 = "DropoutGenMaskV3";
constexpr auto kStatelessDropOutGenMask = "StatelessDropOutGenMask";
constexpr auto kDropoutDoMask = "DropoutDoMask";
constexpr auto kDropoutDoMaskV3 = "DropoutDoMaskV3";
constexpr auto kDropout = "Dropout";
constexpr auto kDropoutGrad = "DropoutGrad";
constexpr auto kDropout2D = "Dropout2D";
constexpr auto kDropout3D = "Dropout3D";
constexpr auto kPadding = "Padding";
constexpr auto kMirrorPad = "MirrorPad";
constexpr auto kConv2DTranspose = "Conv2DTranspose";
constexpr auto kSparseApplyAdadelta = "SparseApplyAdadelta";
constexpr auto kApplyRMSProp = "ApplyRMSProp";
constexpr auto kSparseApplyCenteredRMSProp = "SparseApplyCenteredRMSProp";
constexpr auto kSparseApplyAdagrad = "SparseApplyAdagrad";
constexpr auto kSparseApplyAdagradV2 = "SparseApplyAdagradV2";
constexpr auto kSparseApplyRMSProp = "SparseApplyRMSProp";
constexpr auto kRoll = "Roll";
constexpr auto kTopK = "TopK";
constexpr auto kTanh = "Tanh";
constexpr auto kMish = "Mish";
constexpr auto kLRN = "LRN";
constexpr auto kGridSampler2D = "GridSampler2D";
constexpr auto kGridSampler2DGrad = "GridSampler2DGrad";
constexpr auto kGridSampler3D = "GridSampler3D";
constexpr auto kGridSampler3DGrad = "GridSampler3DGrad";
constexpr auto kAdaptiveMaxPool2D = "AdaptiveMaxPool2D";
constexpr auto kUpsampleTrilinear3D = "UpsampleTrilinear3D";
constexpr auto kUpsampleNearest3D = "UpsampleNearest3D";
constexpr auto kHSwish = "HSwish";
constexpr auto kHSwishGrad = "HSwishGrad";
constexpr auto kNuclearNorm = "NuclearNorm";
constexpr auto kSparseApplyAdagradDA = "SparseApplyAdagradDA";
constexpr auto kMaxPool3DWithArgmax = "MaxPool3DWithArgmax";
constexpr auto kMaxPool3DGradWithArgmax = "MaxPool3DGradWithArgmax";
constexpr auto kUpsampleTrilinear3DGrad = "UpsampleTrilinear3DGrad";
constexpr auto kIFMR = "IFMR";
constexpr auto kSparseApplyMomentum = "SparseApplyMomentum";
constexpr auto kSparseApplyProximalGradientDescent = "SparseApplyProximalGradientDescent";
constexpr auto kAdaptiveMaxPool3DGrad = "AdaptiveMaxPool3DGrad";

// Random
constexpr auto kStandardNormal = "StandardNormal";
constexpr auto kRandomGammaGrad = "RandomGammaGrad";
constexpr auto kUniform = "Uniform";

// RowTensor
constexpr auto kMakeRowTensor = "MakeRowTensor";
constexpr auto kRowTensorGetValues = "RowTensorGetValues";
constexpr auto kRowTensorGetIndices = "RowTensorGetIndices";
constexpr auto kRowTensorGetDenseShape = "RowTensorGetDenseShape";
constexpr auto kRowTensorAdd = "RowTensorAdd";

// CSRTensor
constexpr auto kMakeCSRTensor = "MakeCSRTensor";
constexpr auto kCSRTensorGetValues = "CSRTensorGetValues";
constexpr auto kCSRTensorGetIndptr = "CSRTensorGetIndptr";
constexpr auto kCSRTensorGetIndices = "CSRTensorGetIndices";
constexpr auto kCSRTensorGetDenseShape = "CSRTensorGetDenseShape";
constexpr auto kIsCSRFunc = "IsCSRFunc";

// MapTensor
constexpr auto kMakeMapParameter = "MakeMapParameter";
constexpr auto kMapTensorGet = "MapTensorGet";
constexpr auto kMapTensorPut = "MapTensorPut";
constexpr auto kMapTensorErase = "MapTensorErase";
constexpr auto kMapTensorGetDefaultValue = "MapTensorGetDefaultValue";
constexpr auto kMapTensorGetPermitFilterValue = "MapTensorGetPermitFilterValue";
constexpr auto kMapTensorGetEvictFilterValue = "MapTensorGetEvictFilterValue";
constexpr auto kMapTensorGetKeys = "MapTensorGetKeys";
constexpr auto kMapTensorGetValues = "MapTensorGetValues";
constexpr auto kMapTensorGetData = "MapTensorGetData";
constexpr auto kMapTensorGetGrad = "MapTensorGetGrad";

// COOTensor
constexpr auto kMakeCOOTensor = "MakeCOOTensor";
constexpr auto kCOOTensorGetValues = "COOTensorGetValues";
constexpr auto kCOOTensorGetIndices = "COOTensorGetIndices";
constexpr auto kCOOTensorGetDenseShape = "COOTensorGetDenseShape";
constexpr auto kCOOTensorDenseMatmul = "COOTensorDenseMatmul";
constexpr auto kDenseToSparseSetOperation = "DenseToSparseSetOperation";

// Sparse ops
constexpr auto kSparseCross = "SparseCross";
constexpr auto kRaggedTensorToTensor = "RaggedTensorToTensor";
constexpr auto kSparseTensorDenseMatmul = "SparseTensorDenseMatmul";
constexpr auto kSparseFillEmptyRows = "SparseFillEmptyRows";
constexpr auto kSparseToDenseV2 = "SparseToDenseV2";
constexpr auto kSparseSoftmax = "SparseSoftmax";
constexpr auto kSparseAddmm = "SparseAddmm";
constexpr auto kSparseSparseMaximum = "SparseSparseMaximum";
constexpr auto kCSRReduceSum = "CSRReduceSum";
constexpr auto kCSRMV = "CSRMV";
constexpr auto kCSRMM = "CSRMM";
constexpr auto kCSRMul = "CSRMul";
constexpr auto kCSRGather = "CSRGather";
constexpr auto kCSR2COO = "CSR2COO";
constexpr auto kSparseDenseCwiseAdd = "SparseDenseCwiseAdd";
constexpr auto kSparseDenseCwiseDiv = "SparseDenseCwiseDiv";
constexpr auto kSparseDenseCwiseMul = "SparseDenseCwiseMul";
constexpr auto kCOO2CSR = "COO2CSR";
constexpr auto kCSRDiv = "CSRDiv";
constexpr auto kDenseToDenseSetOperation = "DenseToDenseSetOperation";
constexpr auto kSparseMatrixAdd = "SparseMatrixAdd";
constexpr auto kSparseMatrixMul = "SparseMatrixMul";
constexpr auto kSparseAdd = "SparseAdd";
constexpr auto kSparseSegmentMeanGrad = "SparseSegmentMeanGrad";
constexpr auto kSparseSegmentMeanWithNumSegments = "SparseSegmentMeanWithNumSegments";
constexpr auto kSparseConcat = "SparseConcat";
constexpr auto kSparseMatrixNNZ = "SparseMatrixNNZ";
constexpr auto kSparseMatrixTranspose = "SparseMatrixTranspose";
constexpr auto kSparseMatrixSoftmax = "SparseMatrixSoftmax";
constexpr auto kSparseMatrixMatMul = "SparseMatrixMatMul";
constexpr auto kSparseMatrixSparseMatMul = "SparseMatrixSparseMatMul";
constexpr auto kSparseMatrixOrderingAMD = "SparseMatrixOrderingAMD";
constexpr auto kSparseSegmentSum = "SparseSegmentSum";
constexpr auto kSparseSegmentSumGrad = "SparseSegmentSumGrad";
constexpr auto kSparseSegmentSumWithNumSegments = "SparseSegmentSumWithNumSegments";
constexpr auto kSparseSegmentSqrtN = "SparseSegmentSqrtN";
constexpr auto kSparseSegmentSqrtNGrad = "SparseSegmentSqrtNGrad";
constexpr auto kSparseSegmentSqrtNWithNumSegments = "SparseSegmentSqrtNWithNumSegments";
constexpr auto kRaggedTensorToSparse = "RaggedTensorToSparse";
constexpr auto kChannelShuffle = "ChannelShuffle";

// Sparse Grad ops
constexpr auto kSparseAddGrad = "SparseAddGrad";
constexpr auto kSparseFillEmptyRowsGrad = "SparseFillEmptyRowsGrad";
constexpr auto kSparseTensorDenseAdd = "SparseTensorDenseAdd";
constexpr auto kSparseSlice = "SparseSlice";
constexpr auto kSparseSliceGrad = "SparseSliceGrad";
constexpr auto kSparseReorder = "SparseReorder";

// Meta Function Graph
constexpr auto kJ = "J";
constexpr auto kVmap = "Vmap";
constexpr auto kTaylor = "Taylor";

// Others
constexpr auto kSwitch = "Switch";
constexpr auto kLoad = "Load";
constexpr auto kUpdateState = "UpdateState";
constexpr auto kReturn = "Return";
constexpr auto kDepend = "Depend";
constexpr auto kidentity = "identity";
constexpr auto kSampleDistortedBoundingBoxV2 = "SampleDistortedBoundingBoxV2";
constexpr auto kAssign = "Assign";
constexpr auto kAssignAdd = "AssignAdd";
constexpr auto kAssignSub = "AssignSub";
constexpr auto kVmapStackAssign = "VmapStackAssign";
constexpr auto kVmapUnstackAssign = "VmapUnstackAssign";
constexpr auto kBartlettWindow = "BartlettWindow";
constexpr auto kEnvironCreate = "EnvironCreate";
constexpr auto kEnvironSet = "EnvironSet";
constexpr auto kEnvironGet = "EnvironGet";
constexpr auto kEnvironAdd = "EnvironAdd";
constexpr auto kPopulationCount = "PopulationCount";
constexpr auto kEnvironDestroyAll = "EnvironDestroyAll";
constexpr auto kMutable = "mutable";
constexpr auto kGetGrad = "GetGrad";
constexpr auto kSetSize = "SetSize";
constexpr auto kStandardLaplace = "StandardLaplace";
constexpr auto kLower = "Lower";

constexpr auto kTupleToTensor = "TupleToTensor";
constexpr auto kScalarToTensor = "ScalarToTensor";
constexpr auto kTensorToTuple = "TensorToTuple";
constexpr auto kTensorToScalar = "TensorToScalar";
constexpr auto kRealMakeTuple = "RealMakeTuple";
constexpr auto kRealTupleGetItem = "RealTupleGetItem";
constexpr auto kListToTensor = "ListToTensor";
constexpr auto kTensorToList = "TensorToList";
constexpr auto kTupleGreaterThan = "tuple_greater_than";
constexpr auto kTupleGreaterEqual = "tuple_greater_equal";
constexpr auto kListGreaterThan = "list_greater_than";
constexpr auto kListGreaterEqual = "list_greater_equal";

GVAR_DEF(PrimitivePtr, kPrimExtractGlimpse, std::make_shared<Primitive>(kExtractGlimpse));
//
// Here list all primitives used in backend or some special primitives used by core.
// GetNext
GVAR_DEF(PrimitivePtr, kPrimGetNext, std::make_shared<Primitive>(kGetNext));

// Arithmetic
GVAR_DEF(PrimitivePtr, kPrimScalarAdd, std::make_shared<Primitive>(kScalarAdd));
GVAR_DEF(PrimitivePtr, kPrimScalarSub, std::make_shared<Primitive>(kScalarSub));
GVAR_DEF(PrimitivePtr, kPrimScalarMul, std::make_shared<Primitive>(kScalarMul));
GVAR_DEF(PrimitivePtr, kPrimScalarDiv, std::make_shared<Primitive>(kScalarDiv));
GVAR_DEF(PrimitivePtr, kPrimScalarFloorDiv, std::make_shared<Primitive>(kScalarFloordiv));
GVAR_DEF(PrimitivePtr, kPrimScalarMod, std::make_shared<Primitive>(kScalarMod));
GVAR_DEF(PrimitivePtr, kPrimScalarPow, std::make_shared<Primitive>(kScalarPow));
GVAR_DEF(PrimitivePtr, kPrimScalarTrunc, std::make_shared<Primitive>(kScalarTrunc));
GVAR_DEF(PrimitivePtr, kPrimScalarFloor, std::make_shared<Primitive>(kScalarFloor));
GVAR_DEF(PrimitivePtr, kPrimScalarUadd, std::make_shared<Primitive>(kScalarUadd));
GVAR_DEF(PrimitivePtr, kPrimScalarUsub, std::make_shared<Primitive>(kScalarUsub));
GVAR_DEF(PrimitivePtr, kPrimScalarExp, std::make_shared<Primitive>("scalar_exp"));
GVAR_DEF(PrimitivePtr, kPrimScalarLog, std::make_shared<Primitive>("scalar_log"));
GVAR_DEF(PrimitivePtr, kPrimScalarSin, std::make_shared<Primitive>("scalar_sin"));
GVAR_DEF(PrimitivePtr, kPrimScalarCos, std::make_shared<Primitive>("scalar_cos"));
GVAR_DEF(PrimitivePtr, kPrimScalarTan, std::make_shared<Primitive>("scalar_tan"));
GVAR_DEF(PrimitivePtr, kPrimLinearSumAssignment, std::make_shared<Primitive>(kLinearSumAssignment));

// String
GVAR_DEF(PrimitivePtr, kPrimStringEq, std::make_shared<Primitive>(kStringEq));
GVAR_DEF(PrimitivePtr, kPrimStringLt, std::make_shared<Primitive>(kStringLt));
GVAR_DEF(PrimitivePtr, kPrimStringGt, std::make_shared<Primitive>(kStringGt));
GVAR_DEF(PrimitivePtr, kPrimStringLe, std::make_shared<Primitive>(kStringLe));
GVAR_DEF(PrimitivePtr, kPrimStringGe, std::make_shared<Primitive>(kStringGe));
GVAR_DEF(PrimitivePtr, kPrimStringConcat, std::make_shared<Primitive>(kStringConcat));
GVAR_DEF(PrimitivePtr, kPrimStringNot, std::make_shared<Primitive>(kStringNot));
GVAR_DEF(PrimitivePtr, kPrimStringIn, std::make_shared<Primitive>(kStringIn));
GVAR_DEF(PrimitivePtr, kPrimStringMul, std::make_shared<Primitive>(kStringMul));
GVAR_DEF(PrimitivePtr, kPrimStringGetItem, std::make_shared<Primitive>(kStringGetItem));

// Comparisons
GVAR_DEF(PrimitivePtr, kPrimScalarEq, std::make_shared<Primitive>(kScalarEq));
GVAR_DEF(PrimitivePtr, kPrimScalarLt, std::make_shared<Primitive>(kScalarLt));
GVAR_DEF(PrimitivePtr, kPrimScalarGt, std::make_shared<Primitive>(kScalarGt));
GVAR_DEF(PrimitivePtr, kPrimScalarNe, std::make_shared<Primitive>("scalar_ne"));
GVAR_DEF(PrimitivePtr, kPrimScalarLe, std::make_shared<Primitive>(kScalarLe));
GVAR_DEF(PrimitivePtr, kPrimScalarGe, std::make_shared<Primitive>(kScalarGe));
GVAR_DEF(PrimitivePtr, kPrimScalarBool, std::make_shared<Primitive>(kScalarBool));
GVAR_DEF(PrimitivePtr, kPrimScalarBitwiseAnd, std::make_shared<Primitive>(kScalarBitwiseAnd));
GVAR_DEF(PrimitivePtr, kPrimScalarBitwiseOr, std::make_shared<Primitive>(kScalarBitwiseOr));
GVAR_DEF(PrimitivePtr, kPrimBoolNot, std::make_shared<Primitive>(kBoolNot));
GVAR_DEF(PrimitivePtr, kPrimBoolAnd, std::make_shared<Primitive>("bool_and"));
GVAR_DEF(PrimitivePtr, kPrimBoolOr, std::make_shared<Primitive>("bool_or"));
GVAR_DEF(PrimitivePtr, kPrimBoolEq, std::make_shared<Primitive>("bool_eq"));
GVAR_DEF(PrimitivePtr, kPrimBitAnd, std::make_shared<Primitive>("bit_and"));
GVAR_DEF(PrimitivePtr, kPrimBitOr, std::make_shared<Primitive>("bit_or"));
GVAR_DEF(PrimitivePtr, kPrimBitXor, std::make_shared<Primitive>("bit_xor"));
GVAR_DEF(PrimitivePtr, kPrimBitLeftShift, std::make_shared<Primitive>("bit_left_shift"));
GVAR_DEF(PrimitivePtr, kPrimBitRightShift, std::make_shared<Primitive>("bit_right_shift"));
GVAR_DEF(PrimitivePtr, kPrimGreater, std::make_shared<Primitive>("Greater"));
GVAR_DEF(PrimitivePtr, kPrimGreaterEqual, std::make_shared<Primitive>("GreaterEqual"));
GVAR_DEF(PrimitivePtr, kPrimLess, std::make_shared<Primitive>("Less"));
GVAR_DEF(PrimitivePtr, kPrimLessEqual, std::make_shared<Primitive>("LessEqual"));
GVAR_DEF(PrimitivePtr, kPrimEqual, std::make_shared<Primitive>(kEqual));
GVAR_DEF(PrimitivePtr, kPrimNotEqual, std::make_shared<Primitive>(kNotEqual));
GVAR_DEF(PrimitivePtr, kPrimLogicalAnd, std::make_shared<Primitive>("LogicalAnd"));
GVAR_DEF(PrimitivePtr, kPrimLogicalOr, std::make_shared<Primitive>("LogicalOr"));
GVAR_DEF(PrimitivePtr, kPrimLogicalNot, std::make_shared<Primitive>("LogicalNot"));
GVAR_DEF(PrimitivePtr, kPrimLogicalXor, std::make_shared<Primitive>(kLogicalXor));
GVAR_DEF(PrimitivePtr, kPrimEqualCount, std::make_shared<Primitive>("EqualCount"));
GVAR_DEF(PrimitivePtr, kPrimApproximateEqual, std::make_shared<Primitive>("ApproximateEqual"));

GVAR_DEF(PrimitivePtr, kPrimDistribute, std::make_shared<Primitive>("distribute"));
GVAR_DEF(PrimitivePtr, kPrimIm2Col, std::make_shared<Primitive>("Im2Col"));
GVAR_DEF(PrimitivePtr, kPrimCol2Im, std::make_shared<Primitive>("Col2Im"));
GVAR_DEF(PrimitivePtr, kPrimIm2ColV1, std::make_shared<Primitive>("im2col_v1"));
GVAR_DEF(PrimitivePtr, kPrimCol2ImV1, std::make_shared<Primitive>("col2im_v1"));

GVAR_DEF(PrimitivePtr, kPrimLabelGoto, std::make_shared<Primitive>("LabelGoto"));
GVAR_DEF(PrimitivePtr, kPrimLabelSwitch, std::make_shared<Primitive>("LabelSwitch"));
GVAR_DEF(PrimitivePtr, kPrimLabelSet, std::make_shared<Primitive>("LabelSet"));

// Stack ops
GVAR_DEF(PrimitivePtr, kPrimStackInit, std::make_shared<Primitive>("StackInit"));
GVAR_DEF(PrimitivePtr, kPrimStackDestroy, std::make_shared<Primitive>("StackDestroy"));
GVAR_DEF(PrimitivePtr, kPrimStackPush, std::make_shared<Primitive>("StackPush"));
GVAR_DEF(PrimitivePtr, kPrimStackPop, std::make_shared<Primitive>("StackPop"));

// Arrays
GVAR_DEF(PrimitivePtr, kPrimLeftShift, std::make_shared<Primitive>(kLeftShift));
GVAR_DEF(PrimitivePtr, kPrimFillDiagonal, std::make_shared<Primitive>(kFillDiagonal));
GVAR_DEF(PrimitivePtr, kPrimIdentitys, std::make_shared<Primitive>(kIdentity));
GVAR_DEF(PrimitivePtr, kPrimUnravelIndex, std::make_shared<Primitive>(kUnravelIndex));
GVAR_DEF(PrimitivePtr, kPrimDynamicBroadcastTo, std::make_shared<Primitive>(kDynamicBroadcastTo));
GVAR_DEF(PrimitivePtr, kPrimCummin, std::make_shared<Primitive>("Cummin"));
GVAR_DEF(PrimitivePtr, kPrimBroadcastTo, std::make_shared<Primitive>("BroadcastTo"));
GVAR_DEF(PrimitivePtr, kPrimLogNormalReverse, std::make_shared<Primitive>("LogNormalReverse"));
GVAR_DEF(PrimitivePtr, kPrimTopK, std::make_shared<Primitive>(kTopK));
GVAR_DEF(PrimitivePtr, kPrimInTopK, std::make_shared<Primitive>("InTopK"));
GVAR_DEF(PrimitivePtr, kPrimInTopKD, std::make_shared<Primitive>("InTopKD"));
GVAR_DEF(PrimitivePtr, kPrimArrayToScalar, std::make_shared<Primitive>("array_to_scalar"));
GVAR_DEF(PrimitivePtr, kPrimBroadcastShape, std::make_shared<Primitive>("broadcast_shape"));
GVAR_DEF(PrimitivePtr, kPrimArrayMap, std::make_shared<Primitive>("array_map"));
GVAR_DEF(PrimitivePtr, kPrimArrayReduce, std::make_shared<Primitive>("array_reduce"));
GVAR_DEF(PrimitivePtr, kPrimCast, std::make_shared<Primitive>("Cast"));
GVAR_DEF(PrimitivePtr, kPrimConcat, std::make_shared<Primitive>(kConcat));
GVAR_DEF(PrimitivePtr, kPrimConcatD, std::make_shared<Primitive>("ConcatD"));
GVAR_DEF(PrimitivePtr, kPrimParallelConcat, std::make_shared<Primitive>(kParallelConcat));
GVAR_DEF(PrimitivePtr, kPrimCountNonZero, std::make_shared<Primitive>("CountNonZero"));
GVAR_DEF(PrimitivePtr, kPrimFlattenConcat, std::make_shared<Primitive>(kFlattenConcat));
GVAR_DEF(PrimitivePtr, kPrimSqueeze, std::make_shared<Primitive>("Squeeze"));
GVAR_DEF(PrimitivePtr, kPrimSqueezeV3, std::make_shared<Primitive>("SqueezeV3"));
GVAR_DEF(PrimitivePtr, kPrimUnsqueeze, std::make_shared<Primitive>("Unsqueeze"));
GVAR_DEF(PrimitivePtr, kPrimConjugateTranspose, std::make_shared<Primitive>(kConjugateTranspose));
GVAR_DEF(PrimitivePtr, kPrimTransposeD, std::make_shared<Primitive>("TransposeD"));
GVAR_DEF(PrimitivePtr, kPrimTranspose, std::make_shared<Primitive>(kTranspose));
GVAR_DEF(PrimitivePtr, kPrimGatherV2, std::make_shared<Primitive>("GatherV2"));
GVAR_DEF(PrimitivePtr, kPrimGatherD, std::make_shared<Primitive>("GatherD"));
GVAR_DEF(PrimitivePtr, kPrimGatherDGrad, std::make_shared<Primitive>(kGatherDGrad));
GVAR_DEF(PrimitivePtr, kPrimGatherDGradV2, std::make_shared<Primitive>(kGatherDGradV2));
GVAR_DEF(PrimitivePtr, kPrimGather, std::make_shared<Primitive>("Gather"));
GVAR_DEF(PrimitivePtr, kPrimGatherNd, std::make_shared<Primitive>("GatherNd"));
GVAR_DEF(PrimitivePtr, kPrimSparseGatherV2, std::make_shared<Primitive>("SparseGatherV2"));
GVAR_DEF(PrimitivePtr, kPrimCoalesce, std::make_shared<Primitive>(kCoalesce));
GVAR_DEF(PrimitivePtr, kPrimSparseToDense, std::make_shared<Primitive>("SparseToDense"));
GVAR_DEF(PrimitivePtr, kPrimSparseCountSparseOutput, std::make_shared<Primitive>("SparseCountSparseOutput"));
GVAR_DEF(PrimitivePtr, kPrimSspaddmm, std::make_shared<Primitive>("Sspaddmm"));
GVAR_DEF(PrimitivePtr, kPrimShape, std::make_shared<Primitive>("Shape"));
GVAR_DEF(PrimitivePtr, kPrimShapeCalc, std::make_shared<Primitive>("ShapeCalc"));
GVAR_DEF(PrimitivePtr, kPrimStridedRead, std::make_shared<Primitive>("StridedRead"));
GVAR_DEF(PrimitivePtr, kPrimStridedWrite, std::make_shared<Primitive>("StridedWrite"));
GVAR_DEF(PrimitivePtr, kPrimStridedSlice, std::make_shared<Primitive>(kStridedSlice));
GVAR_DEF(PrimitivePtr, kPrimStridedSliceGrad, std::make_shared<Primitive>(kStridedSliceGrad));
GVAR_DEF(PrimitivePtr, kPrimTensorShape, std::make_shared<Primitive>(kTensorShape));
GVAR_DEF(PrimitivePtr, kPrimDynamicShape, std::make_shared<Primitive>(kDynamicShape));
GVAR_DEF(PrimitivePtr, kPrimCheckNumerics, std::make_shared<Primitive>(kCheckNumerics));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingLookup, std::make_shared<Primitive>("EmbeddingLookup"));
GVAR_DEF(PrimitivePtr, kPrimEmbeddingLookupCommGrad, std::make_shared<Primitive>("EmbeddingLookupCommGrad"));
GVAR_DEF(PrimitivePtr, kPrimSize, std::make_shared<Primitive>("Size"));
GVAR_DEF(PrimitivePtr, kPrimArgMax, std::make_shared<Primitive>("Argmax"));
GVAR_DEF(PrimitivePtr, kPrimArgmin, std::make_shared<Primitive>("Argmin"));
GVAR_DEF(PrimitivePtr, kPrimArgMin, std::make_shared<Primitive>("ArgMin"));
GVAR_DEF(PrimitivePtr, kPrimArgminV2, std::make_shared<Primitive>("ArgminV2"));
GVAR_DEF(PrimitivePtr, kPrimPack, std::make_shared<Primitive>("Pack"));
GVAR_DEF(PrimitivePtr, kPrimStack, std::make_shared<Primitive>(kStack));
GVAR_DEF(PrimitivePtr, kPrimUnpack, std::make_shared<Primitive>("Unpack"));
GVAR_DEF(PrimitivePtr, kPrimUnstack, std::make_shared<Primitive>(kUnstack));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentMax, std::make_shared<Primitive>("UnsortedSegmentMax"));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentProd, std::make_shared<Primitive>(kUnsortedSegmentProd));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentSum, std::make_shared<Primitive>(kUnsortedSegmentSum));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentSumD, std::make_shared<Primitive>(kUnsortedSegmentSumD));
GVAR_DEF(PrimitivePtr, kPrimUnsortedSegmentMin, std::make_shared<Primitive>("UnsortedSegmentMin"));
GVAR_DEF(PrimitivePtr, kPrimConcatOffset, std::make_shared<Primitive>("ConcatOffset"));
GVAR_DEF(PrimitivePtr, kPrimConcatOffsetV1, std::make_shared<Primitive>("ConcatOffsetV1"));
GVAR_DEF(PrimitivePtr, kPrimIdentityN, std::make_shared<Primitive>("IdentityN"));
GVAR_DEF(PrimitivePtr, kPrimReshape, std::make_shared<Primitive>("Reshape"));
GVAR_DEF(PrimitivePtr, kPrimListDiff, std::make_shared<Primitive>(kListDiff));
GVAR_DEF(PrimitivePtr, kPrimSubAndFilter, std::make_shared<Primitive>("SubAndFilter"));
GVAR_DEF(PrimitivePtr, kPrimMapCacheIdx, std::make_shared<Primitive>("MapCacheIdx"));
GVAR_DEF(PrimitivePtr, kPrimUpdateCache, std::make_shared<Primitive>("UpdateCache"));
GVAR_DEF(PrimitivePtr, kPrimComputeAccidentalHits, std::make_shared<Primitive>("ComputeAccidentalHits"));
GVAR_DEF(PrimitivePtr, kPrimCacheSwapTable, std::make_shared<Primitive>("CacheSwapTable"));
GVAR_DEF(PrimitivePtr, kPrimDynamicAssign, std::make_shared<Primitive>("DynamicAssign"));
GVAR_DEF(PrimitivePtr, kPrimPadAndShift, std::make_shared<Primitive>("PadAndShift"));
GVAR_DEF(PrimitivePtr, kPrimSlice, std::make_shared<Primitive>(kSlice));
GVAR_DEF(PrimitivePtr, kPrimSliceGrad, std::make_shared<Primitive>("SliceGrad"));
GVAR_DEF(PrimitivePtr, kPrimSliceFusion, std::make_shared<Primitive>("SliceFusion"));
GVAR_DEF(PrimitivePtr, kPrimTile, std::make_shared<Primitive>(kTile));
GVAR_DEF(PrimitivePtr, kPrimTileD, std::make_shared<Primitive>("TileD"));
GVAR_DEF(PrimitivePtr, kPrimAddN, std::make_shared<Primitive>("AddN"));
GVAR_DEF(PrimitivePtr, kPrimAccumulateNV2, std::make_shared<Primitive>("AccumulateNV2"));
GVAR_DEF(PrimitivePtr, kPrimTransData, std::make_shared<Primitive>("TransData"));
GVAR_DEF(PrimitivePtr, kPrimTransDataRNN, std::make_shared<Primitive>("TransDataRNN"));
GVAR_DEF(PrimitivePtr, kPrimNMSWithMask, std::make_shared<Primitive>("NMSWithMask"));
GVAR_DEF(PrimitivePtr, kPrimPad, std::make_shared<Primitive>("Pad"));
GVAR_DEF(PrimitivePtr, kPrimPadD, std::make_shared<Primitive>("PadD"));
GVAR_DEF(PrimitivePtr, kPrimPadding, std::make_shared<Primitive>(kPadding));
GVAR_DEF(PrimitivePtr, kPrimMirrorPad, std::make_shared<Primitive>(kMirrorPad));
GVAR_DEF(PrimitivePtr, kPrimArgMaxWithValue, std::make_shared<Primitive>("ArgMaxWithValue"));
GVAR_DEF(PrimitivePtr, kPrimArgMinWithValue, std::make_shared<Primitive>("ArgMinWithValue"));
GVAR_DEF(PrimitivePtr, kPrimUnique, std::make_shared<Primitive>("Unique"));
GVAR_DEF(PrimitivePtr, kPrimUniqueWithPad, std::make_shared<Primitive>("UniqueWithPad"));
GVAR_DEF(PrimitivePtr, kPrimUniqueGrad, std::make_shared<Primitive>("UniqueGrad"));
GVAR_DEF(PrimitivePtr, kPrimUniqueConsecutive, std::make_shared<Primitive>("UniqueConsecutive"));
GVAR_DEF(PrimitivePtr, kPrimExtractImagePatches, std::make_shared<Primitive>("ExtractImagePatches"));
GVAR_DEF(PrimitivePtr, kPrimDynamicRNN, std::make_shared<Primitive>("DynamicRNN"));
GVAR_DEF(PrimitivePtr, kPrimCudnnGRU, std::make_shared<Primitive>("CudnnGRU"));
GVAR_DEF(PrimitivePtr, kPrimGRUV2, std::make_shared<Primitive>("GRUV2"));
GVAR_DEF(PrimitivePtr, kPrimGRUV2Grad, std::make_shared<Primitive>("GRUV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimLSTMV2, std::make_shared<Primitive>("LSTMV2"));
GVAR_DEF(PrimitivePtr, kPrimDynamicRNNGrad, std::make_shared<Primitive>("DynamicRNNGrad"));
GVAR_DEF(PrimitivePtr, kPrimDynamicGRUV2, std::make_shared<Primitive>("DynamicGRUV2"));
GVAR_DEF(PrimitivePtr, kPrimDynamicGRUV2Grad, std::make_shared<Primitive>("DynamicGRUV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimSearchSorted, std::make_shared<Primitive>("SearchSorted"));
GVAR_DEF(PrimitivePtr, kPrimScatterAdd, std::make_shared<Primitive>("ScatterAdd"));
GVAR_DEF(PrimitivePtr, kPrimScatterSub, std::make_shared<Primitive>("ScatterSub"));
GVAR_DEF(PrimitivePtr, kPrimScatterMul, std::make_shared<Primitive>("ScatterMul"));
GVAR_DEF(PrimitivePtr, kPrimScatterDiv, std::make_shared<Primitive>("ScatterDiv"));
GVAR_DEF(PrimitivePtr, kPrimScatterMax, std::make_shared<Primitive>("ScatterMax"));
GVAR_DEF(PrimitivePtr, kPrimScatterMin, std::make_shared<Primitive>("ScatterMin"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdAdd, std::make_shared<Primitive>("ScatterNdAdd"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdSub, std::make_shared<Primitive>("ScatterNdSub"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdMax, std::make_shared<Primitive>("ScatterNdMax"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdMin, std::make_shared<Primitive>("ScatterNdMin"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdMul, std::make_shared<Primitive>("ScatterNdMul"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdDiv, std::make_shared<Primitive>("ScatterNdDiv"));
GVAR_DEF(PrimitivePtr, kPrimScatterUpdate, std::make_shared<Primitive>("ScatterUpdate"));
GVAR_DEF(PrimitivePtr, kPrimScatterElements, std::make_shared<Primitive>("ScatterElements"));
GVAR_DEF(PrimitivePtr, kPrimScatterAddWithAxis, std::make_shared<Primitive>(kScatterAddWithAxis));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterElements, std::make_shared<Primitive>("TensorScatterElements"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterUpdate, std::make_shared<Primitive>("TensorScatterUpdate"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterAdd, std::make_shared<Primitive>("TensorScatterAdd"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterSub, std::make_shared<Primitive>("TensorScatterSub"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterMul, std::make_shared<Primitive>("TensorScatterMul"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterDiv, std::make_shared<Primitive>("TensorScatterDiv"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterMax, std::make_shared<Primitive>("TensorScatterMax"));
GVAR_DEF(PrimitivePtr, kPrimTensorScatterMin, std::make_shared<Primitive>("TensorScatterMin"));
GVAR_DEF(PrimitivePtr, kPrimTensorCopySlices, std::make_shared<Primitive>("TensorCopySlices"));
GVAR_DEF(PrimitivePtr, kPrimMapUniform, std::make_shared<Primitive>("MapUniform"));
GVAR_DEF(PrimitivePtr, kPrimSplit, std::make_shared<Primitive>("Split"));
GVAR_DEF(PrimitivePtr, kPrimSplitD, std::make_shared<Primitive>("SplitD"));
GVAR_DEF(PrimitivePtr, kPrimSplitV, std::make_shared<Primitive>(kSplitV));
GVAR_DEF(PrimitivePtr, kPrimSplitVD, std::make_shared<Primitive>("SplitVD"));
GVAR_DEF(PrimitivePtr, kPrimSequenceMask, std::make_shared<Primitive>("SequenceMask"));
GVAR_DEF(PrimitivePtr, kPrimRange, std::make_shared<Primitive>("Range"));
GVAR_DEF(PrimitivePtr, kPrimRangeV2, std::make_shared<Primitive>("RangeV2"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToBatchND, std::make_shared<Primitive>("SpaceToBatchND"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpaceND, std::make_shared<Primitive>("BatchToSpaceND"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpaceNDV2, std::make_shared<Primitive>("BatchToSpaceNDV2"));
GVAR_DEF(PrimitivePtr, kPrimDepthToSpace, std::make_shared<Primitive>("DepthToSpace"));
GVAR_DEF(PrimitivePtr, kPrimBatchToSpace, std::make_shared<Primitive>("BatchToSpace"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantParam, std::make_shared<Primitive>("FakeQuantParam"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToBatch, std::make_shared<Primitive>("SpaceToBatch"));
GVAR_DEF(PrimitivePtr, kPrimScatterNd, std::make_shared<Primitive>("ScatterNd"));
GVAR_DEF(PrimitivePtr, kPrimScatterNdUpdate, std::make_shared<Primitive>("ScatterNdUpdate"));
GVAR_DEF(PrimitivePtr, kPrimScatterNonAliasingAdd, std::make_shared<Primitive>("ScatterNonAliasingAdd"));
GVAR_DEF(PrimitivePtr, kPrimConstantOfShape, std::make_shared<Primitive>("ConstantOfShape"));
GVAR_DEF(PrimitivePtr, kPrimSquaredDifference, std::make_shared<Primitive>("SquaredDifference"));
GVAR_DEF(PrimitivePtr, kPrimReverseV2, std::make_shared<Primitive>("ReverseV2"));
GVAR_DEF(PrimitivePtr, kPrimReverseSequence, std::make_shared<Primitive>("ReverseSequence"));
GVAR_DEF(PrimitivePtr, kPrimRank, std::make_shared<Primitive>("Rank"));
GVAR_DEF(PrimitivePtr, kPrimResizeBilinear, std::make_shared<Primitive>("ResizeBilinear"));
GVAR_DEF(PrimitivePtr, kPrimParallelResizeBilinear, std::make_shared<Primitive>("ParallelResizeBilinear"));
GVAR_DEF(PrimitivePtr, kPrimParallelResizeBilinearGrad, std::make_shared<Primitive>("ParallelResizeBilinearGrad"));
GVAR_DEF(PrimitivePtr, kPrimResizeGrad, std::make_shared<Primitive>("ResizeGrad"));
GVAR_DEF(PrimitivePtr, kPrimResizeNearestNeighbor, std::make_shared<Primitive>("ResizeNearestNeighbor"));
GVAR_DEF(PrimitivePtr, kPrimResizeNearestNeighborGrad, std::make_shared<Primitive>("ResizeNearestNeighborGrad"));
GVAR_DEF(PrimitivePtr, kPrimResizeNearestNeighborV2, std::make_shared<Primitive>("ResizeNearestNeighborV2"));
GVAR_DEF(PrimitivePtr, kPrimResizeNearestNeighborV2Grad, std::make_shared<Primitive>("ResizeNearestNeighborV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimDynamicResizeNearestNeighbor, std::make_shared<Primitive>("DynamicResizeNearestNeighbor"));
GVAR_DEF(PrimitivePtr, kPrimResizeLinear1D, std::make_shared<Primitive>("ResizeLinear1D"));
GVAR_DEF(PrimitivePtr, kPrimResizeLinear1DGrad, std::make_shared<Primitive>("ResizeLinear1DGrad"));
GVAR_DEF(PrimitivePtr, kPrimSort, std::make_shared<Primitive>("Sort"));
GVAR_DEF(PrimitivePtr, kPrimMaskedFill, std::make_shared<Primitive>("MaskedFill"));
GVAR_DEF(PrimitivePtr, kPrimMaskedScatter, std::make_shared<Primitive>("MaskedScatter"));
GVAR_DEF(PrimitivePtr, kPrimMaskedSelect, std::make_shared<Primitive>("MaskedSelect"));
GVAR_DEF(PrimitivePtr, kPrimMaskedSelectGrad, std::make_shared<Primitive>("MaskedSelectGrad"));
GVAR_DEF(PrimitivePtr, kPrimDiag, std::make_shared<Primitive>(kDiag));
GVAR_DEF(PrimitivePtr, kPrimDiagD, std::make_shared<Primitive>("DiagD"));
GVAR_DEF(PrimitivePtr, kPrimDiagPart, std::make_shared<Primitive>(kDiagPart));
GVAR_DEF(PrimitivePtr, kPrimDiagPartD, std::make_shared<Primitive>("DiagPartD"));
GVAR_DEF(PrimitivePtr, kPrimMatrixDiagV3, std::make_shared<Primitive>(kMatrixDiagV3));
GVAR_DEF(PrimitivePtr, kPrimMatrixDiagPartV3, std::make_shared<Primitive>(kMatrixDiagPartV3));
GVAR_DEF(PrimitivePtr, kPrimMatrixSetDiagV3, std::make_shared<Primitive>(kMatrixSetDiagV3));
GVAR_DEF(PrimitivePtr, kPrimMatrixBandPart, std::make_shared<Primitive>(kMatrixBandPart));
GVAR_DEF(PrimitivePtr, kPrimNonZero, std::make_shared<Primitive>("NonZero"));
GVAR_DEF(PrimitivePtr, kPrimNonZeroWithValue, std::make_shared<Primitive>("NonZeroWithValue"));
GVAR_DEF(PrimitivePtr, kPrimNonZeroWithValueShape, std::make_shared<Primitive>("NonZeroWithValueShape"));
GVAR_DEF(PrimitivePtr, kPrimNoRepeatNGram, std::make_shared<Primitive>("NoRepeatNGram"));
GVAR_DEF(PrimitivePtr, kPrimGenerateEodMask, std::make_shared<Primitive>("GenerateEodMask"));
GVAR_DEF(PrimitivePtr, kPrimRealInner, std::make_shared<Primitive>(kRealInner));
GVAR_DEF(PrimitivePtr, kPrimReal, std::make_shared<Primitive>(kReal));
GVAR_DEF(PrimitivePtr, kPrimImag, std::make_shared<Primitive>(kImag));
GVAR_DEF(PrimitivePtr, kPrimConj, std::make_shared<Primitive>(kConj));
GVAR_DEF(PrimitivePtr, kPrimFillV2, std::make_shared<Primitive>(kFillV2));
GVAR_DEF(PrimitivePtr, kPrimFills, std::make_shared<Primitive>(kFills));
GVAR_DEF(PrimitivePtr, kPrimExtractVolumePatches, std::make_shared<Primitive>("ExtractVolumePatches"));
GVAR_DEF(PrimitivePtr, kPrimLstsq, std::make_shared<Primitive>(kLstsq));
GVAR_DEF(PrimitivePtr, kPrimLowerBound, std::make_shared<Primitive>(kLowerBound));
GVAR_DEF(PrimitivePtr, kPrimUpperBound, std::make_shared<Primitive>(kUpperBound));
GVAR_DEF(PrimitivePtr, kPrimResizeBicubic, std::make_shared<Primitive>("ResizeBicubic"));
GVAR_DEF(PrimitivePtr, kPrimResizeBicubicGrad, std::make_shared<Primitive>("ResizeBicubicGrad"));
GVAR_DEF(PrimitivePtr, kPrimCummax, std::make_shared<Primitive>(kCummax));
GVAR_DEF(PrimitivePtr, kPrimMvlgamma, std::make_shared<Primitive>(kMvlgamma));
GVAR_DEF(PrimitivePtr, kPrimMvlgammaGrad, std::make_shared<Primitive>(kMvlgammaGrad));
GVAR_DEF(PrimitivePtr, kPrimRightShift, std::make_shared<Primitive>(kRightShift));
GVAR_DEF(PrimitivePtr, kPrimLogSpace, std::make_shared<Primitive>(kLogSpace));
GVAR_DEF(PrimitivePtr, kPrimTril, std::make_shared<Primitive>(kTril));
GVAR_DEF(PrimitivePtr, kPrimEye, std::make_shared<Primitive>(kEye));
GVAR_DEF(PrimitivePtr, kPrimTriu, std::make_shared<Primitive>(kTriu));
GVAR_DEF(PrimitivePtr, kPrimMeshgrid, std::make_shared<Primitive>(kMeshgrid));
GVAR_DEF(PrimitivePtr, kPrimHammingWindow, std::make_shared<Primitive>("HammingWindow"));
GVAR_DEF(PrimitivePtr, kPrimSegmentMax, std::make_shared<Primitive>(kSegmentMax));
GVAR_DEF(PrimitivePtr, kPrimSegmentMin, std::make_shared<Primitive>(kSegmentMin));
GVAR_DEF(PrimitivePtr, kPrimSegmentSum, std::make_shared<Primitive>(kSegmentSum));
GVAR_DEF(PrimitivePtr, kPrimAffineGrid, std::make_shared<Primitive>(kAffineGrid));
GVAR_DEF(PrimitivePtr, kPrimAffineGridGrad, std::make_shared<Primitive>(kAffineGridGrad));
GVAR_DEF(PrimitivePtr, kPrimSegmentMean, std::make_shared<Primitive>(kSegmentMean));
GVAR_DEF(PrimitivePtr, kPrimSegmentProd, std::make_shared<Primitive>(kSegmentProd));
GVAR_DEF(PrimitivePtr, kPrimSparseSparseMinimum, std::make_shared<Primitive>(kSparseSparseMinimum));
GVAR_DEF(PrimitivePtr, kPrimSparseReshape, std::make_shared<Primitive>(kSparseReshape));
GVAR_DEF(PrimitivePtr, kPrimSparseReorder, std::make_shared<Primitive>(kSparseReorder));
GVAR_DEF(PrimitivePtr, kPrimBincount, std::make_shared<Primitive>(kBincount));

// image
GVAR_DEF(PrimitivePtr, kPrimCropAndResizeGradBoxes, std::make_shared<Primitive>(kCropAndResizeGradBoxes));
GVAR_DEF(PrimitivePtr, kPrimResizeArea, std::make_shared<Primitive>("ResizeArea"));
GVAR_DEF(PrimitivePtr, kPrimResizeBilinearV2, std::make_shared<Primitive>(kResizeBilinearV2));
GVAR_DEF(PrimitivePtr, kPrimResizeBilinearV2Grad, std::make_shared<Primitive>("ResizeBilinearV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimRGBToHSV, std::make_shared<Primitive>("RGBToHSV"));
GVAR_DEF(PrimitivePtr, kPrimResizeBilinearGrad, std::make_shared<Primitive>(kResizeBilinearGrad));
GVAR_DEF(PrimitivePtr, kPrimCropAndResize, std::make_shared<Primitive>(kCropAndResize));
GVAR_DEF(PrimitivePtr, kPrimCropAndResizeGradImage, std::make_shared<Primitive>(kCropAndResizeGradImage));
GVAR_DEF(PrimitivePtr, kPrimResizeV2, std::make_shared<Primitive>(kResizeV2));
GVAR_DEF(PrimitivePtr, kPrimResizeV2Grad, std::make_shared<Primitive>(kResizeV2Grad));

// NN
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPoolWithFixedKsize,
         std::make_shared<Primitive>(kFractionalMaxPoolWithFixedKsize));
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPoolGradWithFixedKsize,
         std::make_shared<Primitive>(kFractionalMaxPoolGradWithFixedKsize));
GVAR_DEF(PrimitivePtr, kPrimCeLU, std::make_shared<Primitive>("CeLU"));
GVAR_DEF(PrimitivePtr, kPrimCeluV2, std::make_shared<Primitive>("CeluV2"));
GVAR_DEF(PrimitivePtr, kPrimAdam, std::make_shared<Primitive>("Adam"));
GVAR_DEF(PrimitivePtr, kPrimAdamWeightDecay, std::make_shared<Primitive>("AdamWeightDecay"));
GVAR_DEF(PrimitivePtr, kPrimAdamNoUpdateParam, std::make_shared<Primitive>("AdamNoUpdateParam"));
GVAR_DEF(PrimitivePtr, kPrimLamb, std::make_shared<Primitive>("Lamb"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdaMax, std::make_shared<Primitive>("ApplyAdaMax"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdaMaxD, std::make_shared<Primitive>("ApplyAdaMaxD"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdam, std::make_shared<Primitive>("ApplyAdam"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdamD, std::make_shared<Primitive>("ApplyAdamD"));
GVAR_DEF(PrimitivePtr, kPrimAudioSpectrogram, std::make_shared<Primitive>("AudioSpectrogram"));
GVAR_DEF(PrimitivePtr, kPrimFlatten, std::make_shared<Primitive>("Flatten"));
GVAR_DEF(PrimitivePtr, kPrimCrop, std::make_shared<Primitive>("Crop"));
GVAR_DEF(PrimitivePtr, kPrimFlattenGrad, std::make_shared<Primitive>("FlattenGrad"));
GVAR_DEF(PrimitivePtr, kPrimSoftmax, std::make_shared<Primitive>("Softmax"));
GVAR_DEF(PrimitivePtr, kPrimSoftmaxV2, std::make_shared<Primitive>("SoftmaxV2"));
GVAR_DEF(PrimitivePtr, kPrimSoftmaxGrad, std::make_shared<Primitive>("SoftmaxGrad"));
GVAR_DEF(PrimitivePtr, kPrimSoftsign, std::make_shared<Primitive>("Softsign"));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmaxCrossEntropy, std::make_shared<Primitive>("SparseSoftmaxCrossEntropy"));
GVAR_DEF(PrimitivePtr, kPrimSoftmaxV2WithDropoutDoMaskV3, std::make_shared<Primitive>("SoftmaxV2WithDropoutDoMaskV3"));
GVAR_DEF(PrimitivePtr, kPrimLogSoftmax, std::make_shared<Primitive>("LogSoftmax"));
GVAR_DEF(PrimitivePtr, kPrimLogSoftmaxV2, std::make_shared<Primitive>("LogSoftmaxV2"));
GVAR_DEF(PrimitivePtr, kPrimLogSoftmaxGrad, std::make_shared<Primitive>("LogSoftmaxGrad"));
GVAR_DEF(PrimitivePtr, kPrimLstm, std::make_shared<Primitive>("LSTM"));
GVAR_DEF(PrimitivePtr, kPrimAttention, std::make_shared<Primitive>("Attention"));
GVAR_DEF(PrimitivePtr, kPrimLstmGrad, std::make_shared<Primitive>("LSTMGrad"));
GVAR_DEF(PrimitivePtr, kPrimLstmGradData, std::make_shared<Primitive>("LSTMGradData"));
GVAR_DEF(PrimitivePtr, kPrimLstmGradWeight, std::make_shared<Primitive>("LSTMGradWeight"));
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
GVAR_DEF(PrimitivePtr, kPrimApplyGradientDescent, std::make_shared<Primitive>("ApplyGradientDescent"));
GVAR_DEF(PrimitivePtr, kPrimApplyPowerSign, std::make_shared<Primitive>("ApplyPowerSign"));
GVAR_DEF(PrimitivePtr, kPrimApplyPowerSignD, std::make_shared<Primitive>("ApplyPowerSignD"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveAvgPool3D, std::make_shared<Primitive>("AdaptiveAvgPool3D"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveAvgPool3DGrad, std::make_shared<Primitive>("AdaptiveAvgPool3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveAvgPool2D, std::make_shared<Primitive>("AdaptiveAvgPool2D"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveAvgPool2DGrad, std::make_shared<Primitive>("AdaptiveAvgPool2DGrad"));
GVAR_DEF(PrimitivePtr, kPrimBesselI0e, std::make_shared<Primitive>("BesselI0e"));
GVAR_DEF(PrimitivePtr, kPrimBesselI1e, std::make_shared<Primitive>("BesselI1e"));
GVAR_DEF(PrimitivePtr, kPrimBesselJ0, std::make_shared<Primitive>("BesselJ0"));
GVAR_DEF(PrimitivePtr, kPrimBesselJ1, std::make_shared<Primitive>("BesselJ1"));
GVAR_DEF(PrimitivePtr, kPrimBesselY0, std::make_shared<Primitive>("BesselY0"));
GVAR_DEF(PrimitivePtr, kPrimBesselY1, std::make_shared<Primitive>("BesselY1"));
GVAR_DEF(PrimitivePtr, kPrimTanhGrad, std::make_shared<Primitive>("TanhGrad"));
GVAR_DEF(PrimitivePtr, kPrimPooling, std::make_shared<Primitive>("Pooling"));
GVAR_DEF(PrimitivePtr, kPrimPoolingGrad, std::make_shared<Primitive>("PoolingGrad"));
GVAR_DEF(PrimitivePtr, kPrimPSROIPooling, std::make_shared<Primitive>("PSROIPooling"));
GVAR_DEF(PrimitivePtr, kPrimPSROIPoolingGrad, std::make_shared<Primitive>("PSROIPoolingGrad"));
GVAR_DEF(PrimitivePtr, kPrimPSROIPoolingV2, std::make_shared<Primitive>("PSROIPoolingV2"));
GVAR_DEF(PrimitivePtr, kPrimPSROIPoolingGradV2D, std::make_shared<Primitive>("PSROIPoolingGradV2D"));
GVAR_DEF(PrimitivePtr, kPrimROIPooling, std::make_shared<Primitive>("ROIPooling"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool, std::make_shared<Primitive>("MaxPool"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool3D, std::make_shared<Primitive>("MaxPool3D"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGrad, std::make_shared<Primitive>("MaxPoolGrad"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool3DGrad, std::make_shared<Primitive>("MaxPool3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolV1, std::make_shared<Primitive>("MaxPoolV1"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradV1, std::make_shared<Primitive>("MaxPoolGradV1"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradGrad, std::make_shared<Primitive>("MaxPoolGradGrad"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool3DGradGrad, std::make_shared<Primitive>("MaxPool3DGradGrad"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolWithArgmax, std::make_shared<Primitive>("MaxPoolWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradWithArgmax, std::make_shared<Primitive>("MaxPoolGradWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradWithArgmaxV2, std::make_shared<Primitive>("MaxPoolGradWithArgmaxV2"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolGradGradWithArgmax, std::make_shared<Primitive>("MaxPoolGradGradWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolWithArgmaxV2, std::make_shared<Primitive>("MaxPoolWithArgmaxV2"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool3DWithArgmax, std::make_shared<Primitive>("MaxPool3DWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxPool3DGradWithArgmax, std::make_shared<Primitive>("MaxPool3DGradWithArgmax"));
GVAR_DEF(PrimitivePtr, kPrimMaxUnpool2D, std::make_shared<Primitive>(kMaxUnpool2D));
GVAR_DEF(PrimitivePtr, kPrimMaxUnpool2DGrad, std::make_shared<Primitive>(kMaxUnpool2DGrad));
GVAR_DEF(PrimitivePtr, kPrimMaxUnpool3D, std::make_shared<Primitive>(kMaxUnpool3D));
GVAR_DEF(PrimitivePtr, kPrimMaxUnpool3DGrad, std::make_shared<Primitive>(kMaxUnpool3DGrad));
GVAR_DEF(PrimitivePtr, kPrimApplyCenteredRMSProp, std::make_shared<Primitive>("ApplyCenteredRMSProp"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool, std::make_shared<Primitive>("AvgPool"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool3D, std::make_shared<Primitive>("AvgPool3D"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool3DD, std::make_shared<Primitive>("AvgPool3DD"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolGrad, std::make_shared<Primitive>("AvgPoolGrad"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool3DGrad, std::make_shared<Primitive>("AvgPool3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimAvgPool3DGradD, std::make_shared<Primitive>("AvgPool3DGradD"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolGradVm, std::make_shared<Primitive>("AvgPoolGradVm"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolGradGe, std::make_shared<Primitive>("AvgPoolGradGe"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolV1, std::make_shared<Primitive>("AvgPoolV1"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolGradV1, std::make_shared<Primitive>("AvgPoolGradV1"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveMaxPool2D, std::make_shared<Primitive>(kAdaptiveMaxPool2D));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveMaxPool2d, std::make_shared<Primitive>("AdaptiveMaxPool2d"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveMaxPool3D, std::make_shared<Primitive>(kAdaptiveMaxPool3D));
GVAR_DEF(PrimitivePtr, kPrimFusedSparseAdam, std::make_shared<Primitive>("FusedSparseAdam"));
GVAR_DEF(PrimitivePtr, kPrimFusedSparseFtrl, std::make_shared<Primitive>("FusedSparseFtrl"));
GVAR_DEF(PrimitivePtr, kPrimFusedSparseLazyAdam, std::make_shared<Primitive>("FusedSparseLazyAdam"));
GVAR_DEF(PrimitivePtr, kPrimFusedSparseProximalAdagrad, std::make_shared<Primitive>("FusedSparseProximalAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimFusedBatchNorm, std::make_shared<Primitive>("FusedBatchNorm"));
GVAR_DEF(PrimitivePtr, kPrimMatrixExp, std::make_shared<Primitive>("MatrixExp"));
GVAR_DEF(PrimitivePtr, kPrimMultiMarginLoss, std::make_shared<Primitive>(kMultiMarginLoss));
GVAR_DEF(PrimitivePtr, kPrimMultiMarginLossGrad, std::make_shared<Primitive>(kMultiMarginLossGrad));
GVAR_DEF(PrimitivePtr, kPrimConv2D, std::make_shared<Primitive>("Conv2D"));
GVAR_DEF(PrimitivePtr, kPrimConv3D, std::make_shared<Primitive>("Conv3D"));
GVAR_DEF(PrimitivePtr, kPrimMultilabelMarginLoss, std::make_shared<Primitive>(kMultilabelMarginLoss));
GVAR_DEF(PrimitivePtr, kPrimMultilabelMarginLossGrad, std::make_shared<Primitive>(kMultilabelMarginLossGrad));
GVAR_DEF(PrimitivePtr, kPrimCTCLossV2, std::make_shared<Primitive>("CTCLossV2"));
GVAR_DEF(PrimitivePtr, kPrimCTCLossV2Grad, std::make_shared<Primitive>("CTCLossV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimCTCLoss, std::make_shared<Primitive>(kCTCLoss));
GVAR_DEF(PrimitivePtr, kPrimNLLLoss, std::make_shared<Primitive>(kNLLLoss));
GVAR_DEF(PrimitivePtr, kPrimNLLLossGrad, std::make_shared<Primitive>(kNLLLossGrad));
GVAR_DEF(PrimitivePtr, kPrimFullConnection, std::make_shared<Primitive>("FullConnection"));
GVAR_DEF(PrimitivePtr, kPrimConv2DTranspose, std::make_shared<Primitive>(kConv2DTranspose));
GVAR_DEF(PrimitivePtr, kPrimConv3DTranspose, std::make_shared<Primitive>("Conv3DTranspose"));
GVAR_DEF(PrimitivePtr, kPrimTripletMarginLoss, std::make_shared<Primitive>(kTripletMarginLoss));
GVAR_DEF(PrimitivePtr, kPrimRoll, std::make_shared<Primitive>(kRoll));
GVAR_DEF(PrimitivePtr, kPrimGroupConv2DGradInput, std::make_shared<Primitive>("GroupConv2DGradInput"));
GVAR_DEF(PrimitivePtr, kPrimBatchNorm, std::make_shared<Primitive>("BatchNorm"));
GVAR_DEF(PrimitivePtr, kPrimBatchNormWithActivation, std::make_shared<Primitive>("BatchNormWithActivation"));
GVAR_DEF(PrimitivePtr, kPrimBatchNormWithAddAndActivation,
         std::make_shared<Primitive>("BatchNormWithAddAndActivation"));
GVAR_DEF(PrimitivePtr, kPrimBNInfer, std::make_shared<Primitive>("BNInfer"));
GVAR_DEF(PrimitivePtr, kPrimBNInferGrad, std::make_shared<Primitive>("BNInferGrad"));
GVAR_DEF(PrimitivePtr, kPrimBatchNormGrad, std::make_shared<Primitive>("BatchNormGrad"));
GVAR_DEF(PrimitivePtr, kPrimBatchNormGradWithActivation, std::make_shared<Primitive>("BatchNormGradWithActivation"));
GVAR_DEF(PrimitivePtr, kPrimBatchNormGradWithAddAndActivation,
         std::make_shared<Primitive>("BatchNormGradWithAddAndActivation"));
GVAR_DEF(PrimitivePtr, kPrimBatchNormGradGrad, std::make_shared<Primitive>("BatchNormGradGrad"));
GVAR_DEF(PrimitivePtr, kPrimInstanceNorm, std::make_shared<Primitive>("InstanceNorm"));
GVAR_DEF(PrimitivePtr, kPrimInstanceNormGrad, std::make_shared<Primitive>("InstanceNormGrad"));
GVAR_DEF(PrimitivePtr, kPrimInstanceNormV2, std::make_shared<Primitive>("InstanceNormV2"));
GVAR_DEF(PrimitivePtr, kPrimInstanceNormV2Grad, std::make_shared<Primitive>("InstanceNormV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimSyncBatchNorm, std::make_shared<Primitive>("SyncBatchNorm"));
GVAR_DEF(PrimitivePtr, kPrimSyncBatchNormGrad, std::make_shared<Primitive>("SyncBatchNormGrad"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingReduce, std::make_shared<Primitive>("BNTrainingReduce"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingReduceGrad, std::make_shared<Primitive>("BNTrainingReduceGrad"));
GVAR_DEF(PrimitivePtr, kPrimReluGrad, std::make_shared<Primitive>(kReLUGrad));
GVAR_DEF(PrimitivePtr, kPrimReluGradV2, std::make_shared<Primitive>("ReluGradV2"));
GVAR_DEF(PrimitivePtr, kPrimReLU6Grad, std::make_shared<Primitive>("ReLU6Grad"));
GVAR_DEF(PrimitivePtr, kPrimRelu6Grad, std::make_shared<Primitive>("Relu6Grad"));
GVAR_DEF(PrimitivePtr, kPrimConv2DBackpropInput, std::make_shared<Primitive>("Conv2DBackpropInput"));
GVAR_DEF(PrimitivePtr, kPrimConv2DBackpropInputD, std::make_shared<Primitive>("Conv2DBackpropInputD"));
GVAR_DEF(PrimitivePtr, kPrimConv2DBackpropFilter, std::make_shared<Primitive>("Conv2DBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimConv2DBackpropFilterD, std::make_shared<Primitive>("Conv2DBackpropFilterD"));
GVAR_DEF(PrimitivePtr, kPrimConv3DBackpropInput, std::make_shared<Primitive>("Conv3DBackpropInput"));
GVAR_DEF(PrimitivePtr, kPrimConv3DBackpropFilter, std::make_shared<Primitive>("Conv3DBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimDilation2D, std::make_shared<Primitive>("Dilation2D"));
GVAR_DEF(PrimitivePtr, kPrimDilation2DBackpropInput, std::make_shared<Primitive>("Dilation2DBackpropInput"));
GVAR_DEF(PrimitivePtr, kPrimDilation2DBackpropFilter, std::make_shared<Primitive>("Dilation2DBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimDeformableOffsetsGrad, std::make_shared<Primitive>("DeformableOffsetsGrad"));
GVAR_DEF(PrimitivePtr, kPrimCustomNormalize, std::make_shared<Primitive>("CustomNormalize"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNative, std::make_shared<Primitive>("DepthwiseConv2dNative"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNativeD, std::make_shared<Primitive>("DepthwiseConv2dNativeD"));
GVAR_DEF(PrimitivePtr, kPrimCTCGreedyDecoder, std::make_shared<Primitive>("CTCGreedyDecoder"));
GVAR_DEF(PrimitivePtr, kPrimDataFormatDimMap, std::make_shared<Primitive>("DataFormatDimMap"));
GVAR_DEF(PrimitivePtr, kPrimDataFormatVecPermute, std::make_shared<Primitive>("DataFormatVecPermute"));
GVAR_DEF(PrimitivePtr, kPrimDynamicStitch, std::make_shared<Primitive>("DynamicStitch"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNativeBackpropFilter,
         std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropFilter"));
GVAR_DEF(PrimitivePtr, kPrimDepthwiseConv2dNativeBackpropInput,
         std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropInput"));
GVAR_DEF(PrimitivePtr, kPrimDetectionPostProcess, std::make_shared<Primitive>("DetectionPostProcess"));
GVAR_DEF(PrimitivePtr, kPrimBiasAddGrad, std::make_shared<Primitive>(kBiasAddGrad));
GVAR_DEF(PrimitivePtr, kPrimBiasAdd, std::make_shared<Primitive>(kBiasAdd));
GVAR_DEF(PrimitivePtr, kPrimBiasSubGrad, std::make_shared<Primitive>("BiasSubGrad"));
GVAR_DEF(PrimitivePtr, kPrimBinaryCrossEntropy, std::make_shared<Primitive>("BinaryCrossEntropy"));
GVAR_DEF(PrimitivePtr, kPrimBinaryCrossEntropyGrad, std::make_shared<Primitive>("BinaryCrossEntropyGrad"));
GVAR_DEF(PrimitivePtr, kPrimSmoothL1Loss, std::make_shared<Primitive>("SmoothL1Loss"));
GVAR_DEF(PrimitivePtr, kPrimSmoothL1LossV2, std::make_shared<Primitive>("SmoothL1LossV2"));
GVAR_DEF(PrimitivePtr, kPrimSmoothL1LossGrad, std::make_shared<Primitive>("SmoothL1LossGrad"));
GVAR_DEF(PrimitivePtr, kPrimSmoothL1LossGradV2, std::make_shared<Primitive>("SmoothL1LossGradV2"));
GVAR_DEF(PrimitivePtr, kPrimSoftMarginLoss, std::make_shared<Primitive>("SoftMarginLoss"));
GVAR_DEF(PrimitivePtr, kPrimSoftMarginLossGrad, std::make_shared<Primitive>("SoftMarginLossGrad"));
GVAR_DEF(PrimitivePtr, kPrimSoftmaxCrossEntropyWithLogits,
         std::make_shared<Primitive>("SoftmaxCrossEntropyWithLogits"));
GVAR_DEF(PrimitivePtr, kPrimL2Loss, std::make_shared<Primitive>("L2Loss"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidCrossEntropyWithLogits,
         std::make_shared<Primitive>("SigmoidCrossEntropyWithLogits"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidCrossEntropyWithLogitsV2,
         std::make_shared<Primitive>("SigmoidCrossEntropyWithLogitsV2"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidCrossEntropyWithLogitsGrad,
         std::make_shared<Primitive>("SigmoidCrossEntropyWithLogitsGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmaxCrossEntropyWithLogits,
         std::make_shared<Primitive>("SparseSoftmaxCrossEntropyWithLogits"));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmaxCrossEntropyWithLogitsV2,
         std::make_shared<Primitive>("SparseSoftmaxCrossEntropyWithLogitsV2"));
GVAR_DEF(PrimitivePtr, kPrimMomentum, std::make_shared<Primitive>("Momentum"));
GVAR_DEF(PrimitivePtr, kPrimApplyMomentum, std::make_shared<Primitive>("ApplyMomentum"));
GVAR_DEF(PrimitivePtr, kPrimApplyMomentumD, std::make_shared<Primitive>("ApplyMomentumD"));
GVAR_DEF(PrimitivePtr, kPrimApplyFtrl, std::make_shared<Primitive>("ApplyFtrl"));
GVAR_DEF(PrimitivePtr, kPrimApplyFtrlD, std::make_shared<Primitive>("ApplyFtrlD"));
GVAR_DEF(PrimitivePtr, kPrimLrn, std::make_shared<Primitive>(kLRN));
GVAR_DEF(PrimitivePtr, kPrimLrnGrad, std::make_shared<Primitive>("LRNGrad"));
GVAR_DEF(PrimitivePtr, kPrimLayerNorm, std::make_shared<Primitive>(kLayerNorm));
GVAR_DEF(PrimitivePtr, kPrimLayerNormGrad, std::make_shared<Primitive>(kLayerNormGrad));
GVAR_DEF(PrimitivePtr, kPrimLayerNormGradGrad, std::make_shared<Primitive>("LayerNormGradGrad"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormXBackprop, std::make_shared<Primitive>("LayerNormXBackprop"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormXBackpropV2, std::make_shared<Primitive>("LayerNormXBackpropV2"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormBetaGammaBackprop, std::make_shared<Primitive>("LayerNormBetaGammaBackprop"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormBetaGammaBackpropV2, std::make_shared<Primitive>("LayerNormBetaGammaBackpropV2"));
GVAR_DEF(PrimitivePtr, kPrimLog1p, std::make_shared<Primitive>("Log1p"));
GVAR_DEF(PrimitivePtr, kPrimDropoutGenMask, std::make_shared<Primitive>(kDropoutGenMask));
GVAR_DEF(PrimitivePtr, kPrimDropoutGenMaskV3, std::make_shared<Primitive>(kDropoutGenMaskV3));
GVAR_DEF(PrimitivePtr, kPrimStatelessDropOutGenMask, std::make_shared<Primitive>(kStatelessDropOutGenMask));
GVAR_DEF(PrimitivePtr, kPrimDropoutDoMask, std::make_shared<Primitive>(kDropoutDoMask));
GVAR_DEF(PrimitivePtr, kPrimDropOutDoMask, std::make_shared<Primitive>("DropOutDoMask"));
GVAR_DEF(PrimitivePtr, kPrimDropoutDoMaskV3, std::make_shared<Primitive>(kDropoutDoMaskV3));
GVAR_DEF(PrimitivePtr, kPrimDropOutDoMaskV3, std::make_shared<Primitive>("DropOutDoMaskV3"));
GVAR_DEF(PrimitivePtr, kPrimDropOutDoMaskV3D, std::make_shared<Primitive>("DropOutDoMaskV3D"));
GVAR_DEF(PrimitivePtr, kPrimDropoutGrad, std::make_shared<Primitive>(kDropoutGrad));
GVAR_DEF(PrimitivePtr, kPrimDropout, std::make_shared<Primitive>(kDropout));
GVAR_DEF(PrimitivePtr, kPrimDropout2D, std::make_shared<Primitive>(kDropout2D));
GVAR_DEF(PrimitivePtr, kPrimDropout3D, std::make_shared<Primitive>(kDropout3D));
GVAR_DEF(PrimitivePtr, kPrimUniformInt, std::make_shared<Primitive>("UniformInt"));
GVAR_DEF(PrimitivePtr, kPrimUniformReal, std::make_shared<Primitive>("UniformReal"));
GVAR_DEF(PrimitivePtr, kPrimCudnnUniformReal, std::make_shared<Primitive>("CudnnUniformReal"));
GVAR_DEF(PrimitivePtr, kPrimOneHot, std::make_shared<Primitive>("OneHot"));
GVAR_DEF(PrimitivePtr, kPrimOneHotD, std::make_shared<Primitive>("OneHotD"));
GVAR_DEF(PrimitivePtr, kPrimGeLU, std::make_shared<Primitive>(kGeLU));
GVAR_DEF(PrimitivePtr, kPrimGelu, std::make_shared<Primitive>("Gelu"));
GVAR_DEF(PrimitivePtr, kPrimGeLUGrad, std::make_shared<Primitive>(kGeLUGrad));
GVAR_DEF(PrimitivePtr, kPrimGeluGrad, std::make_shared<Primitive>("GeluGrad"));
GVAR_DEF(PrimitivePtr, kPrimFastGeLU, std::make_shared<Primitive>(kFastGeLU));
GVAR_DEF(PrimitivePtr, kPrimFastGelu, std::make_shared<Primitive>("FastGelu"));
GVAR_DEF(PrimitivePtr, kPrimFastGeLUGrad, std::make_shared<Primitive>(kFastGeLUGrad));
GVAR_DEF(PrimitivePtr, kPrimFastGeluGrad, std::make_shared<Primitive>("FastGeluGrad"));
GVAR_DEF(PrimitivePtr, kPrimReLU, std::make_shared<Primitive>(kReLU));
GVAR_DEF(PrimitivePtr, kPrimRelu, std::make_shared<Primitive>("Relu"));
GVAR_DEF(PrimitivePtr, kPrimElu, std::make_shared<Primitive>("Elu"));
GVAR_DEF(PrimitivePtr, kPrimEluGrad, std::make_shared<Primitive>("EluGrad"));
GVAR_DEF(PrimitivePtr, kPrimReLU6, std::make_shared<Primitive>(kReLU6));
GVAR_DEF(PrimitivePtr, kPrimReLUV2, std::make_shared<Primitive>(kReLUV2));
GVAR_DEF(PrimitivePtr, kPrimReluV2, std::make_shared<Primitive>("ReluV2"));
GVAR_DEF(PrimitivePtr, kPrimReLUV3, std::make_shared<Primitive>(kReLUV3));
GVAR_DEF(PrimitivePtr, kPrimPReLU, std::make_shared<Primitive>("PReLU"));
GVAR_DEF(PrimitivePtr, kPrimPRelu, std::make_shared<Primitive>("PRelu"));
GVAR_DEF(PrimitivePtr, kPrimPReLUGrad, std::make_shared<Primitive>("PReLUGrad"));
GVAR_DEF(PrimitivePtr, kPrimGLU, std::make_shared<Primitive>(kGLU));
GVAR_DEF(PrimitivePtr, kPrimGluGrad, std::make_shared<Primitive>(kGluGrad));
GVAR_DEF(PrimitivePtr, kPrimSeLU, std::make_shared<Primitive>("SeLU"));
GVAR_DEF(PrimitivePtr, kPrimSelu, std::make_shared<Primitive>("Selu"));
GVAR_DEF(PrimitivePtr, kPrimSiLU, std::make_shared<Primitive>("SiLU"));
GVAR_DEF(PrimitivePtr, kPrimSiLUGrad, std::make_shared<Primitive>("SiLUGrad"));
GVAR_DEF(PrimitivePtr, kPrimSoftplus, std::make_shared<Primitive>("Softplus"));
GVAR_DEF(PrimitivePtr, kPrimQuantile, std::make_shared<Primitive>("Quantile"));
GVAR_DEF(PrimitivePtr, kPrimSoftplusGrad, std::make_shared<Primitive>("SoftplusGrad"));
GVAR_DEF(PrimitivePtr, kPrimZeros, std::make_shared<Primitive>("Zeros"));
GVAR_DEF(PrimitivePtr, kPrimZerosLike, std::make_shared<Primitive>(kZerosLike));
GVAR_DEF(PrimitivePtr, kPrimOnes, std::make_shared<Primitive>(kOnes));
GVAR_DEF(PrimitivePtr, kPrimOnesLike, std::make_shared<Primitive>(kOnesLike));
GVAR_DEF(PrimitivePtr, kPrimBpropCut, std::make_shared<Primitive>("bprop_cut"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantPerLayer, std::make_shared<Primitive>("FakeQuantPerLayer"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantPerChannel, std::make_shared<Primitive>("FakeQuantPerChannel"));
GVAR_DEF(PrimitivePtr, kPrimFakeLearnedScaleQuantPerLayer,
         std::make_shared<Primitive>("FakeLearnedScaleQuantPerLayer"));
GVAR_DEF(PrimitivePtr, kPrimFakeLearnedScaleQuantPerChannel,
         std::make_shared<Primitive>("FakeLearnedScaleQuantPerChannel"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantWithMinMaxVars, std::make_shared<Primitive>("FakeQuantWithMinMaxVars"));
GVAR_DEF(PrimitivePtr, kPrimApplyRMSProp, std::make_shared<Primitive>(kApplyRMSProp));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyFtrl, std::make_shared<Primitive>("SparseApplyFtrl"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyFtrlD, std::make_shared<Primitive>("SparseApplyFtrlD"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyFtrlV2D, std::make_shared<Primitive>("SparseApplyFtrlV2D"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyProximalAdagrad, std::make_shared<Primitive>("SparseApplyProximalAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyProximalAdagradD, std::make_shared<Primitive>("SparseApplyProximalAdagradD"));
GVAR_DEF(PrimitivePtr, kPrimFusedAdam, std::make_shared<Primitive>("FusedAdam"));
GVAR_DEF(PrimitivePtr, kPrimFusedAdaFactor, std::make_shared<Primitive>("FusedAdaFactor"));
GVAR_DEF(PrimitivePtr, kPrimFusedAdaFactorWithGlobalNorm, std::make_shared<Primitive>("FusedAdaFactorWithGlobalNorm"));
GVAR_DEF(PrimitivePtr, kPrimFusedAdamWeightDecay, std::make_shared<Primitive>("FusedAdamWeightDecay"));
GVAR_DEF(PrimitivePtr, kPrimSGD, std::make_shared<Primitive>("SGD"));
GVAR_DEF(PrimitivePtr, kPrimBCEWithLogitsLoss, std::make_shared<Primitive>("BCEWithLogitsLoss"));
GVAR_DEF(PrimitivePtr, kPrimClipByNorm, std::make_shared<Primitive>("ClipByNorm"));
GVAR_DEF(PrimitivePtr, kPrimClipByNormNoDivSum, std::make_shared<Primitive>("ClipByNormNoDivSum"));
GVAR_DEF(PrimitivePtr, kPrimTensorMove, std::make_shared<Primitive>("TensorMove"));
GVAR_DEF(PrimitivePtr, kPrimL2Normalize, std::make_shared<Primitive>("L2Normalize"));
GVAR_DEF(PrimitivePtr, kPrimL2NormalizeGrad, std::make_shared<Primitive>("L2NormalizeGrad"));
GVAR_DEF(PrimitivePtr, kPrimCustomExtractFeatures, std::make_shared<Primitive>("CustomExtractFeatures"));
GVAR_DEF(PrimitivePtr, kLambApplyOptimizerAssign, std::make_shared<Primitive>("LambApplyOptimizerAssign"));
GVAR_DEF(PrimitivePtr, kLambApplyWeightAssign, std::make_shared<Primitive>("LambApplyWeightAssign"));
GVAR_DEF(PrimitivePtr, kSoftmaxGradExt, std::make_shared<Primitive>("SoftmaxGradExt"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyAdadelta, std::make_shared<Primitive>(kSparseApplyAdadelta));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyCenteredRMSProp, std::make_shared<Primitive>(kSparseApplyCenteredRMSProp));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyAdagrad, std::make_shared<Primitive>("SparseApplyAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyAdagradV2, std::make_shared<Primitive>("SparseApplyAdagradV2"));
GVAR_DEF(PrimitivePtr, kSquareSumV1, std::make_shared<Primitive>("SquareSumV1"));
GVAR_DEF(PrimitivePtr, kFusedMulAdd, std::make_shared<Primitive>("FusedMulAdd"));
GVAR_DEF(PrimitivePtr, kPrimSoftShrink, std::make_shared<Primitive>("SoftShrink"));
GVAR_DEF(PrimitivePtr, kPrimSoftShrinkGrad, std::make_shared<Primitive>("SoftShrinkGrad"));
GVAR_DEF(PrimitivePtr, kPrimHShrink, std::make_shared<Primitive>("HShrink"));
GVAR_DEF(PrimitivePtr, kPrimHShrinkGrad, std::make_shared<Primitive>("HShrinkGrad"));
GVAR_DEF(PrimitivePtr, kPrimHSwish, std::make_shared<Primitive>(kHSwish));
GVAR_DEF(PrimitivePtr, kPrimHardSwish, std::make_shared<Primitive>("HardSwish"));
GVAR_DEF(PrimitivePtr, kPrimHardSwishGrad, std::make_shared<Primitive>("HardSwishGrad"));
GVAR_DEF(PrimitivePtr, kPrimHSwishGrad, std::make_shared<Primitive>(kHSwishGrad));
GVAR_DEF(PrimitivePtr, kPrimHSVToRGB, std::make_shared<Primitive>("HSVToRGB"));
GVAR_DEF(PrimitivePtr, kPrimDeformableOffsets, std::make_shared<Primitive>("DeformableOffsets"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagradDA, std::make_shared<Primitive>("ApplyAdagradDA"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagradV2, std::make_shared<Primitive>("ApplyAdagradV2"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagradV2D, std::make_shared<Primitive>("ApplyAdagradV2D"));
GVAR_DEF(PrimitivePtr, kPrimApplyProximalGradientDescent, std::make_shared<Primitive>("ApplyProximalGradientDescent"));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyRMSProp, std::make_shared<Primitive>("SparseApplyRMSProp"));
GVAR_DEF(PrimitivePtr, kPrimApplyKerasMomentum, std::make_shared<Primitive>("ApplyKerasMomentum"));
GVAR_DEF(PrimitivePtr, kPrimLARSUpdate, std::make_shared<Primitive>("LARSUpdate"));
GVAR_DEF(PrimitivePtr, kPrimLarsV2Update, std::make_shared<Primitive>("LarsV2Update"));
GVAR_DEF(PrimitivePtr, kPrimBoundingBoxDecode, std::make_shared<Primitive>("BoundingBoxDecode"));
GVAR_DEF(PrimitivePtr, kPrimBoundingBoxEncode, std::make_shared<Primitive>("BoundingBoxEncode"));
GVAR_DEF(PrimitivePtr, kPrimROIAlign, std::make_shared<Primitive>("ROIAlign"));
GVAR_DEF(PrimitivePtr, kPrimROIAlignGrad, std::make_shared<Primitive>("ROIAlignGrad"));
GVAR_DEF(PrimitivePtr, kPrimApplyAddSign, std::make_shared<Primitive>("ApplyAddSign"));
GVAR_DEF(PrimitivePtr, kPrimApplyAddSignD, std::make_shared<Primitive>("ApplyAddSignD"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagrad, std::make_shared<Primitive>("ApplyAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdagradD, std::make_shared<Primitive>("ApplyAdagradD"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdadelta, std::make_shared<Primitive>("ApplyAdadelta"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdadeltaD, std::make_shared<Primitive>("ApplyAdadeltaD"));
GVAR_DEF(PrimitivePtr, kPrimApplyAdamWithAmsgrad, std::make_shared<Primitive>("ApplyAdamWithAmsgrad"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingUpdate, std::make_shared<Primitive>("BNTrainingUpdate"));
GVAR_DEF(PrimitivePtr, kPrimBNTrainingUpdateGrad, std::make_shared<Primitive>("BNTrainingUpdateGrad"));
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPool, std::make_shared<Primitive>("FractionalMaxPool"));
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPoolGrad, std::make_shared<Primitive>("FractionalMaxPoolGrad"));
GVAR_DEF(PrimitivePtr, kPrimMish, std::make_shared<Primitive>(kMish));
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPool3DWithFixedKsize,
         std::make_shared<Primitive>("FractionalMaxPool3DWithFixedKsize"));
GVAR_DEF(PrimitivePtr, kPrimFractionalMaxPool3DGradWithFixedKsize,
         std::make_shared<Primitive>("FractionalMaxPool3DGradWithFixedKsize"));
GVAR_DEF(PrimitivePtr, kPrimFractionalAvgPool, std::make_shared<Primitive>("FractionalAvgPool"));
GVAR_DEF(PrimitivePtr, kPrimFractionalAvgPoolGrad, std::make_shared<Primitive>("FractionalAvgPoolGrad"));
GVAR_DEF(PrimitivePtr, kPrimNthElement, std::make_shared<Primitive>("NthElement"));
GVAR_DEF(PrimitivePtr, kPrimPadV3, std::make_shared<Primitive>(kPadV3));
GVAR_DEF(PrimitivePtr, kPrimPadV3Grad, std::make_shared<Primitive>(kPadV3Grad));
GVAR_DEF(PrimitivePtr, kPrimMirrorPadGrad, std::make_shared<Primitive>(kMirrorPadGrad));
GVAR_DEF(PrimitivePtr, kPrimGridSampler2D, std::make_shared<Primitive>(kGridSampler2D));
GVAR_DEF(PrimitivePtr, kPrimGridSampler2DGrad, std::make_shared<Primitive>(kGridSampler2DGrad));
GVAR_DEF(PrimitivePtr, kPrimGridSampler3D, std::make_shared<Primitive>(kGridSampler3D));
GVAR_DEF(PrimitivePtr, kPrimGridSampler3DGrad, std::make_shared<Primitive>(kGridSampler3DGrad));
GVAR_DEF(PrimitivePtr, kPrimPdist, std::make_shared<Primitive>("Pdist"));
GVAR_DEF(PrimitivePtr, kPrimPdistGrad, std::make_shared<Primitive>("PdistGrad"));
GVAR_DEF(PrimitivePtr, kPrimRenorm, std::make_shared<Primitive>(kRenorm));
GVAR_DEF(PrimitivePtr, kPrimUpsampleTrilinear3D, std::make_shared<Primitive>("UpsampleTrilinear3D"));
GVAR_DEF(PrimitivePtr, kPrimNuclearNorm, std::make_shared<Primitive>(kNuclearNorm));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyAdagradDA, std::make_shared<Primitive>(kSparseApplyAdagradDA));
GVAR_DEF(PrimitivePtr, kPrimBiasDropoutAdd, std::make_shared<Primitive>("BiasDropoutAdd"));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveMaxPool2DGrad, std::make_shared<Primitive>("AdaptiveMaxPool2DGrad"));
GVAR_DEF(PrimitivePtr, kPrimUpsampleNearest3D, std::make_shared<Primitive>("UpsampleNearest3D"));
GVAR_DEF(PrimitivePtr, kPrimUpsampleNearest3DGrad, std::make_shared<Primitive>("UpsampleNearest3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimUpsampleTrilinear3DGrad, std::make_shared<Primitive>("UpsampleTrilinear3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimIFMR, std::make_shared<Primitive>(kIFMR));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyMomentum, std::make_shared<Primitive>(kSparseApplyMomentum));
GVAR_DEF(PrimitivePtr, kPrimSparseApplyProximalGradientDescent,
         std::make_shared<Primitive>(kSparseApplyProximalGradientDescent));
GVAR_DEF(PrimitivePtr, kPrimAdaptiveMaxPool3DGrad, std::make_shared<Primitive>("AdaptiveMaxPool3DGrad"));
GVAR_DEF(PrimitivePtr, kPrimChannelShuffle, std::make_shared<Primitive>(kChannelShuffle));

// Comm ops
GVAR_DEF(PrimitivePtr, kPrimMirror, std::make_shared<Primitive>("_MirrorOperator"));
GVAR_DEF(PrimitivePtr, kPrimMirrorMiniStep, std::make_shared<Primitive>("_MirrorMiniStepOperator"));
GVAR_DEF(PrimitivePtr, kPrimMiniStepAllGather, std::make_shared<Primitive>("_MiniStepAllGather"));
GVAR_DEF(PrimitivePtr, kPrimMicroStepAllGather, std::make_shared<Primitive>("_MicroStepAllGather"));
GVAR_DEF(PrimitivePtr, kPrimVirtualDiv, std::make_shared<Primitive>("_VirtualDiv"));
GVAR_DEF(PrimitivePtr, kPrimVirtualAdd, std::make_shared<Primitive>("_VirtualAdd"));
GVAR_DEF(PrimitivePtr, kPrimVirtualDataset, std::make_shared<Primitive>("_VirtualDataset"));
GVAR_DEF(PrimitivePtr, kPrimVirtualOutput, std::make_shared<Primitive>("_VirtualOutput"));
GVAR_DEF(PrimitivePtr, kPrimSend, std::make_shared<Primitive>("Send"));
GVAR_DEF(PrimitivePtr, kPrimReceive, std::make_shared<Primitive>("Receive"));
GVAR_DEF(PrimitivePtr, kPrimRpcSend, std::make_shared<Primitive>("RpcSend"));
GVAR_DEF(PrimitivePtr, kPrimRpcRecv, std::make_shared<Primitive>("RpcRecv"));
GVAR_DEF(PrimitivePtr, kPrimAllReduce, std::make_shared<Primitive>("AllReduce"));
GVAR_DEF(PrimitivePtr, kPrimNeighborExchange, std::make_shared<Primitive>("NeighborExchange"));
GVAR_DEF(PrimitivePtr, kPrimNeighborExchangeV2, std::make_shared<Primitive>("NeighborExchangeV2"));
GVAR_DEF(PrimitivePtr, kPrimNeighborExchangeV2Grad, std::make_shared<Primitive>("NeighborExchangeV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimAllToAll, std::make_shared<Primitive>("AlltoAll"));
GVAR_DEF(PrimitivePtr, kPrimAllToAllv, std::make_shared<Primitive>("AllToAllv"));
GVAR_DEF(PrimitivePtr, kPrimAllSwap, std::make_shared<Primitive>("_AllSwap"));
GVAR_DEF(PrimitivePtr, kPrimBroadcast, std::make_shared<Primitive>("Broadcast"));
GVAR_DEF(PrimitivePtr, kPrimAllGather, std::make_shared<Primitive>("AllGather"));
GVAR_DEF(PrimitivePtr, kPrimReduceScatter, std::make_shared<Primitive>("ReduceScatter"));
GVAR_DEF(PrimitivePtr, kPrimMemCpyAsync, std::make_shared<Primitive>("memcpy_async"));
GVAR_DEF(PrimitivePtr, kPrimFill, std::make_shared<Primitive>("Fill"));
GVAR_DEF(PrimitivePtr, kPrimFusedPushWeight, std::make_shared<Primitive>("FusedPushWeight"));
GVAR_DEF(PrimitivePtr, kPrimFusedPullWeight, std::make_shared<Primitive>("FusedPullWeight"));
GVAR_DEF(PrimitivePtr, kPrimInitDataSetQueue, std::make_shared<Primitive>("InitDataSetQueue"));
GVAR_DEF(PrimitivePtr, kPrimVirtualAssignAdd, std::make_shared<Primitive>("_VirtualAssignAdd"));
GVAR_DEF(PrimitivePtr, kPrimVirtualAccuGrad, std::make_shared<Primitive>("_VirtualAccuGrad"));
GVAR_DEF(PrimitivePtr, kPrimVirtualPipelineEnd, std::make_shared<Primitive>("_VirtualPipelineEnd"));
GVAR_DEF(PrimitivePtr, kPrimMirrorMicroStep, std::make_shared<Primitive>("_MirrorMicroStepOperator"));
GVAR_DEF(PrimitivePtr, kPrimApplyProximalAdagrad, std::make_shared<Primitive>("ApplyProximalAdagrad"));
GVAR_DEF(PrimitivePtr, kPrimStreamSend, std::make_shared<Primitive>("StreamSend"));
GVAR_DEF(PrimitivePtr, kPrimStreamRecv, std::make_shared<Primitive>("StreamRecv"));
GVAR_DEF(PrimitivePtr, kPrimDenseToSparseSetOperation, std::make_shared<Primitive>(kDenseToSparseSetOperation));

// Quant ops
GVAR_DEF(PrimitivePtr, kPrimBatchNormFold, std::make_shared<Primitive>("BatchNormFold"));
GVAR_DEF(PrimitivePtr, kPrimFakeQuantWithMinMaxVarsPerChannel,
         std::make_shared<Primitive>("FakeQuantWithMinMaxVarsPerChannel"));
// Control ops
GVAR_DEF(PrimitivePtr, kPrimMerge, std::make_shared<Primitive>("Merge"));

// RowTensor
GVAR_DEF(PrimitivePtr, kPrimMakeRowTensor, std::make_shared<Primitive>(kMakeRowTensor));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetValues, std::make_shared<Primitive>(kRowTensorGetValues));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetIndices, std::make_shared<Primitive>(kRowTensorGetIndices));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetDenseShape, std::make_shared<Primitive>(kRowTensorGetDenseShape));
GVAR_DEF(PrimitivePtr, kPrimRowTensorAdd, std::make_shared<Primitive>(kRowTensorAdd));

// COOTensor
GVAR_DEF(PrimitivePtr, kPrimMakeCOOTensor, std::make_shared<Primitive>(kMakeCOOTensor));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetValues, std::make_shared<Primitive>(kCOOTensorGetValues));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetIndices, std::make_shared<Primitive>(kCOOTensorGetIndices));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetDenseShape, std::make_shared<Primitive>(kCOOTensorGetDenseShape));

// CSRTensor
GVAR_DEF(PrimitivePtr, kPrimMakeCSRTensor, std::make_shared<Primitive>(kMakeCSRTensor));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetValues, std::make_shared<Primitive>(kCSRTensorGetValues));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetIndptr, std::make_shared<Primitive>(kCSRTensorGetIndptr));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetIndices, std::make_shared<Primitive>(kCSRTensorGetIndices));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetDenseShape, std::make_shared<Primitive>(kCSRTensorGetDenseShape));

// MapTensor
GVAR_DEF(PrimitivePtr, kPrimMakeMapParameter, std::make_shared<Primitive>(kMakeMapParameter));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGet, std::make_shared<Primitive>(kMapTensorGet));
GVAR_DEF(PrimitivePtr, kPrimMapTensorPut, std::make_shared<Primitive>(kMapTensorPut));
GVAR_DEF(PrimitivePtr, kPrimMapTensorErase, std::make_shared<Primitive>(kMapTensorErase));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetDefaultValue, std::make_shared<Primitive>(kMapTensorGetDefaultValue));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetPermitFilterValue, std::make_shared<Primitive>(kMapTensorGetPermitFilterValue));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetEvictFilterValue, std::make_shared<Primitive>(kMapTensorGetEvictFilterValue));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetKeys, std::make_shared<Primitive>(kMapTensorGetKeys));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetValues, std::make_shared<Primitive>(kMapTensorGetValues));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetData, std::make_shared<Primitive>(kMapTensorGetData));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetGrad, std::make_shared<Primitive>(kMapTensorGetGrad));

// Sparse ops
GVAR_DEF(PrimitivePtr, kPrimSparseCross, std::make_shared<Primitive>(kSparseCross));
GVAR_DEF(PrimitivePtr, kPrimRaggedTensorToTensor, std::make_shared<Primitive>(kRaggedTensorToTensor));
GVAR_DEF(PrimitivePtr, kPrimSparseTensorDenseMatmul, std::make_shared<Primitive>(kSparseTensorDenseMatmul));
GVAR_DEF(PrimitivePtr, kPrimSparseFillEmptyRows, std::make_shared<Primitive>(kSparseFillEmptyRows));
GVAR_DEF(PrimitivePtr, kPrimSparseToDenseV2, std::make_shared<Primitive>(kSparseToDenseV2));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmax, std::make_shared<Primitive>(kSparseSoftmax));
GVAR_DEF(PrimitivePtr, kPrimSparseAddmm, std::make_shared<Primitive>(kSparseAddmm));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixMul, std::make_shared<Primitive>(kSparseMatrixMul));
GVAR_DEF(PrimitivePtr, kPrimSparseSparseMaximum, std::make_shared<Primitive>(kSparseSparseMaximum));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorDenseMatmul, std::make_shared<Primitive>(kCOOTensorDenseMatmul));
GVAR_DEF(PrimitivePtr, kPrimCSRReduceSum, std::make_shared<Primitive>(kCSRReduceSum));
GVAR_DEF(PrimitivePtr, kPrimCSRMV, std::make_shared<Primitive>(kCSRMV));
GVAR_DEF(PrimitivePtr, kPrimCSRMM, std::make_shared<Primitive>(kCSRMM));
GVAR_DEF(PrimitivePtr, kPrimCSRMul, std::make_shared<Primitive>(kCSRMul));
GVAR_DEF(PrimitivePtr, kPrimSparseDenseCwiseAdd, std::make_shared<Primitive>(kSparseDenseCwiseAdd));
GVAR_DEF(PrimitivePtr, kPrimSparseDenseCwiseDiv, std::make_shared<Primitive>(kSparseDenseCwiseDiv));
GVAR_DEF(PrimitivePtr, kPrimSparseDenseCwiseMul, std::make_shared<Primitive>(kSparseDenseCwiseMul));
GVAR_DEF(PrimitivePtr, kPrimCSRGather, std::make_shared<Primitive>(kCSRGather));
GVAR_DEF(PrimitivePtr, kPrimCSR2COO, std::make_shared<Primitive>(kCSR2COO));
GVAR_DEF(PrimitivePtr, kPrimCOO2CSR, std::make_shared<Primitive>(kCOO2CSR));
GVAR_DEF(PrimitivePtr, kPrimCSRDiv, std::make_shared<Primitive>(kCSRDiv));
GVAR_DEF(PrimitivePtr, kPrimSparseSplit, std::make_shared<Primitive>(kSparseSplit));
GVAR_DEF(PrimitivePtr, kPrimDenseToDenseSetOperation, std::make_shared<Primitive>(kDenseToDenseSetOperation));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixAdd, std::make_shared<Primitive>(kSparseMatrixAdd));
GVAR_DEF(PrimitivePtr, kPrimSparseAdd, std::make_shared<Primitive>(kSparseAdd));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentMeanGrad, std::make_shared<Primitive>("SparseSegmentMeanGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentMeanWithNumSegments,
         std::make_shared<Primitive>("SparseSegmentMeanWithNumSegments"));
GVAR_DEF(PrimitivePtr, kPrimDenseToCSRSparseMatrix, std::make_shared<Primitive>("DenseToCSRSparseMatrix"));
GVAR_DEF(PrimitivePtr, kPrimSparseTensorToCSRSparseMatrix, std::make_shared<Primitive>(kSparseTensorToCSRSparseMatrix));
GVAR_DEF(PrimitivePtr, kPrimCSRSparseMatrixToSparseTensor, std::make_shared<Primitive>(kCSRSparseMatrixToSparseTensor));
GVAR_DEF(PrimitivePtr, kPrimSparseConcat, std::make_shared<Primitive>(kSparseConcat));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixNNZ, std::make_shared<Primitive>(kSparseMatrixNNZ));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixSoftmax, std::make_shared<Primitive>(kSparseMatrixSoftmax));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixMatMul, std::make_shared<Primitive>(kSparseMatrixMatMul));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixSparseMatMul, std::make_shared<Primitive>(kSparseMatrixSparseMatMul));
GVAR_DEF(PrimitivePtr, kPrimCSRSparseMatrixToDense, std::make_shared<Primitive>("CSRSparseMatrixToDense"));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixTranspose, std::make_shared<Primitive>(kSparseMatrixTranspose));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixOrderingAMD, std::make_shared<Primitive>(kSparseMatrixOrderingAMD));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSum, std::make_shared<Primitive>("SparseSegmentSum"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSumGrad, std::make_shared<Primitive>("SparseSegmentSumGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSumWithNumSegments,
         std::make_shared<Primitive>("SparseSegmentSumWithNumSegments"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSqrtN, std::make_shared<Primitive>("SparseSegmentSqrtN"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSqrtNGrad, std::make_shared<Primitive>("SparseSegmentSqrtNGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSqrtNWithNumSegments,
         std::make_shared<Primitive>("SparseSegmentSqrtNWithNumSegments"));
GVAR_DEF(PrimitivePtr, kPrimRaggedTensorToSparse, std::make_shared<Primitive>(kRaggedTensorToSparse));

// Sparse Grad ops
GVAR_DEF(PrimitivePtr, kPrimSparseAddGrad, std::make_shared<Primitive>(kSparseAddGrad));
GVAR_DEF(PrimitivePtr, kPrimSparseFillEmptyRowsGrad, std::make_shared<Primitive>(kSparseFillEmptyRowsGrad));
GVAR_DEF(PrimitivePtr, kPrimSparseTensorDenseAdd, std::make_shared<Primitive>(kSparseTensorDenseAdd));
GVAR_DEF(PrimitivePtr, kPrimSparseSlice, std::make_shared<Primitive>(kSparseSlice));
GVAR_DEF(PrimitivePtr, kPrimSparseSliceGrad, std::make_shared<Primitive>(kSparseSliceGrad));

// TensorList
GVAR_DEF(PrimitivePtr, kPrimTensorListFromTensor, std::make_shared<Primitive>("TensorListFromTensor"));
GVAR_DEF(PrimitivePtr, kPrimTensorListReserve, std::make_shared<Primitive>("TensorListReserve"));
GVAR_DEF(PrimitivePtr, kPrimTensorListStack, std::make_shared<Primitive>("TensorListStack"));
GVAR_DEF(PrimitivePtr, kPrimTensorListSetItem, std::make_shared<Primitive>("TensorListSetItem"));

// Maths
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
GVAR_DEF(PrimitivePtr, kPrimAdd, std::make_shared<Primitive>(kAdd));
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
GVAR_DEF(PrimitivePtr, kPrimExpand, std::make_shared<Primitive>("Expand"));
GVAR_DEF(PrimitivePtr, kPrimExpandDims, std::make_shared<Primitive>("ExpandDims"));
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
GVAR_DEF(PrimitivePtr, kPrimIdentityMath, std::make_shared<Primitive>("Identity", kSideEffectPropagate));
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
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentMean, std::make_shared<Primitive>(kSparseSegmentMean));
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

// linalg
GVAR_DEF(PrimitivePtr, kPrimGeqrf, std::make_shared<Primitive>("Geqrf"));
GVAR_DEF(PrimitivePtr, kPrimSvd, std::make_shared<Primitive>("Svd"));

GVAR_DEF(PrimitivePtr, kPrimCholeskyGrad, std::make_shared<Primitive>("CholeskyGrad"));

// Image
GVAR_DEF(PrimitivePtr, kPrimNonMaxSuppressionV3, std::make_shared<Primitive>("NonMaxSuppressionV3"));
GVAR_DEF(PrimitivePtr, kPrimNonMaxSuppressionWithOverlaps,
         std::make_shared<Primitive>("NonMaxSuppressionWithOverlaps"));
GVAR_DEF(PrimitivePtr, kPrimAdjustHue, std::make_shared<Primitive>(kAdjustHue));
GVAR_DEF(PrimitivePtr, kPrimAdjustContrastv2, std::make_shared<Primitive>(kAdjustContrastv2));
GVAR_DEF(PrimitivePtr, kPrimAdjustSaturation, std::make_shared<Primitive>(kAdjustSaturation));
GVAR_DEF(PrimitivePtr, kPrimCompareAndBitpack, std::make_shared<Primitive>(kCompareAndBitpack));
GVAR_DEF(PrimitivePtr, kPrimScaleAndTranslate, std::make_shared<Primitive>("ScaleAndTranslate"));
GVAR_DEF(PrimitivePtr, kPrimScaleAndTranslateGrad, std::make_shared<Primitive>("ScaleAndTranslateGrad"));
GVAR_DEF(PrimitivePtr, kPrimCombinedNonMaxSuppression, std::make_shared<Primitive>("CombinedNonMaxSuppression"))

// Statements
GVAR_DEF(PrimitivePtr, kPrimReturn, std::make_shared<Primitive>(kReturn));
GVAR_DEF(PrimitivePtr, kPrimUnroll, std::make_shared<Primitive>("Unroll"));
GVAR_DEF(PrimitivePtr, kPrimSwitch, std::make_shared<Primitive>(kSwitch));
GVAR_DEF(PrimitivePtr, kPrimSwitchLayer, std::make_shared<Primitive>("switch_layer"));
GVAR_DEF(PrimitivePtr, kPrimAssign, std::make_shared<Primitive>(kAssign));
GVAR_DEF(PrimitivePtr, kPrimAssignAdd, std::make_shared<Primitive>(kAssignAdd));
GVAR_DEF(PrimitivePtr, kPrimAssignSub, std::make_shared<Primitive>(kAssignSub));
GVAR_DEF(PrimitivePtr, kPrimVmapStackAssign, std::make_shared<Primitive>(kVmapStackAssign));
GVAR_DEF(PrimitivePtr, kPrimVmapUnstackAssign, std::make_shared<Primitive>(kVmapUnstackAssign));
GVAR_DEF(PrimitivePtr, kPrimSelect, std::make_shared<Primitive>(kSelect));
GVAR_DEF(PrimitivePtr, kPrimCall, std::make_shared<Primitive>("call"));
GVAR_DEF(PrimitivePtr, kPrimRaise, std::make_shared<Primitive>("raise"));
GVAR_DEF(PrimitivePtr, kPrimJoinedStr, std::make_shared<Primitive>("joinedstr"));
GVAR_DEF(PrimitivePtr, kPrimCallInline, std::make_shared<Primitive>("call_inline"));

GVAR_DEF(PrimitivePtr, kPrimMakeTuple, std::make_shared<Primitive>(kMakeTuple));
GVAR_DEF(PrimitivePtr, kPrimMakeSlice, std::make_shared<Primitive>("make_slice"));
GVAR_DEF(PrimitivePtr, kPrimTupleGetItem, std::make_shared<Primitive>(kTupleGetItem));
GVAR_DEF(PrimitivePtr, kPrimSliceGetItem, std::make_shared<Primitive>(kSliceGetItem));
GVAR_DEF(PrimitivePtr, kPrimArrayGetItem, std::make_shared<Primitive>("array_getitem"));
GVAR_DEF(PrimitivePtr, kPrimTupleSetItem, std::make_shared<Primitive>(kTupleSetItem));
GVAR_DEF(PrimitivePtr, kPrimArraySetItem, std::make_shared<Primitive>("array_setitem"));
GVAR_DEF(PrimitivePtr, kPrimGetAttr, std::make_shared<Primitive>("getattr"));
GVAR_DEF(PrimitivePtr, kPrimArrayLen, std::make_shared<Primitive>("array_len"));
GVAR_DEF(PrimitivePtr, kPrimTileShape, std::make_shared<Primitive>("tile_shape"));
GVAR_DEF(PrimitivePtr, kPrimGenerateShapeIndex, std::make_shared<Primitive>("generate_shape_index"));
GVAR_DEF(PrimitivePtr, kPrimGenerateInverseIndex, std::make_shared<Primitive>("generate_inverse_index"));

GVAR_DEF(PrimitivePtr, kPrimRealTupleGetItem, std::make_shared<Primitive>(kRealTupleGetItem));
GVAR_DEF(PrimitivePtr, kPrimListToTensor, std::make_shared<Primitive>(kListToTensor));
GVAR_DEF(PrimitivePtr, kPrimScalarToTensor, std::make_shared<Primitive>(kScalarToTensor));
GVAR_DEF(PrimitivePtr, kPrimTensorToTuple, std::make_shared<Primitive>(kTensorToTuple));
GVAR_DEF(PrimitivePtr, kPrimTensorToList, std::make_shared<Primitive>(kTensorToList));
GVAR_DEF(PrimitivePtr, kPrimTensorToScalar, std::make_shared<Primitive>(kTensorToScalar));

// Debug ops
GVAR_DEF(PrimitivePtr, kPrimAssert, std::make_shared<Primitive>("Assert"));
#ifndef ENABLE_SECURITY
GVAR_DEF(PrimitivePtr, kPrimScalarSummary, std::make_shared<Primitive>("ScalarSummary"));
GVAR_DEF(PrimitivePtr, kPrimImageSummary, std::make_shared<Primitive>("ImageSummary"));
GVAR_DEF(PrimitivePtr, kPrimTensorSummary, std::make_shared<Primitive>("TensorSummary"));
GVAR_DEF(PrimitivePtr, kPrimHistogramSummary, std::make_shared<Primitive>("HistogramSummary"));
GVAR_DEF(PrimitivePtr, kPrimHistogramFixedWidth, std::make_shared<Primitive>("HistogramFixedWidth"));
#endif
GVAR_DEF(PrimitivePtr, kPrimDebug, std::make_shared<Primitive>("Debug"));

// Dynamic shape testing
GVAR_DEF(PrimitivePtr, kPrimConvertToDynamic, std::make_shared<Primitive>("ConvertToDynamic"));
GVAR_DEF(PrimitivePtr, kPrimGpuConvertToDynamicShape, std::make_shared<Primitive>("GpuConvertToDynamicShape"));
GVAR_DEF(PrimitivePtr, kPrimErrorOnDynamicShapeInput, std::make_shared<Primitive>("ErrorOnDynamicShapeInput"));

// Dynamic shape.
GVAR_DEF(PrimitivePtr, kPrimIsDimUnknown, std::make_shared<Primitive>("IsDimUnKnown"));
GVAR_DEF(PrimitivePtr, kPrimIsShapeUnknown, std::make_shared<Primitive>("IsShapeUnKnown"));
GVAR_DEF(PrimitivePtr, kPrimIsElementUnknown, std::make_shared<Primitive>("IsElementUnknown"));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimCheckValid, std::make_shared<Primitive>("CheckValid"));
GVAR_DEF(PrimitivePtr, kPrimBartlettWindow, std::make_shared<Primitive>(kBartlettWindow));
GVAR_DEF(PrimitivePtr, kPrimDepend, std::make_shared<Primitive>(kDepend, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimIOU, std::make_shared<Primitive>("IOU"));
GVAR_DEF(PrimitivePtr, kPrimIou, std::make_shared<Primitive>("Iou"));
GVAR_DEF(PrimitivePtr, kPrimReformat, std::make_shared<Primitive>("Reformat"));
GVAR_DEF(PrimitivePtr, kPrimLoad, std::make_shared<Primitive>(kLoad));
GVAR_DEF(PrimitivePtr, kPrimUpdateState, std::make_shared<Primitive>(kUpdateState));
GVAR_DEF(PrimitivePtr, kPrimMutable, std::make_shared<Primitive>(kMutable));
GVAR_DEF(PrimitivePtr, kPrimGetGrad, std::make_shared<Primitive>(kGetGrad));
GVAR_DEF(PrimitivePtr, kPrimPartial, std::make_shared<Primitive>("Partial", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimIdentity, std::make_shared<Primitive>(kidentity, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimHookBackward, std::make_shared<Primitive>("HookBackward"));
GVAR_DEF(PrimitivePtr, kPrimCellBackwardHook, std::make_shared<Primitive>("CellBackwardHook"));
GVAR_DEF(PrimitivePtr, kPrimPrintShapeType, std::make_shared<Primitive>("PrintShapeType"));
GVAR_DEF(PrimitivePtr, kPrimSameTypeShape, std::make_shared<Primitive>("SameTypeShape"));
GVAR_DEF(PrimitivePtr, kPrimPrint, std::make_shared<Primitive>("Print"));
GVAR_DEF(PrimitivePtr, kPrimIs_, std::make_shared<Primitive>("is_"));
GVAR_DEF(PrimitivePtr, kPrimIsNot, std::make_shared<Primitive>("is_not"));
GVAR_DEF(PrimitivePtr, kPrimInDict, std::make_shared<Primitive>("in_dict"));
GVAR_DEF(PrimitivePtr, kPrimNotInDict, std::make_shared<Primitive>("not_in_dict"));
GVAR_DEF(PrimitivePtr, kPrimIsConstant, std::make_shared<Primitive>("is_constant"));
GVAR_DEF(PrimitivePtr, kPrimEquivFormat, std::make_shared<Primitive>("EquivFormat"));
GVAR_DEF(PrimitivePtr, kPrimLshProjection, std::make_shared<Primitive>("LshProjection"));
GVAR_DEF(PrimitivePtr, kPrimHashtableLookup, std::make_shared<Primitive>("HashtableLookup"));
GVAR_DEF(PrimitivePtr, kPrimCustomPredict, std::make_shared<Primitive>("CustomPredict"));
GVAR_DEF(PrimitivePtr, kPrimPriorBox, std::make_shared<Primitive>("PriorBox"));
GVAR_DEF(PrimitivePtr, kPrimQuantDTypeCast, std::make_shared<Primitive>("QuantDTypeCast"));
GVAR_DEF(PrimitivePtr, kPrimWhile, std::make_shared<Primitive>("While"));
GVAR_DEF(PrimitivePtr, kPrimPull, std::make_shared<Primitive>("Pull"));
GVAR_DEF(PrimitivePtr, kPrimPush, std::make_shared<Primitive>("Push"));
GVAR_DEF(PrimitivePtr, kPrimNPUGetFloatStatus, std::make_shared<Primitive>("NPUGetFloatStatus"));
GVAR_DEF(PrimitivePtr, kPrimNPUAllocFloatStatus, std::make_shared<Primitive>("NPUAllocFloatStatus"));
GVAR_DEF(PrimitivePtr, kPrimNPUClearFloatStatus, std::make_shared<Primitive>("NPUClearFloatStatus"));
GVAR_DEF(PrimitivePtr, kPrimNPUGetFloatStatusV2, std::make_shared<Primitive>("NPUGetFloatStatusV2"));
GVAR_DEF(PrimitivePtr, kPrimNPUClearFloatStatusV2, std::make_shared<Primitive>("NPUClearFloatStatusV2"));
GVAR_DEF(PrimitivePtr, kPrimPyFunc, std::make_shared<Primitive>("PyFunc"));
GVAR_DEF(PrimitivePtr, kPrimDynamicLossScale, std::make_shared<Primitive>("_DynamicLossScale"));
GVAR_DEF(PrimitivePtr, kPrimScaleGrad, std::make_shared<Primitive>("ScaleGrad"));
GVAR_DEF(PrimitivePtr, kPrimPopulationCount, std::make_shared<Primitive>("PopulationCount"));
GVAR_DEF(PrimitivePtr, kPrimBlackmanWindow, std::make_shared<Primitive>("BlackmanWindow"));
GVAR_DEF(PrimitivePtr, kPrimOpaquePredicate, std::make_shared<Primitive>("OpaquePredicate"));

// Structures
GVAR_DEF(PrimitivePtr, kPrimMakeList, std::make_shared<Primitive>(kMakeListNew));
GVAR_DEF(PrimitivePtr, kPrimMakeKeywordArg, std::make_shared<Primitive>("make_keyword_arg"));
GVAR_DEF(PrimitivePtr, kPrimListGetItem, std::make_shared<Primitive>(kListGetItem));
GVAR_DEF(PrimitivePtr, kPrimListSetItem, std::make_shared<Primitive>(kListSetItem));
GVAR_DEF(PrimitivePtr, kPrimDictGetItem, std::make_shared<Primitive>("dict_getitem"));
GVAR_DEF(PrimitivePtr, kPrimDictSetItem, std::make_shared<Primitive>("dict_setitem"));
GVAR_DEF(PrimitivePtr, kPrimDictGetKeys, std::make_shared<Primitive>("dict_getkeys"));
GVAR_DEF(PrimitivePtr, kPrimDictGetValues, std::make_shared<Primitive>("dict_getvalues"));
GVAR_DEF(PrimitivePtr, kPrimDictItems, std::make_shared<Primitive>("dict_items"));
GVAR_DEF(PrimitivePtr, kPrimSequenceLen, std::make_shared<Primitive>("sequence_len"));

// Sequence operations.
GVAR_DEF(PrimitivePtr, kPrimListAppend, std::make_shared<Primitive>(kListAppend));
GVAR_DEF(PrimitivePtr, kPrimSequenceAdd, std::make_shared<Primitive>(kSequenceAdd));
GVAR_DEF(PrimitivePtr, kPrimSequenceCount, std::make_shared<Primitive>(kSequenceCount));
GVAR_DEF(PrimitivePtr, kPrimSequenceIndex, std::make_shared<Primitive>(kSequenceIndex));
GVAR_DEF(PrimitivePtr, kPrimSequenceMul, std::make_shared<Primitive>(kSequenceMul));
GVAR_DEF(PrimitivePtr, kPrimSequenceSlice, std::make_shared<Primitive>(kSequenceSlice));
GVAR_DEF(PrimitivePtr, kPrimSequenceSliceSetItem, std::make_shared<Primitive>(kSequenceSliceSetItem));
GVAR_DEF(PrimitivePtr, kPrimSequenceZerosLike, std::make_shared<Primitive>(kSequenceZerosLike));
GVAR_DEF(PrimitivePtr, kPrimSequenceAddOffset, std::make_shared<Primitive>(kSequenceAddOffset));
GVAR_DEF(PrimitivePtr, kPrimSequenceSliceGrad, std::make_shared<Primitive>(kSequenceSliceGrad));
GVAR_DEF(PrimitivePtr, kPrimSequenceMax, std::make_shared<Primitive>(kSequenceMax));
GVAR_DEF(PrimitivePtr, kPrimSequenceMin, std::make_shared<Primitive>(kSequenceMin));
GVAR_DEF(PrimitivePtr, kPrimSequenceAddN, std::make_shared<Primitive>(kSequenceAddN));
GVAR_DEF(PrimitivePtr, kPrimSequenceConcat, std::make_shared<Primitive>(kSequenceConcat));

// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimSampleDistortedBoundingBoxV2, std::make_shared<Primitive>(kSampleDistortedBoundingBoxV2));
GVAR_DEF(PrimitivePtr, kPrimEnvironCreate, std::make_shared<Primitive>(kEnvironCreate));
GVAR_DEF(PrimitivePtr, kPrimEnvironSet, std::make_shared<Primitive>(kEnvironSet));
GVAR_DEF(PrimitivePtr, kPrimEnvironGet, std::make_shared<Primitive>(kEnvironGet));
GVAR_DEF(PrimitivePtr, kPrimEnvironAdd, std::make_shared<Primitive>(kEnvironAdd));
GVAR_DEF(PrimitivePtr, kPrimEnvironDestroyAll, std::make_shared<Primitive>(kEnvironDestroyAll));
GVAR_DEF(PrimitivePtr, kPrimSetSize, std::make_shared<Primitive>(kSetSize));

// JIT Fallback ops
// We add IO side-effect for them in advance.
GVAR_DEF(PrimitivePtr, kPrimPyInterpret,
         std::make_shared<Primitive>("PyInterpret", mindspore::HashMap<std::string, ValuePtr>(
                                                      {{std::string(GRAPH_FLAG_SIDE_EFFECT_IO), MakeValue(true)}})));
GVAR_DEF(PrimitivePtr, kPrimPyExecute,
         std::make_shared<Primitive>("PyExecute", mindspore::HashMap<std::string, ValuePtr>(
                                                    {{std::string(GRAPH_FLAG_SIDE_EFFECT_IO), MakeValue(true)},
                                                     {std::string("primitive_target"), MakeValue("CPU")}})));

// Other primitive not used by backend but used in core;
GVAR_DEF(PrimitivePtr, kPrimStateSetItem, std::make_shared<Primitive>("state_setitem"));
GVAR_DEF(PrimitivePtr, kPrimJ, std::make_shared<Primitive>(kJ, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimVmap, std::make_shared<Primitive>(kVmap, kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimShard, std::make_shared<Primitive>("Shard", kSideEffectPropagate));
GVAR_DEF(PrimitivePtr, kPrimTaylor, std::make_shared<Primitive>(kTaylor));

// Used to build graph which have keyword arguments
GVAR_DEF(PrimitivePtr, kPrimExtractKeywordArg, std::make_shared<Primitive>("extract_keyword_arg"));
GVAR_DEF(PrimitivePtr, kPrimMakeDict, std::make_shared<Primitive>("make_dict"));

// GraphKernel ops
GVAR_DEF(PrimitivePtr, kPrimInplaceAssign, std::make_shared<Primitive>("InplaceAssign"));

// Custom
GVAR_DEF(PrimitivePtr, kPrimCustom, std::make_shared<Primitive>("Custom"));

// Only used in lite
GVAR_DEF(PrimitivePtr, kPrimLeakyRelu, std::make_shared<Primitive>("LeakyRelu"));
GVAR_DEF(PrimitivePtr, kPrimConstant, std::make_shared<Primitive>("Constant"));
GVAR_DEF(PrimitivePtr, kPrimLocalResponseNormalization, std::make_shared<Primitive>("LocalResponseNormalization"));
GVAR_DEF(PrimitivePtr, kPrimFftReal, std::make_shared<Primitive>("FftReal"));
GVAR_DEF(PrimitivePtr, kPrimMfcc, std::make_shared<Primitive>("Mfcc"));
GVAR_DEF(PrimitivePtr, kPrimRfft, std::make_shared<Primitive>("Rfft"));
GVAR_DEF(PrimitivePtr, kPrimFftImag, std::make_shared<Primitive>("FftImag"));
GVAR_DEF(PrimitivePtr, kPrimSkipGram, std::make_shared<Primitive>("SkipGram"));
GVAR_DEF(PrimitivePtr, kPrimConv2DFusion, std::make_shared<Primitive>("Conv2DFusion"));
GVAR_DEF(PrimitivePtr, kPrimConv2dTransposeFusion, std::make_shared<Primitive>("Conv2dTransposeFusion"));
GVAR_DEF(PrimitivePtr, kPrimDepthWiseConv2DFusion, std::make_shared<Primitive>("DepthWiseConv2DFusion"));
GVAR_DEF(PrimitivePtr, kPrimAddFusion, std::make_shared<Primitive>("AddFusion"));
GVAR_DEF(PrimitivePtr, kPrimScaleFusion, std::make_shared<Primitive>("ScaleFusion"));
GVAR_DEF(PrimitivePtr, kPrimSubFusion, std::make_shared<Primitive>("SubFusion"));
GVAR_DEF(PrimitivePtr, kPrimMulFusion, std::make_shared<Primitive>("MulFusion"));
GVAR_DEF(PrimitivePtr, kPrimSigmoid, std::make_shared<Primitive>("Sigmoid"));
GVAR_DEF(PrimitivePtr, kPrimSigmoidGrad, std::make_shared<Primitive>("SigmoidGrad"));
GVAR_DEF(PrimitivePtr, kPrimHSigmoid, std::make_shared<Primitive>("HSigmoid"));
GVAR_DEF(PrimitivePtr, kPrimHSigmoidGrad, std::make_shared<Primitive>("HSigmoidGrad"));
GVAR_DEF(PrimitivePtr, kPrimLogSigmoid, std::make_shared<Primitive>("LogSigmoid"));
GVAR_DEF(PrimitivePtr, kPrimClip, std::make_shared<Primitive>("Clip"));
GVAR_DEF(PrimitivePtr, kPrimHardTanh, std::make_shared<Primitive>("HardTanh"));
GVAR_DEF(PrimitivePtr, kPrimDepthWiseConv2DTransposeFusion,
         std::make_shared<Primitive>("DepthWiseConv2DTransposeFusion"));
GVAR_DEF(PrimitivePtr, kPrimArgMinFusion, std::make_shared<Primitive>("ArgMinFusion"));
GVAR_DEF(PrimitivePtr, kPrimArgMaxFusion, std::make_shared<Primitive>("ArgMaxFusion"));
GVAR_DEF(PrimitivePtr, kPrimSpaceToDepth, std::make_shared<Primitive>("SpaceToDepth"));
GVAR_DEF(PrimitivePtr, kPrimPadFusion, std::make_shared<Primitive>("PadFusion"));
GVAR_DEF(PrimitivePtr, kPrimPowFusion, std::make_shared<Primitive>("PowFusion"));
GVAR_DEF(PrimitivePtr, kPrimResize, std::make_shared<Primitive>("Resize"));
GVAR_DEF(PrimitivePtr, kPrimIf, std::make_shared<Primitive>("If"));
GVAR_DEF(PrimitivePtr, kPrimAvgPoolFusion, std::make_shared<Primitive>("AvgPoolFusion"));
GVAR_DEF(PrimitivePtr, kPrimMaxPoolFusion, std::make_shared<Primitive>("MaxPoolFusion"));
GVAR_DEF(PrimitivePtr, kPrimActivation, std::make_shared<Primitive>("Activation"));
GVAR_DEF(PrimitivePtr, kPrimPReLUFusion, std::make_shared<Primitive>("PReLUFusion"));
GVAR_DEF(PrimitivePtr, kPrimTopKFusion, std::make_shared<Primitive>("TopKFusion"));
GVAR_DEF(PrimitivePtr, kPrimTileFusion, std::make_shared<Primitive>("TileFusion"));
GVAR_DEF(PrimitivePtr, kPrimReduceFusion, std::make_shared<Primitive>("ReduceFusion"));
GVAR_DEF(PrimitivePtr, kPrimReduceSumD, std::make_shared<Primitive>("ReduceSumD"));
GVAR_DEF(PrimitivePtr, kPrimLayerNormFusion, std::make_shared<Primitive>("LayerNormFusion"));
GVAR_DEF(PrimitivePtr, kPrimDType, std::make_shared<Primitive>("DType"));
GVAR_DEF(PrimitivePtr, kPrimDivFusion, std::make_shared<Primitive>("DivFusion"));
GVAR_DEF(PrimitivePtr, kPrimExpFusion, std::make_shared<Primitive>("ExpFusion"));
GVAR_DEF(PrimitivePtr, kPrimErf, std::make_shared<Primitive>("Erf"));
GVAR_DEF(PrimitivePtr, kPrimErfc, std::make_shared<Primitive>("Erfc"));
GVAR_DEF(PrimitivePtr, kPrimSplice, std::make_shared<Primitive>("Splice"));
GVAR_DEF(PrimitivePtr, kPrimAffine, std::make_shared<Primitive>("Affine"));
GVAR_DEF(PrimitivePtr, kPrimEltwise, std::make_shared<Primitive>("Eltwise"));
GVAR_DEF(PrimitivePtr, kPrimMatMulFusion, std::make_shared<Primitive>("MatMulFusion"));
GVAR_DEF(PrimitivePtr, kPrimDynamicQuant, std::make_shared<Primitive>("DynamicQuant"));
GVAR_DEF(PrimitivePtr, kPrimPartialFusion, std::make_shared<Primitive>("PartialFusion"));
GVAR_DEF(PrimitivePtr, kPrimFSEDecode, std::make_shared<Primitive>("FSEDecode"));

// Type introspection
GVAR_DEF(PrimitivePtr, kPrimTypeOf, std::make_shared<Primitive>("typeof"));
GVAR_DEF(PrimitivePtr, kPrimTopTypeOf, std::make_shared<Primitive>("TopTypeof"));
GVAR_DEF(PrimitivePtr, kPrimHasType, std::make_shared<Primitive>("hastype"));
GVAR_DEF(PrimitivePtr, kPrimIsInstance, std::make_shared<Primitive>("isinstance"));

GVAR_DEF(PrimitivePtr, kPrimResolve, std::make_shared<Primitive>("resolve"));
GVAR_DEF(PrimitivePtr, kPrimEmbed, std::make_shared<Primitive>("embed"));
GVAR_DEF(PrimitivePtr, kPrimRefToEmbed, std::make_shared<Primitive>("RefToEmbed"));
GVAR_DEF(PrimitivePtr, kPrimCreateInstance, std::make_shared<Primitive>("create_instance"));
GVAR_DEF(PrimitivePtr, kPrimCallInstance, std::make_shared<Primitive>("call_instance"));
GVAR_DEF(PrimitivePtr, kPrimWithEnter, std::make_shared<Primitive>("with_enter"));
GVAR_DEF(PrimitivePtr, kPrimWithExit, std::make_shared<Primitive>("with_exit"));
// Other miscellaneous
GVAR_DEF(PrimitivePtr, kPrimInsertGradientOf, std::make_shared<Primitive>("InsertGradientOf"));
GVAR_DEF(PrimitivePtr, kPrimCheckBprop, std::make_shared<Primitive>("CheckBprop"));
GVAR_DEF(PrimitivePtr, kPrimMixedPrecisionCast, std::make_shared<Primitive>("MixedPrecisionCast"));

// Structures
GVAR_DEF(PrimitivePtr, kPrimListReduce, std::make_shared<Primitive>("list_reduce"));
GVAR_DEF(PrimitivePtr, kPrimTupleReversed, std::make_shared<Primitive>("tuple_reversed"));
GVAR_DEF(PrimitivePtr, kPrimReducedShape, std::make_shared<Primitive>("reduced_shape"));
GVAR_DEF(PrimitivePtr, kPrimTupleDiv, std::make_shared<Primitive>("tuple_div"));
GVAR_DEF(PrimitivePtr, kPrimTupleToArray, std::make_shared<Primitive>("tuple_to_array"));
GVAR_DEF(PrimitivePtr, kPrimShapeMul, std::make_shared<Primitive>("shape_mul"));
GVAR_DEF(PrimitivePtr, kPrimTupleEqual, std::make_shared<Primitive>(kTupleEqual));
GVAR_DEF(PrimitivePtr, kPrimListEqual, std::make_shared<Primitive>(kListEqual));
GVAR_DEF(PrimitivePtr, kPrimTupleGreaterThan, std::make_shared<Primitive>("tuple_greater_than"));
GVAR_DEF(PrimitivePtr, kPrimListGreaterThan, std::make_shared<Primitive>("list_greater_than"));
GVAR_DEF(PrimitivePtr, kPrimTupleGreaterEqual, std::make_shared<Primitive>("tuple_greater_equal"));
GVAR_DEF(PrimitivePtr, kPrimListGreaterEqual, std::make_shared<Primitive>("list_greater_equal"));
GVAR_DEF(PrimitivePtr, kPrimTupleLessThan, std::make_shared<Primitive>(kTupleLt));
GVAR_DEF(PrimitivePtr, kPrimListLessThan, std::make_shared<Primitive>(kListLt));
GVAR_DEF(PrimitivePtr, kPrimTupleLessEqual, std::make_shared<Primitive>(kTupleLe));
GVAR_DEF(PrimitivePtr, kPrimListLessEqual, std::make_shared<Primitive>(kListLe));
GVAR_DEF(PrimitivePtr, kPrimMakeRange, std::make_shared<Primitive>("make_range"));
GVAR_DEF(PrimitivePtr, kPrimStopGradient, std::make_shared<Primitive>("StopGradient"));
GVAR_DEF(PrimitivePtr, kPrimDictLen, std::make_shared<Primitive>("dict_len"));
GVAR_DEF(PrimitivePtr, kPrimFakeBprop, std::make_shared<Primitive>("fake_bprop"));
GVAR_DEF(PrimitivePtr, kPrimBroadcastGradientArgs, std::make_shared<Primitive>("BroadcastGradientArgs"));
GVAR_DEF(PrimitivePtr, kPrimDynamicBroadcastGradientArgs, std::make_shared<Primitive>(kDynamicBroadcastGradientArgs));
GVAR_DEF(PrimitivePtr, kPrimConvertToAdapterTensor, std::make_shared<Primitive>("ConvertToAdapterTensor"));
GVAR_DEF(PrimitivePtr, kPrimConvertToMsTensor, std::make_shared<Primitive>("ConvertToMsTensor"));

// Random
GVAR_DEF(PrimitivePtr, kPrimStandardLaplace, std::make_shared<Primitive>("StandardLaplace"));
GVAR_DEF(PrimitivePtr, kPrimStandardNormal, std::make_shared<Primitive>(kStandardNormal));
GVAR_DEF(PrimitivePtr, kPrimParameterizedTruncatedNormal, std::make_shared<Primitive>("ParameterizedTruncatedNormal"));
GVAR_DEF(PrimitivePtr, kPrimRandomNormal, std::make_shared<Primitive>("RandomNormal"));
GVAR_DEF(PrimitivePtr, kPrimNonDeterministicInts, std::make_shared<Primitive>("NonDeterministicInts"));
GVAR_DEF(PrimitivePtr, kPrimTruncatedNormal, std::make_shared<Primitive>("TruncatedNormal"));
GVAR_DEF(PrimitivePtr, kPrimRandomPoisson, std::make_shared<Primitive>("RandomPoisson"));
GVAR_DEF(PrimitivePtr, kPrimRandomGamma, std::make_shared<Primitive>("RandomGamma"));
GVAR_DEF(PrimitivePtr, kPrimRandomShuffle, std::make_shared<Primitive>("RandomShuffle"));
GVAR_DEF(PrimitivePtr, kPrimRandomGammaGrad, std::make_shared<Primitive>("RandomGammaGrad"));
GVAR_DEF(PrimitivePtr, kPrimRandomCategorical, std::make_shared<Primitive>("RandomCategorical"));
GVAR_DEF(PrimitivePtr, kPrimRandperm, std::make_shared<Primitive>("Randperm"));
GVAR_DEF(PrimitivePtr, kPrimRandpermV2, std::make_shared<Primitive>("RandpermV2"));
GVAR_DEF(PrimitivePtr, kPrimUniformCandidateSampler, std::make_shared<Primitive>("UniformCandidateSampler"));
GVAR_DEF(PrimitivePtr, kPrimLogUniformCandidateSampler, std::make_shared<Primitive>("LogUniformCandidateSampler"));
GVAR_DEF(PrimitivePtr, kPrimMultinomial, std::make_shared<Primitive>("Multinomial"));
GVAR_DEF(PrimitivePtr, kPrimMultinomialWithReplacement, std::make_shared<Primitive>("MultinomialWithReplacement"));
GVAR_DEF(PrimitivePtr, kPrimRandomChoiceWithMask, std::make_shared<Primitive>("RandomChoiceWithMask"));
GVAR_DEF(PrimitivePtr, kPrimUniform, std::make_shared<Primitive>(kUniform));
GVAR_DEF(PrimitivePtr, kPrimLower, std::make_shared<Primitive>(kLower));

// RL Ops
GVAR_DEF(PrimitivePtr, kPrimTensorArrayStack, std::make_shared<Primitive>("TensorArrayStack"));
GVAR_DEF(PrimitivePtr, kPrimTensorArray, std::make_shared<Primitive>("TensorArray"));
GVAR_DEF(PrimitivePtr, kPrimTensorArrayWrite, std::make_shared<Primitive>("TensorArrayWrite"));
GVAR_DEF(PrimitivePtr, kPrimTensorArrayGather, std::make_shared<Primitive>("TensorArrayGather"));
GVAR_DEF(PrimitivePtr, kPrimPartitionedCall, std::make_shared<Primitive>("PartitionedCall"));
GVAR_DEF(PrimitivePtr, kPrimDecodeImage, std::make_shared<Primitive>("DecodeImage"));
GVAR_DEF(PrimitivePtr, kPrimStridedSliceV2, std::make_shared<Primitive>("StridedSliceV2"));
GVAR_DEF(PrimitivePtr, kPrimStridedSliceV2Grad, std::make_shared<Primitive>("StridedSliceV2Grad"));
GVAR_DEF(PrimitivePtr, kPrimKMeansCentroids, std::make_shared<Primitive>("KMeansCentroids"));
GVAR_DEF(PrimitivePtr, kPrimReservoirReplayBufferCreate, std::make_shared<Primitive>("ReservoirReplayBufferCreate"));
GVAR_DEF(PrimitivePtr, kPrimReservoirReplayBufferPush, std::make_shared<Primitive>("ReservoirReplayBufferPush"));
GVAR_DEF(PrimitivePtr, kPrimReservoirReplayBufferSample, std::make_shared<Primitive>("ReservoirReplayBufferSample"));
GVAR_DEF(PrimitivePtr, kPrimReservoirReplayBufferDestroy, std::make_shared<Primitive>("ReservoirReplayBufferDestroy"));
GVAR_DEF(PrimitivePtr, kPrimOCRDetectionPreHandle, std::make_shared<Primitive>("OCRDetectionPreHandle"));

// Real tuple and list ops.
GVAR_DEF(PrimitivePtr, kPrimTupleToTensor, std::make_shared<Primitive>(kTupleToTensor));
GVAR_DEF(PrimitivePtr, kPrimRealMakeTuple, std::make_shared<Primitive>(kRealMakeTuple));

// AdamApplyOne
GVAR_DEF(PrimitivePtr, kPrimAdamApplyOne, std::make_shared<Primitive>("AdamApplyOne"));
GVAR_DEF(PrimitivePtr, kPrimAdamApplyOneAssign, std::make_shared<Primitive>("AdamApplyOneAssign"));

// AdamApplyOneWithDecay
GVAR_DEF(PrimitivePtr, kPrimAdamApplyOneWithDecay, std::make_shared<Primitive>("AdamApplyOneWithDecay"));
GVAR_DEF(PrimitivePtr, kPrimAdamApplyOneWithDecayAssign, std::make_shared<Primitive>("AdamApplyOneWithDecayAssign"));

// OCR Ops
GVAR_DEF(PrimitivePtr, kPrimOCRRecognitionPreHandle, std::make_shared<Primitive>("OCRRecognitionPreHandle"));

// Sponge Ops
GVAR_DEF(PrimitivePtr, kPrimAngleAtomEnergy, std::make_shared<Primitive>("AngleAtomEnergy"));

// Framework ops
GVAR_DEF(PrimitivePtr, kPrimTileSize, std::make_shared<Primitive>("TileSize"));

class DoSignaturePrimitive : public Primitive {
 public:
  explicit DoSignaturePrimitive(const std::string &name, const ValuePtr &function)
      : Primitive("S-Prim-" + name), function_(function) {}

  ~DoSignaturePrimitive() override = default;

  MS_DECLARE_PARENT(DoSignaturePrimitive, Primitive)

  const ValuePtr function() const { return function_; }

 private:
  ValuePtr function_;
};
using DoSignaturePrimitivePtr = std::shared_ptr<DoSignaturePrimitive>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_CORE_OPS_H_
