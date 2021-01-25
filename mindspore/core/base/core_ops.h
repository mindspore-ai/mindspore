/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPERATOR_OPS_H_
#define MINDSPORE_CORE_OPERATOR_OPS_H_

#include <iostream>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"

namespace mindspore {
namespace prim {
// Here list all primitives used in backend or some special primitives used by core.
// Arithmetic
inline const PrimitivePtr kPrimScalarAdd = std::make_shared<Primitive>("scalar_add");
inline const PrimitivePtr kPrimScalarSub = std::make_shared<Primitive>("scalar_sub");
inline const PrimitivePtr kPrimScalarMul = std::make_shared<Primitive>("scalar_mul");
inline const PrimitivePtr kPrimScalarDiv = std::make_shared<Primitive>("scalar_div");
inline const PrimitivePtr kPrimScalarFloordiv = std::make_shared<Primitive>("scalar_floordiv");
inline const PrimitivePtr kPrimScalarMod = std::make_shared<Primitive>("scalar_mod");
inline const PrimitivePtr kPrimScalarPow = std::make_shared<Primitive>("scalar_pow");
inline const PrimitivePtr kPrimScalarTrunc = std::make_shared<Primitive>("scalar_trunc");
inline const PrimitivePtr kPrimScalarFloor = std::make_shared<Primitive>("scalar_floor");
inline const PrimitivePtr kPrimScalarUadd = std::make_shared<Primitive>("scalar_uadd");
inline const PrimitivePtr kPrimScalarUsub = std::make_shared<Primitive>("scalar_usub");
inline const PrimitivePtr kPrimScalarExp = std::make_shared<Primitive>("scalar_exp");
inline const PrimitivePtr kPrimScalarLog = std::make_shared<Primitive>("scalar_log");
inline const PrimitivePtr kPrimScalarSin = std::make_shared<Primitive>("scalar_sin");
inline const PrimitivePtr kPrimScalarCos = std::make_shared<Primitive>("scalar_cos");
inline const PrimitivePtr kPrimScalarTan = std::make_shared<Primitive>("scalar_tan");

// Comparisons
inline const PrimitivePtr kPrimScalarEq = std::make_shared<Primitive>("scalar_eq");
inline const PrimitivePtr kPrimScalarLt = std::make_shared<Primitive>("scalar_lt");
inline const PrimitivePtr kPrimScalarGt = std::make_shared<Primitive>("scalar_gt");
inline const PrimitivePtr kPrimScalarNe = std::make_shared<Primitive>("scalar_ne");
inline const PrimitivePtr kPrimScalarLe = std::make_shared<Primitive>("scalar_le");
inline const PrimitivePtr kPrimScalarGe = std::make_shared<Primitive>("scalar_ge");
inline const PrimitivePtr kPrimBoolNot = std::make_shared<Primitive>("bool_not");
inline const PrimitivePtr kPrimBoolAnd = std::make_shared<Primitive>("bool_and");
inline const PrimitivePtr kPrimBoolOr = std::make_shared<Primitive>("bool_or");
inline const PrimitivePtr kPrimBoolEq = std::make_shared<Primitive>("bool_eq");
inline const PrimitivePtr kPrimGreater = std::make_shared<Primitive>("Greater");
inline const PrimitivePtr kPrimGreaterEqual = std::make_shared<Primitive>("GreaterEqual");
inline const PrimitivePtr kPrimLess = std::make_shared<Primitive>("Less");
inline const PrimitivePtr kPrimLessEqual = std::make_shared<Primitive>("LessEqual");
inline const PrimitivePtr kPrimEqual = std::make_shared<Primitive>("Equal");
inline const PrimitivePtr kPrimNotEqual = std::make_shared<Primitive>("NotEqual");
inline const PrimitivePtr kPrimLogicalAnd = std::make_shared<Primitive>("LogicalAnd");
inline const PrimitivePtr kPrimLogicalOr = std::make_shared<Primitive>("LogicalOr");
inline const PrimitivePtr kPrimLogicalNot = std::make_shared<Primitive>("LogicalNot");

inline const PrimitivePtr kPrimDistribute = std::make_shared<Primitive>("distribute");
inline const PrimitivePtr kPrimDot = std::make_shared<Primitive>("dot");
inline const PrimitivePtr kPrimIm2Col = std::make_shared<Primitive>("im2col");
inline const PrimitivePtr kPrimCol2Im = std::make_shared<Primitive>("col2im");
inline const PrimitivePtr kPrimIm2ColV1 = std::make_shared<Primitive>("im2col_v1");
inline const PrimitivePtr kPrimCol2ImV1 = std::make_shared<Primitive>("col2im_v1");

inline const PrimitivePtr kPrimLabelGoto = std::make_shared<Primitive>("LabelGoto");
inline const PrimitivePtr kPrimLabelSwitch = std::make_shared<Primitive>("LabelSwitch");
inline const PrimitivePtr kPrimLabelSet = std::make_shared<Primitive>("LabelSet");

// Arrays
inline const PrimitivePtr kPrimScalarToArray = std::make_shared<Primitive>("scalar_to_array");
inline const PrimitivePtr kPrimArrayToScalar = std::make_shared<Primitive>("array_to_scalar");
inline const PrimitivePtr kPrimBroadcastShape = std::make_shared<Primitive>("broadcast_shape");
inline const PrimitivePtr kPrimArrayMap = std::make_shared<Primitive>("array_map");
inline const PrimitivePtr kPrimArrayReduce = std::make_shared<Primitive>("array_reduce");
inline const PrimitivePtr kPrimCast = std::make_shared<Primitive>("Cast");
inline const PrimitivePtr kPrimConcat = std::make_shared<Primitive>("Concat");
inline const PrimitivePtr kPrimSqueeze = std::make_shared<Primitive>("Squeeze");
inline const PrimitivePtr kPrimTranspose = std::make_shared<Primitive>("Transpose");
inline const PrimitivePtr kPrimGatherV2 = std::make_shared<Primitive>("GatherV2");
inline const PrimitivePtr kPrimGatherD = std::make_shared<Primitive>("GatherD");
inline const PrimitivePtr kPrimSparseGatherV2 = std::make_shared<Primitive>("SparseGatherV2");
inline const PrimitivePtr kPrimShape = std::make_shared<Primitive>("Shape");
inline const PrimitivePtr kPrimDynamicShape = std::make_shared<Primitive>("DynamicShape");
inline const PrimitivePtr kPrimEmbeddingLookup = std::make_shared<Primitive>("EmbeddingLookup");
inline const PrimitivePtr kPrimEmbeddingLookupCommGrad = std::make_shared<Primitive>("EmbeddingLookupCommGrad");
inline const PrimitivePtr kPrimSize = std::make_shared<Primitive>("Size");
inline const PrimitivePtr kPrimArgMax = std::make_shared<Primitive>("Argmax");
inline const PrimitivePtr kPrimPack = std::make_shared<Primitive>("Pack");
inline const PrimitivePtr kPrimUnsortedSegmentMax = std::make_shared<Primitive>("UnsortedSegmentMax");
inline const PrimitivePtr kPrimUnsortedSegmentSum = std::make_shared<Primitive>("UnsortedSegmentSum");
inline const PrimitivePtr kPrimUnsortedSegmentMin = std::make_shared<Primitive>("UnsortedSegmentMin");
inline const PrimitivePtr kPrimConcatOffset = std::make_shared<Primitive>("ConcatOffset");
inline const PrimitivePtr kPrimReshape = std::make_shared<Primitive>("Reshape");
inline const PrimitivePtr kPrimSubAndFilter = std::make_shared<Primitive>("SubAndFilter");
inline const PrimitivePtr kPrimMapCacheIdx = std::make_shared<Primitive>("MapCacheIdx");
inline const PrimitivePtr kPrimUpdateCache = std::make_shared<Primitive>("UpdateCache");
inline const PrimitivePtr kPrimComputeAccidentalHits = std::make_shared<Primitive>("ComputeAccidentalHits");
inline const PrimitivePtr kPrimCacheSwapTable = std::make_shared<Primitive>("CacheSwapTable");
inline const PrimitivePtr kPrimDynamicAssign = std::make_shared<Primitive>("DynamicAssign");
inline const PrimitivePtr kPrimPadAndShift = std::make_shared<Primitive>("PadAndShift");
inline const PrimitivePtr kPrimSlice = std::make_shared<Primitive>("Slice");
inline const PrimitivePtr kPrimTile = std::make_shared<Primitive>("Tile");
inline const PrimitivePtr kPrimAddN = std::make_shared<Primitive>("AddN");
inline const PrimitivePtr kPrimAccumulateNV2 = std::make_shared<Primitive>("AccumulateNV2");
inline const PrimitivePtr KPrimTransData = std::make_shared<Primitive>("TransData");
inline const PrimitivePtr kPrimNMSWithMask = std::make_shared<Primitive>("NMSWithMask");
inline const PrimitivePtr kPrimPad = std::make_shared<Primitive>("Pad");
inline const PrimitivePtr kPrimArgMaxWithValue = std::make_shared<Primitive>("ArgMaxWithValue");
inline const PrimitivePtr kPrimUnique = std::make_shared<Primitive>("Unique");
inline const PrimitivePtr kPrimUniqueGrad = std::make_shared<Primitive>("UniqueGrad");
inline const PrimitivePtr kPrimExtractImagePatches = std::make_shared<Primitive>("ExtractImagePatches");
inline const PrimitivePtr kPrimDynamicRNN = std::make_shared<Primitive>("DynamicRNN");
inline const PrimitivePtr kPrimDynamicRNNGrad = std::make_shared<Primitive>("DynamicRNNGrad");
inline const PrimitivePtr kPrimDynamicGRUV2 = std::make_shared<Primitive>("DynamicGRUV2");
inline const PrimitivePtr kPrimDynamicGRUV2Grad = std::make_shared<Primitive>("DynamicGRUV2Grad");
inline const PrimitivePtr kPrimScatterAdd = std::make_shared<Primitive>("ScatterAdd");
inline const PrimitivePtr kPrimScatterUpdate = std::make_shared<Primitive>("ScatterUpdate");
inline const PrimitivePtr kPrimMapUniform = std::make_shared<Primitive>("MapUniform");
inline const PrimitivePtr kPrimSplit = std::make_shared<Primitive>("Split");
inline const PrimitivePtr kPrimSequenceMask = std::make_shared<Primitive>("SequenceMask");
inline const PrimitivePtr kPrimRange = std::make_shared<Primitive>("Range");

// NN
inline const PrimitivePtr kPrimFlatten = std::make_shared<Primitive>("Flatten");
inline const PrimitivePtr kPrimSoftMax = std::make_shared<Primitive>("SoftMax");
inline const PrimitivePtr kPrimLogSoftmax = std::make_shared<Primitive>("LogSoftmax");
inline const PrimitivePtr kPrimLogSoftmaxGrad = std::make_shared<Primitive>("LogSoftmaxGrad");
inline const PrimitivePtr kPrimTanh = std::make_shared<Primitive>("Tanh");
inline const PrimitivePtr kPrimTanhGrad = std::make_shared<Primitive>("TanhGrad");
inline const PrimitivePtr kPrimPooling = std::make_shared<Primitive>("Pooling");
inline const PrimitivePtr kPrimPoolingGrad = std::make_shared<Primitive>("PoolingGrad");
inline const PrimitivePtr kPrimMaxPool = std::make_shared<Primitive>("MaxPool");
inline const PrimitivePtr kPrimMaxPoolGrad = std::make_shared<Primitive>("MaxPoolGrad");
inline const PrimitivePtr kPrimMaxPoolWithArgmax = std::make_shared<Primitive>("MaxPoolWithArgmax");
inline const PrimitivePtr kPrimMaxPoolGradWithArgmax = std::make_shared<Primitive>("MaxPoolGradWithArgmax");
inline const PrimitivePtr kPrimApplyCenteredRMSProp = std::make_shared<Primitive>("ApplyCenteredRMSProp");
inline const PrimitivePtr kPrimAvgPool = std::make_shared<Primitive>("AvgPool");
inline const PrimitivePtr kPrimAvgPoolGrad = std::make_shared<Primitive>("AvgPoolGrad");
inline const PrimitivePtr kPrimAvgPoolGradVm = std::make_shared<Primitive>("AvgPoolGradVm");
inline const PrimitivePtr kPrimAvgPoolGradCpu = std::make_shared<Primitive>("AvgPoolGradCpu");
inline const PrimitivePtr kPrimFusedSparseAdam = std::make_shared<Primitive>("FusedSparseAdam");
inline const PrimitivePtr kPrimFusedBatchNorm = std::make_shared<Primitive>("FusedBatchNorm");
inline const PrimitivePtr kPrimFusedBatchNormEx = std::make_shared<Primitive>("FusedBatchNormEx");
inline const PrimitivePtr kPrimConv2D = std::make_shared<Primitive>("Conv2D");
inline const PrimitivePtr kPrimFusedBatchNormGrad = std::make_shared<Primitive>("FusedBatchNormGrad");
inline const PrimitivePtr kPrimFusedBatchNormGradEx = std::make_shared<Primitive>("FusedBatchNormGradEx");
inline const PrimitivePtr kPrimBatchNorm = std::make_shared<Primitive>("BatchNorm");
inline const PrimitivePtr kPrimBatchNormGrad = std::make_shared<Primitive>("BatchNormGrad");
inline const PrimitivePtr kPrimReluGrad = std::make_shared<Primitive>("ReluGrad");
inline const PrimitivePtr kPrimReluGradV2 = std::make_shared<Primitive>("ReluGradV2");
inline const PrimitivePtr kPrimRelu6Grad = std::make_shared<Primitive>("ReLU6Grad");
inline const PrimitivePtr kPrimConv2DBackpropInput = std::make_shared<Primitive>("Conv2DBackpropInput");
inline const PrimitivePtr kPrimConv2DBackpropFilter = std::make_shared<Primitive>("Conv2DBackpropFilter");
inline const PrimitivePtr kPrimConv3DBackpropInput = std::make_shared<Primitive>("Conv3DBackpropInput");
inline const PrimitivePtr kPrimConv3DBackpropFilter = std::make_shared<Primitive>("Conv3DBackpropFilter");
inline const PrimitivePtr kPrimDepthwiseConv2dNative = std::make_shared<Primitive>("DepthwiseConv2dNative");
inline const PrimitivePtr kPrimCTCGreedyDecoder = std::make_shared<Primitive>("CTCGreedyDecoder");
inline const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropFilter =
  std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropFilter");
inline const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropInput =
  std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropInput");
inline const PrimitivePtr kPrimBiasAdd = std::make_shared<Primitive>("BiasAdd");
inline const PrimitivePtr kPrimBiasAddGrad = std::make_shared<Primitive>("BiasAddGrad");
inline const PrimitivePtr kPrimSoftmaxCrossEntropyWithLogits =
  std::make_shared<Primitive>("SoftmaxCrossEntropyWithLogits");
inline const PrimitivePtr kPrimSparseSoftmaxCrossEntropyWithLogits =
  std::make_shared<Primitive>("SparseSoftmaxCrossEntropyWithLogits");
inline const PrimitivePtr kPrimMomentum = std::make_shared<Primitive>("Momentum");
inline const PrimitivePtr kPrimApplyMomentum = std::make_shared<Primitive>("ApplyMomentum");
inline const PrimitivePtr kPrimLayerNorm = std::make_shared<Primitive>("LayerNorm");
inline const PrimitivePtr kPrimLayerNormGrad = std::make_shared<Primitive>("LayerNormGrad");
inline const PrimitivePtr kPrimLayerNormXBackprop = std::make_shared<Primitive>("LayerNormXBackprop");
inline const PrimitivePtr kPrimLayerNormBetaGammaBackprop = std::make_shared<Primitive>("LayerNormBetaGammaBackprop");
inline const PrimitivePtr kPrimDropoutGenMask = std::make_shared<Primitive>("DropoutGenMask");
inline const PrimitivePtr kPrimDropoutDoMask = std::make_shared<Primitive>("DropoutDoMask");
inline const PrimitivePtr kPrimDropoutGrad = std::make_shared<Primitive>("DropoutGrad");
inline const PrimitivePtr kPrimDropout = std::make_shared<Primitive>("Dropout");
inline const PrimitivePtr kPrimUniformReal = std::make_shared<Primitive>("UniformReal");
inline const PrimitivePtr kPrimCudnnUniformReal = std::make_shared<Primitive>("CudnnUniformReal");
inline const PrimitivePtr kPrimOneHot = std::make_shared<Primitive>("OneHot");
inline const PrimitivePtr kPrimGelu = std::make_shared<Primitive>("Gelu");
inline const PrimitivePtr kPrimGeluGrad = std::make_shared<Primitive>("GeluGrad");
inline const PrimitivePtr kPrimFastGelu = std::make_shared<Primitive>("FastGelu");
inline const PrimitivePtr kPrimFastGeluGrad = std::make_shared<Primitive>("FastGeluGrad");
inline const PrimitivePtr kPrimRelu = std::make_shared<Primitive>("ReLU");
inline const PrimitivePtr kPrimRelu6 = std::make_shared<Primitive>("ReLU6");
inline const PrimitivePtr kPrimReluV2 = std::make_shared<Primitive>("ReLUV2");
inline const PrimitivePtr kPrimZerosLike = std::make_shared<Primitive>("ZerosLike");
inline const PrimitivePtr kPrimOnesLike = std::make_shared<Primitive>("OnesLike");
inline const PrimitivePtr kPrimBpropCut = std::make_shared<Primitive>("bprop_cut");
inline const PrimitivePtr kPrimFakeQuantPerLayer = std::make_shared<Primitive>("FakeQuantPerLayer");
inline const PrimitivePtr kPrimFakeQuantPerChannel = std::make_shared<Primitive>("FakeQuantPerChannel");
inline const PrimitivePtr kPrimApplyRMSProp = std::make_shared<Primitive>("ApplyRMSProp");
inline const PrimitivePtr kPrimSparseApplyFtrl = std::make_shared<Primitive>("SparseApplyFtrl");
inline const PrimitivePtr kPrimSparseApplyProximalAdagrad = std::make_shared<Primitive>("SparseApplyProximalAdagrad");
inline const PrimitivePtr kPrimFusedAdam = std::make_shared<Primitive>("FusedAdam");
inline const PrimitivePtr kPrimFusedAdamWeightDecay = std::make_shared<Primitive>("FusedAdamWeightDecay");
inline const PrimitivePtr kPrimSGD = std::make_shared<Primitive>("SGD");
inline const PrimitivePtr kPrimClipByNormNoDivSum = std::make_shared<Primitive>("ClipByNormNoDivSum");
inline const PrimitivePtr kPrimTensorMove = std::make_shared<Primitive>("TensorMove");

// Comm ops
inline const PrimitivePtr kPrimMirror = std::make_shared<Primitive>("_MirrorOperator");
inline const PrimitivePtr kPrimMirrorMiniStep = std::make_shared<Primitive>("_MirrorMiniStepOperator");
inline const PrimitivePtr kPrimVirtualDiv = std::make_shared<Primitive>("_VirtualDiv");
inline const PrimitivePtr kPrimVirtualDataset = std::make_shared<Primitive>("_VirtualDataset");
inline const PrimitivePtr kPrimSend = std::make_shared<Primitive>("Send");
inline const PrimitivePtr kPrimReceive = std::make_shared<Primitive>("Receive");
inline const PrimitivePtr kPrimAllReduce = std::make_shared<Primitive>("AllReduce");
inline const PrimitivePtr kPrimAllSwap = std::make_shared<Primitive>("AllSwap");
inline const PrimitivePtr kPrimBroadcast = std::make_shared<Primitive>("Broadcast");
inline const PrimitivePtr kPrimAllGather = std::make_shared<Primitive>("AllGather");
inline const PrimitivePtr kPrimReduceScatter = std::make_shared<Primitive>("ReduceScatter");
inline const PrimitivePtr kPrimMemCpyAsync = std::make_shared<Primitive>("memcpy_async");
inline const PrimitivePtr kPrimFill = std::make_shared<Primitive>("Fill");
// RowTensor
inline const PrimitivePtr kPrimMakeRowTensor = std::make_shared<Primitive>("MakeRowTensor");
inline const PrimitivePtr kPrimRowTensorGetValues = std::make_shared<Primitive>("RowTensorGetValues");
inline const PrimitivePtr kPrimRowTensorGetIndices = std::make_shared<Primitive>("RowTensorGetIndices");
inline const PrimitivePtr kPrimRowTensorGetDenseShape = std::make_shared<Primitive>("RowTensorGetDenseShape");

// SparseTensor
inline const PrimitivePtr kPrimMakeSparseTensor = std::make_shared<Primitive>("MakeSparseTensor");
inline const PrimitivePtr kPrimSparseTensorGetValues = std::make_shared<Primitive>("SparseTensorGetValues");
inline const PrimitivePtr kPrimSparseTensorGetIndices = std::make_shared<Primitive>("SparseTensorGetIndices");
inline const PrimitivePtr kPrimSparseTensorGetDenseShape = std::make_shared<Primitive>("SparseTensorGetDenseShape");

// Maths
inline const PrimitivePtr kPrimTensorAdd = std::make_shared<Primitive>("TensorAdd");
inline const PrimitivePtr kPrimMatMul = std::make_shared<Primitive>("MatMul");
inline const PrimitivePtr kPrimBatchMatMul = std::make_shared<Primitive>("BatchMatMul");
inline const PrimitivePtr kPrimMaximumGrad = std::make_shared<Primitive>("MaximumGrad");
inline const PrimitivePtr kPrimMinimumGrad = std::make_shared<Primitive>("MinimumGrad");
inline const PrimitivePtr kPrimReduceMean = std::make_shared<Primitive>("ReduceMean");
inline const PrimitivePtr kPrimReduceSum = std::make_shared<Primitive>("ReduceSum");
inline const PrimitivePtr kPrimReduceAll = std::make_shared<Primitive>("ReduceAll");
inline const PrimitivePtr kPrimReduceAny = std::make_shared<Primitive>("ReduceAny");
inline const PrimitivePtr kPrimReduceMax = std::make_shared<Primitive>("ReduceMax");
inline const PrimitivePtr kPrimReduceMin = std::make_shared<Primitive>("ReduceMin");
inline const PrimitivePtr kPrimNeg = std::make_shared<Primitive>("Neg");
inline const PrimitivePtr kPrimSub = std::make_shared<Primitive>("Sub");
inline const PrimitivePtr kPrimMul = std::make_shared<Primitive>("Mul");
inline const PrimitivePtr kPrimDiv = std::make_shared<Primitive>("Div");
inline const PrimitivePtr kPrimMod = std::make_shared<Primitive>("Mod");
inline const PrimitivePtr kPrimFloor = std::make_shared<Primitive>("Floor");
inline const PrimitivePtr kPrimDivNoNan = std::make_shared<Primitive>("DivNoNan");
inline const PrimitivePtr kPrimMinimum = std::make_shared<Primitive>("Minimum");
inline const PrimitivePtr kPrimMaximum = std::make_shared<Primitive>("Maximum");
inline const PrimitivePtr kPrimSquare = std::make_shared<Primitive>("Square");
inline const PrimitivePtr kPrimCumSum = std::make_shared<Primitive>("CumSum");
inline const PrimitivePtr kPrimCumProd = std::make_shared<Primitive>("CumProd");
inline const PrimitivePtr kPrimSubscalar = std::make_shared<Primitive>("Subscalar");
inline const PrimitivePtr kPrimInplaceAdd = std::make_shared<Primitive>("InplaceAdd");
inline const PrimitivePtr kPrimInplaceSub = std::make_shared<Primitive>("InplaceSub");
inline const PrimitivePtr kPrimPow = std::make_shared<Primitive>("Pow");
inline const PrimitivePtr kPrimRealDiv = std::make_shared<Primitive>("RealDiv");
inline const PrimitivePtr kPrimFloorDiv = std::make_shared<Primitive>("FloorDiv");
inline const PrimitivePtr kPrimSqrt = std::make_shared<Primitive>("Sqrt");
inline const PrimitivePtr kPrimSqrtGrad = std::make_shared<Primitive>("SqrtGrad");
inline const PrimitivePtr kPrimReciprocal = std::make_shared<Primitive>("Reciprocal");
inline const PrimitivePtr kPrimExpandDims = std::make_shared<Primitive>("ExpandDims");
inline const PrimitivePtr kPrimAbs = std::make_shared<Primitive>("Abs");
inline const PrimitivePtr kPrimRound = std::make_shared<Primitive>("Round");
inline const PrimitivePtr kPrimExp = std::make_shared<Primitive>("Exp");
inline const PrimitivePtr kPrimLog = std::make_shared<Primitive>("Log");
inline const PrimitivePtr kPrimRsqrt = std::make_shared<Primitive>("Rsqrt");
inline const PrimitivePtr kPrimSplitV = std::make_shared<Primitive>("SplitV");
inline const PrimitivePtr kPrimLinSpace = std::make_shared<Primitive>("LinSpace");
inline const PrimitivePtr kPrimSign = std::make_shared<Primitive>("Sign");
inline const PrimitivePtr kPrimSquaredDifference = std::make_shared<Primitive>("SquaredDifference");

// Statements
inline const PrimitivePtr kPrimReturn = std::make_shared<Primitive>("return");
inline const PrimitivePtr kPrimSwitch = std::make_shared<Primitive>("switch");
inline const PrimitivePtr kPrimSwitchLayer = std::make_shared<Primitive>("switch_layer");
inline const PrimitivePtr kPrimAssign = std::make_shared<Primitive>("Assign");
inline const PrimitivePtr kPrimAssignAdd = std::make_shared<Primitive>("AssignAdd");
inline const PrimitivePtr kPrimAssignSub = std::make_shared<Primitive>("AssignSub");
inline const PrimitivePtr kPrimSelect = std::make_shared<Primitive>("Select");
inline const PrimitivePtr kPrimCall = std::make_shared<Primitive>("call");

inline const PrimitivePtr kPrimMakeTuple = std::make_shared<Primitive>("make_tuple");
inline const PrimitivePtr kPrimMakeSlice = std::make_shared<Primitive>("make_slice");
inline const PrimitivePtr kPrimTupleGetItem = std::make_shared<Primitive>("tuple_getitem");
inline const PrimitivePtr kPrimArrayGetItem = std::make_shared<Primitive>("array_getitem");
inline const PrimitivePtr kPrimTupleSetItem = std::make_shared<Primitive>("tuple_setitem");
inline const PrimitivePtr kPrimArraySetItem = std::make_shared<Primitive>("array_setitem");
inline const PrimitivePtr kPrimGetAttr = std::make_shared<Primitive>("getattr");
inline const PrimitivePtr kPrimTupleLen = std::make_shared<Primitive>("tuple_len");
inline const PrimitivePtr kPrimArrayLen = std::make_shared<Primitive>("array_len");
inline const PrimitivePtr kPrimTileShape = std::make_shared<Primitive>("tile_shape");
inline const PrimitivePtr kPrimGenerateShapeIndex = std::make_shared<Primitive>("generate_shape_index");
inline const PrimitivePtr kPrimGenerateInverseIndex = std::make_shared<Primitive>("generate_inverse_index");

// Debug ops
inline const PrimitivePtr kPrimScalarSummary = std::make_shared<Primitive>("ScalarSummary");
inline const PrimitivePtr kPrimImageSummary = std::make_shared<Primitive>("ImageSummary");
inline const PrimitivePtr kPrimTensorSummary = std::make_shared<Primitive>("TensorSummary");
inline const PrimitivePtr kPrimHistogramSummary = std::make_shared<Primitive>("HistogramSummary");
inline const PrimitivePtr kPrimDebug = std::make_shared<Primitive>("Debug");

// Dynamic shape testing
inline const PrimitivePtr kPrimGpuConvertToDynamicShape = std::make_shared<Primitive>("GpuConvertToDynamicShape");
inline const PrimitivePtr kPrimErrorOnDynamicShapeInput = std::make_shared<Primitive>("ErrorOnDynamicShapeInput");

// Other miscellaneous
inline const PrimitivePtr kPrimDepend = std::make_shared<Primitive>("Depend");
inline const PrimitivePtr kPrimReformat = std::make_shared<Primitive>("Reformat");
inline const PrimitivePtr kPrimPartial = std::make_shared<Primitive>("Partial");
inline const PrimitivePtr kPrimIdentity = std::make_shared<Primitive>("identity");
inline const PrimitivePtr kPrimHookBackward = std::make_shared<Primitive>("HookBackward");
inline const PrimitivePtr kPrimPrintShapeType = std::make_shared<Primitive>("PrintShapeType");
inline const PrimitivePtr kPrimSameTypeShape = std::make_shared<Primitive>("SameTypeShape");
inline const PrimitivePtr kPrimPrint = std::make_shared<Primitive>("Print");
inline const PrimitivePtr kPrimControlDepend = std::make_shared<Primitive>("ControlDepend");
inline const PrimitivePtr kPrimIs_ = std::make_shared<Primitive>("is_");
inline const PrimitivePtr kPrimIsNot = std::make_shared<Primitive>("is_not");
inline const PrimitivePtr kPrimInDict = std::make_shared<Primitive>("in_dict");
inline const PrimitivePtr kPrimNotInDict = std::make_shared<Primitive>("not_in_dict");
inline const PrimitivePtr kPrimIsConsant = std::make_shared<Primitive>("is_constant");
inline const PrimitivePtr kPrimEquivFormat = std::make_shared<Primitive>("EquivFormat");

// Structures
inline const PrimitivePtr kPrimMakeList = std::make_shared<Primitive>("make_list");
inline const PrimitivePtr kPrimMakeKeywordArg = std::make_shared<Primitive>("make_keyword_arg");
inline const PrimitivePtr kPrimListGetItem = std::make_shared<Primitive>("list_getitem");
inline const PrimitivePtr kPrimListSetItem = std::make_shared<Primitive>("list_setitem");
inline const PrimitivePtr kPrimDictGetItem = std::make_shared<Primitive>("dict_getitem");
inline const PrimitivePtr kPrimDictSetItem = std::make_shared<Primitive>("dict_setitem");
inline const PrimitivePtr kPrimDictGetKeys = std::make_shared<Primitive>("dict_getkeys");
inline const PrimitivePtr kPrimDictGetValues = std::make_shared<Primitive>("dict_getvalues");
inline const PrimitivePtr kPrimListAppend = std::make_shared<Primitive>("list_append");
inline const PrimitivePtr kPrimListLen = std::make_shared<Primitive>("list_len");

// Other miscellaneous
inline const PrimitivePtr kPrimEnvSetItem = std::make_shared<Primitive>("env_setitem");
inline const PrimitivePtr kPrimEnvGetItem = std::make_shared<Primitive>("env_getitem");
inline const PrimitivePtr kPrimEnvAdd = std::make_shared<Primitive>("env_add");
inline const PrimitivePtr kPrimMakeRefKey = std::make_shared<Primitive>("MakeRefKey");
inline const PrimitivePtr kPrimGetRefKey = std::make_shared<Primitive>("get_ref_key");
inline const PrimitivePtr kPrimMakeRef = std::make_shared<Primitive>("make_ref");
inline const PrimitivePtr kPrimGetRefValue = std::make_shared<Primitive>("get_ref_value");

// Other primitve not used by backend but used in core;
inline const PrimitivePtr kPrimStateSetItem = std::make_shared<Primitive>("state_setitem");
inline const PrimitivePtr kPrimJ = std::make_shared<Primitive>("J");

// Used to build graph which have keyword arguments
inline const PrimitivePtr kPrimExtractKeywordArg = std::make_shared<Primitive>("extract_keyword_arg");
inline const PrimitivePtr kPrimMakeDict = std::make_shared<Primitive>("make_dict");

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

#endif  // MINDSPORE_CORE_OPERATOR_OPS_H_
