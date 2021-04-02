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
#include "nnacl/infer/infer_register.h"
#ifdef MS_COMPILE_IOS
extern void _AudioSpectrogramPrimType_AudioSpectrogram();
extern void _EqualPrimType_Equal();
extern void _GreaterPrimType_Greater();
extern void _GreaterEqualPrimType_GreaterEqual();
extern void _LessPrimType_Less();
extern void _LessEqualPrimType_LessEqual();
extern void _NotEqualPrimType_NotEqual();
extern void _DivGradPrimType_DivGrad();
extern void _MulGradPrimType_MulGrad();
extern void _MinimumGradPrimType_MinimumGrad();
extern void _FullConnectionPrimType_FullConnection();
extern void _AdamPrimType_Adam();
extern void _SlicePrimType_SliceFusion();
extern void _LSTMPrimType_LSTM();
extern void _TilePrimType_TileFusion();
extern void _EmbeddingLookupPrimType_EmbeddingLookupFusion();
extern void _AssignAddPrimType_AssignAdd();
extern void _LinSpacePrimType_LinSpace();
extern void _AddGradPrimType_AddGrad();
extern void _SubGradPrimType_SubGrad();
extern void _OneHotPrimType_OneHot();
extern void _RandomStandardNormalPrimType_RandomStandardNormal();
extern void _StridedSlicePrimType_StridedSlice();
extern void _FillPrimType_Fill();
extern void _CastPrimType_Cast();
extern void _SGDPrimType_SGD();
extern void _UniquePrimType_Unique();
extern void _BatchToSpacePrimType_BatchToSpace();
extern void _TensorListFromTensorPrimType_TensorListFromTensor();
extern void _TensorListGetItemPrimType_TensorListGetItem();
extern void _SoftmaxCrossEntropyWithLogitsPrimType_SoftmaxCrossEntropyWithLogits();
extern void _ExpandDimsPrimType_ExpandDims();
extern void _InvertPermutationPrimType_InvertPermutation();
extern void _MergePrimType_Merge();
extern void _RfftPrimType_Rfft();
extern void _Conv2DBackpropInputFusionPrimType_Conv2DBackpropInputFusion();
extern void _ScatterNdPrimType_ScatterNd();
extern void _FusedBatchNormPrimType_FusedBatchNorm();
extern void _PartialPrimType_PartialFusion();
extern void _HashtableLookupPrimType_HashtableLookup();
extern void _ReducePrimType_ReduceFusion();
extern void _GatherPrimType_Gather();
extern void _SplitPrimType_Split();
extern void _RangePrimType_Range();
extern void _BroadcastToPrimType_BroadcastTo();
extern void _UnsortedSegmentSumPrimType_UnsortedSegmentSum();
extern void _SqueezePrimType_Squeeze();
extern void _MaximumGradPrimType_MaximumGrad();
extern void _PowPrimType_PowFusion();
extern void _PriorBoxPrimType_PriorBox();
extern void _SpaceToDepthPrimType_SpaceToDepth();
extern void _RankPrimType_Rank();
extern void _MfccPrimType_Mfcc();
extern void _BinaryCrossEntropyPrimType_BinaryCrossEntropy();
extern void _Conv2DBackpropFilterFusionPrimType_Conv2DBackpropFilterFusion();
extern void _WherePrimType_Where();
extern void _DetectionPostProcessPrimType_DetectionPostProcess();
extern void _FlattenGradPrimType_FlattenGrad();
extern void _ConstantOfShapePrimType_ConstantOfShape();
extern void _SelectPrimType_Select();
extern void _ConcatPrimType_Concat();
extern void _StackPrimType_Stack();
extern void _SwitchPrimType_Switch();
extern void _NonMaxSuppressionPrimType_NonMaxSuppression();
extern void _SkipGramPrimType_SkipGram();
extern void _AddNPrimType_AddN();
extern void _ArgMinPrimType_ArgMinFusion();
extern void _ArgMaxPrimType_ArgMaxFusion();
extern void _LayerNormFusionPrimType_LayerNormFusion();
extern void _MaxPoolPrimType_MaxPoolFusion();
extern void _AvgPoolPrimType_AvgPoolFusion();
extern void _AssignPrimType_Assign();
extern void _CropPrimType_Crop();
extern void _UnsqueezePrimType_Unsqueeze();
extern void _AbsPrimType_Abs();
extern void _AbsGradPrimType_AbsGrad();
extern void _ActivationPrimType_Activation();
extern void _ActivationGradPrimType_ActivationGrad();
extern void _BatchNormPrimType_BatchNorm();
extern void _BinaryCrossEntropyGradPrimType_BinaryCrossEntropyGrad();
extern void _BiasAddPrimType_BiasAdd();
extern void _CeilPrimType_Ceil();
extern void _ClipPrimType_Clip();
extern void _ControlDependPrimType_ControlDepend();
extern void _CosPrimType_Cos();
extern void _DependPrimType_Depend();
extern void _EluPrimType_Elu();
extern void _ErfPrimType_Erf();
extern void _ExpPrimType_ExpFusion();
extern void _FakeQuantWithMinMaxVarsPrimType_FakeQuantWithMinMaxVars();
extern void _FloorPrimType_Floor();
extern void _IfPrimType_If();
extern void _InstanceNormPrimType_InstanceNorm();
extern void _IsFinitePrimType_IsFinite();
extern void _LeakyReluPrimType_LeakyRelu();
extern void _LogPrimType_Log();
extern void _LogGradPrimType_LogGrad();
extern void _LogicalNotPrimType_LogicalNot();
extern void _LRNPrimType_LRN();
extern void _L2NormalizePrimType_L2NormalizeFusion();
extern void _NegPrimType_Neg();
extern void _NegGradPrimType_NegGrad();
extern void _PowerGradPrimType_PowerGrad();
extern void _PReLUPrimType_PReLUFusion();
extern void _ReciprocalPrimType_Reciprocal();
extern void _ReverseSequencePrimType_ReverseSequence();
extern void _ReversePrimType_ReverseV2();
extern void _RoundPrimType_Round();
extern void _RsqrtPrimType_Rsqrt();
extern void _ScalePrimType_ScaleFusion();
extern void _SigmoidCrossEntropyWithLogitsPrimType_SigmoidCrossEntropyWithLogits();
extern void _SigmoidCrossEntropyWithLogitsGradPrimType_SigmoidCrossEntropyWithLogitsGrad();
extern void _SinPrimType_Sin();
extern void _SmoothL1LossPrimType_SmoothL1Loss();
extern void _SmoothL1LossGradPrimType_SmoothL1LossGrad();
extern void _SqrtPrimType_Sqrt();
extern void _SquarePrimType_Square();
extern void _ZerosLikePrimType_ZerosLike();
extern void _AssertPrimType_Assert();
extern void _TensorListSetItemPrimType_TensorListSetItem();
extern void _SplicePrimType_Splice();
extern void _SparseSoftmaxCrossEntropyWithLogitsPrimType_SparseSoftmaxCrossEntropyWithLogits();
extern void _GRUPrimType_GRU();
extern void _SizeOpPrimType_Size();
extern void _CustomExtractFeaturesPrimType_CustomExtractFeatures();
extern void _CustomPredictPrimType_CustomPredict();
extern void _WhilePrimType_While();
extern void _StridedSliceGradPrimType_StridedSliceGrad();
extern void _LshProjectionPrimType_LshProjection();
extern void _SoftmaxPrimType_Softmax();
extern void _CustomNormalizePrimType_CustomNormalize();
extern void _UnstackPrimType_Unstack();
extern void _ROIPoolingPrimType_ROIPooling();
extern void _LayerNormGradPrimType_LayerNormGrad();
extern void _DropoutGradPrimType_DropoutGrad();
extern void _TopKPrimType_TopKFusion();
extern void _ApplyMomentumPrimType_ApplyMomentum();
extern void _AdderPrimType_AdderFusion();
extern void _Conv2DPrimType_Conv2DFusion();
extern void _UniformRealPrimType_UniformReal();
extern void _AvgPoolGradPrimType_AvgPoolGrad();
extern void _MaxPoolGradPrimType_MaxPoolGrad();
extern void _DepthToSpacePrimType_DepthToSpace();
extern void _Conv2dTransposePrimType_Conv2dTransposeFusion();
extern void _QuantDTypeCastPrimType_QuantDTypeCast();
extern void _FftImagPrimType_FftImag();
extern void _FftRealPrimType_FftReal();
extern void _ResizePrimType_Resize();
extern void _SpaceToBatchNDPrimType_SpaceToBatchND();
extern void _TransposePrimType_Transpose();
extern void _TensorListReservePrimType_TensorListReserve();
extern void _ShapePrimType_Shape();
extern void _BiasAddGradPrimType_BiasAddGrad();
extern void _MatMulPrimType_MatMul();
extern void _DropoutPrimType_Dropout();
extern void _ReshapePrimType_Reshape();
extern void _PadPrimType_PadFusion();
extern void _TensorListStackPrimType_TensorListStack();
extern void _FlattenPrimType_Flatten();
extern void _BatchNormGradPrimType_BatchNormGrad();
extern void _SpaceToBatchPrimType_SpaceToBatch();
extern void _GatherNdPrimType_GatherNd();
extern void _CropAndResizePrimType_CropAndResize();
extern void _SparseToDensePrimType_SparseToDense();
extern void _AddPrimType_AddFusion();
extern void _DivPrimType_DivFusion();
extern void _EltwisePrimType_Eltwise();
extern void _FloorDivPrimType_FloorDiv();
extern void _FloorModPrimType_FloorMod();
extern void _LogicalAndPrimType_LogicalAnd();
extern void _LogicalOrPrimType_LogicalOr();
extern void _MaximumPrimType_Maximum();
extern void _MinimumPrimType_Minimum();
extern void _ModPrimType_Mod();
extern void _MulPrimType_MulFusion();
extern void _RealDivPrimType_RealDiv();
extern void _SubPrimType_SubFusion();
extern void _SquaredDifferencePrimType_SquaredDifference();

void RegisterInfer() {
  _AudioSpectrogramPrimType_AudioSpectrogram();
  _EqualPrimType_Equal();
  _GreaterPrimType_Greater();
  _GreaterEqualPrimType_GreaterEqual();
  _LessPrimType_Less();
  _LessEqualPrimType_LessEqual();
  _NotEqualPrimType_NotEqual();
  _DivGradPrimType_DivGrad();
  _MulGradPrimType_MulGrad();
  _MinimumGradPrimType_MinimumGrad();
  _FullConnectionPrimType_FullConnection();
  _AdamPrimType_Adam();
  _SlicePrimType_SliceFusion();
  _LSTMPrimType_LSTM();
  _TilePrimType_TileFusion();
  _EmbeddingLookupPrimType_EmbeddingLookupFusion();
  _AssignAddPrimType_AssignAdd();
  _LinSpacePrimType_LinSpace();
  _AddGradPrimType_AddGrad();
  _SubGradPrimType_SubGrad();
  _OneHotPrimType_OneHot();
  _RandomStandardNormalPrimType_RandomStandardNormal();
  _StridedSlicePrimType_StridedSlice();
  _FillPrimType_Fill();
  _CastPrimType_Cast();
  _SGDPrimType_SGD();
  _UniquePrimType_Unique();
  _BatchToSpacePrimType_BatchToSpace();
  _TensorListFromTensorPrimType_TensorListFromTensor();
  _TensorListGetItemPrimType_TensorListGetItem();
  _SoftmaxCrossEntropyWithLogitsPrimType_SoftmaxCrossEntropyWithLogits();
  _ExpandDimsPrimType_ExpandDims();
  _InvertPermutationPrimType_InvertPermutation();
  _MergePrimType_Merge();
  _RfftPrimType_Rfft();
  _Conv2DBackpropInputFusionPrimType_Conv2DBackpropInputFusion();
  _ScatterNdPrimType_ScatterNd();
  _FusedBatchNormPrimType_FusedBatchNorm();
  _PartialPrimType_PartialFusion();
  _HashtableLookupPrimType_HashtableLookup();
  _ReducePrimType_ReduceFusion();
  _GatherPrimType_Gather();
  _SplitPrimType_Split();
  _RangePrimType_Range();
  _BroadcastToPrimType_BroadcastTo();
  _UnsortedSegmentSumPrimType_UnsortedSegmentSum();
  _SqueezePrimType_Squeeze();
  _MaximumGradPrimType_MaximumGrad();
  _PowPrimType_PowFusion();
  _PriorBoxPrimType_PriorBox();
  _SpaceToDepthPrimType_SpaceToDepth();
  _RankPrimType_Rank();
  _MfccPrimType_Mfcc();
  _BinaryCrossEntropyPrimType_BinaryCrossEntropy();
  _Conv2DBackpropFilterFusionPrimType_Conv2DBackpropFilterFusion();
  _WherePrimType_Where();
  _DetectionPostProcessPrimType_DetectionPostProcess();
  _FlattenGradPrimType_FlattenGrad();
  _ConstantOfShapePrimType_ConstantOfShape();
  _SelectPrimType_Select();
  _ConcatPrimType_Concat();
  _StackPrimType_Stack();
  _SwitchPrimType_Switch();
  _NonMaxSuppressionPrimType_NonMaxSuppression();
  _SkipGramPrimType_SkipGram();
  _AddNPrimType_AddN();
  _ArgMinPrimType_ArgMinFusion();
  _ArgMaxPrimType_ArgMaxFusion();
  _LayerNormFusionPrimType_LayerNormFusion();
  _MaxPoolPrimType_MaxPoolFusion();
  _AvgPoolPrimType_AvgPoolFusion();
  _AssignPrimType_Assign();
  _CropPrimType_Crop();
  _UnsqueezePrimType_Unsqueeze();
  _AbsPrimType_Abs();
  _AbsGradPrimType_AbsGrad();
  _ActivationPrimType_Activation();
  _ActivationGradPrimType_ActivationGrad();
  _BatchNormPrimType_BatchNorm();
  _BinaryCrossEntropyGradPrimType_BinaryCrossEntropyGrad();
  _BiasAddPrimType_BiasAdd();
  _CeilPrimType_Ceil();
  _ClipPrimType_Clip();
  _ControlDependPrimType_ControlDepend();
  _CosPrimType_Cos();
  _DependPrimType_Depend();
  _EluPrimType_Elu();
  _ErfPrimType_Erf();
  _ExpPrimType_ExpFusion();
  _FakeQuantWithMinMaxVarsPrimType_FakeQuantWithMinMaxVars();
  _FloorPrimType_Floor();
  _IfPrimType_If();
  _InstanceNormPrimType_InstanceNorm();
  _IsFinitePrimType_IsFinite();
  _LeakyReluPrimType_LeakyRelu();
  _LogPrimType_Log();
  _LogGradPrimType_LogGrad();
  _LogicalNotPrimType_LogicalNot();
  _LRNPrimType_LRN();
  _L2NormalizePrimType_L2NormalizeFusion();
  _NegPrimType_Neg();
  _NegGradPrimType_NegGrad();
  _PowerGradPrimType_PowerGrad();
  _PReLUPrimType_PReLUFusion();
  _ReciprocalPrimType_Reciprocal();
  _ReverseSequencePrimType_ReverseSequence();
  _ReversePrimType_ReverseV2();
  _RoundPrimType_Round();
  _RsqrtPrimType_Rsqrt();
  _ScalePrimType_ScaleFusion();
  _SigmoidCrossEntropyWithLogitsPrimType_SigmoidCrossEntropyWithLogits();
  _SigmoidCrossEntropyWithLogitsGradPrimType_SigmoidCrossEntropyWithLogitsGrad();
  _SinPrimType_Sin();
  _SmoothL1LossPrimType_SmoothL1Loss();
  _SmoothL1LossGradPrimType_SmoothL1LossGrad();
  _SqrtPrimType_Sqrt();
  _SquarePrimType_Square();
  _ZerosLikePrimType_ZerosLike();
  _AssertPrimType_Assert();
  _TensorListSetItemPrimType_TensorListSetItem();
  _SplicePrimType_Splice();
  _SparseSoftmaxCrossEntropyWithLogitsPrimType_SparseSoftmaxCrossEntropyWithLogits();
  _GRUPrimType_GRU();
  _SizeOpPrimType_Size();
  _CustomExtractFeaturesPrimType_CustomExtractFeatures();
  _CustomPredictPrimType_CustomPredict();
  _WhilePrimType_While();
  _StridedSliceGradPrimType_StridedSliceGrad();
  _LshProjectionPrimType_LshProjection();
  _SoftmaxPrimType_Softmax();
  _CustomNormalizePrimType_CustomNormalize();
  _UnstackPrimType_Unstack();
  _ROIPoolingPrimType_ROIPooling();
  _LayerNormGradPrimType_LayerNormGrad();
  _DropoutGradPrimType_DropoutGrad();
  _TopKPrimType_TopKFusion();
  _ApplyMomentumPrimType_ApplyMomentum();
  _AdderPrimType_AdderFusion();
  _Conv2DPrimType_Conv2DFusion();
  _UniformRealPrimType_UniformReal();
  _AvgPoolGradPrimType_AvgPoolGrad();
  _MaxPoolGradPrimType_MaxPoolGrad();
  _DepthToSpacePrimType_DepthToSpace();
  _Conv2dTransposePrimType_Conv2dTransposeFusion();
  _QuantDTypeCastPrimType_QuantDTypeCast();
  _FftImagPrimType_FftImag();
  _FftRealPrimType_FftReal();
  _ResizePrimType_Resize();
  _SpaceToBatchNDPrimType_SpaceToBatchND();
  _TransposePrimType_Transpose();
  _TensorListReservePrimType_TensorListReserve();
  _ShapePrimType_Shape();
  _BiasAddGradPrimType_BiasAddGrad();
  _MatMulPrimType_MatMul();
  _DropoutPrimType_Dropout();
  _ReshapePrimType_Reshape();
  _PadPrimType_PadFusion();
  _TensorListStackPrimType_TensorListStack();
  _FlattenPrimType_Flatten();
  _BatchNormGradPrimType_BatchNormGrad();
  _SpaceToBatchPrimType_SpaceToBatch();
  _GatherNdPrimType_GatherNd();
  _CropAndResizePrimType_CropAndResize();
  _SparseToDensePrimType_SparseToDense();
  _AddPrimType_AddFusion();
  _DivPrimType_DivFusion();
  _EltwisePrimType_Eltwise();
  _FloorDivPrimType_FloorDiv();
  _FloorModPrimType_FloorMod();
  _LogicalAndPrimType_LogicalAnd();
  _LogicalOrPrimType_LogicalOr();
  _MaximumPrimType_Maximum();
  _MinimumPrimType_Minimum();
  _ModPrimType_Mod();
  _MulPrimType_MulFusion();
  _RealDivPrimType_RealDiv();
  _SubPrimType_SubFusion();
  _SquaredDifferencePrimType_SquaredDifference();
}
#endif
InferShape *g_infer_func = NULL;

__attribute__((constructor(101))) void InitInferFuncBuf() {
  if (g_infer_func != NULL) {
    return;
  }
  g_infer_func = malloc(PrimType_MAX * sizeof(InferShape));
  if (g_infer_func != NULL) {
    memset(g_infer_func, 0, PrimType_MAX * sizeof(InferShape));
  }
#ifdef MS_COMPILE_IOS
  RegisterInfer();
#endif
}

__attribute__((destructor)) void DestroyInferFuncBuf() {
  if (g_infer_func == NULL) {
    return;
  }
  free(g_infer_func);
  g_infer_func = NULL;
}

InferShape GetInferFunc(int prim_type) {
  if (g_infer_func != NULL && prim_type < PrimType_MAX) {
    return g_infer_func[prim_type];
  }
  return NULL;
}

void RegInfer(int prim_type, InferShape func) {
  if (g_infer_func != NULL && prim_type < PrimType_MAX) {
    g_infer_func[prim_type] = func;
  }
}
