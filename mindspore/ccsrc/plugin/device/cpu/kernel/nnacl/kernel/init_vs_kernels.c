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

#include "nnacl/kernel/init_vs_kernels.h"
#include "nnacl/kernel/activation.h"
#include "nnacl/kernel/arithmetic.h"
#include "nnacl/kernel/arithmetic_compare.h"
#include "nnacl/kernel/arithmetic_self.h"
#include "nnacl/kernel/arg_min_max.h"
#include "nnacl/kernel/addn.h"
#include "nnacl/kernel/biasadd.h"
#include "nnacl/kernel/batch_norm.h"
#include "nnacl/kernel/clip.h"
#include "nnacl/kernel/concat.h"
#include "nnacl/kernel/crop.h"
#include "nnacl/kernel/crop_and_resize.h"
#include "nnacl/kernel/exp.h"
#include "nnacl/kernel/depth_to_space.h"
#include "nnacl/kernel/fill.h"
#include "nnacl/kernel/fused_batch_norm.h"
#include "nnacl/kernel/fullconnection.h"
#include "nnacl/kernel/gather.h"
#include "nnacl/kernel/gather_d.h"
#include "nnacl/kernel/gather_nd.h"
#include "nnacl/kernel/group_norm.h"
#include "nnacl/kernel/log_softmax.h"
#include "nnacl/kernel/local_response_norm.h"
#include "nnacl/kernel/layer_norm.h"
#include "nnacl/kernel/matmul.h"
#include "nnacl/kernel/non_max_suppression.h"
#include "nnacl/kernel/non_zero.h"
#include "nnacl/kernel/nllloss.h"
#include "nnacl/kernel/prior_box.h"
#include "nnacl/kernel/prelu.h"
#include "nnacl/kernel/pad.h"
#include "nnacl/kernel/pow.h"
#include "nnacl/kernel/reshape.h"
#include "nnacl/kernel/reverse.h"
#include "nnacl/kernel/range.h"
#include "nnacl/kernel/rank.h"
#include "nnacl/kernel/scale.h"
#include "nnacl/kernel/shape.h"
#include "nnacl/kernel/reduce.h"
#include "nnacl/kernel/ragged_range.h"
#include "nnacl/kernel/stack.h"
#include "nnacl/kernel/strided_slice.h"
#include "nnacl/kernel/softmax.h"
#include "nnacl/kernel/size.h"
#include "nnacl/kernel/splice.h"
#include "nnacl/kernel/tile.h"
#include "nnacl/kernel/tril.h"
#include "nnacl/kernel/triu.h"
#include "nnacl/kernel/transpose.h"
#include "nnacl/kernel/slice.h"
#include "nnacl/kernel/unique.h"
#ifdef ENABLE_FP16
#include "nnacl/kernel/f16/arithmetic_f16.h"
#include "nnacl/kernel/f16/arithmetic_compare_f16.h"
#include "nnacl/kernel/f16/concat_f16.h"
#include "nnacl/kernel/f16/reduce_f16.h"
#include "nnacl/kernel/f16/stack_f16.h"
#endif

void InitVSKernelF16(KernelCreator **creators) {
#ifdef ENABLE_FP16
  creators[PrimType_Abs][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_Activation][REGIST_DT(kNumberTypeFloat16)] = CreateActivation;
  creators[PrimType_AddFusion][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_AddN][REGIST_DT(kNumberTypeFloat16)] = CreateAddN;
  creators[PrimType_ArgMinFusion][REGIST_DT(kNumberTypeFloat16)] = CreateArgMinMax;
  creators[PrimType_ArgMaxFusion][REGIST_DT(kNumberTypeFloat16)] = CreateArgMinMax;
  creators[PrimType_BatchNorm][REGIST_DT(kNumberTypeFloat16)] = CreateBatchNorm;
  creators[PrimType_Ceil][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_Concat][REGIST_DT(kNumberTypeFloat16)] = CreateConcatF16;
  creators[PrimType_Cos][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_Crop][REGIST_DT(kNumberTypeFloat16)] = CreateCrop;
  creators[PrimType_DivFusion][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_DepthToSpace][REGIST_DT(kNumberTypeFloat16)] = CreateDepthToSpace;
  creators[PrimType_Eltwise][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_Erf][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_Equal][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticCompareF16;
  creators[PrimType_ExpandDims][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_Fill][REGIST_DT(kNumberTypeFloat16)] = CreateFill;
  creators[PrimType_Flatten][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_FlattenGrad][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_Floor][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_FloorMod][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_FloorDiv][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_FusedBatchNorm][REGIST_DT(kNumberTypeFloat16)] = CreateFusedBatchNorm;
  creators[PrimType_Gather][REGIST_DT(kNumberTypeFloat16)] = CreateGather;
  creators[PrimType_GatherD][REGIST_DT(kNumberTypeFloat16)] = CreateGatherD;
  creators[PrimType_GatherNd][REGIST_DT(kNumberTypeFloat16)] = CreateGatherNd;
  creators[PrimType_Greater][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticCompareF16;
  creators[PrimType_GreaterEqual][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticCompareF16;
  creators[PrimType_Less][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticCompareF16;
  creators[PrimType_LessEqual][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticCompareF16;
  creators[PrimType_LayerNormFusion][REGIST_DT(kNumberTypeFloat16)] = CreateLayerNorm;
  creators[PrimType_LogicalAnd][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_LogicalOr][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_Log][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_LogSoftmax][REGIST_DT(kNumberTypeFloat16)] = CreateLogSoftmax;
  creators[PrimType_LogicalNot][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_Maximum][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_Minimum][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_MulFusion][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_Neg][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_NotEqual][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticCompareF16;
  creators[PrimType_PadFusion][REGIST_DT(kNumberTypeFloat16)] = CreatePad;
  creators[PrimType_PReLUFusion][REGIST_DT(kNumberTypeFloat16)] = CreatePRelu;
  creators[PrimType_PowFusion][REGIST_DT(kNumberTypeFloat16)] = CreatePow;
  creators[PrimType_Reshape][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_RealDiv][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_ReduceFusion][REGIST_DT(kNumberTypeFloat16)] = CreateReduceF16;
  creators[PrimType_Rsqrt][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_Round][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_Reciprocal][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_ScaleFusion][REGIST_DT(kNumberTypeFloat16)] = CreateScale;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeFloat16)] = CreateShape;
  creators[PrimType_Softmax][REGIST_DT(kNumberTypeFloat16)] = CreateSoftmax;
  creators[PrimType_Stack][REGIST_DT(kNumberTypeFloat16)] = CreateStackF16;
  creators[PrimType_StridedSlice][REGIST_DT(kNumberTypeFloat16)] = CreateStridedSlice;
  creators[PrimType_Squeeze][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_SubFusion][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_SquaredDifference][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticF16;
  creators[PrimType_Splice][REGIST_DT(kNumberTypeFloat16)] = CreateSplice;
  creators[PrimType_Sin][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_Size][REGIST_DT(kNumberTypeFloat16)] = CreateSize;
  creators[PrimType_SliceFusion][REGIST_DT(kNumberTypeFloat16)] = CreateSlice;
  creators[PrimType_Square][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_Sqrt][REGIST_DT(kNumberTypeFloat16)] = CreateArithmeticSelf;
  creators[PrimType_TileFusion][REGIST_DT(kNumberTypeFloat16)] = CreateTile;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeFloat16)] = CreateTriu;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeFloat16)] = CreateTril;
  creators[PrimType_Transpose][REGIST_DT(kNumberTypeFloat16)] = CreateTranspose;
  creators[PrimType_Unsqueeze][REGIST_DT(kNumberTypeFloat16)] = CreateReshape;
  creators[PrimType_Unique][REGIST_DT(kNumberTypeFloat16)] = CreateUnique;
#endif
}

void InitVSKernelA(KernelCreator **creators) {
  creators[PrimType_Abs][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_Abs][REGIST_DT(kNumberTypeInt32)] = CreateArithmeticSelf;
  creators[PrimType_Activation][REGIST_DT(kNumberTypeFloat32)] = CreateActivation;
  creators[PrimType_Activation][REGIST_DT(kNumberTypeUInt32)] = CreateActivation;
  creators[PrimType_AddFusion][REGIST_DT(kNumberTypeBool)] = CreateArithmetic;
  creators[PrimType_AddFusion][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_AddFusion][REGIST_DT(kNumberTypeInt32)] = CreateArithmetic;
  creators[PrimType_AddN][REGIST_DT(kNumberTypeFloat32)] = CreateAddN;
  creators[PrimType_ArgMinFusion][REGIST_DT(kNumberTypeInt32)] = CreateArgMinMax;
  creators[PrimType_ArgMinFusion][REGIST_DT(kNumberTypeFloat32)] = CreateArgMinMax;
  creators[PrimType_ArgMaxFusion][REGIST_DT(kNumberTypeInt32)] = CreateArgMinMax;
  creators[PrimType_ArgMaxFusion][REGIST_DT(kNumberTypeFloat32)] = CreateArgMinMax;
  creators[PrimType_BiasAdd][REGIST_DT(kNumberTypeFloat32)] = CreateBiasAdd;
  creators[PrimType_BatchNorm][REGIST_DT(kNumberTypeFloat32)] = CreateBatchNorm;
  creators[PrimType_Ceil][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_Cos][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_Clip][REGIST_DT(kNumberTypeFloat)] = CreateClip;
  creators[PrimType_Clip][REGIST_DT(kNumberTypeFloat32)] = CreateClip;
  creators[PrimType_Clip][REGIST_DT(kNumberTypeInt)] = CreateClip;
  creators[PrimType_Clip][REGIST_DT(kNumberTypeInt32)] = CreateClip;
  creators[PrimType_Concat][REGIST_DT(kNumberTypeBool)] = CreateConcat;
  creators[PrimType_Concat][REGIST_DT(kNumberTypeInt32)] = CreateConcat;
  creators[PrimType_Concat][REGIST_DT(kNumberTypeFloat32)] = CreateConcat;
  creators[PrimType_Crop][REGIST_DT(kNumberTypeInt32)] = CreateCrop;
  creators[PrimType_Crop][REGIST_DT(kNumberTypeFloat32)] = CreateCrop;
  creators[PrimType_CropAndResize][REGIST_DT(kNumberTypeFloat32)] = CreateCropAndResize;
  creators[PrimType_DepthToSpace][REGIST_DT(kNumberTypeFloat32)] = CreateDepthToSpace;
  creators[PrimType_DivFusion][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_DivFusion][REGIST_DT(kNumberTypeInt32)] = CreateArithmetic;
  creators[PrimType_Eltwise][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_Equal][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticCompare;
  creators[PrimType_Equal][REGIST_DT(kNumberTypeInt32)] = CreateArithmeticCompare;
  creators[PrimType_Erf][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_ExpFusion][REGIST_DT(kNumberTypeFloat32)] = CreateExp;
  creators[PrimType_ExpandDims][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
  creators[PrimType_ExpandDims][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_ExpandDims][REGIST_DT(kNumberTypeBool)] = CreateReshape;
  creators[PrimType_ExpandDims][REGIST_DT(kNumberTypeInt8)] = CreateReshape;
  creators[PrimType_Fill][REGIST_DT(kNumberTypeBool)] = CreateFill;
  creators[PrimType_Fill][REGIST_DT(kNumberTypeInt32)] = CreateFill;
  creators[PrimType_Fill][REGIST_DT(kNumberTypeFloat32)] = CreateFill;
  creators[PrimType_Flatten][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
  creators[PrimType_Flatten][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_FlattenGrad][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_Floor][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_FloorDiv][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_FloorDiv][REGIST_DT(kNumberTypeInt32)] = CreateArithmetic;
  creators[PrimType_FloorMod][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_FloorMod][REGIST_DT(kNumberTypeInt32)] = CreateArithmetic;
  creators[PrimType_FullConnection][REGIST_DT(kNumberTypeFloat32)] = CreateFullconnection;
  creators[PrimType_FusedBatchNorm][REGIST_DT(kNumberTypeFloat32)] = CreateFusedBatchNorm;
  creators[PrimType_Gather][REGIST_DT(kNumberTypeFloat32)] = CreateGather;
  creators[PrimType_Gather][REGIST_DT(kNumberTypeInt32)] = CreateGather;
  creators[PrimType_Gather][REGIST_DT(kNumberTypeBool)] = CreateGather;
  creators[PrimType_GatherD][REGIST_DT(kNumberTypeFloat32)] = CreateGatherD;
  creators[PrimType_GatherD][REGIST_DT(kNumberTypeInt32)] = CreateGatherD;
  creators[PrimType_GatherNd][REGIST_DT(kNumberTypeBool)] = CreateGatherNd;
  creators[PrimType_GatherNd][REGIST_DT(kNumberTypeInt32)] = CreateGatherNd;
  creators[PrimType_GatherNd][REGIST_DT(kNumberTypeFloat32)] = CreateGatherNd;
  creators[PrimType_Greater][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticCompare;
  creators[PrimType_Greater][REGIST_DT(kNumberTypeInt32)] = CreateArithmeticCompare;
  creators[PrimType_GreaterEqual][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticCompare;
  creators[PrimType_GreaterEqual][REGIST_DT(kNumberTypeInt32)] = CreateArithmeticCompare;
  creators[PrimType_GroupNormFusion][REGIST_DT(kNumberTypeFloat32)] = CreateGroupNorm;
}

void InitVSKernelI(KernelCreator **creators) {
  creators[PrimType_IsFinite][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_LayerNormFusion][REGIST_DT(kNumberTypeFloat32)] = CreateLayerNorm;
  creators[PrimType_Less][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticCompare;
  creators[PrimType_Less][REGIST_DT(kNumberTypeInt32)] = CreateArithmeticCompare;
  creators[PrimType_LessEqual][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticCompare;
  creators[PrimType_LessEqual][REGIST_DT(kNumberTypeInt32)] = CreateArithmeticCompare;
  creators[PrimType_LogicalAnd][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_LogicalAnd][REGIST_DT(kNumberTypeBool)] = CreateArithmetic;
  creators[PrimType_LogicalAnd][REGIST_DT(kNumberTypeInt32)] = CreateArithmetic;
  creators[PrimType_LogicalOr][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_LogicalOr][REGIST_DT(kNumberTypeBool)] = CreateArithmetic;
  creators[PrimType_Log][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_LogSoftmax][REGIST_DT(kNumberTypeFloat32)] = CreateLogSoftmax;
  creators[PrimType_Log1p][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_LogicalNot][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_LogicalNot][REGIST_DT(kNumberTypeBool)] = CreateArithmeticSelf;
  creators[PrimType_LRN][REGIST_DT(kNumberTypeFloat32)] = CreateLocalResponseNorm;
  creators[PrimType_Maximum][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_Maximum][REGIST_DT(kNumberTypeInt32)] = CreateArithmetic;
  creators[PrimType_MatMulFusion][REGIST_DT(kNumberTypeFloat32)] = CreateMatmul;
  creators[PrimType_Mod][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_Mod][REGIST_DT(kNumberTypeInt32)] = CreateArithmetic;
  creators[PrimType_MulFusion][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_MulFusion][REGIST_DT(kNumberTypeInt32)] = CreateArithmetic;
  creators[PrimType_Minimum][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_Minimum][REGIST_DT(kNumberTypeInt32)] = CreateArithmetic;
  creators[PrimType_NLLLoss][REGIST_DT(kNumberTypeFloat32)] = CreateNLLLoss;
  creators[PrimType_Neg][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_Neg][REGIST_DT(kNumberTypeInt32)] = CreateArithmeticSelf;
  creators[PrimType_NotEqual][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticCompare;
  creators[PrimType_NotEqual][REGIST_DT(kNumberTypeInt32)] = CreateArithmeticCompare;
  creators[PrimType_NotEqual][REGIST_DT(kNumberTypeInt64)] = CreateArithmeticCompare;
  creators[PrimType_NonZero][REGIST_DT(kNumberTypeBool)] = CreateNonZero;
  creators[PrimType_NonMaxSuppression][REGIST_DT(kNumberTypeFloat32)] = CreateNonMaxSuppression;
  creators[PrimType_PadFusion][REGIST_DT(kNumberTypeFloat32)] = CreatePad;
  creators[PrimType_PriorBox][REGIST_DT(kNumberTypeFloat32)] = CreatePriorBox;
  creators[PrimType_PriorBox][REGIST_DT(kNumberTypeInt8)] = CreatePriorBox;
  creators[PrimType_PowFusion][REGIST_DT(kNumberTypeFloat32)] = CreatePow;
  creators[PrimType_PReLUFusion][REGIST_DT(kNumberTypeFloat32)] = CreatePRelu;
}

void InitVSKernelR(KernelCreator **creators) {
  creators[PrimType_RaggedRange][REGIST_DT(kNumberTypeInt32)] = CreateRaggedRange;
  creators[PrimType_RaggedRange][REGIST_DT(kNumberTypeFloat32)] = CreateRaggedRange;
  creators[PrimType_Range][REGIST_DT(kNumberTypeFloat32)] = CreateRange;
  creators[PrimType_Range][REGIST_DT(kNumberTypeInt32)] = CreateRange;
  creators[PrimType_Range][REGIST_DT(kNumberTypeFloat16)] = CreateRange;
  creators[PrimType_Rank][REGIST_DT(kNumberTypeFloat32)] = CreateRank;
  creators[PrimType_Rank][REGIST_DT(kNumberTypeFloat32)] = CreateRank;
  creators[PrimType_Reshape][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
  creators[PrimType_Reshape][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_Reshape][REGIST_DT(kNumberTypeBool)] = CreateReshape;
  creators[PrimType_RealDiv][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_ReduceFusion][REGIST_DT(kNumberTypeBool)] = CreateReduce;
  creators[PrimType_ReduceFusion][REGIST_DT(kNumberTypeInt32)] = CreateReduce;
  creators[PrimType_ReduceFusion][REGIST_DT(kNumberTypeFloat32)] = CreateReduce;
  creators[PrimType_Reciprocal][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_ReverseV2][REGIST_DT(kNumberTypeInt32)] = CreateReverse;
  creators[PrimType_ReverseV2][REGIST_DT(kNumberTypeFloat32)] = CreateReverse;
  creators[PrimType_Round][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_Rsqrt][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_ScaleFusion][REGIST_DT(kNumberTypeFloat32)] = CreateScale;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeInt32)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeBool)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeFloat32)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeInt8)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeUInt8)] = CreateShape;
  creators[PrimType_Shape][REGIST_DT(kNumberTypeInt64)] = CreateShape;
  creators[PrimType_Softmax][REGIST_DT(kNumberTypeFloat32)] = CreateSoftmax;
  creators[PrimType_SquaredDifference][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_Stack][REGIST_DT(kNumberTypeFloat32)] = CreateStack;
  creators[PrimType_Stack][REGIST_DT(kNumberTypeInt32)] = CreateStack;
  creators[PrimType_StridedSlice][REGIST_DT(kNumberTypeFloat32)] = CreateStridedSlice;
  creators[PrimType_StridedSlice][REGIST_DT(kNumberTypeInt64)] = CreateStridedSlice;
  creators[PrimType_StridedSlice][REGIST_DT(kNumberTypeInt32)] = CreateStridedSlice;
  creators[PrimType_StridedSlice][REGIST_DT(kNumberTypeInt8)] = CreateStridedSlice;
  creators[PrimType_Square][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_Sqrt][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_Sin][REGIST_DT(kNumberTypeFloat32)] = CreateArithmeticSelf;
  creators[PrimType_Size][REGIST_DT(kNumberTypeInt32)] = CreateSize;
  creators[PrimType_Size][REGIST_DT(kNumberTypeFloat32)] = CreateSize;
  creators[PrimType_SliceFusion][REGIST_DT(kNumberTypeInt32)] = CreateSlice;
  creators[PrimType_SliceFusion][REGIST_DT(kNumberTypeFloat32)] = CreateSlice;
  creators[PrimType_SubFusion][REGIST_DT(kNumberTypeFloat32)] = CreateArithmetic;
  creators[PrimType_SubFusion][REGIST_DT(kNumberTypeInt32)] = CreateArithmetic;
  creators[PrimType_Squeeze][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_Squeeze][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
  creators[PrimType_Squeeze][REGIST_DT(kNumberTypeBool)] = CreateReshape;
  creators[PrimType_Splice][REGIST_DT(kNumberTypeFloat32)] = CreateSplice;
  creators[PrimType_TileFusion][REGIST_DT(kNumberTypeInt32)] = CreateTile;
  creators[PrimType_TileFusion][REGIST_DT(kNumberTypeFloat32)] = CreateTile;
  creators[PrimType_TileFusion][REGIST_DT(kNumberTypeBool)] = CreateTile;
  creators[PrimType_TileFusion][REGIST_DT(kNumberTypeUInt8)] = CreateTile;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeDouble)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeFloat)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeFloat64)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeFloat32)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeInt)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeInt64)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeInt32)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeInt16)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeInt8)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeUInt64)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeUInt32)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeUInt16)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeUInt8)] = CreateTriu;
  creators[PrimType_Triu][REGIST_DT(kNumberTypeBool)] = CreateTriu;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeDouble)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeFloat)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeFloat64)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeFloat32)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeInt)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeInt64)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeInt32)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeInt16)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeInt8)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeUInt64)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeUInt32)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeUInt16)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeUInt8)] = CreateTril;
  creators[PrimType_Tril][REGIST_DT(kNumberTypeBool)] = CreateTril;
  creators[PrimType_Transpose][REGIST_DT(kNumberTypeFloat32)] = CreateTranspose;
  creators[PrimType_Transpose][REGIST_DT(kNumberTypeInt32)] = CreateTranspose;
  creators[PrimType_Unsqueeze][REGIST_DT(kNumberTypeFloat32)] = CreateReshape;
  creators[PrimType_Unsqueeze][REGIST_DT(kNumberTypeInt32)] = CreateReshape;
  creators[PrimType_Unsqueeze][REGIST_DT(kNumberTypeInt64)] = CreateReshape;
  creators[PrimType_Unsqueeze][REGIST_DT(kNumberTypeBool)] = CreateReshape;
  creators[PrimType_Unique][REGIST_DT(kNumberTypeInt32)] = CreateUnique;
  creators[PrimType_Unique][REGIST_DT(kNumberTypeFloat32)] = CreateUnique;
}

void init_vs_kernels(KernelCreator **creators) {
  InitVSKernelA(creators);
  InitVSKernelI(creators);
  InitVSKernelR(creators);
  InitVSKernelF16(creators);
}
