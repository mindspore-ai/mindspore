/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License", true);
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

#include "abstract/ops/infer_functions.h"
#include "abstract/ops/primitive_infer_map.h"
namespace mindspore {
namespace abstract {
// using R = PrimitiveEvalImplMap::mapped_type;
static PrimitiveEvalImplMap deprecated_infer_map{
  // deprecated infer functions, new implement please refer to op_infer.h
};
PrimitiveEvalImplMap *GetDeprecatedPrimitiveInferMapPtr() { return &deprecated_infer_map; }
const PrimitiveEvalImplMap &GetDeprecatedPrimitiveInferMap() { return deprecated_infer_map; }
REG_PRIM_INFER_FUNC(Return, true)
REG_PRIM_INFER_FUNC(Switch, true)
REG_PRIM_INFER_FUNC(SwitchLayer, true)
REG_PRIM_INFER_FUNC(Is_, true)
REG_PRIM_INFER_FUNC(IsNot, true)
REG_PRIM_INFER_FUNC(InDict, true)
REG_PRIM_INFER_FUNC(NotInDict, true)
REG_PRIM_INFER_FUNC(IsConstant, true)
REG_PRIM_INFER_FUNC(Pooling, true)
REG_PRIM_INFER_FUNC(PoolingGrad, true)
REG_PRIM_INFER_FUNC(BatchNorm, true)
//  REG_PRIM_INFER_FUNC(BiasAddGrad, true)
REG_PRIM_INFER_FUNC(BpropCut, true)
// REG_PRIM_INFER_FUNC(Sqrt, true)
// REG_PRIM_INFER_FUNC(SqrtGrad, true)
REG_PRIM_INFER_FUNC(ArrayToScalar, true)
REG_PRIM_INFER_FUNC(BroadcastShape, true)
REG_PRIM_INFER_FUNC(MakeDict, true)
// REG_PRIM_INFER_FUNC(UnsortedSegmentSum, true)
REG_PRIM_INFER_FUNC(UnsortedSegmentMax, true)
REG_PRIM_INFER_FUNC(UnsortedSegmentMin, true)
REG_PRIM_INFER_FUNC(MakeKeywordArg, true)
REG_PRIM_INFER_FUNC(ExtractKeywordArg, true)
REG_PRIM_INFER_FUNC(DictGetItem, true)
REG_PRIM_INFER_FUNC(DictSetItem, true)
REG_PRIM_INFER_FUNC(DictGetKeys, true)
REG_PRIM_INFER_FUNC(DictGetValues, true)
REG_PRIM_INFER_FUNC(DictItems, true)
REG_PRIM_INFER_FUNC(ArrayLen, true)
REG_PRIM_INFER_FUNC(Mutable, true)
REG_PRIM_INFER_FUNC(GetGrad, true)
REG_PRIM_INFER_FUNC(Identity, true)
REG_PRIM_INFER_FUNC(EnvironCreate, true)
REG_PRIM_INFER_FUNC(EnvironGet, true)
REG_PRIM_INFER_FUNC(EnvironSet, true)
REG_PRIM_INFER_FUNC(EnvironAdd, true)
REG_PRIM_INFER_FUNC(EnvironDestroyAll, true)
REG_PRIM_INFER_FUNC(StateSetItem, true)
REG_PRIM_INFER_FUNC(Depend, true)
REG_PRIM_INFER_FUNC(UpdateState, true)
REG_PRIM_INFER_FUNC(Debug, true)
REG_PRIM_INFER_FUNC(MakeRowTensor, true)
REG_PRIM_INFER_FUNC(RowTensorGetValues, true)
REG_PRIM_INFER_FUNC(RowTensorGetIndices, true)
REG_PRIM_INFER_FUNC(RowTensorGetDenseShape, true)
REG_PRIM_INFER_FUNC(RowTensorAdd, true)
REG_PRIM_INFER_FUNC(UniqueGrad, true)
REG_PRIM_INFER_FUNC(Unique, true)
REG_PRIM_INFER_FUNC(OCRRecognitionPreHandle, true)
REG_PRIM_INFER_FUNC(ScatterAdd, true)
REG_PRIM_INFER_FUNC(ScatterSub, true)
// REG_PRIM_INFER_FUNC(Div, true)
REG_PRIM_INFER_FUNC(SubAndFilter, true)
REG_PRIM_INFER_FUNC(MapCacheIdx, true)
REG_PRIM_INFER_FUNC(CacheSwapTable, true)
REG_PRIM_INFER_FUNC(UpdateCache, true)
REG_PRIM_INFER_FUNC(ComputeAccidentalHits, true)
REG_PRIM_INFER_FUNC(PadAndShift, true)
REG_PRIM_INFER_FUNC(DynamicAssign, true)
REG_PRIM_INFER_FUNC(SparseApplyProximalAdagrad, true)
REG_PRIM_INFER_FUNC(AllSwap, true)
// REG_PRIM_INFER_FUNC(AllReduce, true)
// REG_PRIM_INFER_FUNC(Broadcast, true)
// REG_PRIM_INFER_FUNC(AllGather, true)
// REG_PRIM_INFER_FUNC(ReduceScatter, true)
// REG_PRIM_INFER_FUNC(SGD, true)
// REG_PRIM_INFER_FUNC(Transpose, true)
REG_PRIM_INFER_FUNC(MemCpyAsync, true)
REG_PRIM_INFER_FUNC(EmbeddingLookup, true)
// REG_PRIM_INFER_FUNC(Cast, true)
// REG_PRIM_INFER_FUNC(Minimum, true)
// REG_PRIM_INFER_FUNC(DivNoNan, true)
// REG_PRIM_INFER_FUNC(LinSpace, true)
REG_PRIM_INFER_FUNC(IsDimUnknown, true)
REG_PRIM_INFER_FUNC(IsShapeUnknown, true)
// REG_PRIM_INFER_FUNC(Pad, true)
REG_PRIM_INFER_FUNC(MapUniform, true)
REG_PRIM_INFER_FUNC(Split, true)
REG_PRIM_INFER_FUNC(SequenceMask, true)
// REG_PRIM_INFER_FUNC(Concat, true)
// REG_PRIM_INFER_FUNC(ConcatOffset, true)
REG_PRIM_INFER_FUNC(FlattenConcat, true)
REG_PRIM_INFER_FUNC(MatMul, true)
// REG_PRIM_INFER_FUNC(Less, true)
REG_PRIM_INFER_FUNC(Load, true)
// REG_PRIM_INFER_FUNC(TransData, true)
// REG_PRIM_INFER_FUNC(TensorMove, true)
REG_PRIM_INFER_FUNC(TensorCopySlices, true)
REG_PRIM_INFER_FUNC(RealInner, true)
REG_PRIM_INFER_FUNC(BiasDropoutAdd, true)
REG_PRIM_INFER_FUNC(TensorArrayStack, true)
REG_PRIM_INFER_FUNC(KMeansCentroids, true)
// REG_PRIM_INFER_FUNC(AdamApplyOne, true)
// REG_PRIM_INFER_FUNC(AdamApplyOneWithDecay, true)
}  // namespace abstract
}  // namespace mindspore
