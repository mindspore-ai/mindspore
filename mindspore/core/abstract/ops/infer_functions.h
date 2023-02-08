/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_ABSTRACT_INFER_FUNCTIONS_H_
#define MINDSPORE_CORE_ABSTRACT_INFER_FUNCTIONS_H_
#include <string>
#include <memory>
#include "abstract/abstract_value.h"
#include "abstract/param_validator.h"
#include "mindspore/core/ops/core_ops.h"
#include "abstract/ops/primitive_infer_map.h"
namespace mindspore {
namespace abstract {
MIND_API AbstractBasePtr InferImplReturn(const AnalysisEnginePtr &, const PrimitivePtr &,
                                         const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplSwitch(const AnalysisEnginePtr &, const PrimitivePtr &,
                                         const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplSwitchLayer(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplIs_(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplIsNot(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplInDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplNotInDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplIsConstant(const AnalysisEnginePtr &, const PrimitivePtr &,
                                             const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplPooling(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplPoolingGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplBatchNorm(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplBiasAddGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplBpropCut(const AnalysisEnginePtr &, const PrimitivePtr &,
                                           const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplSqrt(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplSqrtGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const AbstractBasePtrList &args_spec_list);

MIND_API AbstractBasePtr InferImplArrayToScalar(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplBroadcastShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const AbstractBasePtrList &args_spec_list);

MIND_API AbstractBasePtr InferImplMakeDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplUnsortedSegmentSum(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplUnsortedSegmentMax(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplUnsortedSegmentMin(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMakeKeywordArg(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplExtractKeywordArg(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplDictGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplDictSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplDictGetKeys(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplDictGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplDictItems(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplArrayLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMutable(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplGetGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplIdentity(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const AbstractBasePtrList &args_spec_list);

MIND_API AbstractBasePtr InferImplEnvironCreate(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplEnvironGet(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplEnvironSet(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplEnvironAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplEnvironDestroyAll(const AnalysisEnginePtr &, const PrimitivePtr &,
                                                    const AbstractBasePtrList &);

MIND_API AbstractBasePtr InferImplStateSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplDepend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplUpdateState(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplDebug(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMakeRowTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplRowTensorGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                     const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplRowTensorGetIndices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                      const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplRowTensorGetDenseShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                         const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplRowTensorAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplUniqueGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplUnique(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplOCRRecognitionPreHandle(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                          const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplScatterAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplScatterSub(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);

MIND_API AbstractBasePtr InferImplSubAndFilter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMapCacheIdx(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplCacheSwapTable(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplUpdateCache(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplComputeAccidentalHits(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                        const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplPadAndShift(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplDynamicAssign(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplSparseApplyProximalAdagrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                             const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplAllSwap(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplAllReduce(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplBroadcast(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplAllGather(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplReduceScatter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplSGD(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplTranspose(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMemCpyAsync(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplEmbeddingLookup(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplCast(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMinimum(const AnalysisEnginePtr &engine_ptr, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplDivNoNan(const AnalysisEnginePtr &engine_ptr, const PrimitivePtr &primitive,
                                           const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplLinSpace(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplIsDimUnknown(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplIsShapeUnknown(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplIsElementUnknown(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplPad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMapUniform(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplSplit(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplSequenceMask(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplConcat(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplConcatOffset(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplFlattenConcat(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMatMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplLess(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplLoad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplTransData(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplTensorMove(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplTensorCopySlices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplRealInner(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplBiasDropoutAdd(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplTensorArrayStack(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const AbstractBasePtrList &);
MIND_API AbstractBasePtr InferImplKMeansCentroids(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplAdamApplyOne(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplAdamApplyOneWithDecay(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                        const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMapTensorGetDefaultValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                           const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMapTensorGetPermitFilterValue(const AnalysisEnginePtr &,
                                                                const PrimitivePtr &primitive,
                                                                const AbstractBasePtrList &args_spec_list);
MIND_API AbstractBasePtr InferImplMapTensorGetEvictFilterValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                               const AbstractBasePtrList &args_spec_list);

template <typename T>
AbstractBasePtr InferTupleOrListOrDictLen(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list or dict.
  constexpr size_t len_input_size = 1;
  CheckArgsSize(op_name, args_spec_list, len_input_size);
  auto arg = CheckArg<T>(op_name, args_spec_list, 0);
  auto abs = dyn_cast<AbstractSequence>(args_spec_list[0]);
  if (abs != nullptr && abs->dynamic_len()) {
    // If the sequence is dynamic length, return any value scalar.
    return std::make_shared<AbstractScalar>(kAnyValue, kInt64);
  }
  return std::make_shared<AbstractScalar>(SizeToLong(arg->size()));
}
#define REG_PRIM_INFER_FUNC(name, in_white_list)                                  \
  static auto helper_eval_##name = abstract::RegisterStandardPrimitiveEvalHelper( \
    abstract::GetDeprecatedPrimitiveInferMapPtr(), prim::kPrim##name, InferImpl##name, nullptr, in_white_list);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_INFER_FUNCTIONS_H_
