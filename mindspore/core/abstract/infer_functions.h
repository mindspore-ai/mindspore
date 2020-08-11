/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "base/core_ops.h"
namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplReturn(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDot(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSwitch(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSwitchLayer(const AnalysisEnginePtr &, const PrimitivePtr &,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplIs_(const AnalysisEnginePtr &, const PrimitivePtr &,
                             const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplIsNot(const AnalysisEnginePtr &, const PrimitivePtr &,
                               const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplInDict(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplNotInDict(const AnalysisEnginePtr &, const PrimitivePtr &,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplIsConstant(const AnalysisEnginePtr &, const PrimitivePtr &,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplPooling(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplPoolingGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplFusedBatchNorm(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplFusedBatchNormGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplReluGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplConv2DBackpropInput(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplConv2DBackpropFilter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplBiasAddGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGelu(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGeluGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplRelu(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplZerosLike(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplBpropCut(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplLayerNorm(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplLayerNormGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDropoutGenMask(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplMinOrMaxGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplScalarToArray(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplArrayToScalar(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplBroadCastShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplPack(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplMakeTuple(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeList(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeSlice(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeKwarg(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplExtractKwarg(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeRecord(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTupleGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplListGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTupleSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplListSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDictGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDictSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplListAppend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTupleLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplListLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplArrayLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGenShapeIndex(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGenInverseIndex(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplIdentity(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplEnvGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplEnvSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplEnvAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeRefKey(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeRef(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGetRefKey(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGetRefValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGetRefOrigin(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplStateSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDepend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplControlDepend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDebug(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeSparseTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSparseTensorGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSparseTensorGetIndices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSparseTensorGetDenseShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplMakeRowTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplRowTensorGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplRowTensorGetIndices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplRowTensorGetDenseShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplUniqueGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplUnique(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list);

template <typename T>
AbstractBasePtr InferTupleOrListOrDictLen(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list or dict.
  CheckArgsSize(op_name, args_spec_list, 1);
  auto arg = CheckArg<T>(op_name, args_spec_list, 0);
  return std::make_shared<AbstractScalar>(SizeToInt(arg->size()));
}
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_INFER_FUNCTIONS_H_
