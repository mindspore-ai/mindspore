/**
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
#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATE_OPS_FRONT_INFER_FUNCTION_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATE_OPS_FRONT_INFER_FUNCTION_H_

#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace abstract {
const std::vector<std::string> kSparsePrimStr = {"PrimitiveAbstractClosure: S_Prim_MakeCSRTensor",
                                                 "PrimitiveAbstractClosure: S_Prim_MakeCOOTensor",
                                                 "PrimitiveAbstractClosure: S_Prim_MakeRowTensor"};
// String
AbstractBasePtr InferImplStringMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplStringGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_abs_list);
// Tuple
AbstractBasePtr InferImplTupleReversed(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplTupleDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplTuple2Array(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_abs_list);
// List
AbstractBasePtr InferImplListReduce(const AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_abs_list);
// Dict
AbstractBasePtr InferImplDictLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_abs_list);
// Slice
AbstractBasePtr InferImplMakeSlice(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplSliceGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_abs_list);
// Type checking
AbstractBasePtr InferImplTypeof(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplHasType(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplIsInstance(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_abs_list);
// Shape processing
AbstractBasePtr InferImplReduceShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_abs_list);
// Auto-Grad
AbstractBasePtr InferImplStopGradient(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplFakeBprop(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplJ(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplBroadcastGradientArgs(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_abs_list);
// Other
AbstractBasePtr InferImplReusing(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplTaylor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplShard(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplVmap(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplDtypeToEnum(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_abs_list);
AbstractBasePtr InferImplAddAttr(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtr &arg_abs_list);
// Delete this when the infer value can be mapped to the CPU backend operator.
bool PrimNeedFrontendInferValue(const PrimitivePtr &primitive);

PrimitiveEvalImplMap *GetFrontendPrimitiveInferMapPtr();
#define REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(name, primitive, infer_impl, infer_value_impl)                         \
  auto helper_##name = abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), \
                                                                     primitive, infer_impl, infer_value_impl, false);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATE_OPS_FRONT_INFER_FUNCTION_H_
