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
#include <optional>
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
namespace mindspore {
namespace abstract {
const std::vector<std::string> kSparsePrimStr = {"Prim: S-Prim-MakeCSRTensor", "Prim: S-Prim-MakeCOOTensor",
                                                 "Prim: S-Prim-MakeRowTensor"};
// String
AbstractBasePtr InferImplStringMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplStringGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
// Tuple
AbstractBasePtr InferImplTupleReversed(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTupleDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTuple2Array(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTupleEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
// List
AbstractBasePtr InferImplListReduce(const AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplListEqual(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
// Dict
AbstractBasePtr InferImplDictLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
// Slice
AbstractBasePtr InferImplMakeSlice(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSliceGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
// Type checking
AbstractBasePtr InferImplTypeof(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplHasType(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplIsInstance(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
// Shape processing
AbstractBasePtr InferImplReduceShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplShapeMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
// Auto-Grad
AbstractBasePtr InferImplStopGradient(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplFakeBprop(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplJ(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplBroadcastGradientArgs(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list);
// Other
AbstractBasePtr InferImplTaylor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplShard(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplVmap(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplConvertToAdapterTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplConvertToMsTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const AbstractBasePtrList &args_spec_list);

// Delete this when the infer value can be mapped to the CPU backend operator.
bool PrimNeedFrontendInferValue(const PrimitivePtr &primitive);

const PrimitiveEvalImplMap &GetFrontendPrimitiveInferMap();
PrimitiveEvalImplMap *GetFrontendPrimitiveInferMapPtr();
// get prim infer from core/ops infer map or frontend infer map
std::optional<StandardPrimitiveImplReg> GetFrontendPrimitiveInferImpl(const PrimitivePtr &primitive);
#define REGISTER_PRIMITIVE_FRONT_EVAL_IMPL(name, primitive, infer_impl, infer_value_impl)                         \
  auto helper_##name = abstract::RegisterStandardPrimitiveEvalHelper(abstract::GetFrontendPrimitiveInferMapPtr(), \
                                                                     primitive, infer_impl, infer_value_impl, false);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATE_OPS_FRONT_INFER_FUNCTION_H_
