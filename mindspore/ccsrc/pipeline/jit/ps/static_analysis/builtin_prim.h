/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_BUILTIN_PRIM_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_BUILTIN_PRIM_H_

#include <memory>
#include <string>

#include "utils/hash_map.h"
#include "pipeline/jit/ps/static_analysis/evaluator.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace abstract {
class InnerAbsEvaluator : public TransitionPrimEvaluator {
 public:
  InnerAbsEvaluator() : TransitionPrimEvaluator("InnerAbsEvaluator") {}
  ~InnerAbsEvaluator() override = default;
  MS_DECLARE_PARENT(InnerAbsEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;
  bool CheckConst(const AbstractBasePtrList &args_abs_list) const;
};

class InnerRoundEvaluator : public TransitionPrimEvaluator {
 public:
  InnerRoundEvaluator() : TransitionPrimEvaluator("InnerRoundEvaluator") {}
  ~InnerRoundEvaluator() override = default;
  MS_DECLARE_PARENT(InnerRoundEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;
  bool CheckConst(const AbstractBasePtrList &args_abs_list) const;
};

class InnerLenEvaluator : public TransitionPrimEvaluator {
 public:
  InnerLenEvaluator() : TransitionPrimEvaluator("InnerLenEvaluator") {}
  ~InnerLenEvaluator() override = default;
  MS_DECLARE_PARENT(InnerLenEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;
};
}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_BUILTIN_PRIM_H_
