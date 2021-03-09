/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_PRIM_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_PRIM_H_

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pipeline/jit/static_analysis/evaluator.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace abstract {
class StandardPrimEvaluator : public TrivialPrimEvaluator {
 public:
  StandardPrimEvaluator(const PrimitivePtr primitive, StandardPrimitiveEvalImpl eval_impl)
      : TrivialPrimEvaluator("StandardPrimEvaluator"), prim_(primitive), eval_impl_(eval_impl) {}
  ~StandardPrimEvaluator() override = default;
  MS_DECLARE_PARENT(StandardPrimEvaluator, TrivialPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) override;
  PrimitivePtr prim() { return prim_; }

  std::string ToString() const override { return identifier_ + prim_->name(); }

 private:
  EvalResultPtr EvalPyCheckPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args);

  PrimitivePtr prim_;
  const StandardPrimitiveEvalImpl eval_impl_;
};

using StandardPrimEvaluatorPtr = std::shared_ptr<StandardPrimEvaluator>;

class PythonPrimEvaluator : public TrivialPrimEvaluator {
 public:
  explicit PythonPrimEvaluator(const PrimitivePyPtr primitive)
      : TrivialPrimEvaluator("PythonPrimEvaluator"), prim_py_(primitive) {}
  ~PythonPrimEvaluator() override = default;
  MS_DECLARE_PARENT(PythonPrimEvaluator, TrivialPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) override;
  PrimitivePtr prim() { return dyn_cast<Primitive>(prim_py_); }

  std::string ToString() const override { return identifier_ + prim_py_->name(); }

 private:
  PrimitivePyPtr prim_py_;
};

class DoSignatureEvaluator : public Evaluator {
 public:
  explicit DoSignatureEvaluator(const PrimitivePtr primitive) : Evaluator("DoSignatureEvaluator"), prim_(primitive) {}
  ~DoSignatureEvaluator() override = default;
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &argrefs,
                    AnfNodeConfigPtr out_config = nullptr) override;

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &) override {
    MS_LOG(EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

 private:
  PrimitivePtr prim_;
};

class UnpackGraphEvaluator : public Evaluator {
 public:
  explicit UnpackGraphEvaluator(const PrimitivePtr primitive) : Evaluator("UnpackGraphEvaluator"), prim_(primitive) {}
  ~UnpackGraphEvaluator() override = default;
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &argrefs,
                    AnfNodeConfigPtr out_config = nullptr) override;

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &) override {
    MS_LOG(EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

 private:
  PrimitivePtr prim_;
};

class MixedPrecisionCastEvaluator : public Evaluator {
 public:
  explicit MixedPrecisionCastEvaluator(const PrimitivePtr primitive)
      : Evaluator("MixedPrecisionCastEvaluator"), prim_(primitive) {}
  ~MixedPrecisionCastEvaluator() override = default;
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &argrefs,
                    AnfNodeConfigPtr out_config = nullptr) override;

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &) override {
    MS_LOG(EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

 private:
  PrimitivePtr prim_;
};

bool IsInWhiteList(const PrimitivePtr &primitive);

using ValuePtrList = std::vector<ValuePtr>;
using PrimitiveImpl = ValuePtr (*)(const ValuePtrList &);

class UniformPrimEvaluator : public TrivialPrimEvaluator {
 public:
  UniformPrimEvaluator(const FunctionPtr func_desc, PrimitiveImpl impl, bool eval_value, const TypePtr specify_out_type)
      : TrivialPrimEvaluator("UniformPrimEvaluator"),
        impl_(impl),
        eval_value_(eval_value),
        func_desc_(func_desc),
        nargs_(func_desc_->args().size()),
        return_value_type_(func_desc_->retval()),
        specify_out_type_(specify_out_type) {
    for (size_t i = 0; i < nargs_; ++i) {
      TypePtr type = func_desc_->args()[i];
      if (type_map_[type]) {
        type_map_[type]->push_back(i);
      } else {
        type_map_[type] = std::make_shared<std::vector<size_t>>();
        type_map_[type]->push_back(i);
      }
    }
  }
  ~UniformPrimEvaluator() override = default;
  MS_DECLARE_PARENT(UniformPrimEvaluator, TrivialPrimEvaluator);

  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) override;
  ValuePtr RunImpl(const ValuePtrList &args) const;

  // If eval_value_ is False, return broadened arguments.
  AbstractBasePtrList NormalizeArgs(const AbstractBasePtrList &args_spec_list) const override {
    if (!eval_value_) {
      AbstractBasePtrList broadened_args_spec_list;
      (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(broadened_args_spec_list),
                           [](const AbstractBasePtr &arg) -> AbstractBasePtr { return arg->Broaden(); });
      return broadened_args_spec_list;
    }
    return args_spec_list;
  }

 private:
  PrimitiveImpl impl_;
  bool eval_value_;
  const FunctionPtr func_desc_;
  const std::size_t nargs_;
  const TypePtr return_value_type_;
  const TypePtr specify_out_type_;
  std::unordered_map<TypePtr, std::shared_ptr<std::vector<size_t>>, TypeHasher, TypeEqual> type_map_;
};

PrimEvaluatorMap &GetPrimEvaluatorConstructors();

// Check whether type x is a subtype of model.
bool IsSubtype(const AbstractBasePtr x, const TypePtr model);

void ClearPrimEvaluatorMap();

py::dict ConvertAbstractToPython(const AbstractBasePtr &abs_base);
}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_PRIM_H_
