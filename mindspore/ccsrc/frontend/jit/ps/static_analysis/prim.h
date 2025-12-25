/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_STATIC_ANALYSIS_PRIM_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_STATIC_ANALYSIS_PRIM_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "utils/hash_map.h"
#include "frontend/jit/ps/static_analysis/evaluator.h"
#include "frontend/jit/ps/static_analysis/prim_to_function.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_def.h"
#include "ops/ops_frontend_func_impl.h"

namespace mindspore {
namespace abstract {
class PrimitiveFunctionEvaluator final : public TrivialPrimEvaluator {
 public:
  explicit PrimitiveFunctionEvaluator(const PrimitivePtr &primitive);
  ~PrimitiveFunctionEvaluator() override = default;
  MS_DECLARE_PARENT(PrimitiveFunctionEvaluator, TrivialPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) override;
  std::string ToString() const override { return identifier_ + "_PrimitiveFunction_" + prim_func_->name(); }

 protected:
  bool inplace_prim() const override { return prim_func_->inplace_prim(); }
  bool graph_view_prim() const {
    MS_EXCEPTION_IF_NULL(op_def_);
    return op_def_->is_graph_view_;
  }
  std::vector<size_t> rw_write_input_indexes() const { return prim_func_->rw_write_input_indexes(); }
  std::vector<int64_t> inplace_input_indexes() const { return prim_func_->inplace_input_indexes(); }

 private:
  AbstractBasePtr CheckAndInfer(const AbstractBasePtrList &args);
  AbstractBasePtr ProcessViewInplaceAbstract(const AbstractBasePtrList &args, const AbstractBasePtr &res);
  void CheckArgsSizeAndType(const AbstractBasePtrList &args);
  PrimitivePtr prim_func_;
  mindspore::ops::OpDefPtr op_def_{nullptr};
  mindspore::ops::OpFrontendFuncImplPtr frontend_func_impl_{nullptr};
};

class StandardPrimEvaluator final : public TrivialPrimEvaluator {
 public:
  StandardPrimEvaluator(const PrimitivePtr &primitive, const StandardPrimitiveImplReg &eval_impl)
      : TrivialPrimEvaluator("StandardPrimEvaluator"), prim_(primitive), eval_impl_(eval_impl) {}
  explicit StandardPrimEvaluator(const PrimitivePtr &primitive)
      : TrivialPrimEvaluator("StandardPrimEvaluator"), prim_(primitive) {}
  ~StandardPrimEvaluator() override = default;
  MS_DECLARE_PARENT(StandardPrimEvaluator, TrivialPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) override;
  PrimitivePtr prim() { return prim_; }

  std::string ToString() const override { return identifier_ + "_" + prim_->name(); }

 protected:
  bool inplace_prim() const override { return prim_->inplace_prim(); }
  std::vector<size_t> rw_write_input_indexes() const { return prim_->rw_write_input_indexes(); }
  std::vector<int64_t> inplace_input_indexes() const { return prim_->inplace_input_indexes(); }

 private:
  EvalResultPtr EvalPyCheckPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args);
  EvalResultPtr RunPyInferValue(const AnalysisEnginePtr &engine, const AbstractBasePtr &abs_base,
                                const AbstractBasePtrList &args);
  PrimitivePtr prim_;
  const StandardPrimitiveImplReg eval_impl_;
};

using StandardPrimEvaluatorPtr = std::shared_ptr<StandardPrimEvaluator>;

class PythonPrimEvaluator final : public TrivialPrimEvaluator {
 public:
  explicit PythonPrimEvaluator(const PrimitivePyPtr &primitive)
      : TrivialPrimEvaluator("PythonPrimEvaluator"), prim_py_(primitive) {}
  ~PythonPrimEvaluator() override = default;
  MS_DECLARE_PARENT(PythonPrimEvaluator, TrivialPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) override;
  PrimitivePtr prim() { return dyn_cast<Primitive>(prim_py_); }

  std::string ToString() const override { return identifier_ + "_" + prim_py_->name(); }

 protected:
  bool inplace_prim() const override { return dyn_cast<Primitive>(prim_py_)->inplace_prim(); }

 private:
  PrimitivePyPtr prim_py_;
};

using ValuePtrList = std::vector<ValuePtr>;
using PrimitiveImpl = ValuePtr (*)(const ValuePtrList &);

class UniformPrimEvaluator final : public TrivialPrimEvaluator {
 public:
  UniformPrimEvaluator(const PrimitivePtr &primitive, const PrimitiveImpl &impl, bool eval_value,
                       const TypePtr &specify_out_type)
      : TrivialPrimEvaluator("UniformPrimEvaluator"),
        prim_(primitive),
        impl_(impl),
        eval_value_(eval_value),
        specify_out_type_(specify_out_type) {
    FunctionPtr func = nullptr;
    (void)prim::PrimToFunction::GetInstance().GetFunction(primitive, &func);
    MS_EXCEPTION_IF_NULL(func);
    func_desc_ = func;
    nargs_ = func->args().size();
    return_value_type_ = func->retval();

    for (size_t i = 0; i < nargs_; ++i) {
      const TypePtr &type = func_desc_->args()[i];
      type_map_[type].push_back(i);
    }
  }
  ~UniformPrimEvaluator() override { impl_ = nullptr; };
  MS_DECLARE_PARENT(UniformPrimEvaluator, TrivialPrimEvaluator);

  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args) override;
  ValuePtr RunImpl(const ValuePtrList &args) const;

  // If eval_value_ is False, return broadened arguments.
  AbstractBasePtrList NormalizeArgs(const AbstractBasePtrList &args_abs_list) const override {
    if (!eval_value_) {
      AbstractBasePtrList broadened_args_abs_list;
      (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(broadened_args_abs_list),
                           [](const AbstractBasePtr &arg) -> AbstractBasePtr { return arg->Broaden(); });
      return broadened_args_abs_list;
    }
    return args_abs_list;
  }

  std::string ToString() const override { return identifier_ + "_" + prim_->name(); }

 protected:
  bool inplace_prim() const override { return false; }

 private:
  PrimitivePtr prim_;
  PrimitiveImpl impl_;
  bool eval_value_;
  FunctionPtr func_desc_;
  std::size_t nargs_;
  TypePtr return_value_type_;
  TypePtr specify_out_type_;
  mindspore::HashMap<TypePtr, std::vector<size_t>, TypeHashById, TypeEqualById> type_map_;
};

class DoSignatureEvaluator final : public Evaluator {
 public:
  explicit DoSignatureEvaluator(const PrimitivePtr primitive) : Evaluator("DoSignatureEvaluator"), prim_(primitive) {}
  ~DoSignatureEvaluator() override = default;
  MS_DECLARE_PARENT(DoSignatureEvaluator, Evaluator);
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override;

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(INTERNAL_EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

 private:
  PrimitivePtr prim_;
  CNodePtr GenerateNewNodeBySignatures(const ValuePtr &func, const AbstractBasePtrList &args_abs_list,
                                       const AnalysisEnginePtr &engine, const AnfNodeConfigPtr &out_conf);
};

class UnpackGraphEvaluator final : public Evaluator {
 public:
  explicit UnpackGraphEvaluator(const PrimitivePtr primitive) : Evaluator("UnpackGraphEvaluator"), prim_(primitive) {}
  ~UnpackGraphEvaluator() override = default;
  MS_DECLARE_PARENT(UnpackGraphEvaluator, Evaluator);
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override;

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(INTERNAL_EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

 private:
  PrimitivePtr prim_;
};

class MixedPrecisionCastEvaluator final : public Evaluator {
 public:
  explicit MixedPrecisionCastEvaluator(const PrimitivePtr primitive)
      : Evaluator("MixedPrecisionCastEvaluator"), prim_(primitive) {}
  ~MixedPrecisionCastEvaluator() override = default;
  MS_DECLARE_PARENT(MixedPrecisionCastEvaluator, Evaluator);
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override;

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(INTERNAL_EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

 private:
  PrimitivePtr prim_;
};

class SwitchEvaluator final : public Evaluator {
 public:
  SwitchEvaluator() : Evaluator("SwitchEvaluator") {}
  ~SwitchEvaluator() override = default;
  MS_DECLARE_PARENT(SwitchEvaluator, Evaluator);
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override;

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(INTERNAL_EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }
};

class PrimitiveArgsToInputsEvaluator final : public TransitionPrimEvaluator {
 public:
  explicit PrimitiveArgsToInputsEvaluator(const PrimitivePtr primitive)
      : TransitionPrimEvaluator("PrimitiveArgsToInputsEvaluator"), prim_(primitive) {}
  ~PrimitiveArgsToInputsEvaluator() override = default;
  MS_DECLARE_PARENT(PrimitiveArgsToInputsEvaluator, TransitionPrimEvaluator)
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;

 private:
  PrimitivePtr prim_;
};

class DoTransPrimitiveFunctionEvaluator final : public TransitionPrimEvaluator {
 public:
  explicit DoTransPrimitiveFunctionEvaluator(const PrimitivePtr primitive)
      : TransitionPrimEvaluator("DoTransPrimitiveFunctionEvaluator"), prim_(primitive) {}
  ~DoTransPrimitiveFunctionEvaluator() override = default;
  MS_DECLARE_PARENT(DoTransPrimitiveFunctionEvaluator, TransitionPrimEvaluator)
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;

 private:
  PrimitivePtr prim_;
};

class PrimInstanceEvaluator final : public TransitionPrimEvaluator {
 public:
  explicit PrimInstanceEvaluator(const std::string &prim_name, const AnfNodePtr node)
      : TransitionPrimEvaluator("PrimInstanceEvaluator"), prim_name_(prim_name), instance_node_(AnfNodePtr(node)) {}
  ~PrimInstanceEvaluator() override = default;
  MS_DECLARE_PARENT(PrimInstanceEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;

 private:
  std::string prim_name_;
  AnfNodeWeakPtr instance_node_;
};

class PrimitiveToMetaEvaluator final : public TransitionPrimEvaluator {
 public:
  explicit PrimitiveToMetaEvaluator(const PrimitivePtr primitive)
      : TransitionPrimEvaluator("PrimitiveToMetaEvaluator"), prim_(primitive) {}
  ~PrimitiveToMetaEvaluator() override = default;
  MS_DECLARE_PARENT(PrimitiveToMetaEvaluator, TransitionPrimEvaluator)
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;

 private:
  PrimitivePtr prim_;
};

class FunctionalEvaluator final : public TransitionPrimEvaluator {
 public:
  explicit FunctionalEvaluator(const std::string &name, const FunctionalPtr &functional, bool is_method)
      : TransitionPrimEvaluator("FunctionalEvaluator"), name_(name), functional_(functional), is_method_(is_method) {}
  ~FunctionalEvaluator() override = default;
  MS_DECLARE_PARENT(FunctionalEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;

 private:
  std::string name_;
  FunctionalPtr functional_{nullptr};
  bool is_method_{false};
};

class ConstexprEvaluator final : public TransitionPrimEvaluator {
 public:
  explicit ConstexprEvaluator(const PrimitivePyPtr primitive)
      : TransitionPrimEvaluator("ConstexprEvaluator"), prim_py_(primitive) {}
  ~ConstexprEvaluator() override = default;
  MS_DECLARE_PARENT(ConstexprEvaluator, TransitionPrimEvaluator)
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;

 private:
  PrimitivePyPtr prim_py_;
};

class MakeTupleEvaluator final : public TransitionPrimEvaluator {
 public:
  MakeTupleEvaluator() : TransitionPrimEvaluator("MakeTupleEvaluator") {}
  ~MakeTupleEvaluator() override = default;
  MS_DECLARE_PARENT(MakeTupleEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;
};

class MakeListEvaluator final : public TransitionPrimEvaluator {
 public:
  MakeListEvaluator() : TransitionPrimEvaluator("MakeListEvaluator") {}
  ~MakeListEvaluator() override = default;
  MS_DECLARE_PARENT(MakeListEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;
};

class PyExecuteEvaluator final : public TransitionPrimEvaluator {
 public:
  PyExecuteEvaluator() : TransitionPrimEvaluator("PyExecuteEvaluator") {}
  ~PyExecuteEvaluator() override = default;
  MS_DECLARE_PARENT(PyExecuteEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override;
};

bool IsInWhiteList(const PrimitivePtr &primitive);

PrimEvaluatorMap &GetPrimEvaluatorConstructors();

void ClearPrimEvaluatorMap();

// Return an abstract value for the sensitivity of x.
// The sensitivity of a function is an Env
// The sensitivity of J(x) is x
// else self.Clone;
AbstractBasePtr SensitivityTransform(const AbstractBasePtr &spec);

}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_STATIC_ANALYSIS_PRIM_H_
