/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_ABSTRACT_PRIMITIVE_INFER_MAP_H_
#define MINDSPORE_CORE_ABSTRACT_PRIMITIVE_INFER_MAP_H_

#include <vector>
#include <set>
#include <string>
#include <memory>
#include "utils/hash_map.h"
#include "ir/primitive.h"
#include "ops/primitive_c.h"
#include "mindspore/core/ops/core_ops.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "abstract/ops/op_infer.h"

namespace mindspore {
namespace abstract {
using InferAbstractImpl = AbstractBasePtr (*)(const abstract::AnalysisEnginePtr &, const PrimitivePtr &,
                                              const AbstractBasePtrList &);
using InferValueImpl = ValuePtr (*)(const PrimitivePtr &, const AbstractBasePtrList &);

class MS_CORE_API StandardPrimitiveImplReg {
 public:
  StandardPrimitiveImplReg() = default;
  StandardPrimitiveImplReg(const InferAbstractImpl &infer_abstract, const InferValueImpl &infer_value,
                           bool in_white_list);
  StandardPrimitiveImplReg(const OpInferBasePtr &op_infer, bool is_impl_infer_value)
      : op_infer_(op_infer), is_impl_infer_value_(is_impl_infer_value) {}
  ~StandardPrimitiveImplReg() = default;

  const OpInferBasePtr Get() const { return op_infer_; }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const;
  BaseShapePtr InferShape(const PrimitivePtr &prim, const AbstractBasePtrList &args) const;
  TypePtr InferType(const PrimitivePtr &prim, const AbstractBasePtrList &args) const;
  ValuePtr InferValue(const PrimitivePtr &prim, const AbstractBasePtrList &args) const;

  bool IsImplInferShapeAndType() const { return is_impl_infer_shape_and_type_ && op_infer_ != nullptr; }
  bool IsImplInferValue() const { return is_impl_infer_value_ && op_infer_ != nullptr; }
  bool IsInWhileList() const { return in_white_list_; }

 private:
  OpInferBasePtr op_infer_{nullptr};  // Infer shape, type and value.
  bool is_impl_infer_shape_and_type_{true};
  bool is_impl_infer_value_{false};
  // in_white_list_ is true means this primitive can be executed by vm backend
  // else will be optimized by frontend
  bool in_white_list_{true};
};

void IsImplInferShapeAndType(const OpInferBasePtr &op_infer);
void IsImplInferValue(const OpInferBasePtr &op_infer);

using PrimitiveEvalImplMap =
  mindspore::HashMap<PrimitivePtr, StandardPrimitiveImplReg, PrimitiveHasher, PrimitiveEqual>;

using PrimShapeDependMap = mindspore::HashMap<std::string, std::set<int64_t>>;

MS_CORE_API PrimitiveEvalImplMap &GetPrimitiveToEvalImplMap();

MS_CORE_API PrimitiveEvalImplMap &GetPrimitiveToBackendEvalImplMap();

MS_CORE_API StandardPrimitiveImplReg GetPrimitiveInferImpl(const PrimitivePtr &primitive);

MS_CORE_API std::set<int64_t> GetValueDependArgIndices(const CNodePtr &cnode);

MS_CORE_API void RegisterStandardPrimitiveImpl(const PrimitivePtr &primitive, const StandardPrimitiveImplReg &impl_reg);

class RegisterStandardPrimitiveEvalHelper {
 public:
  RegisterStandardPrimitiveEvalHelper(const PrimitivePtr &primitive, const InferAbstractImpl &infer_shape_and_type_impl,
                                      const InferValueImpl &infer_value_impl, const bool is_white_list = true) {
    const StandardPrimitiveImplReg impl_reg{infer_shape_and_type_impl, infer_value_impl, is_white_list};
    RegisterStandardPrimitiveImpl(primitive, impl_reg);
  }

  RegisterStandardPrimitiveEvalHelper(const PrimitivePtr &primitive, const OpInferBasePtr &op_infer,
                                      bool is_impl_infer_value = false) {
    const StandardPrimitiveImplReg impl_reg{op_infer, is_impl_infer_value};
    RegisterStandardPrimitiveImpl(primitive, impl_reg);
  }
  ~RegisterStandardPrimitiveEvalHelper() = default;
};

#define REGISTER_PRIMITIVE_EVAL_IMPL(name, primitive, infer_shape_and_type_impl, infer_value_impl, is_white_list)      \
  static auto helper_eval_##name = abstract::RegisterStandardPrimitiveEvalHelper(primitive, infer_shape_and_type_impl, \
                                                                                 infer_value_impl, is_white_list);     \
  std::shared_ptr<ops::PrimitiveC> GetDefaultPrimC##name() {                                                           \
    name out;                                                                                                          \
    return std::dynamic_pointer_cast<ops::PrimitiveC>(out.impl());                                                     \
  }                                                                                                                    \
  ops::OpPrimCRegisterHelper primc_gen_##name(#name, GetDefaultPrimC##name);

#define REGISTER_PRIMITIVE_OP_INFER_IMPL(name, primitive, OP_INFER_ClASS, is_impl_infer_value)                         \
  const auto helper_op_infer_##name =                                                                                  \
    abstract::RegisterStandardPrimitiveEvalHelper(primitive, std::make_shared<OP_INFER_ClASS>(), is_impl_infer_value); \
  std::shared_ptr<ops::PrimitiveC> GetDefaultPrimC##name() {                                                           \
    name out;                                                                                                          \
    return std::dynamic_pointer_cast<ops::PrimitiveC>(out.impl());                                                     \
  }                                                                                                                    \
  ops::OpPrimCRegisterHelper primc_gen_##name(#name, GetDefaultPrimC##name)

MS_CORE_API void RegisterHostDependsImpl(const std::string &name, const std::set<int64_t> &host_depends);

MS_CORE_API void RegisterHostDependsImpl(const std::string &prim_name, const std::set<int64_t> &host_depends);

class RegisterHostDependsHelper {
 public:
  RegisterHostDependsHelper(const std::string &name, const std::set<int64_t> &depends) {
    RegisterHostDependsImpl(name, depends);
  }
  ~RegisterHostDependsHelper() = default;
};

// Processes such as InferShape need to obtain some inputs value on the host
#define REGISTER_HOST_DEPENDS(name, ...) \
  static auto helper_host_depends_##name = abstract::RegisterHostDependsHelper(name, __VA_ARGS__);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_PRIMITIVE_INFER_MAP_H_
