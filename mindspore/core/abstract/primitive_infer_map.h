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
#ifndef MINDSPORE_CORE_ABSTRACT_PRIMITIVE_INFER_MAP_H_
#define MINDSPORE_CORE_ABSTRACT_PRIMITIVE_INFER_MAP_H_
#include <unordered_map>
#include <vector>
#include <memory>
#include "ir/primitive.h"
#include "ops/primitive_c.h"
#include "base/core_ops.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"

namespace mindspore {
namespace abstract {
using InferShapeImpl = AbstractBasePtr (*)(const abstract::AnalysisEnginePtr &, const PrimitivePtr &,
                                           const AbstractBasePtrList &);
using InferValueImpl = ValuePtr (*)(const PrimitivePtr &, const AbstractBasePtrList &);

struct StandardPrimitiveImplReg {
  InferShapeImpl infer_shape_impl_;  // infer shape and type for ops
  InferValueImpl infer_value_impl_;  // infer value for ops
  // in_white_list_ is true means this primitive can be executed by vm backend
  // else will be optimized by frontend
  bool in_white_list_;
};

using PrimitiveEvalImplMap =
  std::unordered_map<PrimitivePtr, StandardPrimitiveImplReg, PrimitiveHasher, PrimitiveEqual>;

PrimitiveEvalImplMap &GetPrimitiveToEvalImplMap();

PrimitiveEvalImplMap &GetPrimitiveToBackendEvalImplMap();

StandardPrimitiveImplReg GetPrimitiveInferImpl(const PrimitivePtr &primitive);

std::vector<int64_t> GetDependsFormMap(const CNodePtr &cnode);

void RegisterStandardPrimitiveImpl(const PrimitivePtr &primitive, const StandardPrimitiveImplReg &impl_reg);

class RegisterStandardPrimitiveEvalHelper {
 public:
  RegisterStandardPrimitiveEvalHelper(const PrimitivePtr &primitive, const InferShapeImpl &infer_impl,
                                      const InferValueImpl &infer_value_impl, const bool is_wight_list = true) {
    const StandardPrimitiveImplReg impl_reg{infer_impl, infer_value_impl, is_wight_list};
    RegisterStandardPrimitiveImpl(primitive, impl_reg);
  }
  ~RegisterStandardPrimitiveEvalHelper() = default;
};

#define REGISTER_PRIMITIVE_EVAL_IMPL(name, primitive, infer_impl, infer_value_impl, is_wight_list)         \
  static auto helper_##name =                                                                              \
    abstract::RegisterStandardPrimitiveEvalHelper(primitive, infer_impl, infer_value_impl, is_wight_list); \
  std::shared_ptr<ops::PrimitiveC> GetDefaultPrimC##name() {                                               \
    auto out = std::make_shared<name>();                                                                   \
    return out;                                                                                            \
  }                                                                                                        \
  ops::OpPrimCRegisterHelper primc_gen_##name(#name, GetDefaultPrimC##name);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_PRIMITIVE_INFER_MAP_H_
