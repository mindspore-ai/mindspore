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
#include "ir/primitive.h"
#include "base/core_ops.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"

namespace mindspore {
namespace abstract {
using StandardPrimitiveEvalImpl = AbstractBasePtr (*)(const abstract::AnalysisEnginePtr &, const PrimitivePtr &,
                                                      const AbstractBasePtrList &);
using InferValueEvalImpl = ValuePtr (*)(const PrimitivePtr &, const AbstractBasePtrList &, const AbstractBasePtr &);

struct StandardPrimitiveImplReg {
  StandardPrimitiveEvalImpl impl_;       // Implement function of Primitive
  InferValueEvalImpl infer_value_func_;  // infer value of primitive
  // true means this primitive can be executed by vm backend else will be constant folded by frontend
  bool in_white_list_;
};

using PrimitiveEvalImplMap =
  std::unordered_map<PrimitivePtr, StandardPrimitiveImplReg, PrimitiveHasher, PrimitiveEqual>;

PrimitiveEvalImplMap &GetPrimitiveToEvalImplMap();

PrimitiveEvalImplMap &GetPrimitiveToBackendEvalImplMap();

StandardPrimitiveEvalImpl GetPrimitiveInferImpl(const PrimitivePtr &primitive);

std::vector<int64_t> GetDependsFormMap(const CNodePtr &cnode);

void RegisterStandardPrimitiveImpl(const PrimitivePtr &primitive, const StandardPrimitiveImplReg &impl_reg);

class RegisterStandardPrimitiveEvalHelper {
 public:
  RegisterStandardPrimitiveEvalHelper(const PrimitivePtr &primitive, const StandardPrimitiveEvalImpl &impl,
                                      const InferValueEvalImpl &infer_value_impl, const bool is_wight_list = true) {
    const StandardPrimitiveImplReg impl_reg{impl, infer_value_impl, is_wight_list};
    RegisterStandardPrimitiveImpl(primitive, impl_reg);
  }
  ~RegisterStandardPrimitiveEvalHelper() = default;
};

#define REGISTER_PRIMITIVE_EVAL_IMPL(name, primitive, impl, infer_value_impl, is_wight_list) \
  static auto helper_##name =                                                                \
    abstract::RegisterStandardPrimitiveEvalHelper(primitive, impl, infer_value_impl, is_wight_list)
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_PRIMITIVE_INFER_MAP_H_
