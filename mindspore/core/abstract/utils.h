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

#ifndef MINDSPORE_CORE_ABSTRACT_UTILS_H_
#define MINDSPORE_CORE_ABSTRACT_UTILS_H_

#include <vector>
#include <utility>
#include <memory>
#include <string>
#include <functional>
#include "abstract/abstract_value.h"
#include "utils/any.h"
#include "utils/misc.h"
#include "utils/shape_utils.h"
#include "mindapi/base/macros.h"

namespace mindspore {
namespace abstract {
ValuePtr ValueJoin(const ValuePtr &value1, const ValuePtr &value2);
MS_CORE_API TypePtr TypeJoin(const TypePtr &type1, const TypePtr &type2);
ShapePtr ShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2);

MS_CORE_API AbstractBasePtr AbstractJoin(const AbstractBasePtrList &args_spec_list);
MS_CORE_API AbstractBasePtrList AbstractJoin(const AbstractBasePtrList &spec1, const AbstractBasePtrList &spec2);
MS_CORE_API AbstractBasePtr AbstractBroaden(const AbstractBasePtr &abs);

// Return an abstract value for the sensitivity of x.
// The sensitivity of a function is an Env
// The sensitivity of J(x) is x
// else self.Clone;
MS_CORE_API AbstractBasePtr SensitivityTransform(const AbstractBasePtr &spec);

ShapeVector BroadcastShape(ShapeVector shpx, ShapeVector shpy);
MS_CORE_API size_t TypeIdSize(const TypeId data_type);
template <typename T>
T ShapeSize(const std::vector<T> &shape) {
  return std::accumulate(shape.begin(), shape.end(), static_cast<T>(1), std::multiplies<T>());
}

AbstractBasePtr MakeAbstract(const BaseShapePtr &base_shape, const TypePtr &type);
MS_CORE_API AbstractBasePtr MakeMonadAbstract(const MonadTypePtr &type);
MS_CORE_API AbstractBasePtr MakeAbstractTensor(const ShapePtr &shape, const TypePtr &type);
MS_CORE_API std::vector<FuncGraphPtr> GetFuncGraphsFromAbs(const AnfNodePtr &anf_node);

class MS_CORE_API EnvSetSparseResultMgr {
 public:
  static EnvSetSparseResultMgr &GetInstance() noexcept {
    static EnvSetSparseResultMgr instance;
    return instance;
  }
  EnvSetSparseResultMgr(const EnvSetSparseResultMgr &) = delete;
  EnvSetSparseResultMgr &operator=(const EnvSetSparseResultMgr &) = delete;
  ~EnvSetSparseResultMgr() = default;

  bool Get() const { return env_set_sparse_result_; }
  void Set(bool env_set_sparse_result) { env_set_sparse_result_ = env_set_sparse_result; }

 private:
  EnvSetSparseResultMgr() = default;
  bool env_set_sparse_result_{false};
};
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_UTILS_H_
