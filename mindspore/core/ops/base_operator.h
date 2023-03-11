/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_BASE_OPERATOR_
#define MINDSPORE_CORE_OPS_BASE_OPERATOR_

#include <memory>
#include <string>
#include <vector>
#include <map>

#include "mindapi/ir/primitive.h"

namespace mindspore {
namespace abstract {
class AnalysisEngine;
using AnalysisEnginePtr = std::shared_ptr<AnalysisEngine>;

class AbstractBase;
using AbstractBasePtr = std::shared_ptr<AbstractBase>;
}  // namespace abstract
}  // namespace mindspore

namespace mindspore {
class Primitive;
using PrimitivePtr = std::shared_ptr<Primitive>;
}  // namespace mindspore

namespace mindspore {
namespace ops {
class PrimitiveC;
using PrimitiveCPtr = std::shared_ptr<PrimitiveC>;
class MIND_API BaseOperator : public api::Primitive {
 public:
  MIND_API_BASE_MEMBER(BaseOperator);
  explicit BaseOperator(const std::string &name);
  PrimitiveCPtr GetPrim();

  void set_batch_rank(int64_t batch_rank);
  int64_t get_batch_rank() const;

 protected:
  void InitIOName(const std::vector<std::string> &inputs_name, const std::vector<std::string> &outputs_name);
};

using OperatorDefineFunc = std::function<std::shared_ptr<BaseOperator>(const std::shared_ptr<mindspore::Base> &)>;
class MIND_API OperatorRegister {
 public:
  ~OperatorRegister() {}

  static OperatorRegister &GetInstance();

  const std::map<std::string, OperatorDefineFunc> &GetOperatorMap();

  void SetOperatorMap(const std::string &kname, const OperatorDefineFunc &fn);

 private:
  OperatorRegister() {}
  std::map<std::string, OperatorDefineFunc> operator_fns_;
};

class MIND_API OperatorRegisterHelper {
 public:
  OperatorRegisterHelper(const std::string &kname, const OperatorDefineFunc &fn) {
    OperatorRegister::GetInstance().SetOperatorMap(kname, fn);
    (void)id_;  // make compiler happy on macos
  }

  ~OperatorRegisterHelper() = default;

 private:
  int id_{0};
};

#define OPERATOR_CREATOR_REG(K_NAME, OP_CLASS)                                                                   \
  std::shared_ptr<BaseOperator> GetDefaultBaseOperator##OP_CLASS(const std::shared_ptr<mindspore::Base> &impl) { \
    return std::make_shared<OP_CLASS>(impl);                                                                     \
  }                                                                                                              \
  OperatorRegisterHelper operator_gen_##OP_CLASS(K_NAME, GetDefaultBaseOperator##OP_CLASS)

#define MIND_API_OPERATOR_IMPL(ClassName, ParentClassName)    \
  MIND_API_BASE_IMPL(ClassName, PrimitiveC, ParentClassName); \
  OPERATOR_CREATOR_REG(#ClassName, ClassName)

// This macro is for operator whose name is not same as its class name.
#define MIND_API_OPERATOR_NAME_IMPL(ClassName, OpName, ParentClassName) \
  MIND_API_BASE_IMPL(ClassName, PrimitiveC, ParentClassName);           \
  OPERATOR_CREATOR_REG(OpName, ClassName)
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BASE_OPERATOR_
