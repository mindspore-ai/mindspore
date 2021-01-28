/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_PRIMITIVE_C_H_
#define MINDSPORE_CORE_OPS_PRIMITIVE_C_H_
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ir/primitive.h"
#include "abstract/primitive_infer_map.h"
#include "ir/value.h"
namespace mindspore {
namespace ops {
class PrimitiveC : public Primitive {
 public:
  explicit PrimitiveC(const std::string &name) : Primitive(name) {}
  MS_DECLARE_PARENT(PrimitiveC, Primitive);
  ~PrimitiveC() = default;
  AbstractBasePtr Infer(const AbstractBasePtrList &abstract_list);

 protected:
  void InitIOName(const std::vector<std::string> &inputs_name, const std::vector<std::string> &outputs_name);
};

using OpPrimCDefineFunc = std::function<std::shared_ptr<PrimitiveC>()>;
class OpPrimCRegister {
 public:
  ~OpPrimCRegister() {}
  static OpPrimCRegister &GetInstance();
  std::map<std::string, OpPrimCDefineFunc> GetPrimCMap();
  void SetPrimCMap(const std::string &kname, const OpPrimCDefineFunc &fn);

 private:
  OpPrimCRegister() {}
  std::map<std::string, OpPrimCDefineFunc> op_primc_fns_;
};

class OpPrimCRegisterHelper {
 public:
  OpPrimCRegisterHelper(const std::string &kname, const OpPrimCDefineFunc &fn) {
    OpPrimCRegister::GetInstance().SetPrimCMap(kname, fn);
  }
  ~OpPrimCRegisterHelper() = default;
};

#define REGISTER_PRIMITIVE_C(kname, primc)               \
  std::shared_ptr<PrimitiveC> GetDefaultPrimC##primc() { \
    auto out = std::make_shared<primc>();                \
    return out;                                          \
  }                                                      \
  OpPrimCRegisterHelper primc_gen_##kname(kname, GetDefaultPrimC##primc);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_PRIMITIVE_C_H_
