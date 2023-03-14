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
#include "ir/value.h"
#include "utils/hash_map.h"
namespace mindspore {
namespace ops {
/// \brief PrimitiveC defines the base class for end side operators.
class MS_CORE_API PrimitiveC : public Primitive {
 public:
  /// \brief Constructor for PrimitiveC.
  ///
  /// \param[in] name The name of the end side operator.
  explicit PrimitiveC(const std::string &name) : Primitive(name) {}
  MS_DECLARE_PARENT(PrimitiveC, Primitive);

  /// \brief Destructor of PrimitiveC.
  ~PrimitiveC() = default;

  /// \brief Derive the abstract of the PrimitiveC object.
  ///
  /// \param[in] abstract_list The abstract of the inputs of the PrimitiveC object.
  /// \return The abstract of the PrimitiveC object.
  AbstractBasePtr Infer(const AbstractBasePtrList &abstract_list);

 protected:
  void InitIOName(const std::vector<std::string> &inputs_name, const std::vector<std::string> &outputs_name);
};

using OpPrimCDefineFunc = std::function<std::shared_ptr<PrimitiveC>()>;
/// \brief OpPrimCRegister defines the singleton to save the end side operators.
class MS_CORE_API OpPrimCRegister {
 public:
  /// \brief Destructor of OpPrimCRegister.
  ~OpPrimCRegister() {}

  /// \brief Get the OpPrimCRegister singleton.
  ///
  /// \return The OpPrimCRegister singleton.
  static OpPrimCRegister &GetInstance();

  /// \brief Get PrimCMap of the OpPrimCRegister singleton.
  ///
  /// \return The PrimCMap of the OpPrimCRegister singleton.
  const HashMap<std::string, OpPrimCDefineFunc> &GetPrimCMap();

  /// \brief Add an element into the PrimCMap of the OpPrimCRegister singleton.
  ///
  /// param[in] kname The name of the input end side operator.
  /// param[in] fn The input end side operator.
  void SetPrimCMap(const std::string &kname, const OpPrimCDefineFunc &fn);

 private:
  OpPrimCRegister() {}
  HashMap<std::string, OpPrimCDefineFunc> op_primc_fns_;
};

/// \brief OpPrimCRegisterHelper defines the helper class for the OpPrimCRegister singleton.
class MS_CORE_API OpPrimCRegisterHelper {
 public:
  /// \brief Constructor for OpPrimCRegisterHelper.
  ///
  /// param[in] kname The name of the input end side operator.
  /// param[in] fn The input end side operator.
  OpPrimCRegisterHelper(const std::string &kname, const OpPrimCDefineFunc &fn) {
    OpPrimCRegister::GetInstance().SetPrimCMap(kname, fn);
    (void)id_;  // make compiler happy on macos
  }

  /// Destructor of OpPrimCRegisterHelper.
  ~OpPrimCRegisterHelper() = default;

 private:
  int id_{0};
};

#define REGISTER_PRIMITIVE_C(kname, primc)                    \
  std::shared_ptr<PrimitiveC> GetDefaultPrimC##primc() {      \
    primc out;                                                \
    return std::dynamic_pointer_cast<PrimitiveC>(out.impl()); \
  }                                                           \
  OpPrimCRegisterHelper primc_gen_##kname(kname, GetDefaultPrimC##primc);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_PRIMITIVE_C_H_
