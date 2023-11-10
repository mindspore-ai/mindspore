/**
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

#ifndef MINDSPORE_CORE_OPS_FRONTEND_FUNC_IMPL_H
#define MINDSPORE_CORE_OPS_FRONTEND_FUNC_IMPL_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include "ir/cell.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "mindapi/base/macros.h"

namespace mindspore::ops {
class OpFrontendFuncImpl {
 public:
  OpFrontendFuncImpl() = default;
  virtual ~OpFrontendFuncImpl() = default;

  /// \brief Infer the output value for target operator. Only override when needed.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's inputs.
  ///
  /// \return Inferred Value based on given inputs.
  virtual ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    return nullptr;
  }

  /// \brief Infer the related Abstract for target operator.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's inputs.
  ///
  /// \return AbstractBasePtr with inferred shape and inferred type.
  virtual AbstractBasePtr InferAbstract(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
    return nullptr;
  }
};

using OpFrontendFuncImplPtr = std::shared_ptr<OpFrontendFuncImpl>;

class FrontendFuncImplHolder {
 public:
  explicit FrontendFuncImplHolder(const OpFrontendFuncImplPtr &func_impl) : func_impl_(func_impl) {}
  ~FrontendFuncImplHolder() = default;
  OpFrontendFuncImplPtr get_func_impl() { return func_impl_; }

 private:
  OpFrontendFuncImplPtr func_impl_{nullptr};
};

using OpsFrontendFuncImplMap = std::unordered_map<std::string, FrontendFuncImplHolder>;

MS_CORE_API OpFrontendFuncImplPtr GetOpFrontendFuncImplPtr(const std::string &name);

class MS_CORE_API RegFrontendFuncImplHelper {
 public:
  RegFrontendFuncImplHelper(const std::string &name, const OpFrontendFuncImplPtr &func_impl);
  ~RegFrontendFuncImplHelper() = default;
};

#define REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL(name, func_impl_class) \
  static auto helper_##func_impl_class = RegFrontendFuncImplHelper(name, std::make_shared<func_impl_class>());

using InferValueFunc = std::function<ValuePtr(const std::string &, const AbstractBasePtrList &)>;
class MS_CORE_API InferValueCallback {
 public:
  InferValueCallback(const InferValueCallback &) = delete;
  InferValueCallback &operator=(const InferValueCallback &) = delete;

  static InferValueCallback &GetInstance();

  void RegImpl(const std::string &impl_type, const InferValueFunc &py_func);
  ValuePtr CallPyInferValue(const std::string &op_name, const AbstractBasePtrList &input_args);
  ValuePtr CallKernelInferValue(const std::string &op_name, const AbstractBasePtrList &input_args);

 private:
  InferValueCallback() = default;
  ~InferValueCallback() {}

 private:
  InferValueFunc python_impl_{nullptr};
  InferValueFunc kernel_impl_{nullptr};
};

class MS_CORE_API InferValueImplRegister {
 public:
  InferValueImplRegister(const std::string &impl_type, const InferValueFunc &fn);
  ~InferValueImplRegister() = default;
};

#define INFER_VALUE_IMPL_REGISTER(impl_type, func) \
  static auto reg_##impl_type##_##func = mindspore::ops::InferValueImplRegister(#impl_type, func)
}  //  namespace mindspore::ops
#endif  //  MINDSPORE_CORE_OPS_FRONTEND_FUNC_IMPL_H
