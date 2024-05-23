/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SIMPLE_INFER_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_SIMPLE_INFER_H_

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <optional>
#include "utils/simple_info.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ops/op_def.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore::ops {
class MIND_API SimpleInfer {
 public:
  DISABLE_COPY_AND_ASSIGN(SimpleInfer);
  static SimpleInfer &Instance() noexcept;
  ops::OpFuncImplPtr GetFunc(const string &op_name);
  void Register(const std::string &op_name, ops::OpFuncImplPtr &&func);

  void DoSimpleInfer(const PrimitivePtr &primitive, const ValueSimpleInfoPtr &value_simple_info,
                     const ops::OpFuncImplPtr &simple_infer_func, const ValuePtrList &input_values);

 private:
  SimpleInfer() = default;
  ~SimpleInfer() = default;

  std::map<std::string, ops::OpFuncImplPtr> simple_infer_fun_;
};

template <typename T>
ValuePtr ConvertValuePtr(const std::optional<T> &t) {
  if (!t.has_value()) {
    return mindspore::kNone;
  }
  return t.value();
}

template <typename T>
ValuePtr ConvertValuePtr(const T &t) {
  return t;
}

// Api for vector input values
ValueSimpleInfoPtr InferBySimple(const PrimitivePtr &primitive, const ValuePtrList &input_values);

template <typename... T>
ValueSimpleInfoPtr InferBySimple(const PrimitivePtr &primitive, const T &... t) {
  const auto &simple_infer_func = SimpleInfer::Instance().GetFunc(primitive->name());
  if (simple_infer_func == nullptr) {
    return nullptr;
  }
  auto value_simple_info = std::make_shared<ValueSimpleInfo>();
  ValuePtrList input_values;
  input_values.reserve(sizeof...(t));
  (input_values.emplace_back(ConvertValuePtr(t)), ...);
  SimpleInfer::Instance().DoSimpleInfer(primitive, value_simple_info, simple_infer_func, input_values);
  return value_simple_info;
}

class SimpleInferRegHelper {
 public:
  SimpleInferRegHelper(const std::string &op_name, ops::OpFuncImplPtr op_func) {
    SimpleInfer::Instance().Register(op_name, std::move(op_func));
  }
  ~SimpleInferRegHelper() = default;
};

#define REGISTER_SIMPLE_INFER(op_name, OpFuncImpl) \
  static const auto op_simple_infer_##op_name =    \
    mindspore::ops::SimpleInferRegHelper(op_name, std::make_shared<OpFuncImpl>());
}  // namespace mindspore::ops
#endif
