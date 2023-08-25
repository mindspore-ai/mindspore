/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/print.h"

#include <memory>

#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/framework_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
constexpr auto kStringValue = "string_value";
constexpr auto kStringPos = "string_pos";
constexpr auto kValueType = "value_type";
constexpr auto kValueTypePos = "value_type_pos";

void Print::set_string_value(const std::vector<std::string> &string_value) {
  (void)this->AddAttr(kStringValue, api::MakeValue(string_value));
}

void Print::set_string_pos(const std::vector<int64_t> &string_pos) {
  (void)this->AddAttr(kStringPos, api::MakeValue(string_pos));
}

void Print::set_value_type(const std::vector<int64_t> &value_type) {
  (void)this->AddAttr(kValueType, api::MakeValue(value_type));
}

void Print::set_value_type_pos(const std::vector<int64_t> &value_type_pos) {
  (void)this->AddAttr(kValueTypePos, api::MakeValue(value_type_pos));
}

std::vector<std::string> Print::get_string_value() const {
  auto value_ptr = this->GetAttr(kStringValue);
  return GetValue<std::vector<std::string>>(value_ptr);
}

std::vector<int64_t> Print::get_string_pos() const {
  auto value_ptr = this->GetAttr(kStringPos);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Print::get_value_type() const {
  auto value_ptr = this->GetAttr(kValueType);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Print::get_value_type_pos() const {
  auto value_ptr = this->GetAttr(kValueTypePos);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::string PrintValueToString(const ValuePtr &value) {
  if (value == nullptr) {
    return "UnknownValue";
  }
  if (value == kValueAny) {
    return "UnknownValue";
  }
  if (value->isa<StringImm>()) {
    return value->ToString();
  }
  if (value->isa<Scalar>()) {
    std::ostringstream buffer;
    buffer << "Scalar(" << value->ToString() << ")";
    return buffer.str();
  }
  return value->ToString();
}

std::string PrintAbstractToString(const AbstractBasePtr &abstract) {
  if (abstract->isa<abstract::AbstractScalar>()) {
    auto value = abstract->BuildValue();
    return PrintValueToString(value);
  }
  if (abstract->isa<abstract::AbstractTensor>()) {
    std::ostringstream buffer;
    auto abs_tensor = abstract->cast<abstract::AbstractTensorPtr>();
    buffer << "Tensor(shape:" << abs_tensor->GetShapeTrack()->ToString()
           << ", dtype:" << abs_tensor->GetTypeTrack()->ToString()
           << ", value:" << PrintValueToString(abs_tensor->BuildValue()) << ")";
    return buffer.str();
  }
  if (abstract->isa<abstract::AbstractSequence>()) {
    auto abs_list = abstract->cast<abstract::AbstractSequencePtr>();
    std::ostringstream buffer;
    buffer << (abstract->isa<abstract::AbstractList>() ? "List[" : "Tuple(");
    if (abs_list->dynamic_len()) {
      buffer << PrintAbstractToString(abs_list->dynamic_len_element_abs());
      buffer << "......";
    } else {
      for (const auto &element : abs_list->elements()) {
        buffer << PrintAbstractToString(element) << ", ";
      }
    }
    buffer << (abstract->isa<abstract::AbstractList>() ? "]" : ")");
    return buffer.str();
  }
  return abstract->ToString();
}

MIND_API_OPERATOR_IMPL(Print, BaseOperator);

class PrintInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    ShapeVector shape = {1};
    return std::make_shared<abstract::Shape>(shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    std::ostringstream buffer;
    if (common::GetEnv("MS_DEV_COMPILE_PRINT") == "1") {
      for (const auto &input_arg : input_args) {
        buffer << PrintAbstractToString(input_arg);
      }
      std::cout << buffer.str() << std::endl;
    }
    return std::make_shared<TensorType>(kInt32);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Print, prim::kPrimPrint, PrintInfer, false);
}  // namespace ops
}  // namespace mindspore
