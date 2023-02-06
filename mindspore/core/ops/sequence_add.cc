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

#include "ops/sequence_add.h"

#include <vector>
#include <string>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
// Take out the abstract of element.
// The elements of input should have same shape and type. Dynamic length sequence already satisfies this requirement.
// For constant length sequence, this requirement need to be checked in this function.
AbstractBasePtr CheckAndGetElementType(const abstract::AbstractSequencePtr input, const std::string &prim_name) {
  if (input->dynamic_len()) {
    return input->dynamic_len_element_abs();
  }
  const auto &elements = input->elements();
  if (elements.empty()) {
    return nullptr;
  }
  CheckAndConvertUtils::CheckAbstractTypeAndShapeSame(elements, "For primitive " + prim_name);
  return elements[0];
}

AbstractBasePtr SequenceAddInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_len = 2;
  constexpr size_t input_1_index = 0;
  constexpr size_t input_2_index = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);
  auto input_1 = dyn_cast<abstract::AbstractSequence>(input_args[input_1_index]);
  MS_EXCEPTION_IF_NULL(input_1);
  auto input_2 = dyn_cast<abstract::AbstractSequence>(input_args[input_2_index]);
  MS_EXCEPTION_IF_NULL(input_2);
  if ((input_1->isa<abstract::AbstractTuple>() && input_2->isa<abstract::AbstractList>()) ||
      (input_1->isa<abstract::AbstractList>() && input_2->isa<abstract::AbstractTuple>())) {
    MS_EXCEPTION(TypeError) << "Can not concatenate list and tuple together. For sequence append operator, "
                            << "the first input is: " << input_1->ToString()
                            << " and the second input is: " << input_2->ToString();
  }

  // All elements of sequence add should have same element type.
  auto abs_1 = CheckAndGetElementType(input_1, prim_name);
  auto abs_2 = CheckAndGetElementType(input_2, prim_name);

  // all of elements is known
  if (!input_1->dynamic_len() && !input_2->dynamic_len()) {
    abstract::AbstractBasePtrList abs;
    for (size_t i = 0; i < input_1->size(); i++) {
      abs.push_back(input_1->elements()[i]);
    }
    for (size_t i = 0; i < input_2->size(); i++) {
      abs.push_back(input_2->elements()[i]);
    }
    auto ret = std::make_shared<abstract::AbstractTuple>(abs);
    return ret;
  }

  // abs_1 is nullptr represents that the input_1 is empty.
  // input_1 can be either dynamic length sequence or constant length sequence.
  if (abs_1 == nullptr) {
    if (input_2->dynamic_len()) {
      return input_2->Clone();
    }
    // input_1 is dynamic length.
    auto ret = input_1->Clone()->cast<abstract::AbstractSequencePtr>();
    ret->set_dynamic_len_element_abs(abs_2);
    return ret;
  }
  // abs_2 is nullptr represents that the input_2 is empty.
  // input_2 can be either dynamic length sequence or constant length sequence.
  if (abs_2 == nullptr) {
    if (input_1->dynamic_len()) {
      return input_1->Clone();
    }
    // input_2 is dynamic length.
    auto ret = input_2->Clone()->cast<abstract::AbstractSequencePtr>();
    ret->set_dynamic_len_element_abs(abs_1);
    return ret;
  }
  CheckAndConvertUtils::CheckAbstractTypeAndShapeSame({abs_1, abs_2}, "For primitive " + prim_name,
                                                      "element of first input", "element of second input");
  if (input_1->dynamic_len()) {
    return input_1->Clone();
  }
  return input_2->Clone();
}
}  // namespace
MIND_API_OPERATOR_IMPL(SequenceAdd, BaseOperator);
class SequenceAddInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceAddInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceAddInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceAddInferInner(primitive, input_args);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceAdd, prim::kPrimSequenceAdd, SequenceAddInfer, true);
}  // namespace ops
}  // namespace mindspore
