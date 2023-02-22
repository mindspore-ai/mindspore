/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/einsum.h"

#include <unordered_map>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int kEinsumEllVal = 52;
constexpr int kEinsumLableNum = 52;
constexpr int kEinsumEllLen = 3;
static int64_t char_to_index(char cur_char) {
  if (cur_char <= 'z' && cur_char >= 'a') {
    return static_cast<int64_t>(cur_char - 'a');
  }
  constexpr int kBigCBegin = 26;
  return static_cast<int64_t>(cur_char - 'A' + kBigCBegin);
}

static void seg_left_equation(const std::string &left_equation, const std::string &prim_name,
                              const std::vector<std::vector<int64_t>> &input_shapes,
                              std::vector<std::vector<int64_t>> *left_elements, std::vector<int64_t> *element_count) {
  size_t cur_element = 0;
  auto found_ell = false;
  for (size_t idx = 0; idx < left_equation.length(); ++idx) {
    auto label = left_equation[idx];
    if (isalpha(label)) {
      (*left_elements)[cur_element].emplace_back(char_to_index(label));
      (*element_count)[char_to_index(label)] += 1;
    } else if (label == '.') {
      if (found_ell) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', each operand can contain only one ellipsis, but it has been found again.";
      }
      if (idx + kEinsumEllLen - 1 >= left_equation.length() || left_equation[idx + 1] != label ||
          left_equation[idx + kEinsumEllLen - 1] != label) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', an ellipsis in the equation must include three \'.\', but got less than 3.";
      }
      idx += (kEinsumEllLen - 1);
      found_ell = true;
      (void)(*left_elements)[cur_element].emplace_back(kEinsumEllVal);
    } else if (label == ',') {
      if ((found_ell && (*left_elements)[cur_element].size() > input_shapes[cur_element].size() + 1) ||
          (!found_ell && (*left_elements)[cur_element].size() != input_shapes[cur_element].size())) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the number of subscript in " << cur_element
                                 << " operand in the eqaution does not match inputs[" << cur_element << "].dim().";
      }
      ++cur_element;
      if (cur_element >= input_shapes.size()) {
        MS_EXCEPTION(ValueError)
          << "For '" << prim_name
          << "', the number of inputs must be equal to the number of inputs and equation's operand.";
      }
      found_ell = false;
    } else {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', operand " << cur_element
                               << " in the equation can only contain [a-zA-z], but got: " << cur_element << ".";
    }
  }
  if (cur_element != input_shapes.size() - 1) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the number of inputs must be equal to the number of equation's operand.";
  }
  for (size_t i = 0; i < (*left_elements).size(); ++i) {
    auto it = std::find((*left_elements)[i].begin(), (*left_elements)[i].end(), kEinsumEllVal);
    if ((*left_elements)[i].size() != input_shapes[i].size() && it == (*left_elements)[i].end()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the number of subscript in " << i
                               << " operand in the eqaution does not match inputs[" << i << "].dim().";
    }
  }
}

static void seg_right_equation_with_arrow(const std::string &left_equation, const std::string &right_equation,
                                          const std::string &prim_name,
                                          std::unordered_map<int64_t, std::vector<int64_t>> *element_shape_map,
                                          std::vector<int64_t> *out_shape) {
  bool found_ell = false;
  if (right_equation.length() == 0) {
    (void)out_shape->emplace_back(1);
    return;
  }
  std::vector<bool> exit_flag(kEinsumLableNum, false);
  for (size_t idx = 0; idx < right_equation.length(); ++idx) {
    if (left_equation.find(right_equation[idx]) == std::string::npos) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', the label to the right of arrow in the equation must appear on the left, but the "
                               << right_equation[idx] << " does not.";
    }

    if (right_equation[idx] == '.') {
      if (found_ell) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', each operand can contain only one ellipsis, but it has been found again.";
      }
      if ((idx + kEinsumEllLen - 1 >= right_equation.length()) ||
          (right_equation[idx + 1] != '.' || right_equation[idx + kEinsumEllLen - 1] != '.')) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', an ellipsis in the equation must include three \'.\', but got less than 3.";
      }
      idx += (kEinsumEllLen - 1);
      found_ell = true;
      (void)out_shape->insert(out_shape->end(), (*element_shape_map)[kEinsumEllVal].begin(),
                              (*element_shape_map)[kEinsumEllVal].end());
    } else if (isalpha(right_equation[idx])) {
      auto val = char_to_index(right_equation[idx]);
      if (exit_flag[val]) {
        MS_EXCEPTION(ValueError)
          << "For '" << prim_name
          << "', each character in the right of arrow in equation can only exist only once, but got"
          << right_equation[idx] << " at least twice.";
      }
      exit_flag[val] = true;
      (void)out_shape->insert(out_shape->end(), (*element_shape_map)[val].begin(), (*element_shape_map)[val].end());
    } else {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', operand " << right_equation
                               << " in the equation can only be consist of [a-zA-z], but found invalid character(s).";
    }
  }
}

static void seg_right_equation_without_arrow(const std::string &left_equation,
                                             std::unordered_map<int64_t, std::vector<int64_t>> *element_shape_map,
                                             const std::vector<int64_t> &element_count,
                                             std::vector<int64_t> *out_shape) {
  if (left_equation.find('.') != std::string::npos) {
    (void)out_shape->insert(out_shape->begin(), (*element_shape_map)[kEinsumEllVal].begin(),
                            (*element_shape_map)[kEinsumEllVal].end());
  }
  for (size_t idx = 0; idx < element_count.size(); ++idx) {
    if (element_count[idx] == 1) {
      (void)out_shape->insert(out_shape->end(), (*element_shape_map)[idx].begin(), (*element_shape_map)[idx].end());
    }
  }
  if (out_shape->size() == 0) {
    (void)out_shape->emplace_back(1);
  }
}

static void element_map_shape(const std::string &prim_name, const std::vector<std::vector<int64_t>> &left_elements,
                              const std::vector<std::vector<int64_t>> &input_shapes,
                              std::unordered_map<int64_t, std::vector<int64_t>> *element_shape_map) {
  for (size_t idx_input = 0; idx_input < input_shapes.size(); ++idx_input) {
    size_t idx_left = 0;
    while (idx_left < left_elements[idx_input].size() && left_elements[idx_input][idx_left] != kEinsumEllVal) {
      auto cur_element = left_elements[idx_input][idx_left];
      if (element_shape_map->find(cur_element) != element_shape_map->end()) {
        if ((*element_shape_map)[cur_element][0] != input_shapes[idx_input][idx_left]) {
          MS_EXCEPTION(ValueError)
            << "For '" << prim_name
            << "', the same label in equation can only represent the same dim in inputs, but got "
            << static_cast<char>(cur_element + 'a') << " in equation represented different dims.";
        }
      } else {
        (*element_shape_map)[cur_element] = {input_shapes[idx_input][idx_left]};
      }
      ++idx_left;
    }

    if (idx_left != left_elements[idx_input].size()) {
      auto idx_element_right = left_elements[idx_input].size() - 1;
      auto idx_shape_right = input_shapes[idx_input].size() - 1;
      while (idx_element_right > idx_left && left_elements[idx_input][idx_element_right] != kEinsumEllVal) {
        auto cur_element = left_elements[idx_input][idx_element_right];
        if (element_shape_map->find(cur_element) != element_shape_map->end()) {
          if ((*element_shape_map)[cur_element][0] != input_shapes[idx_input][idx_shape_right]) {
            MS_EXCEPTION(ValueError)
              << "For '" << prim_name
              << "', the same label in equation can only represent the same dimension in inputs, but got "
              << static_cast<char>(cur_element + 'a') << " in equation represented different dims.";
          }
        } else {
          (*element_shape_map)[cur_element] = {input_shapes[idx_input][idx_shape_right]};
        }
        --idx_shape_right;
        --idx_element_right;
      }
      std::vector<int64_t> temp_vec(input_shapes[idx_input].begin() + idx_left,
                                    input_shapes[idx_input].begin() + idx_shape_right + 1);
      if (element_shape_map->find(kEinsumEllVal) != element_shape_map->end()) {
        if ((*element_shape_map)[kEinsumEllVal] != temp_vec) {
          MS_EXCEPTION(ValueError)
            << "For '" << prim_name
            << "', the same ellipsis in equation can only represent the same dimension in inputs.";
        }
      } else {
        (*element_shape_map)[kEinsumEllVal] = temp_vec;
      }
    }
  }
}
}  // namespace

MIND_API_OPERATOR_IMPL(Einsum, BaseOperator);
void Einsum::Init(const std::string &equation) { this->set_equation(equation); }

void Einsum::set_equation(const std::string &equation) { (void)this->AddAttr(kEquation, api::MakeValue(equation)); }

std::string Einsum::get_equation() const {
  auto value_ptr = this->GetAttr(kEquation);
  return GetValue<std::string>(value_ptr);
}

abstract::ShapePtr EinsumInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto equation = GetValue<std::string>(primitive->GetAttr(kEquation));
  (void)equation.erase(std::remove(equation.begin(), equation.end(), ' '), equation.end());
  if (equation.length() == 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the equation is required, but got none.";
  }
  const std::string seg_arrow = "->";
  const auto seg_pos = equation.find(seg_arrow);
  if (seg_pos == 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the equation must contain characters to the left of the arrow, but got none.";
  }

  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  if (!input_args[0]->isa<abstract::AbstractTuple>() && !input_args[0]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input must be a list or tuple of tensors.";
  }
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  std::vector<std::vector<int64_t>> input_shapes;
  for (size_t idx = 0; idx < elements.size(); ++idx) {
    auto shape = elements[idx]->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->IsDimZero()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the dim of inputs' shape can not be zero, but got input["
                               << idx << "] shape: " << shape->ToString() << ".";
    }
    auto &shape_vec = shape->cast<abstract::ShapePtr>()->shape();
    for (auto &val : shape_vec) {
      if (val == 0) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the shape can not contain zero, but got input[" << idx
                                 << "] shape: " << shape->ToString() << ".";
      }
    }
    (void)input_shapes.emplace_back(shape_vec);
  }

  const auto left_equation = equation.substr(0, seg_pos);
  std::vector<std::vector<int64_t>> left_elements(input_shapes.size());
  std::vector<int64_t> element_count(kEinsumLableNum, 0);
  std::unordered_map<int64_t, std::vector<int64_t>> element_shape_map;
  std::vector<int64_t> out_shape;
  seg_left_equation(left_equation, prim_name, input_shapes, &left_elements, &element_count);
  element_map_shape(prim_name, left_elements, input_shapes, &element_shape_map);

  if (seg_pos == std::string::npos) {
    seg_right_equation_without_arrow(left_equation, &element_shape_map, element_count, &out_shape);
  } else {
    auto right_equation = equation.substr(seg_pos + 2, equation.length() - seg_pos - 2);
    seg_right_equation_with_arrow(left_equation, right_equation, prim_name, &element_shape_map, &out_shape);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}
TypePtr EinsumInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("out_type", elements[0]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
}
AbstractBasePtr EinsumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto res = std::make_shared<abstract::AbstractTensor>(EinsumInferType(primitive, input_args),
                                                        EinsumInferShape(primitive, input_args));
  return res;
}

// AG means auto generated
class MIND_API AGEinsumInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return EinsumInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return EinsumInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return EinsumInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Einsum, prim::kPrimEinsum, AGEinsumInfer, false);
}  // namespace ops
}  // namespace mindspore
