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
#include "ops/op_utils.h"
#include "ir/dtype/tensor_type.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
namespace mindspore {
namespace ops {
namespace {
constexpr int ELL_VAL = 52;
constexpr int LABEL_NUM = 52;
constexpr int ELL_LEN = 3;
constexpr int BIG_C_BEGIN = 26;
static int64_t char_to_index(char cur_char) {
  if (cur_char <= 'z' && cur_char >= 'a') {
    return static_cast<int64_t>(cur_char - 'a');
  }
  return static_cast<int64_t>(cur_char - 'A' + BIG_C_BEGIN);
}

static void seg_left_equation(const std::string &left_equation, const std::vector<std::vector<int64_t>> &input_shapes,
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
        MS_EXCEPTION(ValueError) << "The \'.\' has been found again in " << cur_element
                                 << " operand, for which an operand object can contain only one ellipsis!";
      }
      if (idx + ELL_LEN - 1 >= left_equation.length() || left_equation[idx + 1] != label ||
          left_equation[idx + ELL_LEN - 1] != label) {
        MS_EXCEPTION(ValueError) << "An ellipsis can consist only three \'.\'";
      }
      idx += (ELL_LEN - 1);
      found_ell = true;
      (*left_elements)[cur_element].emplace_back(ELL_VAL);
    } else if (label == ',') {
      if ((found_ell && (*left_elements)[cur_element].size() > input_shapes[cur_element].size() + 1) ||
          (!found_ell && (*left_elements)[cur_element].size() != input_shapes[cur_element].size())) {
        MS_EXCEPTION(ValueError) << "The number of subscript in " << cur_element
                                 << " operand of eqaution does not match inputs[" << cur_element << "].dim()!";
      }
      ++cur_element;
      if (cur_element >= input_shapes.size()) {
        MS_EXCEPTION(ValueError) << "The number of inputs and equation's operand number does not match!";
      }
      found_ell = false;
    } else {
      MS_EXCEPTION(ValueError) << "Operand " << cur_element
                               << " in equation contains invalid subscript, which can only consist of [a-zA-z]!";
    }
  }
  if (cur_element != input_shapes.size() - 1) {
    MS_EXCEPTION(ValueError) << "The number of inputs and equation's operand number does not match!";
  }
}

static void seg_right_equation_with_arrow(const std::string &left_equation, const std::string &right_equation,
                                          std::unordered_map<int64_t, std::vector<int64_t>> *element_shape_map,
                                          std::vector<int64_t> *out_shape) {
  bool found_ell = false;
  if (right_equation.length() == 0) {
    out_shape->emplace_back(1);
    return;
  }
  std::vector<bool> exit_flag(LABEL_NUM, false);
  for (size_t idx = 0; idx < right_equation.length(); ++idx) {
    if (left_equation.find(right_equation[idx]) == std::string::npos) {
      MS_EXCEPTION(ValueError) << "The label to the right of the arrow must have appeared on the left!";
    }

    if (right_equation[idx] == '.') {
      if (found_ell) {
        MS_EXCEPTION(ValueError) << "The \'.\' has been found again in right of arrow!";
      }
      if ((idx + ELL_LEN - 1 >= right_equation.length()) ||
          (right_equation[idx + 1] != '.' || right_equation[idx + ELL_LEN - 1] != '.')) {
        MS_EXCEPTION(ValueError) << "An ellipsis can consist only three \'.\'";
      }
      idx += (ELL_LEN - 1);
      found_ell = true;
      out_shape->insert(out_shape->end(), (*element_shape_map)[ELL_VAL].begin(), (*element_shape_map)[ELL_VAL].end());
    } else if (isalpha(right_equation[idx])) {
      auto val = char_to_index(right_equation[idx]);
      if (exit_flag[val]) {
        MS_EXCEPTION(ValueError) << "output subsrcipt " << right_equation[idx] << " more than once in the output";
      }
      exit_flag[val] = true;
      out_shape->insert(out_shape->end(), (*element_shape_map)[val].begin(), (*element_shape_map)[val].end());
    } else {
      MS_EXCEPTION(ValueError) << "Invalid label in equation representing output (right of arrow)!";
    }
  }
}

static void seg_right_equation_without_arrow(const std::string &left_equation,
                                             std::unordered_map<int64_t, std::vector<int64_t>> *element_shape_map,
                                             const std::vector<int64_t> &element_count,
                                             std::vector<int64_t> *out_shape) {
  if (left_equation.find('.') != std::string::npos) {
    out_shape->insert(out_shape->begin(), (*element_shape_map)[ELL_VAL].begin(), (*element_shape_map)[ELL_VAL].end());
  }
  for (size_t idx = 0; idx < element_count.size(); ++idx) {
    if (element_count[idx] == 1) {
      out_shape->insert(out_shape->end(), (*element_shape_map)[idx].begin(), (*element_shape_map)[idx].end());
    }
  }
  if (out_shape->size() == 0) {
    out_shape->emplace_back(1);
  }
}

static void element_map_shape(const std::vector<std::vector<int64_t>> &left_elements,
                              const std::vector<std::vector<int64_t>> &input_shapes,
                              std::unordered_map<int64_t, std::vector<int64_t>> *element_shape_map) {
  for (size_t idx_input = 0; idx_input < input_shapes.size(); ++idx_input) {
    auto cur_shape = input_shapes[idx_input];
    size_t idx_left = 0;
    while (idx_left < left_elements[idx_input].size() && left_elements[idx_input][idx_left] != ELL_VAL) {
      auto cur_element = left_elements[idx_input][idx_left];
      if (element_shape_map->find(cur_element) != element_shape_map->end()) {
        if ((*element_shape_map)[cur_element][0] != input_shapes[idx_input][idx_left]) {
          MS_EXCEPTION(ValueError) << "The same label in equation can only represent the same dimension in inputs!";
        }
      } else {
        (*element_shape_map)[cur_element] = {input_shapes[idx_input][idx_left]};
      }
      ++idx_left;
    }

    if (idx_left != left_elements[idx_input].size()) {
      auto idx_element_right = left_elements[idx_input].size() - 1;
      auto idx_shape_right = input_shapes[idx_input].size() - 1;
      while (idx_element_right > idx_left && left_elements[idx_input][idx_element_right] != ELL_VAL) {
        auto cur_element = left_elements[idx_input][idx_element_right];
        if (element_shape_map->find(cur_element) != element_shape_map->end()) {
          if ((*element_shape_map)[cur_element][0] != input_shapes[idx_input][idx_shape_right]) {
            MS_EXCEPTION(ValueError) << "The same label in equation can only represent the same dimension in inputs!";
          }
        } else {
          (*element_shape_map)[cur_element] = {input_shapes[idx_input][idx_shape_right]};
        }
        --idx_shape_right;
        --idx_element_right;
      }
      std::vector<int64_t> temp_vec(input_shapes[idx_input].begin() + idx_left,
                                    input_shapes[idx_input].begin() + idx_shape_right + 1);
      if (element_shape_map->find(ELL_VAL) != element_shape_map->end()) {
        if ((*element_shape_map)[ELL_VAL] != temp_vec) {
          MS_EXCEPTION(ValueError) << "The same ellipsis in equation can only represent the same dimension in inputs!";
        }
      } else {
        (*element_shape_map)[ELL_VAL] = temp_vec;
      }
    }
  }
}
}  // namespace

void Einsum::Init(const std::string &equation) { this->set_equation(equation); }

void Einsum::set_equation(const std::string &equation) { (void)this->AddAttr(kEquation, MakeValue(equation)); }

std::string Einsum::get_equation() const {
  auto value_ptr = this->GetAttr(kEquation);
  return GetValue<std::string>(value_ptr);
}

abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto equation = GetValue<std::string>(primitive->GetAttr(kEquation));
  if (!input_args[0]->isa<abstract::AbstractTuple>() && !input_args[0]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(TypeError) << "The input of Einsum must be list or tuple of tensors.";
  }
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  std::vector<std::vector<int64_t>> input_shapes;
  for (size_t idx = 0; idx < elements.size(); ++idx) {
    auto shape = elements[idx]->BuildShape();
    auto &shape_vec = shape->cast<abstract::ShapePtr>()->shape();
    input_shapes.emplace_back(shape_vec);
  }
  equation.erase(std::remove(equation.begin(), equation.end(), ' '), equation.end());
  const std::string seg_arrow = "->";
  const auto seg_pos = equation.find(seg_arrow);
  const auto left_equation = equation.substr(0, seg_pos);
  std::vector<std::vector<int64_t>> left_elements(input_shapes.size());
  std::vector<int64_t> element_count(LABEL_NUM, 0);
  std::unordered_map<int64_t, std::vector<int64_t>> element_shape_map;
  std::vector<int64_t> out_shape;
  seg_left_equation(left_equation, input_shapes, &left_elements, &element_count);
  element_map_shape(left_elements, input_shapes, &element_shape_map);

  if (seg_pos == std::string::npos) {
    seg_right_equation_without_arrow(left_equation, &element_shape_map, element_count, &out_shape);
  } else {
    auto right_equation = equation.substr(seg_pos + 2, equation.length() - seg_pos - 2);
    seg_right_equation_with_arrow(left_equation, right_equation, &element_shape_map, &out_shape);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}
TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
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
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(Einsum, prim::kPrimEinsum, EinsumInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
