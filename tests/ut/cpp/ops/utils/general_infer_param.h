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
#ifndef MINDSPORE_TESTS_UT_CPP_OPS_UTILS_GENERAL_INFER_PARAM
#define MINDSPORE_TESTS_UT_CPP_OPS_UTILS_GENERAL_INFER_PARAM

#include <iostream>
#include <memory>
#include <string>
#include <variant>
#include <sstream>

#include "ir/anf.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "utils/anf_utils.h"
#include "utils/shape_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/utils.h"

namespace mindspore::ops {
constexpr int kSequenceParamVaIndex = 1;
using TypeIdList = std::vector<TypeId>;
struct InferInfoParam {
  std::variant<ShapeVector, ShapeArray> shape;  // ShapeArray for tuple/list input
  std::variant<TypeId, TypeIdList> type;
  std::variant<ValuePtr, ValuePtrList> value{kValueAny};
  bool is_dynamic_seq{false};  // only meaningful for tuple/list input
};

// Stream operator for InferInfoParam
inline std::ostream &operator<<(std::ostream &os, const InferInfoParam &param) {
  os << "InferInfoParam{shape: ";

  // Handle shape variant
  if (std::holds_alternative<ShapeVector>(param.shape)) {
    os << VectorToString<int64_t>(std::get<ShapeVector>(param.shape));
  } else {
    os << "[";
    for (const auto &shape : std::get<ShapeArray>(param.shape)) {
      os << VectorToString<int64_t>(shape) << ", ";
    }
    os << "]";
  }

  os << ", type: ";

  // Handle type variant
  if (std::holds_alternative<TypeId>(param.type)) {
    os << TypeIdToString(std::get<TypeId>(param.type));
  } else {
    os << "[";
    for (const auto &type : std::get<TypeIdList>(param.type)) {
      os << TypeIdToString(type) << ", ";
    }
    os << "]";
  }

  // Only output is_dynamic_seq if it's true
  if (param.is_dynamic_seq) {
    os << ", is_dynamic_seq: true";
  }

  os << "}";
  return os;
}

inline std::string ToString(const InferInfoParam &param) {
  std::ostringstream os;
  os << param;
  return os.str();
}

struct InferOutput {
  ShapeArray shapes;
  std::vector<TypeId> types;

  InferOutput(const ShapeArray &shapes, const std::vector<TypeId> &types) : shapes(shapes), types(types) {}
};

struct GeneralInferParam {
  std::vector<InferInfoParam> arg_params;
  InferOutput expected_output;
  bool expect_throw{false};

  GeneralInferParam(const std::vector<InferInfoParam> &arg_params, const InferOutput &expected_output,
                    bool expect_throw = false)
      : arg_params(arg_params), expected_output(expected_output), expect_throw{expect_throw} {}
};

// The Generator class
class GeneralInferParamGenerator {
 public:
  // Method to feed an input argument list
  GeneralInferParamGenerator &FeedInputArgs(const std::vector<InferInfoParam> &args) {
    input_args_.push_back(args);
    return *this;
  }

  // Method to feed an expected output
  GeneralInferParamGenerator &FeedExpectedOutput(const ShapeArray &shapes, const std::vector<TypeId> types) {
    expected_outputs_.emplace_back(shapes, types);
    return *this;
  }

  // Method to indicate case should throw
  GeneralInferParamGenerator &CaseShouldThrow() {
    expected_outputs_.emplace_back(ShapeArray{}, TypeIdList{});
    throw_case_indices.insert(expected_outputs_.size() - 1);
    return *this;
  }

  // Method to generate the output
  std::vector<GeneralInferParam> Generate() {
    // Ensure that input_args_ and expected_outputs_ sizes match
    if (input_args_.size() != expected_outputs_.size()) {
      throw std::runtime_error("Mismatched sizes between input arguments and expected outputs.");
    }

    // Prepare the result
    std::vector<GeneralInferParam> result;
    for (size_t i = 0; i < input_args_.size(); ++i) {
      result.emplace_back(input_args_[i], expected_outputs_[i], (throw_case_indices.count(i) == 1));
    }

    return result;
  }

 private:
  std::vector<std::vector<InferInfoParam>> input_args_;
  std::vector<InferOutput> expected_outputs_;
  std::set<size_t> throw_case_indices{};
};
}  // namespace mindspore::ops
#endif  //  MINDSPORE_TESTS_UT_CPP_OPS_UTILS_GENERAL_INFER_PARAM