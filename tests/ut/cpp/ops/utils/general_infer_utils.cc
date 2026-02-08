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

#include "ops/utils/general_infer_utils.h"

#include <iostream>
#include <memory>
#include <string>

#include "common/common_test.h"
#include "ops/utils/general_infer_param.h"
#include "ops/test_value_utils.h"
#include "operator/meta_dsl/meta_dsl_utils.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "ir/tensor_new.h"
#include "ops/op_def.h"
#include "utils/anf_utils.h"
#include "utils/shape_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/utils.h"
#include "ops_utils/op_utils.h"
#include "mindspore/core/ops/infer_info/value_infer_info_adapter.h"
#include "ops/infer_info/abstract_infer_info_adapter.h"
#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"

namespace UT {
void InitPythonPath();
}
namespace mindspore::ops {
static bool is_sequence_input(const InferInfoParam &param) { return param.shape.index() == kSequenceParamVaIndex; }

static AbstractBasePtr MakeAbstract(const ShapeVector &shape, const TypeId type, const ValuePtr &value) {
  if (value == mindspore::kNone) {
    return std::make_shared<abstract::AbstractNone>();
  }
  if (shape.empty() && value != nullptr && value != kValueAny) {
    return std::make_shared<abstract::AbstractScalar>(value);
  }
  auto abs = abstract::MakeAbstract(shape, type);
  if (value) {
    abs->set_value(value);
  }
  return abs;
}

AbstractBasePtr param_to_abstract(InferInfoParam param) {
  AbstractBasePtr abs;
  if (is_sequence_input(param)) {
    AbstractBasePtrList abs_list;
    const auto &shapes = std::get<ShapeArray>(param.shape);
    const auto &types = std::get<std::vector<TypeId>>(param.type);
    const auto &values = std::get<ValuePtrList>(param.value);
    for (size_t i = 0; i < shapes.size(); ++i) {
      auto abs_ = MakeAbstract(shapes[i], types[i], values[i]);
      abs_list.push_back(abs_);
    }
    abstract::AbstractSequencePtr sequence_abs = std::make_shared<abstract::AbstractList>(abs_list);
    auto sequence_value = std::make_shared<ValueSequence>(values);
    sequence_abs->set_value(sequence_value);
    if (param.is_dynamic_seq) {
      sequence_abs->CheckAndConvertToDynamicLenSequence();
    }
    abs = sequence_abs;
  } else {
    abs =
      MakeAbstract(std::get<ShapeVector>(param.shape), std::get<TypeId>(param.type), std::get<ValuePtr>(param.value));
  }
  if (!abs) {
    throw std::runtime_error("Failed to create abstract from param: " + ToString(param));
  }
  return abs;
}

InferInfoPtr param_to_abstract_info(InferInfoParam param, const std::string &op_type, const std::string &arg_name) {
  const auto &abs = param_to_abstract(param);
  return std::make_unique<AbstractInferInfoAdapter>(abs, op_type, arg_name);
}

static ValuePtr MakeValue(const ShapeVector &shape, TypeId type, ValuePtr value, const std::string &arg_name) {
  if (value) {
    return value;
  }
  if (shape.empty()) {
    throw std::runtime_error("Value should be provided for scalar input '" + arg_name + "'");
  }
  return tensor::from_spec(type, shape, device::DeviceType::kCPU);
}

InferInfoPtr param_to_value_info(InferInfoParam param, const std::string &op_type, const std::string &arg_name) {
  ValuePtr value;
  if (is_sequence_input(param)) {
    const auto &shapes = std::get<ShapeArray>(param.shape);
    const auto &types = std::get<std::vector<TypeId>>(param.type);
    const auto &values = std::get<ValuePtrList>(param.value);
    ValuePtrList values_;
    for (size_t i = 0; i < shapes.size(); ++i) {
      values_.push_back(MakeValue(shapes[i], types[i], values[i], arg_name));
    }
    value = std::make_shared<ValueSequence>(values_);
  } else {
    const auto &shape = std::get<ShapeVector>(param.shape);
    const auto &type = std::get<TypeId>(param.type);
    const auto &value_ = std::get<ValuePtr>(param.value);
    value = MakeValue(shape, type, value_, arg_name);
  }
  return std::make_unique<ValueInferInfoAdapter>(value, op_type, arg_name);
}

/*
    Turn mixture of list/non-list args to all lists
*/
void align_param(InferInfoParam &param) {
  bool is_sequence_shape = param.shape.index() == kSequenceParamVaIndex;
  bool is_sequence_type = param.type.index() == kSequenceParamVaIndex;
  bool is_sequence_value = param.value.index() == kSequenceParamVaIndex;
  if (is_sequence_shape || is_sequence_type || is_sequence_value) {
    // turn non-list to list of single element
    if (!is_sequence_shape) {
      param.shape = ShapeArray{std::get<ShapeVector>(param.shape)};
    }
    if (!is_sequence_type) {
      param.type = TypeIdList{std::get<TypeId>(param.type)};
    }
    if (!is_sequence_value) {
      param.value = ValuePtrList{std::get<ValuePtr>(param.value)};
    }

    // turn list of single element to list of size n same elements
    auto &shapes = std::get<ShapeArray>(param.shape);
    auto &types = std::get<std::vector<TypeId>>(param.type);
    auto &values = std::get<ValuePtrList>(param.value);
    size_t n = std::max(shapes.size(), std::max(types.size(), values.size()));
    if (shapes.size() == 1) {
      shapes = std::move(ShapeArray(n, shapes[0]));
    }
    if (types.size() == 1) {
      types = std::move(TypeIdList(n, types[0]));
    }
    if (values.size() == 1) {
      values = std::move(ValuePtrList(n, values[0]));
    }
  }
}

void process_params(std::vector<InferInfoParam> &arg_params, const OpDefPtr op_def) {
  if (op_def->args_.size() != arg_params.size()) {
    throw std::runtime_error("Param size" + std::to_string(arg_params.size()) + "doesn't match input size " +
                             std::to_string(op_def->args_.size()));
  }
  for (auto &param : arg_params) {
    align_param(param);
    if (param.shape.index() == kSequenceParamVaIndex) {
      const auto &shapes = std::get<ShapeArray>(param.shape);
      const auto &types = std::get<std::vector<TypeId>>(param.type);
      const auto &values = std::get<ValuePtrList>(param.value);
      if (shapes.size() != types.size() || shapes.size() != values.size()) {
        throw std::runtime_error("There should be the same number of shape, type and values for a sequence input.");
      }
      if (param.is_dynamic_seq) {
        bool shape_all_same = std::all_of(shapes.begin(), shapes.end(),
                                          [&shapes](const ShapeVector &shape) { return shape == shapes.front(); });
        bool type_all_same =
          std::all_of(types.begin(), types.end(), [&types](const TypeId &type) { return type == types.front(); });
        if (!shape_all_same || !type_all_same) {
          throw std::runtime_error("Shape and Type should be the same for dynamic sequence.");
        }
      }
    }
  }
}

void params_to_infos(const std::vector<InferInfoParam> &arg_params, const OpDefPtr op_def,
                     InferInfoPtrList &value_infos, InferInfoPtrList &abstract_infos, bool is_dynamic) {
  const auto &op_type = op_def->name_;
  for (size_t i = 0; i < arg_params.size(); ++i) {
    if (!is_dynamic) {
      try {
        value_infos.push_back(param_to_value_info(arg_params[i], op_type, op_def->args_[i].arg_name_));
      } catch (const std::exception &e) {
        throw std::runtime_error("Failed to convert param [" + std::to_string(i) + "] to value_info: " + e.what());
      }
    }
    try {
      abstract_infos.push_back(param_to_abstract_info(arg_params[i], op_type, op_def->args_[i].arg_name_));
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to convert param [" + std::to_string(i) + "] to abstract_info: " + e.what());
    }
  }
}

static std::vector<std::string> ToTypeName(const std::vector<TypeId> &types) {
  std::vector<std::string> names;
  std::transform(types.begin(), types.end(), std::back_inserter(names),
                 [](TypeId type) { return TypeIdToString(type); });
  return names;
}

static bool is_dynamic_case(const std::vector<InferInfoParam> &arg_params) {
  // Dynamic if contains dynamic shape or kValueAny
  for (const auto &arg_param : arg_params) {
    if (arg_param.shape.index() == kSequenceParamVaIndex) {
      if (arg_param.is_dynamic_seq) {
        return true;
      }
      const auto &shapes = std::get<ShapeArray>(arg_param.shape);
      const auto &values = std::get<ValuePtrList>(arg_param.value);
      for (size_t i = 0; i < shapes.size(); ++i) {
        if (mindspore::IsDynamic(shapes[i]) || values[i] == kValueAny) {
          return true;
        }
      }
    } else {
      if (mindspore::IsDynamic(std::get<ShapeVector>(arg_param.shape)) ||
          std::get<ValuePtr>(arg_param.value) == kValueAny) {
        return true;
      }
    }
  }
  return false;
}

std::vector<AbstractBasePtr> params_to_abstracts(const std::vector<InferInfoParam> &arg_params) {
  std::vector<AbstractBasePtr> abstracts;
  for (const auto &arg_param : arg_params) {
    abstracts.push_back(param_to_abstract(arg_param));
  }
  return abstracts;
}

void get_shape_and_type_from_abs(const AbstractBasePtr &out_abs, ShapeArray &infer_shapes,
                                 std::vector<TypeId> &infer_types, const std::string &op_type) {
  std::vector<AbstractBasePtr> abs_list{};
  if (out_abs->isa<abstract::AbstractTuple>()) {
    abs_list = out_abs->cast<abstract::AbstractTuplePtr>()->elements();
  } else {
    abs_list.push_back(out_abs);
  }
  std::vector<AbstractInferInfoAdapterPtr> infer_infos{};
  const auto &name = "MetaOpOutput - " + op_type;
  for (size_t i = 0; i < abs_list.size(); ++i) {
    infer_infos.push_back(std::make_shared<AbstractInferInfoAdapter>(abs_list[i], name, "output-" + std::to_string(i)));
  }
  for (const auto &infer_info : infer_infos) {
    infer_shapes.push_back(infer_info->GetShape());
    infer_types.push_back(infer_info->GetType());
  }
}

TEST_P(GeneralInferTest, test_infer) {
  const ::testing::TestInfo *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string &test_suite_name = test_info->test_suite_name();  // Foo/GeneralInferTest
  const auto &op_type = test_suite_name.substr(0, test_suite_name.find('/'));
  const auto &param = GetParam();
  auto arg_params = param.arg_params;
  const auto &expect_shapes = param.expected_output.shapes;
  const auto &expect_types = ToTypeName(param.expected_output.types);
  const bool expect_throw = param.expect_throw;

  const auto prim = std::make_shared<Primitive>(op_type);
  const auto op_def = GetOpDef(op_type);
  ASSERT_NE(op_def, nullptr) << "OpDef not defined for " << op_type;
  ASSERT_NO_THROW(process_params(arg_params, op_def)) << "Param check failed";

  // MetaOp infer test
  bool is_meta_impl = prim::RegMetaImplFactory::GetInstance().IsMetaImpl(op_type);
  if (is_meta_impl) {
    UT::InitPythonPath();  // required by RunMetaImpl();
    const auto &abstracts = params_to_abstracts(arg_params);
    try {
      const auto &out_abs = prim::RunMetaImpl(abstracts, prim);
      if (expect_throw) {
        FAIL() << "Expected exception, but none was thrown";
      }
      ASSERT_NE(out_abs, nullptr) << "MetaImpl returns nullptr as output for" << op_type;
      ShapeArray infer_shapes;
      std::vector<TypeId> infer_types;
      get_shape_and_type_from_abs(out_abs, infer_shapes, infer_types, op_type);
      EXPECT_EQ(expect_shapes, infer_shapes) << "Inferred wrong shape for MetaOp infer.";
      EXPECT_EQ(expect_types, ToTypeName(infer_types)) << "Inferred wrong type for MetaOp infer.";
    } catch (const std::exception &e) {
      ASSERT_TRUE(expect_throw) << "Unexpected exception: " << e.what();
    }
    SUCCEED() << "MetaOp infer test case end.";
    return;
  }

  const auto &op_func = op_def->func_impl_;
  ASSERT_TRUE(op_func.GeneralInferRegistered()) << "GeneralInfer not registered for " << op_type;

  InferInfoPtrList value_infos;
  InferInfoPtrList abstract_infos;
  bool is_dynamic = is_dynamic_case(arg_params);
  ASSERT_NO_THROW(params_to_infos(arg_params, op_def, value_infos, abstract_infos, is_dynamic))
    << "Convert param to InferInfo failed.";

  try {
    (void)op_func.CheckValidation(prim, abstract_infos);
    const auto &abstract_infer_shapes = op_func.InferShape(prim, abstract_infos);
    const auto &abstract_infer_types = ToTypeName(op_func.InferType(prim, abstract_infos));
    if (expect_throw) {
      FAIL() << "Expected exception, but none was thrown";
    }
    EXPECT_EQ(expect_shapes, abstract_infer_shapes) << "Inferred wrong shape for abstract infer.";
    EXPECT_EQ(expect_types, abstract_infer_types) << "Inferred wrong type for abstract infer.";
  } catch (const std::exception &e) {
    ASSERT_TRUE(expect_throw) << "Unexpected exception: " << e.what();
  }

  if (!is_dynamic) {
    try {
      (void)op_func.CheckValidation(prim, value_infos);
      const auto &value_infer_shapes = op_func.InferShape(prim, value_infos);
      const auto &value_infer_types = ToTypeName(op_func.InferType(prim, value_infos));
      if (expect_throw) {
        FAIL() << "Expected exception, but none was thrown";
      }
      EXPECT_EQ(expect_shapes, value_infer_shapes) << "Inferred wrong shape for value infer.";
      EXPECT_EQ(expect_types, value_infer_types) << "Inferred wrong type for value infer.";
    } catch (const std::exception &e) {
      ASSERT_TRUE(expect_throw) << "Unexpected exception: " << e.what();
    }
  }
}
}  // namespace mindspore::ops
