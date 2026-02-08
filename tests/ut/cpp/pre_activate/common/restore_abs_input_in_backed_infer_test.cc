/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include <memory>
#include <vector>
#include <string>
#include "ops/base_operator.h"
#include "ir/primitive.h"
#include "include/utils/utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/backend/common/pass_manager/helper.h"
#include "common/common_test.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace opt {
class TestAttr : public ops::BaseOperator {
 public:
  MIND_API_BASE_MEMBER(TestAttr);
  TestAttr() : BaseOperator("") {}
};
class TestDynamicInput : public ops::BaseOperator {
 public:
  MIND_API_BASE_MEMBER(TestDynamicInput);
  TestDynamicInput() : BaseOperator("") {}
};
constexpr auto kAttrConvertTestName = "attr_convert_test";
constexpr auto kDynamicInputTestName = "dynamic_input_test";
inline const PrimitivePtr kPrimAttrConvertTest = std::make_shared<Primitive>(kAttrConvertTestName);
inline const PrimitivePtr kPrimDynamicInputTest = std::make_shared<Primitive>("dynamic_input_test");
AbstractBasePtr InferImplAttrTest(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  // CppInferShapeAndType does not convert attr to input
  EXPECT_EQ(args_spec_list.size(), 2);
  EXPECT_NE(args_spec_list[1], nullptr);
  EXPECT_EQ(args_spec_list[1]->isa<abstract::AbstractTensor>(), true);
  return args_spec_list[0];
}
REGISTER_PRIMITIVE_EVAL_IMPL(TestAttr, kPrimAttrConvertTest, InferImplAttrTest, nullptr, true);
AbstractBasePtr InferImplDynamicInputTest(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list) {
  EXPECT_EQ(args_spec_list.size(), 3);
  EXPECT_NE(args_spec_list[1], nullptr);
  EXPECT_EQ(args_spec_list[1]->isa<abstract::AbstractTuple>(), true);
  auto item = args_spec_list[1]->cast<abstract::AbstractTuplePtr>();
  return args_spec_list[0];
}
REGISTER_PRIMITIVE_EVAL_IMPL(TestDynamicInput, kPrimDynamicInputTest, InferImplDynamicInputTest, nullptr, true);
}  // namespace opt
}  // namespace mindspore