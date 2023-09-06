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
#ifndef MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_H_
#define MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_H_

#include <string>
#include <vector>
#include "common/common_test.h"
#include "utils/ms_context.h"
#include "mindapi/base/shape_vector.h"
#include "base/base.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
namespace{
constexpr int64_t kUnknown = 0;
}
class TestOps : public UT::Common {
 public:
  TestOps() {}
  void SetUp() {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    origin_device_target_ = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  }
  void TearDown() {
    auto context_ptr = MsContext::GetInstance();
    if (context_ptr != nullptr) {
      context_ptr->set_param<std::string>(MS_CTX_DEVICE_TARGET, origin_device_target_);
    }
  }

 private:
  std::string origin_device_target_;
};

struct EltwiseOpParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector out_shape;
  TypePtr out_type;
  std::vector<ValuePtr> attr_list;
};

struct BroadcastOpParams {
  ShapeVector x_shape;
  TypePtr x_type;
  ShapeVector y_shape;
  TypePtr y_type;
  ShapeVector out_shape;
  TypePtr out_type;
};

struct MultiInputOpParams {
  std::vector<ShapeVector> in_shape_array;
  std::vector<TypePtr> in_type_list;
  std::vector<ShapeVector> out_shape_array;
  std::vector<TypePtr> out_type_list;
  std::vector<ValuePtr> attr_list;
};

struct EltwiseOpShapeParams {
  ShapeVector x_shape;
  ShapeVector out_shape;
};

struct EltwiseOpTypeParams {
  TypePtr x_type;
  TypePtr out_type;
};

struct EltwiseGradOpShapeParams {
  ShapeVector grad_shape;
  ShapeVector x_shape;
  ShapeVector out_shape;
};

struct EltwiseGradOpTypeParams {
  TypePtr grad_type;
  TypePtr x_type;
  TypePtr out_type;
};

struct BroadcastOpShapeParams {
  ShapeVector x_shape;
  ShapeVector y_shape;
  ShapeVector out_shape;
};

struct BroadcastOpTypeParams {
  TypePtr x_type;
  TypePtr y_type;
  TypePtr out_type;
};

}  // namespace mindspore::ops
#endif  // MINDSPORE_TESTS_UT_CPP_OPS_TEST_OPS_H_