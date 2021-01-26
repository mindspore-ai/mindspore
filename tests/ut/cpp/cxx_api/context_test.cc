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
#include <memory>
#include "common/common_test.h"
#include "include/api/context.h"

namespace mindspore {
class TestCxxApiContext : public UT::Common {
 public:
  TestCxxApiContext() = default;
};

TEST_F(TestCxxApiContext, test_context_global_context_SUCCESS) {
  std::string device_target = "2333";
  uint32_t device_id = 2333;
  GlobalContext::SetGlobalDeviceTarget(device_target);
  ASSERT_EQ(GlobalContext::GetGlobalDeviceTarget(), device_target);
  GlobalContext::SetGlobalDeviceID(device_id);
  ASSERT_EQ(GlobalContext::GetGlobalDeviceID(), device_id);
}

TEST_F(TestCxxApiContext, test_context_ascend310_context_SUCCESS) {
  std::string option_1 = "aaa";
  std::string option_2 = "vvv";
  std::string option_3 = "www";
  auto option_4 = DataType::kNumberTypeEnd;
  std::string option_5 = "rrr";
  std::string option_6 = "ppp";
  auto ctx = std::make_shared<ModelContext>();
  ModelContext::SetInsertOpConfigPath(ctx, option_1);
  ModelContext::SetInputFormat(ctx, option_2);
  ModelContext::SetInputShape(ctx, option_3);
  ModelContext::SetOutputType(ctx, option_4);
  ModelContext::SetPrecisionMode(ctx, option_5);
  ModelContext::SetOpSelectImplMode(ctx, option_6);

  ASSERT_EQ(ModelContext::GetInsertOpConfigPath(ctx), option_1);
  ASSERT_EQ(ModelContext::GetInputFormat(ctx), option_2);
  ASSERT_EQ(ModelContext::GetInputShape(ctx), option_3);
  ASSERT_EQ(ModelContext::GetOutputType(ctx), option_4);
  ASSERT_EQ(ModelContext::GetPrecisionMode(ctx), option_5);
  ASSERT_EQ(ModelContext::GetOpSelectImplMode(ctx), option_6);
}

TEST_F(TestCxxApiContext, test_context_ascend310_context_nullptr_FAILED) {
  auto ctx = std::make_shared<ModelContext>();
  EXPECT_ANY_THROW(ModelContext::GetInsertOpConfigPath(nullptr));
}

TEST_F(TestCxxApiContext, test_context_ascend310_context_wrong_type_SUCCESS) {
  auto ctx = std::make_shared<ModelContext>();
  ctx->params["mindspore.option.op_select_impl_mode"] = 5;
  ASSERT_EQ(ModelContext::GetOpSelectImplMode(ctx), "");
}

TEST_F(TestCxxApiContext, test_context_ascend310_context_default_value_SUCCESS) {
  auto ctx = std::make_shared<ModelContext>();
  ASSERT_EQ(ModelContext::GetOpSelectImplMode(ctx), "");
}
}  // namespace mindspore
