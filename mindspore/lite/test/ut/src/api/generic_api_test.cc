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
#include <memory>
#include "include/api/types.h"
#include "include/api/context.h"
#include "common/common_test.h"
#include "src/runtime/cxx_api/converters.h"
#include "src/common/context_util.h"

namespace mindspore {
class GenericApiTest : public mindspore::CommonTest {
 public:
  GenericApiTest() {}
};

class Tda4DeviceInfo : public mindspore::DeviceInfoContext {
 public:
  mindspore::DeviceType GetDeviceType() const override { return mindspore::DeviceType::kCustomDevice; };
};

TEST_F(GenericApiTest, TestConvertContextToInnerContext) {
  mindspore::Context *context = new (std::nothrow) mindspore::Context();
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  auto tda4_device_info = std::make_shared<mindspore::Tda4DeviceInfo>();
  device_list.push_back(device_info);
  device_list.push_back(tda4_device_info);

  lite::InnerContext *inner_ctx = ContextUtils::Convert(context);

  ASSERT_EQ(inner_ctx->device_list_.size(), device_list.size());
  ASSERT_EQ(inner_ctx->device_list_[0].device_type_, mindspore::lite::DT_CPU);
  ASSERT_EQ(inner_ctx->device_list_[1].device_type_, mindspore::lite::DT_CUSTOM);
  delete context;
  delete inner_ctx;
}

TEST_F(GenericApiTest, TestConvertInnerContextToContext) {
  mindspore::Context *context = new (std::nothrow) mindspore::Context();
  auto &device_list = context->MutableDeviceInfo();
  auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
  auto tda4_device_info = std::make_shared<mindspore::Tda4DeviceInfo>();
  device_list.push_back(device_info);
  device_list.push_back(tda4_device_info);

  lite::InnerContext *inner_ctx = ContextUtils::Convert(context);
  mindspore::Context *ctx = MSContextFromContext(inner_ctx);
  auto &new_device_list = ctx->MutableDeviceInfo();

  ASSERT_EQ(new_device_list.size(), device_list.size());
  ASSERT_EQ(new_device_list[0]->GetDeviceType(), mindspore::DeviceType::kCPU);
  ASSERT_EQ(new_device_list[1]->GetDeviceType(), mindspore::DeviceType::kCustomDevice);
  delete context;
  delete inner_ctx;
  delete ctx;
}
}  // namespace mindspore
