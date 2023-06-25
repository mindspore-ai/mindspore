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

#include "gtest/gtest.h"
#include "include/c_api/context_c.h"
#include "include/c_api/model_c.h"
#include "src/common/file_utils.h"

namespace {
void CreateCpuContext(MSContextHandle *context) {
  ASSERT_NE(context, nullptr);

  *context = MSContextCreate();
  ASSERT_NE(*context, nullptr);

  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  ASSERT_NE(cpu_device_info, nullptr);
  MSContextAddDeviceInfo(*context, cpu_device_info);
}
}  // namespace

TEST(ModelCApiTest, BuildFromBufferTwice) {
  MSContextHandle context;
  const std::string model_file = "./ml_face_isface.ms";

  // 1. Create CPU context.
  CreateCpuContext(&context);

  // 2. Build the model once. Expect returning success.
  auto model = MSModelCreate();
  ASSERT_NE(model, nullptr);

  size_t model_size = 0;
  char *model_buffer = mindspore::lite::ReadFile(model_file.c_str(), &model_size);
  ASSERT_NE(model_buffer, nullptr);

  auto status = MSModelBuild(model, model_buffer, model_size, kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusSuccess);

  // 3. Build the model twice. Expect specific error code: kMSStatusLiteModelRebuild.
  status = MSModelBuildFromFile(model, model_file.c_str(), kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusLiteModelRebuild);

  MSModelDestroy(&model);
}

TEST(ModelCApiTest, BuildFromFileTwice) {
  MSContextHandle context;
  const std::string model_file = "./ml_face_isface.ms";

  // 1. Create CPU context.
  CreateCpuContext(&context);

  // 2. Build the model once. Expect returning success.
  auto model = MSModelCreate();
  ASSERT_NE(model, nullptr);

  auto status = MSModelBuildFromFile(model, model_file.c_str(), kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusSuccess);

  // 3. Build the model twice. Expect specific error code: kMSStatusLiteModelRebuild.
  status = MSModelBuildFromFile(model, model_file.c_str(), kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusLiteModelRebuild);

  MSModelDestroy(&model);
}

TEST(ModelCApiTest, BuildFromBufferAndFileTwice) {
  MSContextHandle context;
  const std::string model_file = "./ml_face_isface.ms";

  // 1. Create CPU context.
  CreateCpuContext(&context);

  // 2. Build the model once. Expect returning success.
  auto model = MSModelCreate();
  ASSERT_NE(model, nullptr);

  size_t model_size = 0;
  char *model_buffer = mindspore::lite::ReadFile(model_file.c_str(), &model_size);
  ASSERT_NE(model_buffer, nullptr);

  auto status = MSModelBuild(model, model_buffer, model_size, kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusSuccess);

  // 3. Build the model twice. Expect specific error code: kMSStatusLiteModelRebuild.
  status = MSModelBuildFromFile(model, model_file.c_str(), kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusLiteModelRebuild);

  MSModelDestroy(&model);
}
