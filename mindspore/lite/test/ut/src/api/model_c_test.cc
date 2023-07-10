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

std::vector<MSShapeInfo> TransformTensorShapes(const std::vector<std::vector<int32_t>> &dims) {
  std::vector<MSShapeInfo> shape_infos;
  std::transform(dims.begin(), dims.end(), std::back_inserter(shape_infos), [&](auto &shapes) {
    MSShapeInfo shape_info;
    shape_info.shape_num = shapes.size();
    for (size_t i = 0; i < shape_info.shape_num; i++) {
      shape_info.shape[i] = shapes[i];
    }
    return shape_info;
  });
  return shape_infos;
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
  status = MSModelBuild(model, model_buffer, model_size, kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusLiteModelRebuild);
  delete model_buffer;

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
  delete model_buffer;

  // 3. Build the model twice. Expect specific error code: kMSStatusLiteModelRebuild.
  status = MSModelBuildFromFile(model, model_file.c_str(), kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusLiteModelRebuild);

  MSModelDestroy(&model);
}

TEST(ModelCApiTest, BuildFromFileAndBufferTwice) {
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
  size_t model_size = 0;
  char *model_buffer = mindspore::lite::ReadFile(model_file.c_str(), &model_size);
  ASSERT_NE(model_buffer, nullptr);

  status = MSModelBuild(model, model_buffer, model_size, kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusLiteModelRebuild);
  delete model_buffer;

  MSModelDestroy(&model);
}

TEST(ModelCApiTest, Resize) {
  MSContextHandle context;
  const std::string model_file = "./ml_face_isface.ms";

  // 1. Create CPU context and build the model.
  CreateCpuContext(&context);

  auto model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  auto status = MSModelBuildFromFile(model, model_file.c_str(), kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusSuccess);

  // 2. Invoke Resize() with GOOD input shapes. Expect specific ret code: kMSStatusSuccess.
  auto inputs = MSModelGetInputs(model);
  std::vector<MSShapeInfo> shape_infos = TransformTensorShapes({{3, 48, 48, 3}});
  ASSERT_EQ(MSModelResize(model, inputs, shape_infos.data(), inputs.handle_num), kMSStatusSuccess);

  MSModelDestroy(&model);
}

TEST(ModelCApiTest, ResizeWithBadDims) {
  MSContextHandle context;
  const std::string model_file = "./ml_face_isface.ms";

  // 1. Create CPU context and build the model.
  CreateCpuContext(&context);

  auto model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  auto status = MSModelBuildFromFile(model, model_file.c_str(), kMSModelTypeMindIR, context);
  ASSERT_EQ(status, kMSStatusSuccess);

  // 2. Invoke Resize() with BAD input shapes. Expect specific error code: kMSStatusLiteError.
  auto inputs = MSModelGetInputs(model);
  std::vector<MSShapeInfo> shape_infos = TransformTensorShapes({{1, 96, 96, 3}});
  ASSERT_EQ(MSModelResize(model, inputs, shape_infos.data(), inputs.handle_num), kMSStatusLiteError);

  MSModelDestroy(&model);
}
