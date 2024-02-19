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

#include <random>
#include "gtest/gtest.h"
#include "include/c_api/context_c.h"
#include "include/c_api/model_c.h"
#include "src/common/file_utils.h"
#include "common/common_test.h"

namespace mindspore {
class ModelCApiTest : public mindspore::CommonTest {
 public:
  ModelCApiTest() {}
};

int GenerateInputDataWithRandom(MSTensorHandleArray inputs) {
  auto generator = std::uniform_real_distribution<float>(0.0f, 1.0f);
  std::mt19937 random_engine_;
  for (size_t i = 0; i < inputs.handle_num; ++i) {
    float *input_data = reinterpret_cast<float *>(MSTensorGetMutableData(inputs.handle_list[i]));
    if (input_data == NULL) {
      std::cout << "MSTensorGetMutableData failed." << std::endl;
      return kMSStatusLiteError;
    }
    size_t num = static_cast<size_t>(MSTensorGetElementNum(inputs.handle_list[i]));
    (void)std::generate_n(input_data, num, [&]() { return static_cast<float>(generator(random_engine_)); });
  }
  return kMSStatusSuccess;
}

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

char **TransStrVectorToCharArrays(const std::vector<std::string> &s) {
  char **char_arr = static_cast<char **>(malloc(s.size() * sizeof(char *)));
  for (size_t i = 0; i < s.size(); i++) {
    char_arr[i] = static_cast<char *>(malloc((s[i].size() + 1)));
    memcpy(char_arr[i], s[i].c_str(), s[i].size() + 1);
  }
  return char_arr;
}

std::vector<std::string> TransCharArraysToStrVector(char **c, const size_t &num) {
  std::vector<std::string> str;
  for (size_t i = 0; i < num; i++) {
    str.push_back(std::string(c[i]));
  }
  return str;
}

void PrintTrainLossName(MSTrainCfgHandle trainCfg) {
  size_t num = 0;
  char **lossName = MSTrainCfgGetLossName(trainCfg, &num);
  std::vector<std::string> trainCfgLossName = TransCharArraysToStrVector(lossName, num);
  for (auto ele : trainCfgLossName) {
    std::cout << "loss_name:" << ele << std::endl;
  }
  for (size_t i = 0; i < num; i++) {
    free(lossName[i]);
  }
  free(lossName);
}

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

TEST(ModelCApiTest, TrainCfgCreateDestroy) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);

  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  ASSERT_NE(cpu_device_info, nullptr);
  MSDeviceType device_type = MSDeviceInfoGetDeviceType(cpu_device_info);
  ASSERT_EQ(device_type, kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);

  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);
  MSTrainCfgDestroy(&train_cfg);
  ASSERT_EQ(train_cfg, nullptr);
}

TEST(ModelCApiTest, TrainLossName) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);

  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);

  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);
  PrintTrainLossName(train_cfg);

  std::vector<std::string> set_train_cfg_loss_name = {"loss_fct", "_loss_fn"};
  char **set_loss_name = TransStrVectorToCharArrays(set_train_cfg_loss_name);
  MSTrainCfgSetLossName(train_cfg, const_cast<const char **>(set_loss_name), set_train_cfg_loss_name.size());
  PrintTrainLossName(train_cfg);
  size_t num = 0;
  char **lossName = MSTrainCfgGetLossName(train_cfg, &num);
  for (size_t i = 0; i < num; i++) {
    ASSERT_EQ(strcmp(set_train_cfg_loss_name[i].c_str(), lossName[i]), 0);
    std::cout << "cmp loss name: " << lossName[i] << std::endl;
  }
  for (size_t i = 0; i < num; i++) {
    free(lossName[i]);
  }
  free(lossName);
}

TEST(ModelCApiTest, TrainOptimizationLevel) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);
  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);
  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);

  MSTrainCfgSetOptimizationLevel(train_cfg, kMSKO2);
  auto opt_level = MSTrainCfgGetOptimizationLevel(train_cfg);
  std::cout << "Get optimization level: " << opt_level << std::endl;
  ASSERT_EQ(opt_level, kMSKO2);
}

TEST(ModelCApiTest, TrainBuildFromFile) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);
  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);
  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);

  std::string model_file = "lenet_train.ms";
  auto ret = MSTrainModelBuildFromFile(model, model_file.c_str(), kMSModelTypeMindIR, context, train_cfg);
  ASSERT_EQ(ret, kMSStatusSuccess);
  MSModelDestroy(&model);
}

TEST(ModelCApiTest, TrainBuildBuffer) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);
  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);
  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);

  std::string model_file = "lenet_train.ms";
  size_t model_size = 0;
  char *model_buffer = mindspore::lite::ReadFile(model_file.c_str(), &model_size);
  ASSERT_NE(model_buffer, nullptr);
  auto ret = MSTrainModelBuild(model, model_buffer, model_size, kMSModelTypeMindIR, context, train_cfg);
  ASSERT_EQ(ret, kMSStatusSuccess);
  MSModelDestroy(&model);
}

TEST(ModelCApiTest, TrainLearningRate) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);
  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);
  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);
  auto status = MSTrainModelBuildFromFile(model, "lenet_train.ms", kMSModelTypeMindIR, context, train_cfg);
  ASSERT_EQ(status, kMSStatusSuccess);

  auto learing_rate = MSModelGetLearningRate(model);
  std::cout << "learing_rate:" << learing_rate << std::endl;
  status = MSModelSetLearningRate(model, 0.01f);
  ASSERT_EQ(status, kMSStatusSuccess);
  learing_rate = MSModelGetLearningRate(model);
  std::cout << "get_learing_rate:" << learing_rate << std::endl;
  ASSERT_EQ(learing_rate, 0.01f);

  MSTensorHandleArray inputs = MSModelGetInputs(model);
  ASSERT_NE(inputs.handle_list, nullptr);
  GenerateInputDataWithRandom(inputs);
  status = MSModelSetTrainMode(model, true);
  ASSERT_EQ(status, kMSStatusSuccess);
  auto train_mode = MSModelGetTrainMode(model);
  ASSERT_EQ(train_mode, true);
  status = MSRunStep(model, nullptr, nullptr);
  ASSERT_EQ(status, kMSStatusSuccess);
  MSModelDestroy(&model);
}

TEST(ModelCApiTest, TrainUpdateWeights) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);
  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);
  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);
  auto status = MSTrainModelBuildFromFile(model, "lenet_train.ms", kMSModelTypeMindIR, context, train_cfg);
  ASSERT_EQ(status, kMSStatusSuccess);

  auto GenRandomData = [](size_t size, void *data) {
    auto generator = std::uniform_real_distribution<float>(0.0f, 1.0f);
    std::mt19937 random_engine_;
    size_t elements_num = size / sizeof(float);
    (void)std::generate_n(static_cast<float *>(data), elements_num,
                          [&]() { return static_cast<float>(generator(random_engine_)); });
  };
  std::vector<MSTensorHandle> vec_inputs;
  constexpr size_t create_shape_num = 1;
  int64_t create_shape[create_shape_num] = {10};
  MSTensorHandle tensor =
    MSTensorCreate("fc3.bias", kMSDataTypeNumberTypeFloat32, create_shape, create_shape_num, nullptr, 0);
  ASSERT_NE(tensor, nullptr);
  GenRandomData(MSTensorGetDataSize(tensor), MSTensorGetMutableData(tensor));
  vec_inputs.push_back(tensor);
  MSTensorHandleArray update_weights = {1, vec_inputs.data()};
  status = MSModelUpdateWeights(model, update_weights);
  ASSERT_EQ(status, kMSStatusSuccess);

  MSTensorHandleArray get_update_weights = MSModelGetWeights(model);
  for (size_t i = 0; i < get_update_weights.handle_num; ++i) {
    MSTensorHandle weights_tensor = get_update_weights.handle_list[i];
    if (strcmp(MSTensorGetName(weights_tensor), "fc3.bias") == 0) {
      float *input_data = reinterpret_cast<float *>(MSTensorGetMutableData(weights_tensor));
      std::cout << "fc3.bias:" << input_data[0] << std::endl;
    }
  }

  MSTensorHandleArray inputs = MSModelGetInputs(model);
  ASSERT_NE(inputs.handle_list, nullptr);
  GenerateInputDataWithRandom(inputs);
  status = MSModelSetTrainMode(model, true);
  ASSERT_EQ(status, kMSStatusSuccess);
  status = MSRunStep(model, nullptr, nullptr);
  ASSERT_EQ(status, kMSStatusSuccess);
  MSModelDestroy(&model);
}

TEST(ModelCApiTest, TrainSetupVirtualBatch) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);
  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);
  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);
  auto status = MSTrainModelBuildFromFile(model, "lenet_train.ms", kMSModelTypeMindIR, context, train_cfg);
  ASSERT_EQ(status, kMSStatusSuccess);
  status = MSModelSetupVirtualBatch(model, 2, -1.0f, -1.0f);
  ASSERT_EQ(status, kMSStatusSuccess);

  MSTensorHandleArray inputs = MSModelGetInputs(model);
  ASSERT_NE(inputs.handle_list, nullptr);
  GenerateInputDataWithRandom(inputs);
  status = MSModelSetTrainMode(model, true);
  ASSERT_EQ(status, kMSStatusSuccess);
  status = MSRunStep(model, nullptr, nullptr);
  ASSERT_EQ(status, kMSStatusSuccess);
  MSModelDestroy(&model);
}

TEST(ModelCApiTest, TrainExportModel) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);
  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);
  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);
  auto status = MSTrainModelBuildFromFile(model, "lenet_train.ms", kMSModelTypeMindIR, context, train_cfg);
  ASSERT_EQ(status, kMSStatusSuccess);

  MSTensorHandleArray inputs = MSModelGetInputs(model);
  ASSERT_NE(inputs.handle_list, nullptr);
  GenerateInputDataWithRandom(inputs);
  status = MSModelSetTrainMode(model, true);
  ASSERT_EQ(status, kMSStatusSuccess);
  status = MSRunStep(model, nullptr, nullptr);
  ASSERT_EQ(status, kMSStatusSuccess);

  status = MSExportModel(model, kMSModelTypeMindIR, "lenet_train_infer.ms", kMSNO_QUANT, true, nullptr, 0);
  ASSERT_EQ(status, kMSStatusSuccess);

  MSModelDestroy(&model);
}

TEST(ModelCApiTest, TrainExportModelBuffer) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);
  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);
  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);
  auto status = MSTrainModelBuildFromFile(model, "lenet_train.ms", kMSModelTypeMindIR, context, train_cfg);
  ASSERT_EQ(status, kMSStatusSuccess);

  MSTensorHandleArray inputs = MSModelGetInputs(model);
  ASSERT_NE(inputs.handle_list, nullptr);
  GenerateInputDataWithRandom(inputs);
  status = MSModelSetTrainMode(model, true);
  ASSERT_EQ(status, kMSStatusSuccess);
  status = MSRunStep(model, nullptr, nullptr);
  ASSERT_EQ(status, kMSStatusSuccess);

  char *modelData;
  size_t data_size;
  status = MSExportModelBuffer(model, kMSModelTypeMindIR, &modelData, &data_size, kMSNO_QUANT, true, nullptr, 0);
  ASSERT_EQ(status, kMSStatusSuccess);
  ASSERT_NE(modelData, nullptr);
  ASSERT_NE(data_size, 0);
  MSModelDestroy(&model);
}

TEST(ModelCApiTest, TrainExportWeightsMicro) {
  MSContextHandle context = MSContextCreate();
  ASSERT_NE(context, nullptr);
  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  MSContextAddDeviceInfo(context, cpu_device_info);
  MSModelHandle model = MSModelCreate();
  ASSERT_NE(model, nullptr);
  MSTrainCfgHandle train_cfg = MSTrainCfgCreate();
  ASSERT_NE(train_cfg, nullptr);
  auto status = MSTrainModelBuildFromFile(model, "lenet_train.ms", kMSModelTypeMindIR, context, train_cfg);
  ASSERT_EQ(status, kMSStatusSuccess);

  MSTensorHandleArray inputs = MSModelGetInputs(model);
  ASSERT_NE(inputs.handle_list, nullptr);
  GenerateInputDataWithRandom(inputs);
  status = MSModelSetTrainMode(model, true);
  ASSERT_EQ(status, kMSStatusSuccess);
  status = MSRunStep(model, nullptr, nullptr);
  ASSERT_EQ(status, kMSStatusSuccess);

  status = MSExportWeightsCollaborateWithMicro(model, kMSModelTypeMindIR, "lenet_train.bin", true, true, nullptr, 0);
  ASSERT_EQ(status, kMSStatusSuccess);
  MSModelDestroy(&model);
}
}  // namespace mindspore
