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

#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include "include/errorcode.h"
#include "include/context.h"
#include "include/api/types.h"
#include "include/api/model.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kNumPrintOfOutData = 20;
std::string RealPath(const char *path) {
  const size_t max = 4096;
  if (path == nullptr) {
    std::cerr << "path is nullptr" << std::endl;
    return "";
  }
  if ((strlen(path)) >= max) {
    std::cerr << "path is too long" << std::endl;
    return "";
  }
  auto resolved_path = std::make_unique<char[]>(max);
  if (resolved_path == nullptr) {
    std::cerr << "new resolved_path failed" << std::endl;
    return "";
  }

  char *real_path = realpath(path, resolved_path.get());
  if (real_path == nullptr || strlen(real_path) == 0) {
    std::cerr << "file path is not valid : " << path << std::endl;
    return "";
  }
  std::string res = resolved_path.get();
  return res;
}

char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    std::cerr << "file is nullptr." << std::endl;
    return nullptr;
  }

  std::ifstream ifs(file);
  if (!ifs.good()) {
    std::cerr << "file: " << file << " is not exist." << std::endl;
    return nullptr;
  }

  if (!ifs.is_open()) {
    std::cerr << "file: " << file << " open failed." << std::endl;
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char[]> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    std::cerr << "malloc buf failed, file: " << file << std::endl;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}
}  // namespace

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
}

void InitMSContext(const std::shared_ptr<mindspore::Context> &context) {
  context->SetThreadNum(1);
  context->SetEnableParallel(false);
  context->SetThreadAffinity(HIGHER_CPU);
  auto &device_list = context->MutableDeviceInfo();

  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_info->SetEnableFP16(false);
  device_list.push_back(device_info);

  std::shared_ptr<GPUDeviceInfo> provider_gpu_device_info = std::make_shared<GPUDeviceInfo>();
  provider_gpu_device_info->SetEnableFP16(false);
  provider_gpu_device_info->SetProviderDevice("GPU");
  provider_gpu_device_info->SetProvider("Tutorial");
  device_list.push_back(provider_gpu_device_info);
}

int CompileAndRun(int argc, const char **argv) {
  if (argc < 2) {
    std::cerr << "Model file must be provided.\n";
    return RET_ERROR;
  }
  // Read model file.
  auto model_path = RealPath(argv[1]);
  if (model_path.empty()) {
    std::cerr << "model path " << argv[1] << " is invalid.";
    return RET_ERROR;
  }

  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return RET_ERROR;
  }

  (void)InitMSContext(context);

  mindspore::Model ms_model;
  size_t size = 0;
  char *model_buf = ReadFile(model_path.c_str(), &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return RET_ERROR;
  }
  auto ret = ms_model.Build(model_buf, size, kMindIR, context);
  delete[](model_buf);
  if (ret != kSuccess) {
    std::cerr << "ms_model.Build failed." << std::endl;
    return RET_ERROR;
  }
  std::vector<mindspore::MSTensor> ms_inputs_for_api = ms_model.GetInputs();
  for (auto tensor : ms_inputs_for_api) {
    auto input_data = tensor.MutableData();
    if (input_data == nullptr) {
      std::cerr << "MallocData for inTensor failed." << std::endl;
      return RET_ERROR;
    }
    GenerateRandomData<float>(tensor.DataSize(), input_data, std::uniform_real_distribution<float>(1.0f, 1.0f));
  }

  std::cout << "\n------- print inputs ----------" << std::endl;
  for (auto tensor : ms_inputs_for_api) {
    std::cout << "in tensor name is:" << tensor.Name() << "\nin tensor size is:" << tensor.DataSize()
              << "\nin tensor elements num is:" << tensor.ElementNum() << std::endl;
    auto out_data = reinterpret_cast<float *>(tensor.MutableData());
    std::cout << "input data is:";
    for (int i = 0; i < tensor.ElementNum() && i <= kNumPrintOfOutData; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "------- print end ----------\n" << std::endl;

  std::vector<MSTensor> outputs;
  auto status = ms_model.Predict(ms_inputs_for_api, &outputs);
  if (status != kSuccess) {
    std::cerr << "Inference error." << std::endl;
    return RET_ERROR;
  }

  // Get Output Tensor Data.
  auto out_tensors = ms_model.GetOutputs();
  std::cout << "\n------- print outputs ----------" << std::endl;
  for (auto tensor : out_tensors) {
    std::cout << "out tensor name is:" << tensor.Name() << "\nout tensor size is:" << tensor.DataSize()
              << "\nout tensor elements num is:" << tensor.ElementNum() << std::endl;
    auto out_data = reinterpret_cast<float *>(tensor.MutableData());
    std::cout << "output data is:";
    for (int i = 0; i < tensor.ElementNum() && i <= kNumPrintOfOutData; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "------- print end ----------\n" << std::endl;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

int main(int argc, const char **argv) { return mindspore::lite::CompileAndRun(argc, argv); }
