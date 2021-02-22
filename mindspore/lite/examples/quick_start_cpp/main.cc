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
#include "include/errorcode.h"
#include "include/model.h"
#include "include/context.h"
#include "include/lite_session.h"

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

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&]() { return static_cast<T>(distribution(random_engine)); });
}

int GenerateInputDataWithRandom(std::vector<mindspore::tensor::MSTensor *> inputs) {
  for (auto tensor : inputs) {
    auto input_data = tensor->MutableData();
    void *random_data = malloc(tensor->Size());
    if (input_data == nullptr) {
      std::cerr << "MallocData for inTensor failed." << std::endl;
      return -1;
    }
    GenerateRandomData<float>(tensor->Size(), random_data, std::uniform_real_distribution<float>(0.1f, 1.0f));
    // Copy data to input tensor.
    memcpy(input_data, random_data, tensor->Size());
  }
  return mindspore::lite::RET_OK;
}

int Run(mindspore::session::LiteSession *session) {
  auto inputs = session->GetInputs();
  auto ret = GenerateInputDataWithRandom(inputs);
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return ret;
  }

  ret = session->RunGraph();
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }

  auto out_tensors = session->GetOutputs();
  for (auto tensor : out_tensors) {
    std::cout << "tensor name is:" << tensor.first << " tensor size is:" << tensor.second->Size()
              << " tensor elements num is:" << tensor.second->ElementsNum() << std::endl;
    auto out_data = reinterpret_cast<float *>(tensor.second->MutableData());
    std::cout << "output data is:";
    for (int i = 0; i < tensor.second->ElementsNum() && i <= 50; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
  return mindspore::lite::RET_OK;
}

mindspore::session::LiteSession *Compile(mindspore::lite::Model *model) {
  // Create and init context.
  auto context = std::make_shared<mindspore::lite::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed while." << std::endl;
    return nullptr;
  }

  // Create the session.
  mindspore::session::LiteSession *session = mindspore::session::LiteSession::CreateSession(context.get());
  if (session == nullptr) {
    std::cerr << "CreateSession failed while running." << std::endl;
    return nullptr;
  }

  // Compile graph.
  auto ret = session->CompileGraph(model);
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Compile failed while running." << std::endl;
    return nullptr;
  }

  // Note: when use model->Free(), the model can not be compiled again.
  if (model != nullptr) {
    model->Free();
  }
  return session;
}

int CompileAndRun(int argc, const char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: ./mindspore_quick_start_cpp ../model/mobilenetv2.ms\n";
    return -1;
  }
  // Read model file.
  auto model_path = argv[1];
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }
  // Load the .ms model.
  auto model = mindspore::lite::Model::Import(model_buf, size);
  delete[](model_buf);
  if (model == nullptr) {
    std::cerr << "Import model file failed." << std::endl;
    return -1;
  }
  // Compile MindSpore Lite model.
  auto session = Compile(model);
  if (session == nullptr) {
    std::cerr << "Create session failed." << std::endl;
    return -1;
  }
  // Run inference.
  auto ret = Run(session);
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "MindSpore Lite run failed." << std::endl;
    return -1;
  }
  // Delete model buffer.
  delete model;
  // Delete session buffer.
  delete session;
  return mindspore::lite::RET_OK;
}

int main(int argc, const char **argv) { return CompileAndRun(argc, argv); }
