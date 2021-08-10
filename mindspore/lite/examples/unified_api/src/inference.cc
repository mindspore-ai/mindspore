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

#include <getopt.h>
#include <string>
#include <iostream>
#include <fstream>
#include "src/utils.h"
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/graph.h"
#include "include/api/serialization.h"

static void Usage() { std::cout << "Usage: infer -f <.ms model file>" << std::endl; }

static std::string ReadArgs(int argc, char *argv[]) {
  std::string infer_model_fn;
  int opt;
  while ((opt = getopt(argc, argv, "f:")) != -1) {
    switch (opt) {
      case 'f':
        infer_model_fn = std::string(optarg);
        break;
      default:
        break;
    }
  }
  return infer_model_fn;
}

int main(int argc, char **argv) {
  std::string infer_model_fn = ReadArgs(argc, argv);
  if (infer_model_fn.size() == 0) {
    Usage();
    return -1;
  }

  auto context = std::make_shared<mindspore::Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  cpu_context->SetEnableFP16(false);
  context->MutableDeviceInfo().push_back(cpu_context);

  mindspore::Graph graph;
  auto status = mindspore::Serialization::Load(infer_model_fn, mindspore::kMindIR, &graph);
  if (status != mindspore::kSuccess) {
    std::cout << "Error " << status << " during serialization of graph " << infer_model_fn;
    MS_ASSERT(status != mindspore::kSuccess);
  }

  mindspore::Model model;
  status = model.Build(mindspore::GraphCell(graph), context);
  if (status != mindspore::kSuccess) {
    std::cout << "Error " << status << " during build of model " << infer_model_fn;
    MS_ASSERT(status != mindspore::kSuccess);
  }

  auto inputs = model.GetInputs();
  MS_ASSERT(inputs.size() >= 1);

  int index = 0;
  std::cout << "There are " << inputs.size() << " input tensors with sizes: " << std::endl;
  for (auto tensor : inputs) {
    std::cout << "tensor " << index++ << ": shape is [";
    for (auto dim : tensor.Shape()) {
      std::cout << dim << " ";
    }
    std::cout << "]" << std::endl;
  }

  mindspore::MSTensor *input_tensor = inputs.at(0).Clone();
  auto *input_data = reinterpret_cast<float *>(input_tensor->MutableData());
  std::ifstream in;
  in.open("dataset/batch_of32.dat", std::ios::in | std::ios::binary);
  if (in.fail()) {
    std::cout << "error loading dataset/batch_of32.dat file reading" << std::endl;
    MS_ASSERT(!in.fail());
  }
  in.read(reinterpret_cast<char *>(input_data), inputs.at(0).ElementNum() * sizeof(float));
  in.close();

  std::vector<mindspore::MSTensor> outputs;
  status = model.Predict({*input_tensor}, &outputs);
  if (status != mindspore::kSuccess) {
    std::cout << "Error " << status << " during running predict of model " << infer_model_fn;
    MS_ASSERT(status != mindspore::kSuccess);
  }

  index = 0;
  std::cout << "There are " << outputs.size() << " output tensors with sizes: " << std::endl;
  for (auto tensor : outputs) {
    std::cout << "tensor " << index++ << ": shape is [";
    for (auto dim : tensor.Shape()) {
      std::cout << dim << " ";
    }
    std::cout << "]" << std::endl;
  }

  if (outputs.size() > 0) {
    std::cout << "The predicted classes are:" << std::endl;
    auto predictions = reinterpret_cast<float *>(outputs.at(0).MutableData());
    int i = 0;
    for (int b = 0; b < outputs.at(0).Shape().at(0); b++) {
      int max_c = 0;
      float max_p = predictions[i];
      for (int c = 0; c < outputs.at(0).Shape().at(1); c++, i++) {
        if (predictions[i] > max_p) {
          max_c = c;
          max_p = predictions[i];
        }
      }
      std::cout << max_c << ", ";
    }
    std::cout << std::endl;
  }
  return 0;
}
