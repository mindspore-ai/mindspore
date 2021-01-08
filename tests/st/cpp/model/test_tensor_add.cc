/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <string>
#include <vector>
#include "common/common_test.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using namespace mindspore::api;

static const char tensor_add_file[] = "/home/workspace/mindspore_dataset/mindir/tensor_add/tensor_add.mindir";
static const std::vector<float> input_data_1 = {1, 2, 3, 4};
static const std::vector<float> input_data_2 = {2, 3, 4, 5};

class TestTensorAdd : public ST::Common {
 public:
  TestTensorAdd() {}
};

TEST_F(TestTensorAdd, InferMindIR) {
  ContextAutoSet();

  auto graph = Serialization::LoadModel(tensor_add_file, ModelType::kMindIR);
  Model tensor_add((GraphCell(graph)));
  Status ret = tensor_add.Build({});
  ASSERT_TRUE(ret == SUCCESS);

  // prepare input
  std::vector<Buffer> outputs;
  std::vector<Buffer> inputs;
  inputs.emplace_back(Buffer(input_data_1.data(), sizeof(float) * input_data_1.size()));
  inputs.emplace_back(Buffer(input_data_2.data(), sizeof(float) * input_data_2.size()));

  // infer
  ret = tensor_add.Predict(inputs, &outputs);
  ASSERT_TRUE(ret == SUCCESS);

  // print
  for (auto &buffer : outputs) {
    const float *p = reinterpret_cast<const float *>(buffer.Data());
    for (size_t i = 0; i <  buffer.DataSize() / sizeof(float); ++i) {
      ASSERT_LE(std::abs(p[i] - (input_data_1[i] + input_data_2[i])), 1e-4);
    }
  }
}
