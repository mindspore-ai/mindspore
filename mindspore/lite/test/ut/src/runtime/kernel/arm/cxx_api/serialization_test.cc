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
#include <string>
#include <iostream>
#include "common/common_test.h"
#include "include/api/serialization.h"

namespace mindspore {
class TestCxxApiLiteSerialization : public mindspore::CommonTest {
 public:
  TestCxxApiLiteSerialization() = default;
};

TEST_F(TestCxxApiLiteSerialization, test_load_file_not_exist_FAILED) {
  Graph graph;
  auto status = Serialization::Load("./nets/file_not_exist.mindir", ModelType::kMindIR, &graph);
  ASSERT_TRUE(status != kSuccess);
}

TEST_F(TestCxxApiLiteSerialization, test_load_file_not_exist_x2_FAILED) {
  std::vector<Graph> graphs;
  auto status =
    Serialization::Load(std::vector<std::string>(2, "./nets/file_not_exist.mindir"), ModelType::kMindIR, &graphs);
  ASSERT_TRUE(status != kSuccess);
}

TEST_F(TestCxxApiLiteSerialization, test_export_uninitialized_FAILED) {
  Model model;
  ASSERT_TRUE(Serialization::ExportModel(model, ModelType::kMindIR, "./nets/export.ms") != kSuccess);
}

TEST_F(TestCxxApiLiteSerialization, test_export_to_buffer) {
  auto context = std::make_shared<mindspore::Context>();
  auto cpu_context = std::make_shared<mindspore::CPUDeviceInfo>();
  context->MutableDeviceInfo().push_back(cpu_context);

  Graph graph;
  std::string file_name = "../../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_train.ms";
  auto status = mindspore::Serialization::Load(file_name, mindspore::kMindIR, &graph);
  ASSERT_TRUE(status == mindspore::kSuccess);

  Model model;
  auto cfg = std::make_shared<mindspore::TrainCfg>();

  status = model.Build(mindspore::GraphCell(graph), context, cfg);
  ASSERT_TRUE(status == mindspore::kSuccess);

  std::string exported_file = "./export.ms";
  status = Serialization::ExportModel(model, mindspore::kMindIR, exported_file, mindspore::kNoQuant, false);
  ASSERT_TRUE(status == mindspore::kSuccess);

  mindspore::Buffer modef_buffer_infer;
  status = Serialization::ExportModel(model, mindspore::kMindIR, &modef_buffer_infer, mindspore::kNoQuant, false);
  ASSERT_TRUE(status == mindspore::kSuccess);

  std::ifstream file(exported_file.c_str(), std::ifstream::binary);
  ASSERT_TRUE(file);

  file.seekg(0, std::ifstream::end);
  size_t file_size = file.tellg();
  file.seekg(0);

  const int kMaxSize = 1024 * 1024;
  char buf[kMaxSize] = {0};
  file.read(buf, file_size);
  file.close();

  int result = memcmp(buf, modef_buffer_infer.Data(), file_size);
  ASSERT_EQ(result, 0);
}

}  // namespace mindspore
