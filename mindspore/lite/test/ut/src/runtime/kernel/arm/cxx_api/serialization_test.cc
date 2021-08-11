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
#include "include/api/serialization.h"

namespace mindspore {
class TestCxxApiLiteSerialization : public mindspore::CommonTest {
 public:
  TestCxxApiLiteSerialization() = default;
};

TEST_F(TestCxxApiLiteSerialization, test_load_no_encrpty_mindir_SUCCESS) {
  Graph graph;
  ASSERT_TRUE(Serialization::Load("./nets/retinaface1.ms", ModelType::kMindIR, &graph) == kSuccess);
}

TEST_F(TestCxxApiLiteSerialization, test_load_file_not_exist_FAILED) {
  Graph graph;
  auto status = Serialization::Load("./nets/file_not_exist.mindir", ModelType::kMindIR, &graph);
  ASSERT_TRUE(status != kSuccess);
}

TEST_F(TestCxxApiLiteSerialization, test_load_file_not_exist_x2_FAILED) {
  std::vector<Graph> graphs;
  auto status =
    Serialization::Load(std::vector<std::string>(2, "./nets/file_not_exist.mindir"), ModelType::kFlatBuffer, &graphs);
  ASSERT_TRUE(status != kSuccess);
}

TEST_F(TestCxxApiLiteSerialization, test_export_uninitialized_FAILED) {
  Model model;
  ASSERT_TRUE(Serialization::ExportModel(model, ModelType::kFlatBuffer, "./nets/export.ms") != kSuccess);
}

}  // namespace mindspore
