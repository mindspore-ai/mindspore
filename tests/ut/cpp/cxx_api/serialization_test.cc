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
class TestCxxApiSerialization : public UT::Common {
 public:
  TestCxxApiSerialization() = default;
};

TEST_F(TestCxxApiSerialization, test_load_no_encrpty_mindir_SUCCESS) {
  Graph graph;
  ASSERT_TRUE(Serialization::Load("./data/mindir/add_no_encrpty.mindir", ModelType::kMindIR, &graph) == kSuccess);
}

TEST_F(TestCxxApiSerialization, test_load_output_args_nullptr_FAILED) {
  auto status = Serialization::Load("./data/mindir/add_no_encrpty.mindir", ModelType::kMindIR, nullptr);
  ASSERT_TRUE(status != kSuccess);
  auto err_mst = status.GetErrDescription();
  ASSERT_TRUE(err_mst.find("null") != std::string::npos);
}

TEST_F(TestCxxApiSerialization, test_load_file_not_exist_FAILED) {
  Graph graph;
  auto status = Serialization::Load("./data/mindir/file_not_exist.mindir", ModelType::kMindIR, &graph);
  ASSERT_TRUE(status != kSuccess);
  auto err_mst = status.GetErrDescription();
  ASSERT_TRUE(err_mst.find("exist") != std::string::npos);
}

TEST_F(TestCxxApiSerialization, test_load_encrpty_mindir_SUCCESS) {
  Graph graph;
  std::string key_str = "0123456789ABCDEF";
  Key key;
  memcpy(key.key, key_str.data(), key_str.size());
  key.len = key_str.size();
  ASSERT_TRUE(Serialization::Load("./data/mindir/add_encrpty_key_0123456789ABCDEF.mindir", ModelType::kMindIR, &graph,
                                  key, kDecModeAesGcm) == kSuccess);
}

TEST_F(TestCxxApiSerialization, test_load_encrpty_mindir_without_key_FAILED) {
  Graph graph;
  auto status =
    Serialization::Load("./data/mindir/add_encrpty_key_0123456789ABCDEF.mindir", ModelType::kMindIR, &graph);
  ASSERT_TRUE(status != kSuccess);
  auto err_mst = status.GetErrDescription();
  ASSERT_TRUE(err_mst.find("be encrypted") != std::string::npos);
}

TEST_F(TestCxxApiSerialization, test_load_encrpty_mindir_with_wrong_key_FAILED) {
  Graph graph;
  std::string key_str = "WRONG_KEY";
  Key key;
  memcpy(key.key, key_str.data(), key_str.size());
  key.len = key_str.size();
  auto status = Serialization::Load("./data/mindir/add_encrpty_key_0123456789ABCDEF.mindir", ModelType::kMindIR, &graph,
                                    key, kDecModeAesGcm);
  ASSERT_TRUE(status != kSuccess);
}

TEST_F(TestCxxApiSerialization, test_load_no_encrpty_mindir_with_wrong_key_FAILED) {
  Graph graph;
  std::string key_str = "WRONG_KEY";
  Key key;
  memcpy(key.key, key_str.data(), key_str.size());
  key.len = key_str.size();
  auto status = Serialization::Load("./data/mindir/add_no_encrpty.mindir", ModelType::kMindIR, &graph,
                                    key, kDecModeAesGcm);
  ASSERT_TRUE(status != kSuccess);
}

TEST_F(TestCxxApiSerialization, test_load_no_encrpty_mindir_x1_SUCCESS) {
  std::vector<Graph> graphs;
  ASSERT_TRUE(Serialization::Load(std::vector<std::string>(1, "./data/mindir/add_no_encrpty.mindir"),
                                  ModelType::kMindIR, &graphs) == kSuccess);
}

TEST_F(TestCxxApiSerialization, test_load_no_encrpty_mindir_x2_SUCCESS) {
  std::vector<Graph> graphs;
  ASSERT_TRUE(Serialization::Load(std::vector<std::string>(2, "./data/mindir/add_no_encrpty.mindir"),
                                  ModelType::kMindIR, &graphs) == kSuccess);
}

TEST_F(TestCxxApiSerialization, test_load_file_not_exist_x2_FAILED) {
  std::vector<Graph> graphs;
  auto status = Serialization::Load(std::vector<std::string>(2, "./data/mindir/file_not_exist.mindir"),
                                    ModelType::kMindIR, &graphs);
  ASSERT_TRUE(status != kSuccess);
  auto err_mst = status.GetErrDescription();
  ASSERT_TRUE(err_mst.find("exist") != std::string::npos);
}

TEST_F(TestCxxApiSerialization, test_load_encrpty_mindir_without_key_x2_FAILED) {
  std::vector<Graph> graphs;
  auto status = Serialization::Load(
    std::vector<std::string>(2, "./data/mindir/add_encrpty_key_0123456789ABCDEF.mindir"), ModelType::kMindIR, &graphs);
  ASSERT_TRUE(status != kSuccess);
  auto err_mst = status.GetErrDescription();
  ASSERT_TRUE(err_mst.find("be encrypted") != std::string::npos);
}
}  // namespace mindspore
