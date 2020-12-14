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

#include "common/common_test.h"
#include "ps/embedding_table_shard_metadata.h"

namespace mindspore {
namespace ps {
class TestEmbeddingTableShardMetadata : public UT::Common {
 public:
  TestEmbeddingTableShardMetadata() = default;
  virtual ~TestEmbeddingTableShardMetadata() = default;

  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestEmbeddingTableShardMetadata, EmbeddingTable) {
  EmbeddingTableShardMetadata embedding_table_shard(1, 100);
  EXPECT_EQ(embedding_table_shard.begin(), 1);
  EXPECT_EQ(embedding_table_shard.end(), 100);
  EXPECT_EQ(embedding_table_shard.size(), 99);
}
}  // namespace ps
}  // namespace mindspore
