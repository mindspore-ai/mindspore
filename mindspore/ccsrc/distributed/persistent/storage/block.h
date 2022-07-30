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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_BLOCK_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_BLOCK_H_

#include <memory>
#include <string>

#include "distributed/persistent/storage/json_utils.h"
#include "nlohmann/json.hpp"

namespace mindspore {
namespace distributed {
namespace storage {
// Using json to store and get meta info of a block, the content of meta info can be customized,
// such as shard shape, shard range, field length, etc.
using BlockMeta = JsonUtils;

// Class Block corresponds to the block file, saves the path of the block file,
// and provides block file integrity verification.
class Block {
 public:
  explicit Block(const std::string &block_name) : block_file_name_(block_name) {}
  ~Block() = default;

  // The following two methods are used to file integrity check.
  // Generate sha256 hash sequence.
  void GenSha256Seq() const;

  // Check sha256 hash sequence.
  bool CheckSha256Seq() const;

  // Set the block meta pointer associated with the block file.
  void set_block_meta(const std::shared_ptr<BlockMeta> &block_meta) { block_meta_ = block_meta; }

  // Get block meta file path.
  const std::string &block_file_name() const { return block_file_name_; }

 private:
  // The block meta information corresponding to the block.
  std::shared_ptr<BlockMeta> block_meta_;
  // The block file path.
  std::string block_file_name_;
};
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_BLOCK_H_
