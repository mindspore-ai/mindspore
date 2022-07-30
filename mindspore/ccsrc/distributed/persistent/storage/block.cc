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

#include "distributed/persistent/storage/block.h"
#include "utils/system/sha256.h"
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"
#include "distributed/persistent/storage/constants.h"

namespace mindspore {
namespace distributed {
namespace storage {
void Block::GenSha256Seq() const {
  std::string sha256_cal = system::sha256::GetHashFromFile(block_file_name_);
  MS_EXCEPTION_IF_NULL(block_meta_);
  block_meta_->Insert(kHashSeq, sha256_cal);
}

bool Block::CheckSha256Seq() const {
  MS_EXCEPTION_IF_NULL(block_meta_);
  std::string sha256_gen = block_meta_->Get<std::string>(kHashSeq);
  if (sha256_gen != system::sha256::GetHashFromFile(block_file_name_)) {
    MS_LOG(ERROR) << "The block file has been modified, file name: " << block_file_name_;
    return false;
  }
  return true;
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
