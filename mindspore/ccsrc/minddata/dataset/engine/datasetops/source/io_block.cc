/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/io_block.h"

#include <string>
#include <vector>

namespace mindspore {
namespace dataset {
// IOBlock Class //

// Constructor of the IOBlock (1).  A simpler one for the case when the block only has 1 key.
IOBlock::IOBlock(int64_t inKey, IOBlockFlags io_block_flags) : index_keys_(1, inKey), io_block_flags_(io_block_flags) {}

// Constructor of the IOBlock (2)
IOBlock::IOBlock(const std::vector<int64_t> &in_keys, IOBlockFlags io_block_flags) : io_block_flags_(io_block_flags) {
  index_keys_.insert(index_keys_.end(), in_keys.begin(), in_keys.end());
}

// Constructor of the IOBlock (3).  A special IOBlock that is used for control messaging.
IOBlock::IOBlock(IOBlockFlags io_block_flags) : io_block_flags_(io_block_flags) {}

// Fetches the first key from this block
Status IOBlock::GetKey(int64_t *out_key) const {
  if (out_key == nullptr || index_keys_.empty()) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to get the key from IOBlock.");
  }
  *out_key = index_keys_[0];
  return Status::OK();
}

// Fetches the list of keys from this block.
Status IOBlock::GetKeys(std::vector<int64_t> *out_keys) const {
  if (out_keys == nullptr) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Output arg for GetKeys is null.");
  }
  *out_keys = index_keys_;  // vector copy assign
  return Status::OK();
}

// FilenameBlock derived class //

// Constructor of the FilenameBlock (1)
FilenameBlock::FilenameBlock(int64_t key, int64_t start_offset, int64_t end_offset, IOBlockFlags io_block_flags)
    : IOBlock(key, io_block_flags), start_offset_(start_offset), end_offset_(end_offset) {}

// Constructor of the FilenameBlock (2).  A special IOBlock that is used for control messaging.
FilenameBlock::FilenameBlock(IOBlockFlags io_block_flags)
    : IOBlock(io_block_flags), start_offset_(kInvalidOffset), end_offset_(kInvalidOffset) {}

// Gets the filename from the block using the provided index container
Status FilenameBlock::GetFilename(std::string *out_filename, const AutoIndexObj<std::string> &index) const {
  if (out_filename == nullptr) {
    RETURN_STATUS_UNEXPECTED("Failed to get filename from FilenameBlock.");
  }

  // a FilenameBlock only has one key.  Call base class method to fetch that key
  int64_t fetched_key;
  RETURN_IF_NOT_OK(IOBlock::GetKey(&fetched_key));

  // Do an index lookup using that key to get the filename.
  auto r = index.Search(fetched_key);
  if (r.second) {
    auto &it = r.first;
    *out_filename = it.value();
  } else {
    RETURN_STATUS_UNEXPECTED("Could not find filename from index.");
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
