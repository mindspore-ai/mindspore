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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IO_BLOCK_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IO_BLOCK_H_

#include <string>
#include <vector>

#include "minddata/dataset/util/auto_index.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// The IOBlock class is used to describe a "unit of work" that a storage leaf operator worker thread
// is responsible for acting on.
// The IOBlocks and it's derived classes abstracts a key-store and key-lookup interface where each
// block contains 1 to n keys, and the keys are used in conjunction with an index to provide the meta
// information for satisfying an IO request.
class IOBlock {
 public:
  enum IOBlockFlags : uint32_t {
    kDeIoBlockNone = 0,
    kDeIoBlockFlagEoe = 1u,       // end of IOBlocks for one epoch
    kDeIoBlockFlagEof = 1u << 1,  // end of IOBlocks for entire program
    kDeIoBlockFlagWait = 1u << 2  // control signal for workers to suspend operations
  };

  // Constructor of the IOBlock (1).  A simpler one for the case when the block only has 1 key.
  // @param inKey - A single key to add into the block
  // @param io_block_flags - The flag setting for the block
  IOBlock(int64_t inKey, IOBlockFlags io_block_flags);

  // Constructor of the IOBlock (2).
  // @param in_keys - A vector of keys to add into the block
  // @param io_block_flags - The flag setting for the block
  IOBlock(const std::vector<int64_t> &in_keys, IOBlockFlags io_block_flags);

  // Constructor of the IOBlock (3).  A special IOBlock that is used for control messaging.
  // @param io_block_flags - The flag setting for the block
  explicit IOBlock(IOBlockFlags io_block_flags);

  // Destructor
  virtual ~IOBlock() = default;

  // Fetches the first key from the block.
  // @note Only useful if you know the block only has 1 key.
  // @return  A copy of the first key from the block
  // @return Status The status code returned
  Status GetKey(int64_t *out_key) const;

  // Fetches the list of keys from this block.
  // @param out_keys - A copy of the vector of keys from the block.
  // @return Status The status code returned
  Status GetKeys(std::vector<int64_t> *out_keys) const;

  // Does this block have the eoe flag turned on?
  // @return T/F if the IOBlock is eoe
  bool eoe() const { return static_cast<uint32_t>(io_block_flags_) & static_cast<uint32_t>(kDeIoBlockFlagEoe); }

  // Does this block have the eof flag turned on?
  // @return T/F if the IOBlock is eof
  bool eof() const { return static_cast<uint32_t>(io_block_flags_) & static_cast<uint32_t>(kDeIoBlockFlagEof); }

  // Does this block have the wait flag turned on?
  // @return T/F is the IOBlock is wait
  bool wait() const { return static_cast<uint32_t>(io_block_flags_) & static_cast<uint32_t>(kDeIoBlockFlagWait); }

  // Adds a key to this block
  // @param key - The key to add to this block
  void AddKey(int64_t key) { index_keys_.push_back(key); }

 protected:
  std::vector<int64_t> index_keys_;  // keys used for lookups to the meta info for the data
  IOBlockFlags io_block_flags_;
};  // class IOBlock

const int64_t kInvalidOffset = -1;

// The Filename block derived class implements a style of IO block where each block contains only a
// single key that maps to a filename.
class FilenameBlock : public IOBlock {
 public:
  // Constructor of the FilenameBlock (1)
  // @param key - The key identifier that can be used to find the data for this block
  // @param start_offset - Start offset
  // @param end_offset - End offset
  // @param io_block_flags - The flag setting for the block
  FilenameBlock(int64_t key, int64_t start_offset, int64_t end_offset, IOBlockFlags io_block_flags);

  // Constructor of the FilenameBlock (2).  A special IOBlock that is used for control messaging.
  // @param io_block_flags - The flag setting for the block
  explicit FilenameBlock(IOBlockFlags io_block_flags);

  // Destructor
  ~FilenameBlock() = default;

  // Gets the filename from the block using the provided index container
  // @param out_filename - The filename to add to the block
  // @param index - The index to perform lookup against
  // @return Status The status code returned
  Status GetFilename(std::string *out_filename, const AutoIndexObj<std::string> &index) const;

  // Get the start offset of file
  // @return int64_t - Start offset
  int64_t GetStartOffset() const { return start_offset_; }

  // Get the end offset of the file
  // @return int64_t - Start offset
  int64_t GetEndOffset() const { return end_offset_; }

 private:
  int64_t start_offset_;
  int64_t end_offset_;
};  // class TFBlock
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IO_BLOCK_H_
