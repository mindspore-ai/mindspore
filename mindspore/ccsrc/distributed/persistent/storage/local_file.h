/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_LOCAL_FILE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_LOCAL_FILE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "distributed/persistent/storage/storage.h"
#include "distributed/persistent/storage/block.h"
#include "distributed/persistent/storage/file_io_utils.h"
#include "distributed/persistent/storage/constants.h"
#include "utils/hash_map.h"
#include "utils/system/file_system.h"
#include "utils/system/env.h"

namespace mindspore {
namespace distributed {
namespace storage {
// The default maximum block length : 128MB.
constexpr size_t DEFAULT_MAX_BLOCK_LENGTH = 128 << 20;

// The default page size : 16KB.
constexpr size_t DEFAULT_PAGE_SIZE = 16 << 10;

// The default file size to save pages: 1GB.
constexpr size_t DEFAULT_BLOCK_FILE_SIZE = 1 << 30;

// Get a num is nearest bigger then n and can be divided by base
inline size_t RoundUp(size_t n, size_t base) { return (n + base - 1) / base * base; }

// File type persistence storage implementation class.
class LocalFile : public StorageBase {
 public:
  explicit LocalFile(const std::map<std::string, std::string> &storage_config, const size_t id_size = 0,
                     const size_t feature_size = 0)
      : id_size_(id_size), feature_size_(feature_size) {
    auto file_path_iter = storage_config.find(kFileStoragePath);
    if (file_path_iter != storage_config.end()) {
      file_path_ = file_path_iter->second;
    }

    auto block_length_iter = storage_config.find(kMaxBlockLength);
    if (block_length_iter != storage_config.end() && !(block_length_iter->second).empty()) {
      max_block_length_ = std::stoul(block_length_iter->second);
    } else {
      max_block_length_ = DEFAULT_MAX_BLOCK_LENGTH;
    }
  }

  ~LocalFile() override = default;

  // Init some data member:
  // 1. Calculate how many value can be write in a page or block.
  // 2. Init file system handle.
  void Initialize() override;

  // The following two methods are override version function for Write:
  // 1. Create blocks and block metas.
  // 2. Write input data to block files and Generate sha256 sequence for every block file.
  // Write the entire blob data of tensor to the block files on disk:
  void Write(const InputData &input, const DirtyInfo &dirty_info) override;
  // Write the entire blob data composed of multiple tensors to the block files on disk:
  void Write(const std::vector<InputData> &inputs, const DirtyInfo &dirty_info) override;
  // Write data of ids to block files.
  void Write(const void *input, const std::vector<int> &ids) override;

  // The following two methods are override version function for Read:
  // 1.Tamper proof check.
  // 2.Read all block files and merge them into contiguous memory.
  // Read data from all block files in file_path_(dir):
  void Read(const OutputData &output) override;
  // Read data from all block files in file_path_(dir) for multiple tensors.
  void Read(const std::vector<OutputData> &outputs) override;
  // Read ids from block files.
  void Read(const std::vector<int> &ids, void *output, std::vector<int> *missing) override;

 private:
  // Create blocks and block metas and write input data to block files.
  void WriteBlockFiles(const std::vector<InputData> &inputs);

  // Write shardding data to one specific block file by block index and generate sha256.
  void WriteOneBlockFile(size_t block_index, const std::vector<InputData> &inputs) const;

  // Obtain the corresponding file block index according to dirty info, only need to rewrite these file blocks, and
  // dirty info needs to be sorted in ascending order.
  void TransformDirtyInfoToBlockIndices(const DirtyInfo &dirty_info, std::vector<int> *block_indices) const;

  // Load file list info of block files and block meta files in the 'file_path_' to block list and block meta list.
  bool LoadBlocksInfo();

  void ReadPages(size_t num_ids, const std::vector<int> &ids, void *pages_ptr, size_t *offsets);

  void WritePages(size_t num_ids, const std::vector<int> &ids);

  std::string BlockFilePath(size_t block_id) const;

  // The local file is composed of many block files, and each block file corresponds to a Block object in memory.
  std::vector<std::shared_ptr<Block>> block_list_;

  // Container used to store meta info for every block in member variable 'block_list_', meta info can be customized,
  // such as shard shape, shard range, field length, etc.
  std::vector<std::shared_ptr<BlockMeta>> block_meta_list_;

  // Size of ID of a one-dim value in tensor data.
  size_t id_size_;

  // Size of o one-dim value(called feature) in tensor data.
  size_t feature_size_;

  // Folder path to save all block files.
  std::string file_path_;

  // Maximum size of each block file.
  size_t max_block_length_;

  // Indicates whether block files has been created.
  bool finish_create_block_files_{false};

  // Size of a read/write page in block.
  size_t page_size_;

  size_t num_features_per_page_;
  size_t num_pages_per_block_file_;
  size_t num_features_per_block_file_;

  // Current stored feature numbers of all block files.
  size_t curr_storage_size_{0};

  std::vector<system::WriteFilePtr> feature_block_files_;

  // Buffer to save pages for read/write.
  std::vector<uint8_t> pages_buf_;

  // Buffer to save offset of feature in page for read/write.
  std::vector<size_t> offsets_buf_;

  // Map to record id in which page.
  mindspore::HashMap<int, size_t> id_to_page_loc_;

  // File System Ops handle.
  std::shared_ptr<system::FileSystem> fs_;
};
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_LOCAL_FILE_H_
