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
#include <utility>

#include "include/backend/distributed/persistent/storage/storage.h"
#include "distributed/persistent/storage/block.h"
#include "distributed/persistent/storage/file_io_utils.h"
#include "distributed/persistent/storage/constants.h"
#include "utils/system/file_system.h"

namespace mindspore {
namespace distributed {
namespace storage {
// The default maximum block length : 128MB.
constexpr size_t DEFAULT_MAX_BLOCK_LENGTH = 128 << 20;

// File type persistence storage implementation class.
template <typename KeyType = int32_t, typename ValueType = float>
class LocalFile : public StorageBase {
 public:
  explicit LocalFile(const std::map<std::string, std::string> &storage_config);
  ~LocalFile() override = default;

  // Initialize local file storage, such as creating file system handle and check data legitimacy.
  void Initialize() override;

  // Release the resource used by the local file storage.
  void Finalize() override;

  // The following two methods are override version function for Write:
  // 1. Create blocks and block metas.
  // 2. Write input data to block files and Generate sha256 sequence for every block file.
  // Write the entire blob data of tensor to the block files on disk:
  void Write(const InputData &input, const DirtyInfo &dirty_info) override;
  // Write the entire blob data composed of multiple tensors to the block files on disk:
  void Write(const std::vector<InputData> &inputs, const DirtyInfo &dirty_info) override;

  // Write key-value pairs data into local file storage.
  // Parameter[in] `keys`: The keys need to write, containing data pointer and data buffer length.
  // Parameter[in] `values`: The values corresponding to keys need to write, containing data pointer and data buffer
  // length.
  void Write(const ConstDataWithLen &keys, const ConstDataWithLen &values) override;

  // The following two methods are override version function for Read:
  // 1.Tamper proof check.
  // 2.Read all block files and merge them into contiguous memory.
  // Read data from all block files in file_path_(dir):
  void Read(const OutputData &output) override;
  // Read data from all block files in file_path_(dir) for multiple tensors.
  void Read(const std::vector<OutputData> &outputs) override;

  // Read key-value pairs' values data from local file storage.
  // Parameter[in] `keys`: The keys whose values need to read, containing data pointer and data buffer length.
  // Parameter[out] `values`: The values corresponding to keys need to read, containing data pointer and data buffer
  // length.
  void Read(const ConstDataWithLen &keys, const DataWithLen &values) override;

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

  // The local file is composed of many block files, and each block file corresponds to a Block object in memory.
  std::vector<std::shared_ptr<Block>> block_list_;

  // Container used to store meta info for every block in member variable 'block_list_', meta info can be customized,
  // such as shard shape, shard range, field length, etc.
  std::vector<std::shared_ptr<BlockMeta>> block_meta_list_;

  // Folder path to save all block files.
  std::string file_path_;

  // Maximum size of each block file.
  size_t max_block_length_;

  // Indicates whether block files has been created.
  bool finish_create_block_files_{false};

  // File system of create or delete file.
  std::shared_ptr<system::FileSystem> fs_;

  // All write-read helper for all block files.
  std::vector<system::WriteFilePtr> block_files_;

  // For key-value data storage, the value size (such as the number of floating values)for one key-value pair.
  size_t element_size_;

  // The number of elements that a block can hold.
  size_t block_size_{1};

  // Record all key-value pairs' positions in block files, You can query which block file the value corresponding to a
  // key is stored in, and the offset location of the block file.
  // Data structure for this map: key -> pair{block index, offset in block}, offset in block is measured in bytes from
  // the beginning of this file.
  HashMap<KeyType, std::pair<size_t, size_t>> keys_to_locations_;

  // Record latest used position in latest created block file.
  size_t current_offset_in_block_{0};
};
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_LOCAL_FILE_H_
