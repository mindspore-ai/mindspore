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

#include "distributed/persistent/storage/local_file.h"

#include <dirent.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>

#include "utils/convert_utils_base.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"
#include "distributed/persistent/storage/constants.h"

namespace mindspore {
namespace distributed {
namespace storage {
void LocalFile::Write(const InputData &input, const DirtyInfo &dirty_info) {
  std::vector<InputData> inputs = {input};
  Write(inputs, dirty_info);
}

void LocalFile::Write(const std::vector<InputData> &inputs, const DirtyInfo &dirty_info) {
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "The inputs is empty";
  }

  // The block file has been created, only the blocks related to the dirty information need to be rewritten.
  if (finish_create_block_files_) {
    std::vector<int> block_indices;
    TransformDirtyInfoToBlockIndices(dirty_info, &block_indices);

    for (const auto &block_index : block_indices) {
      WriteOneBlockFile(IntToSize(block_index), inputs);
    }
    return;
  }

  // Create block files and write inputs_data to block files.
  WriteBlockFiles(inputs);
}

void LocalFile::TransformDirtyInfoToBlockIndices(const DirtyInfo &dirty_info, std::vector<int> *block_indices) const {
  MS_EXCEPTION_IF_NULL(block_indices);
  if (block_meta_list_.empty()) {
    MS_LOG(EXCEPTION) << "The block meta list is empty";
  }

  size_t block_index = 0;
  bool block_index_alread_insert_vec = false;
  auto block_meta_ptr = block_meta_list_.at(block_index);
  MS_EXCEPTION_IF_NULL(block_meta_ptr);
  int cur_lower_bound = block_meta_ptr->Get<int>(kShardRangeLowerBound);
  int cur_upper_bound = block_meta_ptr->Get<int>(kShardRangeUpperBound);

  for (const auto &dirty_value : dirty_info) {
    if (dirty_value >= cur_lower_bound && dirty_value < cur_upper_bound) {
      if (!block_index_alread_insert_vec) {
        block_index_alread_insert_vec = true;
        block_indices->push_back(block_index);
      }
      continue;
    }

    while (!(dirty_value >= cur_lower_bound && dirty_value < cur_upper_bound)) {
      if (++block_index >= block_meta_list_.size()) {
        break;
      }
      block_meta_ptr = block_meta_list_[block_index];
      MS_EXCEPTION_IF_NULL(block_meta_ptr);
      cur_lower_bound = block_meta_ptr->Get<int>(kShardRangeLowerBound);
      cur_upper_bound = block_meta_ptr->Get<int>(kShardRangeUpperBound);
    }

    if (block_index < block_meta_list_.size()) {
      block_indices->push_back(block_index);
    }
  }
}

void LocalFile::WriteBlockFiles(const std::vector<InputData> &inputs) {
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "The inputs is empty";
  }

  const std::vector<int> &shape = std::get<0>(inputs.front());
  size_t first_dim = 0;
  if (shape.size() > 0) {
    first_dim = IntToSize(shape[0]);
  }
  if (first_dim == 0) {
    MS_LOG(EXCEPTION) << "The dimension of input shape contain zero.";
  }

  size_t non_first_dims_size = std::get<2>(inputs.front()) / first_dim;
  if (non_first_dims_size == 0) {
    MS_LOG(EXCEPTION) << "The size of input tensor is zero.";
  }

  size_t tensor_num = inputs.size();
  size_t slice_size = static_cast<size_t>(
    std::floor(static_cast<float>(static_cast<float>(max_block_length_) / tensor_num) / non_first_dims_size));
  if (slice_size == 0) {
    MS_LOG(EXCEPTION) << "The slice size in block is zero.";
  }

  size_t block_num = static_cast<size_t>(std::ceil(static_cast<float>(first_dim) / slice_size));

  size_t offset = 0;
  for (size_t block_index = 0; block_index < block_num; ++block_index) {
    // Create block meta.
    std::string block_meta_file_name =
      file_path_ + "/" + kBlockMetaFilePrefix + std::to_string(block_index) + kJsonSuffix;
    auto block_meta_ptr = std::make_shared<BlockMeta>(block_meta_file_name);
    if (!block_meta_ptr->Initialize()) {
      MS_LOG(EXCEPTION) << "Initialize block meta failed, file name [" << block_meta_file_name << "]";
    }

    size_t cur_lower_bound = slice_size * block_index;
    block_meta_ptr->Insert(kShardRangeLowerBound, cur_lower_bound);
    size_t cur_upper_bound = std::min(cur_lower_bound + slice_size, first_dim);
    block_meta_ptr->Insert(kShardRangeUpperBound, cur_upper_bound);

    size_t field_length = (cur_upper_bound - cur_lower_bound) * non_first_dims_size;
    block_meta_ptr->Insert(kFieldsLength, field_length);
    block_meta_ptr->Insert(kOffset, offset);
    offset += field_length;
    block_meta_list_.push_back(block_meta_ptr);

    // Create block.
    auto block_ptr = std::make_shared<Block>(file_path_ + "/" + kBlockFilePrefix + std::to_string(block_index));
    block_ptr->set_block_meta(block_meta_ptr);
    block_list_.push_back(block_ptr);
  }

  finish_create_block_files_ = true;

  // Write inputs_data to block files and Gen Sha256 seq.
  for (size_t block_index = 0; block_index < block_num; ++block_index) {
    WriteOneBlockFile(block_index, inputs);
  }
}

void LocalFile::WriteOneBlockFile(size_t block_index, const std::vector<InputData> &inputs) const {
  const auto &block_meta_ptr = block_meta_list_.at(block_index);
  MS_EXCEPTION_IF_NULL(block_meta_ptr);
  size_t field_size = block_meta_ptr->Get<size_t>(kFieldsLength);
  size_t offset = block_meta_ptr->Get<size_t>(kOffset);
  std::vector<std::pair<const void *, size_t>> block_inputs_data;

  for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    const void *data_ptr = reinterpret_cast<const char *>(std::get<1>(inputs.at(input_index))) + offset;
    size_t data_size = field_size;
    (void)block_inputs_data.emplace_back(data_ptr, data_size);
  }

  const auto &block_ptr = block_list_.at(block_index);
  MS_EXCEPTION_IF_NULL(block_ptr);
  // Rewrite the current block file.
  if (!FileIOUtils::Write(block_ptr->block_file_name(), block_inputs_data)) {
    MS_LOG(EXCEPTION) << "Write to block file[" << block_ptr->block_file_name() << "] failed.";
  }

  ChangeFileMode(block_ptr->block_file_name(), S_IRWXU | S_IRWXG | S_IRWXO);

  // Generate sha256 hash sequence.
  block_ptr->GenSha256Seq();
}

void LocalFile::Read(const OutputData &output) {
  std::vector<OutputData> outputs = {output};
  Read(outputs);
}

void LocalFile::Read(const std::vector<OutputData> &outputs) {
  if (block_list_.empty() || block_meta_list_.empty()) {
    // Load file list info of block files and block meta files in the current folder to block list and block meta list.
    if (!LoadBlocksInfo()) {
      MS_LOG(EXCEPTION) << "LoadBlocksInfo failed";
    }
  }

  // Read all block files.
  for (size_t block_index = 0; block_index < block_list_.size(); ++block_index) {
    std::vector<std::pair<void *, size_t>> block_output_data;
    const auto &block_meta_ptr = block_meta_list_[block_index];
    MS_EXCEPTION_IF_NULL(block_meta_ptr);
    size_t field_size = block_meta_ptr->Get<size_t>(kFieldsLength);
    size_t offset = block_meta_ptr->Get<size_t>(kOffset);

    for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
      void *data_ptr = reinterpret_cast<char *>(std::get<0>(outputs[output_index])) + offset;
      size_t data_size = field_size;
      (void)block_output_data.emplace_back(data_ptr, data_size);
    }

    const auto &block_ptr = block_list_[block_index];
    MS_EXCEPTION_IF_NULL(block_ptr);
    if (!block_ptr->CheckSha256Seq()) {
      MS_LOG(EXCEPTION) << "CheckSha256 failed, file name [" << block_ptr->block_file_name() << "]";
    }

    if (!FileIOUtils::Read(block_ptr->block_file_name(), block_output_data)) {
      MS_LOG(EXCEPTION) << "Read block file failed, file name [" << block_ptr->block_file_name() << "]";
    }
  }
}

bool LocalFile::LoadBlocksInfo() {
  DIR *dir = opendir(file_path_.c_str());
  if (dir == nullptr) {
    MS_LOG(ERROR) << "The file path [" << file_path_ << "] is not exist";
    return false;
  }
  std::vector<std::string> block_file_name_list;
  std::vector<std::string> block_meta_file_name_list;
  struct dirent *entry;

  // Get file names of all block file and block meta file in the current folder.
  while ((entry = readdir(dir)) != nullptr) {
    std::string file_name = entry->d_name;
    if (file_name.length() <= JSON_SUFFIX_LENS) {
      continue;
    }

    std::string real_storage_file_path = file_path_ + "/" + file_name;
    auto suffix = file_name.substr(file_name.length() - JSON_SUFFIX_LENS);
    if (suffix == kJsonSuffix) {
      block_meta_file_name_list.push_back(real_storage_file_path);
    } else {
      block_file_name_list.push_back(real_storage_file_path);
    }
  }
  (void)closedir(dir);

  if (block_file_name_list.size() != block_meta_file_name_list.size()) {
    MS_LOG(ERROR) << "The block file number[" << block_file_name_list.size()
                  << "] is not equal to block meta file number[" << block_meta_file_name_list.size() << "]";
    return false;
  }

  sort(block_file_name_list.begin(), block_file_name_list.end());
  sort(block_meta_file_name_list.begin(), block_meta_file_name_list.end());
  for (size_t i = 0; i < block_file_name_list.size(); i++) {
    auto block_meta_ptr = std::make_shared<BlockMeta>(block_meta_file_name_list[i]);
    if (!block_meta_ptr->Initialize()) {
      MS_LOG(ERROR) << "Initialize block meta failed, file name [" << block_meta_file_name_list[i] << "]";
      return false;
    }
    block_meta_list_.push_back(block_meta_ptr);

    auto block_ptr = std::make_shared<Block>(block_file_name_list[i]);
    block_ptr->set_block_meta(block_meta_ptr);
    block_list_.push_back(block_ptr);
  }
  return true;
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
