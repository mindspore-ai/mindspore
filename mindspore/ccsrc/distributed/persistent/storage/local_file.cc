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
#include "securec/include/securec.h"
#include "include/common/utils/utils.h"
#include "distributed/persistent/storage/constants.h"

namespace mindspore {
namespace distributed {
namespace storage {
std::string GetBlockFileName(size_t block_id) {
  const std::string name_no_leading_zero = std::to_string(block_id);
  if (name_no_leading_zero.size() > kBlockFileNameLens) {
    MS_LOG(EXCEPTION) << "Not a right block id: " << name_no_leading_zero;
  }
  return std::string(kBlockFileNameLens - name_no_leading_zero.size(), '0') + name_no_leading_zero;
}

size_t GetBlockId(const std::string &block_file_name) {
  size_t pos = 0;
  const size_t block_id = LongToSize(std::stoull(block_file_name, &pos, 10));
  if (pos != kBlockFileNameLens) {
    MS_LOG(EXCEPTION) << "Not a right block file name: " << block_file_name;
  }
  return block_id;
}

void LocalFile::Initialize() {
  MS_EXCEPTION_IF_ZERO("feature_size_", feature_size_);
  MS_EXCEPTION_IF_ZERO("page_size_", page_size_);

  num_features_per_page_ = page_size_ / feature_size_;
  num_pages_per_block_file_ = DEFAULT_BLOCK_FILE_SIZE / page_size_;
  num_features_per_block_file_ = num_features_per_page_ * num_pages_per_block_file_;

  MS_LOG(DEBUG) << "Local File meta, id_size: " << id_size_ << ", feature_size: " << feature_size_
                << ", page_size: " << page_size_ << ", num_features_per_page: " << num_features_per_page_
                << ", num_pages_per_block_file: " << num_pages_per_block_file_
                << ", num_features_per_block_file: " << num_features_per_block_file_;

  fs_ = system::Env::GetFileSystem();
}

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

void LocalFile::Read(size_t ids_num, const int32_t *ids, void *output, size_t *miss_num, size_t *miss_indices) {
  MS_EXCEPTION_IF_NULL(miss_indices);

  offsets_buf_.resize(ids_num);
  pages_buf_.resize(ids_num * page_size_);
  void *pages_ptr = pages_buf_.data();

  ReadPages(ids_num, ids, pages_ptr, offsets_buf_.data());

  size_t miss_count = 0;
  for (uint32_t i = 0; i < ids_num; ++i) {
    if (offsets_buf_.at(i) == page_size_) {
      miss_indices[miss_count] = i;
      miss_count++;
      continue;
    }
    auto ret = memcpy_s(AddressOffset(output, i * feature_size_), feature_size_,
                        AddressOffset(pages_ptr, (i * page_size_) + offsets_buf_[i]), feature_size_);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "Failed to copy output when read block, ret = " << ret;
    }
  }

  *miss_num = miss_count;
}

void LocalFile::ReadPages(size_t ids_num, const int32_t *ids, void *pages_ptr, size_t *offsets) {
  MS_EXCEPTION_IF_NULL(pages_ptr);
  MS_EXCEPTION_IF_NULL(offsets);
  MS_EXCEPTION_IF_ZERO("num_features_per_page_", num_features_per_page_);
  MS_EXCEPTION_IF_ZERO("num_pages_per_block_file_", num_pages_per_block_file_);

  for (size_t i = 0; i < ids_num; ++i) {
    const int32_t id = ids[i];
    auto it = id_to_page_loc_.find(id);
    if (it == id_to_page_loc_.end()) {
      offsets[i] = page_size_;
    } else {
      const size_t id_index = it->second;
      const size_t page_id = id_index / num_features_per_page_;
      const size_t id_in_page = id_index - page_id * num_features_per_page_;
      const size_t offset_in_page = id_in_page * feature_size_;
      const size_t block_file_id = page_id / num_pages_per_block_file_;
      const size_t page_in_block_file = page_id - block_file_id * num_pages_per_block_file_;
      const size_t block_file_offset = page_in_block_file * page_size_;
      system::WriteFilePtr file = feature_block_files_.at(block_file_id);
      offsets[i] = offset_in_page;
      file->PRead(AddressOffset(pages_ptr, i * page_size_), page_size_, block_file_offset);
    }
  }
}

void LocalFile::Write(const void *input, size_t ids_num, const int32_t *ids) {
  MS_EXCEPTION_IF_NULL(input);

  pages_buf_.resize(ids_num * page_size_);

  // Copy data at input to pages buf, page by page.
  for (size_t i = 0; i < ids_num; i += num_features_per_page_) {
    const size_t page_id = i / num_features_per_page_;
    const size_t copy_size = (ids_num - i) < num_features_per_page_ ? (ids_num - i) * feature_size_ : page_size_;
    auto ret = memcpy_s(AddressOffset(pages_buf_.data(), page_id * page_size_), copy_size,
                        AddressOffset(const_cast<void *>(input), i * feature_size_), copy_size);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "Failed to copy input when write block, ret = " << ret;
    }
  }
  WritePages(ids_num, ids);
}

// This function write page buf to a block file, a block have many pages, a page have many feature, a feature is a
// one-dim tensor. Block file is append only, so need use curr_storage_size_ to record the write pos at block file.
void LocalFile::WritePages(size_t ids_num, const int32_t *ids) {
  MS_EXCEPTION_IF_ZERO("num_features_per_page_", num_features_per_page_);
  MS_EXCEPTION_IF_ZERO("num_pages_per_block_file_", num_pages_per_block_file_);

  const void *pages_ptr = pages_buf_.data();

  // 1. Calculate how many pages will be written, we need aligned by features per page.
  const size_t num_pages = RoundUp(ids_num, num_features_per_page_) / num_features_per_page_;
  const size_t num_padded_ids = num_pages * num_features_per_page_;
  const size_t start_index = curr_storage_size_;
  curr_storage_size_ += num_padded_ids;
  if (start_index % num_features_per_page_ != 0) {
    MS_LOG(EXCEPTION) << "Not right start index when persistent to local file.";
  }
  const size_t start_page_id = start_index / num_features_per_page_;
  size_t written_pages = 0;

  while (written_pages < num_pages) {
    // 2. Then find the newest block file to write. Open exist block file or create new one if full.
    const size_t batch_start_page_id = start_page_id + written_pages;
    const size_t batch_block_file_id = batch_start_page_id / num_pages_per_block_file_;
    if (batch_block_file_id == feature_block_files_.size()) {
      feature_block_files_.emplace_back(fs_->CreateWriteFile(BlockFilePath(batch_block_file_id), "wb+"));
    } else {
      if (batch_block_file_id > feature_block_files_.size()) {
        MS_LOG(EXCEPTION) << "WRONG batch block id: " << batch_block_file_id
                          << " , bigger then block file size: " << feature_block_files_.size();
      }
    }
    system::WriteFilePtr feature_block_file = feature_block_files_.at(batch_block_file_id);

    // 3. Find the start write pos at the block file, write the pages to block file at right pos.
    const size_t page_id_in_block = batch_start_page_id - batch_block_file_id * num_pages_per_block_file_;
    const size_t pages_to_write =
      std::min(num_pages - written_pages, (batch_block_file_id + 1) * num_pages_per_block_file_ - batch_start_page_id);
    const size_t features_bytes = pages_to_write * page_size_;
    const size_t features_offset_in_file = page_id_in_block * page_size_;
    if (!feature_block_file->Trunc(features_offset_in_file + features_bytes)) {
      MS_LOG(EXCEPTION) << "Failed to trunc block file!";
    }
    if (!feature_block_file->PWrite(AddressOffset(const_cast<void *>(pages_ptr), written_pages * page_size_),
                                    features_bytes, features_offset_in_file)) {
      MS_LOG(EXCEPTION) << "Failed to write feature pages to block file!";
    }

    written_pages += pages_to_write;
  }

  // 4. Record the index of page in block file of a feature.
  for (size_t i = 0; i < ids_num; ++i) {
    id_to_page_loc_[ids[i]] = start_index + i;
  }
}

std::string LocalFile::BlockFilePath(size_t block_id) const {
  return file_path_ + "/" + kBlockFilePrefix + GetBlockFileName(block_id);
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
