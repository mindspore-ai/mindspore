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
#include "minddata/dataset/engine/datasetops/source/yes_no_op.h"

#include <algorithm>
#include <iomanip>
#include <regex>
#include <set>

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
constexpr float kMaxShortVal = 32767.0;
constexpr char kExtension[] = ".wav";
constexpr int kStrLen = 15;  // the length of name.
#ifndef _WIN32
constexpr char kSplitSymbol[] = "/";
#else
constexpr char kSplitSymbol[] = "\\";
#endif

YesNoOp::YesNoOp(const std::string &file_dir, int32_t num_workers, int32_t queue_size,
                 std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      dataset_dir_(file_dir),
      data_schema_(std::move(data_schema)) {}

Status YesNoOp::PrepareData() {
  auto realpath = FileUtils::GetRealPath(dataset_dir_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << dataset_dir_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + dataset_dir_ + " does not exist.");
  }
  Path dir(realpath.value());
  if (dir.Exists() == false || dir.IsDirectory() == false) {
    RETURN_STATUS_UNEXPECTED("Invalid directory, " + dataset_dir_ + " does not exist or is not a directory.");
  }
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(&dir);
  RETURN_UNEXPECTED_IF_NULL(dir_itr);
  while (dir_itr->HasNext()) {
    Path file = dir_itr->Next();
    if (file.Extension() == kExtension) {
      all_wave_files_.emplace_back(file.ToString());
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!all_wave_files_.empty(), "Invalid file, no .wav files found under " + dataset_dir_);
  num_rows_ = all_wave_files_.size();
  return Status::OK();
}

void YesNoOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    ParallelOp::Print(out, show_all);
    out << "\n";
  } else {
    ParallelOp::Print(out, show_all);
    out << "\nNumber of rows: " << num_rows_ << "\nYesNo directory: " << dataset_dir_ << "\n\n";
  }
}

Status YesNoOp::Split(const std::string &line, std::vector<int32_t> *split_num) {
  RETURN_UNEXPECTED_IF_NULL(split_num);
  std::string str = line;
  int dot_pos = str.find_last_of(kSplitSymbol);
  std::string sub_line = line.substr(dot_pos + 1, kStrLen);  // (dot_pos + 1) because the index start from 0.
  std::string::size_type pos;
  std::vector<std::string> split;
  sub_line += "_";  // append to sub_line indicating the end of the string.
  uint32_t size = sub_line.size();
  for (uint32_t index = 0; index < size;) {
    pos = sub_line.find("_", index);
    if (pos != index) {
      std::string s = sub_line.substr(index, pos - index);
      split.emplace_back(s);
    }
    index = pos + 1;
  }
  try {
    for (int i = 0; i < split.size(); i++) {
      split_num->emplace_back(stoi(split[i]));
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "[Internal ERROR] Converting char to int confront with an error in function stoi: " << e.what();
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Converting char to int confront with an error in function stoi: " +
                             std::string(e.what()));
  }
  return Status::OK();
}

Status YesNoOp::LoadTensorRow(row_id_type index, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::shared_ptr<Tensor> waveform, sample_rate_scalar, label_scalar;
  int32_t sample_rate;
  std::string file_name = all_wave_files_[index];
  std::vector<int32_t> label;
  std::vector<float> waveform_vec;
  RETURN_IF_NOT_OK(Split(file_name, &label));
  RETURN_IF_NOT_OK(ReadWaveFile(file_name, &waveform_vec, &sample_rate));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(waveform_vec, &waveform));
  RETURN_IF_NOT_OK(waveform->ExpandDim(0));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(sample_rate, &sample_rate_scalar));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(label, &label_scalar));
  (*trow) = TensorRow(index, {waveform, sample_rate_scalar, label_scalar});
  trow->setPath({file_name, file_name, file_name});
  return Status::OK();
}

Status YesNoOp::CountTotalRows(int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  if (all_wave_files_.size() == 0) {
    RETURN_IF_NOT_OK(PrepareData());
  }
  *count = static_cast<int64_t>(all_wave_files_.size());
  return Status::OK();
}

Status YesNoOp::ComputeColMap() {
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
