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
#include "minddata/dataset/engine/datasetops/source/lj_speech_op.h"

#include <fstream>
#include <iomanip>
#include <utility>

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/path.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
LJSpeechOp::LJSpeechOp(const std::string &file_dir, int32_t num_workers, int32_t queue_size,
                       std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(file_dir),
      data_schema_(std::move(data_schema)) {}

Status LJSpeechOp::PrepareData() {
  auto real_path = FileUtils::GetRealPath(folder_path_.c_str());
  if (!real_path.has_value()) {
    RETURN_STATUS_UNEXPECTED("Invalid file path, LJSpeech Dataset folder: " + folder_path_ + " does not exist.");
  }
  Path root_folder(real_path.value());
  Path metadata_file_path = root_folder / "metadata.csv";
  CHECK_FAIL_RETURN_UNEXPECTED(metadata_file_path.Exists() && !metadata_file_path.IsDirectory(),
                               "Invalid file, failed to find LJSpeech metadata file: " + metadata_file_path.ToString());
  std::ifstream csv_reader(metadata_file_path.ToString());
  CHECK_FAIL_RETURN_UNEXPECTED(csv_reader.is_open(),
                               "Invalid file, failed to open LJSpeech metadata file: " + metadata_file_path.ToString() +
                                 ", make sure file not damaged or permission denied.");
  std::string line = "";
  while (getline(csv_reader, line)) {
    int32_t last_pos = 0, curr_pos = 0;
    std::vector<std::string> row;
    while (curr_pos < line.size()) {
      if (line[curr_pos] == '|') {
        row.emplace_back(line.substr(last_pos, curr_pos - last_pos));
        last_pos = curr_pos + 1;
      }
      ++curr_pos;
    }
    row.emplace_back(line.substr(last_pos, curr_pos - last_pos));
    meta_info_list_.emplace_back(row);
  }
  if (meta_info_list_.empty()) {
    csv_reader.close();
    RETURN_STATUS_UNEXPECTED("Reading failed, unable to read valid data from the LJSpeech metadata file: " +
                             metadata_file_path.ToString() + ".");
  }
  num_rows_ = meta_info_list_.size();
  csv_reader.close();
  return Status::OK();
}

// Load 1 TensorRow (waveform, sample_rate, transcription, normalized_transcription).
// 1 function call produces 1 TensorTow
Status LJSpeechOp::LoadTensorRow(row_id_type index, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  int32_t num_items = meta_info_list_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(index >= 0 && index < num_items, "[Internal ERROR] The input index is out of range.");
  std::shared_ptr<Tensor> waveform;
  std::shared_ptr<Tensor> sample_rate_scalar;
  std::shared_ptr<Tensor> transcription, normalized_transcription;
  std::string file_name_pref = meta_info_list_[index][0], transcription_str = meta_info_list_[index][1],
              normalized_transcription_str = meta_info_list_[index][2];
  int32_t sample_rate;
  std::string file_name = file_name_pref + ".wav";
  Path root_folder(folder_path_);
  Path wav_file_path = root_folder / "wavs" / file_name;
  Path metadata_file_path = root_folder / "metadata.csv";
  std::vector<float> waveform_vec;
  RETURN_IF_NOT_OK(ReadWaveFile(wav_file_path.ToString(), &waveform_vec, &sample_rate));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(waveform_vec, &waveform));
  RETURN_IF_NOT_OK(waveform->ExpandDim(0));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(sample_rate, &sample_rate_scalar));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(transcription_str, &transcription));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(normalized_transcription_str, &normalized_transcription));
  (*trow) = TensorRow(index, {waveform, sample_rate_scalar, transcription, normalized_transcription});
  // Add file path info
  trow->setPath({wav_file_path.ToString(), metadata_file_path.ToString(), metadata_file_path.ToString(),
                 metadata_file_path.ToString()});
  return Status::OK();
}

void LJSpeechOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows: " << num_rows_ << "\nLJSpeech directory: " << folder_path_ << "\n\n";
  }
}

Status LJSpeechOp::CountTotalRows(const std::string &dir, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  auto real_path = FileUtils::GetRealPath(dir.c_str());
  if (!real_path.has_value()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, " + dir + " does not exist.");
  }
  Path root_folder(real_path.value());
  Path metadata_file_path = root_folder / "metadata.csv";
  CHECK_FAIL_RETURN_UNEXPECTED(metadata_file_path.Exists() && !metadata_file_path.IsDirectory(),
                               "Invalid file, failed to find metadata file: " + metadata_file_path.ToString());
  std::ifstream csv_reader(metadata_file_path.ToString());
  CHECK_FAIL_RETURN_UNEXPECTED(csv_reader.is_open(),
                               "Invalid file, failed to open metadata file: " + metadata_file_path.ToString() +
                                 ", make sure file not damaged or permission denied.");
  std::string line = "";
  int64_t cnt = 0;
  while (getline(csv_reader, line)) {
    ++cnt;
  }
  *count = cnt;
  csv_reader.close();
  return Status::OK();
}

Status LJSpeechOp::ComputeColMap() {
  // set the column name map (base class field)
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
