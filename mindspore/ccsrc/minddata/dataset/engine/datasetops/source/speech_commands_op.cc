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
#include "minddata/dataset/engine/datasetops/source/speech_commands_op.h"

#include <fstream>
#include <iomanip>
#include <regex>

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
constexpr char kTestFiles[] = "testing_list.txt";
constexpr char kValFiles[] = "validation_list.txt";
constexpr char kExtension[] = ".wav";
#ifndef _WIN32
constexpr char kSplitSymbol[] = "/";
#else
constexpr char kSplitSymbol[] = "\\";
#endif

SpeechCommandsOp::SpeechCommandsOp(const std::string &dataset_dir, const std::string &usage, int32_t num_workers,
                                   int32_t queue_size, std::unique_ptr<DataSchema> data_schema,
                                   std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      data_schema_(std::move(data_schema)) {}

Status SpeechCommandsOp::PrepareData() {
  // Get file lists according to usage.
  // When usage == "train", need to get all filenames then subtract files of usage: "test" and "valid".
  std::set<std::string> selected_files;
  auto real_dataset_dir = FileUtils::GetRealPath(dataset_dir_.c_str());
  if (!real_dataset_dir.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << dataset_dir_;
    RETURN_STATUS_UNEXPECTED("Get real path failed, path=" + dataset_dir_);
  }
  std::string real_path = real_dataset_dir.value();
  if (usage_ == "all") {
    RETURN_IF_NOT_OK(WalkAllFiles(real_path));
    selected_files = all_wave_files;
  } else if (usage_ == "test" || usage_ == "valid") {
    RETURN_IF_NOT_OK(ParseFileList(real_path, usage_));
    selected_files = loaded_names;
  } else {
    RETURN_IF_NOT_OK(WalkAllFiles(real_path));
    RETURN_IF_NOT_OK(ParseFileList(real_path, "test"));
    RETURN_IF_NOT_OK(ParseFileList(real_path, "valid"));
    set_difference(all_wave_files.begin(), all_wave_files.end(), loaded_names.begin(), loaded_names.end(),
                   inserter(selected_files, selected_files.begin()));
  }
  selected_files_vec.assign(selected_files.begin(), selected_files.end());
  num_rows_ = selected_files_vec.size();
  return Status::OK();
}

Status SpeechCommandsOp::ParseFileList(const std::string &pf_path, const std::string &pf_usage) {
  std::string line;
  std::string file_list = (pf_usage == "test" ? kTestFiles : kValFiles);
  Path path(pf_path);
  std::string list_path = (Path(pf_path) / Path(file_list)).ToString();
  std::ifstream file_reader(list_path);
  while (getline(file_reader, line)) {
    Path file_path(path / line);
    loaded_names.insert(file_path.ToString());
  }
  file_reader.close();
  return Status::OK();
}

Status SpeechCommandsOp::WalkAllFiles(const std::string &walk_path) {
  Path dir(walk_path);
  if (dir.IsDirectory() == false) {
    RETURN_STATUS_UNEXPECTED("Invalid parameter, no folder found in: " + walk_path);
  }
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(&dir);
  RETURN_UNEXPECTED_IF_NULL(dir_itr);
  std::vector<std::string> folder_names;
  while (dir_itr->HasNext()) {
    Path sub_dir = dir_itr->Next();
    if (sub_dir.IsDirectory() && (sub_dir.ToString().find("_background_noise_") == std::string::npos)) {
      folder_names.emplace_back(sub_dir.ToString());
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!folder_names.empty(), "Invalid file, failed to open directory: " + dir.ToString());
  for (int i = 0; i < folder_names.size(); i++) {
    Path folder_path(folder_names[i]);
    if (folder_path.IsDirectory()) {
      auto folder_it = Path::DirIterator::OpenDirectory(&folder_path);
      while (folder_it->HasNext()) {
        Path file = folder_it->Next();
        if (file.Extension() == kExtension) {
          all_wave_files.insert(file.ToString());
        }
      }
    } else {
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to open directory: " + folder_path.ToString());
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!all_wave_files.empty(), "Invalid file, no .wav files found under " + dataset_dir_);
  return Status::OK();
}

Status SpeechCommandsOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::string file_name = selected_files_vec[row_id];
  std::shared_ptr<Tensor> waveform, sample_rate_scalar, label_scalar, speaker_id_scalar, utterance_number_scalar;
  std::string label, speaker_id;
  int32_t utterance_number, sample_rate;
  std::vector<float> waveform_vec;
  RETURN_IF_NOT_OK(ReadWaveFile(file_name, &waveform_vec, &sample_rate));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(waveform_vec, &waveform));
  RETURN_IF_NOT_OK(waveform->ExpandDim(0));
  RETURN_IF_NOT_OK(GetFileInfo(file_name, &label, &speaker_id, &utterance_number));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(sample_rate, &sample_rate_scalar));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(label, &label_scalar));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(speaker_id, &speaker_id_scalar));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(utterance_number, &utterance_number_scalar));
  (*trow) = TensorRow(row_id, {waveform, sample_rate_scalar, label_scalar, speaker_id_scalar, utterance_number_scalar});
  trow->setPath({file_name, file_name, file_name, file_name, file_name});
  return Status::OK();
}

void SpeechCommandsOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying and common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show and custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows: " << num_rows_ << "\nSpeechCommands directory: " << dataset_dir_ << "\n\n";
  }
}

Status SpeechCommandsOp::ComputeColMap() {
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status SpeechCommandsOp::GetFileInfo(const std::string &file_path, std::string *label, std::string *speaker_id,
                                     int32_t *utterance_number) {
  // Using regex to get wave infos from filename.
  RETURN_UNEXPECTED_IF_NULL(label);
  RETURN_UNEXPECTED_IF_NULL(speaker_id);
  RETURN_UNEXPECTED_IF_NULL(utterance_number);
  int32_t split_index = 0;
  split_index = file_path.find_last_of(kSplitSymbol);
  std::string label_string = file_path.substr(0, split_index);
  *label = label_string.substr(label_string.find_last_of(kSplitSymbol) + 1);  // plus "1" for index start from 0.
  std::string filename = file_path.substr(split_index + 1);
  std::smatch result;
  {
    std::unique_lock<std::mutex> _lock(mux_);
    regex_match(filename, result, std::regex("(.*)_nohash_(\\d+)\\.wav"));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!(result[0] == "" || result[1] == ""),
                               "Invalid file name, failed to get file info: " + filename);
  *speaker_id = result[1];
  std::string utt_id = result[2];
  *utterance_number = atoi(utt_id.c_str());
  return Status::OK();
}

Status SpeechCommandsOp::CountTotalRows(int64_t *num_rows) {
  RETURN_UNEXPECTED_IF_NULL(num_rows);
  if (all_wave_files.size() == 0) {
    auto real_path = FileUtils::GetRealPath(dataset_dir_.c_str());
    if (!real_path.has_value()) {
      MS_LOG(ERROR) << "Get real path failed, path=" << dataset_dir_;
      RETURN_STATUS_UNEXPECTED("Get real path failed, path=" + dataset_dir_);
    }
    RETURN_IF_NOT_OK(WalkAllFiles(real_path.value()));
  }
  (*num_rows) = static_cast<int64_t>(all_wave_files.size());
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
