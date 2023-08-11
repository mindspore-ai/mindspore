/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/source/libri_tts_op.h"

#include <fstream>
#include <iomanip>
#include <set>

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
const int32_t label_file_suffix_len = 10;
const char label_file_suffix[] = ".trans.tsv";
const char audio_file_suffix[] = ".wav";
const std::vector<std::string> usage_list = {"dev-clean",       "dev-other",       "test-clean",     "test-other",
                                             "train-clean-100", "train-clean-360", "train-other-500"};

LibriTTSOp::LibriTTSOp(const std::string &dataset_dir, const std::string &usage, int32_t num_workers,
                       int32_t queue_size, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      data_schema_(std::move(data_schema)) {}

Status LibriTTSOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  LibriTTSLabelTuple audio_tuple = audio_label_tuples_[row_id];
  const uint32_t rate = 24000;
  std::shared_ptr<Tensor> waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id;
  Path dir(real_path_);
  std::string file_name = audio_tuple.utterance_id + audio_file_suffix;
  Path full_dir = dir / audio_tuple.usage / std::to_string(audio_tuple.speaker_id) /
                  std::to_string(audio_tuple.chapter_id) / file_name;
  RETURN_IF_NOT_OK(ReadAudio(full_dir.ToString(), &waveform));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(rate, &sample_rate));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.original_text, &original_text));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.normalized_text, &normalized_text));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.speaker_id, &speaker_id));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.chapter_id, &chapter_id));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.utterance_id, &utterance_id));
  (*trow) = TensorRow(
    row_id, {std::move(waveform), std::move(sample_rate), std::move(original_text), std::move(normalized_text),
             std::move(speaker_id), std::move(chapter_id), std::move(utterance_id)});
  std::string label_path = audio_tuple.label_path;
  trow->setPath({full_dir.ToString(), full_dir.ToString(), label_path, label_path, label_path, label_path, label_path});
  return Status::OK();
}

void LibriTTSOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    ParallelOp::Print(out, show_all);
    out << "\n";
  } else {
    ParallelOp::Print(out, show_all);
    out << "\nNumber of rows: " << num_rows_ << "\nLibriTTS directory: " << dataset_dir_ << "\n\n";
  }
}

Status LibriTTSOp::CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  auto schema = std::make_unique<DataSchema>();

  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("waveform", DataType(DataType::DE_FLOAT32), TensorImpl::kCv, 1)));
  TensorShape scalar_rate = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("sample_rate", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar_rate)));
  TensorShape scalar_original_text = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("original_text", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &scalar_original_text)));
  TensorShape scalar_normalized_text = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("normalized_text", DataType(DataType::DE_STRING),
                                                   TensorImpl::kFlexible, 0, &scalar_normalized_text)));
  TensorShape scalar_speaker_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("speaker_id", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar_speaker_id)));
  TensorShape scalar_chapter_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("chapter_id", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar_chapter_id)));
  TensorShape scalar_utterance_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("utterance_id", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &scalar_utterance_id)));
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  auto op =
    std::make_shared<LibriTTSOp>(dir, usage, num_workers, op_connect_size, std::move(schema), std::move(sampler));
  RETURN_IF_NOT_OK(op->PrepareData());
  *count = op->audio_label_tuples_.size();
  return Status::OK();
}

Status LibriTTSOp::ComputeColMap() {
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status LibriTTSOp::ReadAudio(const std::string &audio_dir, std::shared_ptr<Tensor> *waveform) {
  RETURN_UNEXPECTED_IF_NULL(waveform);
  const int32_t kWavFileSampleRate = 24000;
  int32_t sample_rate = 0;
  std::vector<float> waveform_vec;
  RETURN_IF_NOT_OK(ReadWaveFile(audio_dir, &waveform_vec, &sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(
    sample_rate == kWavFileSampleRate,
    "Invalid file, sampling rate of LibriTTS wav file must be 24000, file path: " + audio_dir);
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(waveform_vec, waveform));
  RETURN_IF_NOT_OK((*waveform)->ExpandDim(0));
  return Status::OK();
}

Status LibriTTSOp::PrepareData() {
  auto realpath = FileUtils::GetRealPath(dataset_dir_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, LibriTTS dataset dir: " << dataset_dir_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, LibriTTS dataset dir: " + dataset_dir_ + " does not exist.");
  }
  real_path_ = realpath.value();
  Path dir(real_path_);
  if (usage_ != "all") {
    Path full_dir = dir / usage_;
    cur_usage_ = usage_;
    RETURN_IF_NOT_OK(GetPaths(&full_dir));
    RETURN_IF_NOT_OK(GetLabels());
  } else {
    for (std::string usage_iter : usage_list) {
      cur_usage_ = usage_iter;
      Path full_dir = dir / cur_usage_;
      RETURN_IF_NOT_OK(GetPaths(&full_dir));
      RETURN_IF_NOT_OK(GetLabels());
    }
  }
  num_rows_ = audio_label_tuples_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows_ > 0,
                               "Invalid data, no valid data matching the dataset API LibriTTSDataset. "
                               "Please check dataset API or file path: " +
                                 dataset_dir_ + ".");
  return Status::OK();
}

Status LibriTTSOp::GetPaths(Path *dir) {
  RETURN_UNEXPECTED_IF_NULL(dir);
  auto iter = Path::DirIterator::OpenDirectory(dir);
  if (iter == nullptr) {
    MS_LOG(WARNING) << "Invalid file path, unable to open directory: " << dir->ToString() << ".";
  } else {
    while (iter->HasNext()) {
      Path sub_dir = iter->Next();
      if (sub_dir.IsDirectory()) {
        RETURN_IF_NOT_OK(GetPaths(&sub_dir));
      } else {
        Path file_path = sub_dir;
        std::string file_name = file_path.Basename();
        int32_t length = file_name.size();
        if (length > label_file_suffix_len && file_name.substr(length - label_file_suffix_len) == label_file_suffix) {
          label_files_.push_back(sub_dir.ToString());
          return Status::OK();
        }
      }
    }
  }
  return Status::OK();
}

Status LibriTTSOp::GetLabels() {
  std::string utterance_id_body = "";
  std::string original_text_body = "";
  std::string normalized_text_body = "";
  const uint32_t base = 10;
  const uint32_t ascii_zero = 48;
  const size_t underline_exact = 3;
  for (std::string label_file : label_files_) {
    std::ifstream label_reader(label_file);
    while (getline(label_reader, utterance_id_body, '\t')) {
      getline(label_reader, original_text_body, '\t');
      getline(label_reader, normalized_text_body, '\n');
      uint32_t speaker_id = 0;
      uint32_t chapter_id = 0;
      size_t underline_num = 0;
      size_t underline_inx[4] = {0};
      for (size_t i = 0; i < utterance_id_body.size() && underline_num <= underline_exact; i++) {
        if (utterance_id_body[i] == '_') {
          underline_inx[underline_num++] = i;
        }
      }
      if (underline_num != underline_exact) {
        label_reader.close();
        RETURN_STATUS_UNEXPECTED("Invalid file, the file may not be a LibriTTS dataset file: " + label_file);
      }
      for (size_t i = 0; i < underline_inx[0]; i++) {
        speaker_id = speaker_id * base + utterance_id_body[i] - ascii_zero;
      }
      for (size_t i = underline_inx[0] + 1; i < underline_inx[1]; i++) {
        chapter_id = chapter_id * base + utterance_id_body[i] - ascii_zero;
      }
      audio_label_tuples_.push_back(
        {cur_usage_, utterance_id_body, original_text_body, normalized_text_body, speaker_id, chapter_id, label_file});
    }
    label_reader.close();
  }
  label_files_.clear();
  return Status::OK();
}
}  // namespace dataset.
}  // namespace mindspore.
