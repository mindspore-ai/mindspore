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
#include "minddata/dataset/engine/datasetops/source/tedlium_op.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
TedliumOp::TedliumOp(const std::string &dataset_dir, const std::string &release, const std::string &usage,
                     const std::string &extensions, int32_t num_parallel_workers,
                     std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler, int32_t queue_size)
    : MappableLeafOp(num_parallel_workers, queue_size, std::move(sampler)),
      dataset_dir_(dataset_dir),
      release_(release),
      usage_(usage),
      extensions_(extensions),
      data_schema_(std::move(data_schema)),
      audio_files_({}),
      usage_list_({}) {}

void TedliumOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nNumber of rows: " << num_rows_ << "\nTedliumOp directory: " << dataset_dir_;
  }
}

Status TedliumOp::PrepareData() {
  auto real_path = FileUtils::GetRealPath(dataset_dir_.c_str());
  if (!real_path.has_value()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, get real path failed, path=" + dataset_dir_);
  }
  Path root_folder(real_path.value());

  if (release_ == "release1" || release_ == "release2") {
    if (usage_ == "train" || usage_ == "test" || usage_ == "dev") {
      usage_list_.push_back(usage_);
    } else if (usage_ == "all") {
      usage_list_ = {"train", "test", "dev"};
    } else {
      RETURN_STATUS_UNEXPECTED(
        "Invalid parameter, usage should be \"train\", \"test\", \"dev\" or \"all\" when "
        "specify \"release1\" or \"release2\" , got " +
        usage_);
    }
    for (int32_t i = 0; i < usage_list_.size(); ++i) {
      Path stm_folder = root_folder / usage_list_[i] / "stm";
      RETURN_IF_NOT_OK(ReadStmFolderRows(stm_folder, usage_list_[i]));
    }
  } else if (release_ == "release3") {
    if (usage_ == "all") {
      Path stm_folder = root_folder / "data" / "stm";
      RETURN_IF_NOT_OK(ReadStmFolderRows(stm_folder, "data"));
    } else {
      RETURN_STATUS_UNEXPECTED("Invalid parameter, usage should be \"all\" when specify \"release3\" , got " + usage_);
    }
  }
  std::sort(audio_files_.begin(), audio_files_.end());
  num_rows_ = audio_files_.size();
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API TedliumDataset. Please check file path or dataset API.");
  }
  return Status::OK();
}

Status TedliumOp::ReadStmFolderRows(const Path &stm_folder, const std::string &release_usage) {
  Path dir(stm_folder);
  std::shared_ptr<Path::DirIterator> dirItr = Path::DirIterator::OpenDirectory(&dir);
  if (!dir.Exists() || dirItr == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open folder: " + dir.ToString());
  }
  MS_LOG(DEBUG) << "Tedlium " + release_ + " stm folder Path found: " << dir << ".";
  while (dirItr->HasNext()) {
    Path file = dirItr->Next();
    if (file.Extension() == ".stm") {
      std::ifstream handle(file.ToString());
      if (!handle.is_open()) {
        RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + file.ToString());
      }
      std::string line;
      int32_t numline = 0;
      while (getline(handle, line)) {
        std::string filename = line.substr(0, line.find(" "));
        std::stringstream ss;
        ss << numline;
        audio_files_.push_back({ss.str(), filename, release_usage});
        ++numline;
      }
      handle.close();
    }
  }
  return Status::OK();
}

Status TedliumOp::ReadStm(const Path &file_stm_path, int32_t row_line, std::string *talk_id, std::string *speaker_id,
                          std::string *start_time, std::string *end_time, std::string *identifier,
                          std::string *transcript) {
  std::ifstream handle(file_stm_path.ToString().c_str());
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, get real path failed, path=" + file_stm_path.ToString());
  }
  std::string line;
  int32_t i = 0;
  while (i <= row_line && getline(handle, line)) {
    ++i;
  }
  handle.close();
  std::vector<std::string> temp;
  i = 0;
  const int32_t data_stm_number = 7;
  // There are seven pieces of data in each row, which need to be read out and stored
  // with a space as a separator.
  // Talk_id, _, speaker_id, start_time, end_time, identifier, transcript.
  // "_" is the data we don't need.
  while (i < data_stm_number - 1) {
    std::string s = line.substr(0, line.find(" "));
    temp.push_back(s);
    line.erase(0, line.find(" ") + 1);  // to delete space, so use s.find(" ") + 1.
    ++i;
  }
  temp.push_back(line);
  if (temp.size() != data_stm_number) {
    RETURN_STATUS_UNEXPECTED("Invalid data, stm data was broken.");
  }

  const int32_t talk_id_num = 0, speaker_id_num = 2, start_time_num = 3, end_time_num = 4, identifier_num = 5,
                transcript_num = 6;
  *talk_id = temp[talk_id_num];
  // temp[1] is "_", which is the data we don't need.
  *speaker_id = temp[speaker_id_num];
  *start_time = temp[start_time_num];
  *end_time = temp[end_time_num];
  *identifier = temp[identifier_num];
  *transcript = temp[transcript_num];

  return Status::OK();
}

Status TedliumOp::ReadSph(const Path &file_sph_path, double start_time, double end_time, int32_t *sample_rate,
                          std::vector<float> *result) {
  std::ifstream handle(file_sph_path.ToString().c_str(), std::ios::in | std::ios::binary);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + file_sph_path.ToString());
  }

  char head[1024];
  handle.read(head, sizeof(head));
  CHECK_FAIL_RETURN_UNEXPECTED(!handle.fail(),
                               "Invalid data, failed to read head part from sph file: " + file_sph_path.ToString() +
                                 ", re-download dataset(make sure the data is true).");
  std::vector<std::string> vec;
  for (int32_t i = 0, j = 0; i < strlen(head); ++i) {
    if (head[i] == '\n' || head[i] == ' ') {
      while (head[i + 1] == ' ') {
        i++;
      }
      std::string strTemp(head + j, i - j);
      vec.push_back(strTemp);
      j = i + 1;
    }
  }
  const int32_t dataToBytes = 2;
  for (int32_t i = 0; i < vec.size(); ++i) {
    if (vec[i] == "sample_rate") {
      *sample_rate = atoi(vec[i + dataToBytes].c_str());
    }
  }

  int32_t start = static_cast<int32_t>(start_time * (*sample_rate));
  int32_t end = static_cast<int32_t>(end_time * (*sample_rate));
  const int32_t size = (end - start);
  std::vector<char> temp(size * dataToBytes);
  handle.seekg(start, std::ios::beg);
  int32_t j = 0;
  char c;
  while (j < size * dataToBytes) {
    handle.read(&c, 1);
    CHECK_FAIL_RETURN_UNEXPECTED(!handle.fail(),
                                 "Invalid data, failed to read data part from sph file: " + file_sph_path.ToString() +
                                   ", re-download dataset(make sure the data is true).");
    temp.push_back(c);
    ++j;
  }

  const float kMaxVal = 32767.0;
  for (int32_t i = 0; i < size; ++i) {
    char bh = temp[2 * i];
    char bl = temp[2 * i + 1];
    // SPH audio files is big-endian, so we should convert the two bytes of data into int16_t based
    // on the high 8 bits and the low 8 bits.
    int16_t s = static_cast<int16_t>(((bh & 0x00FF) << 8) | (bl & 0x00FF));
    // Data normalization: Convert the data from the interval [-32768,32767] to the interval [-1,1].
    double t = s / kMaxVal;
    (*result).push_back(t);
  }
  handle.close();

  return Status::OK();
}

Status TedliumOp::LoadTensorRow(row_id_type row_id, TensorRow *row) {
  int32_t row_line = atoi(audio_files_[row_id][0].c_str());
  std::string file_name = audio_files_[row_id][1];
  std::string file_usage_or3_none_ = audio_files_[row_id][2];
  Path dir_path(dataset_dir_);
  Path file_stm_path = dir_path / file_usage_or3_none_ / "stm" / (file_name + ".stm");
  Path file_sph_path = dir_path / file_usage_or3_none_ / "sph" / (file_name + extensions_);
  std::string talk_id, speaker_id, start_time, end_time, identifier, transcript;
  std::vector<float> result;
  int32_t sample_rate;
  RETURN_IF_NOT_OK(
    ReadStm(file_stm_path, row_line, &talk_id, &speaker_id, &start_time, &end_time, &identifier, &transcript));
  RETURN_IF_NOT_OK(ReadSph(file_sph_path, atof(start_time.c_str()), atof(end_time.c_str()), &sample_rate, &result));

  std::shared_ptr<Tensor> sample_rate_tensor, talk_id_tensor, speaker_id_tensor, identifier_tensor, transcript_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(sample_rate, &sample_rate_tensor));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(talk_id, &talk_id_tensor));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(speaker_id, &speaker_id_tensor));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(identifier, &identifier_tensor));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(transcript, &transcript_tensor));

  std::shared_ptr<Tensor> audio_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(result, &audio_tensor));
  RETURN_IF_NOT_OK(audio_tensor->ExpandDim(0));
  (*row) = TensorRow(row_id, {audio_tensor, sample_rate_tensor, transcript_tensor, talk_id_tensor, speaker_id_tensor,
                              identifier_tensor});
  row->setPath({file_sph_path.ToString(), file_sph_path.ToString(), file_stm_path.ToString(), file_stm_path.ToString(),
                file_stm_path.ToString(), file_stm_path.ToString()});

  return Status::OK();
}

Status TedliumOp::CountTotalRows(const std::string &dataset_dir, const std::string &release, const std::string &usage,
                                 const std::string &extensions, int64_t *count) {
  // the logic of counting the number of samples is copied from PrepareData()
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto new_sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);

  // build a new unique schema object
  auto new_schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(
    new_schema->AddColumn(ColDescriptor("waveform", DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
  TensorShape sample_rate_scalar = TensorShape::CreateScalar();
  TensorShape trans_scalar = TensorShape::CreateScalar();
  TensorShape talk_id_scalar = TensorShape::CreateScalar();
  TensorShape speaker_id_scalar = TensorShape::CreateScalar();
  TensorShape identi_scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(new_schema->AddColumn(
    ColDescriptor("sample_rate", DataType(DataType::DE_INT32), TensorImpl::kFlexible, 0, &sample_rate_scalar)));
  RETURN_IF_NOT_OK(new_schema->AddColumn(
    ColDescriptor("transcript", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &trans_scalar)));
  RETURN_IF_NOT_OK(new_schema->AddColumn(
    ColDescriptor("talk_id", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &talk_id_scalar)));
  RETURN_IF_NOT_OK(new_schema->AddColumn(
    ColDescriptor("speaker_id", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &speaker_id_scalar)));
  RETURN_IF_NOT_OK(new_schema->AddColumn(
    ColDescriptor("identifier", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &identi_scalar)));

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  std::shared_ptr<TedliumOp> op =
    std::make_shared<TedliumOp>(dataset_dir, release, usage, extensions, num_workers, std::move(new_schema),
                                std::move(new_sampler), op_connect_size);
  RETURN_IF_NOT_OK(op->PrepareData());
  *count = static_cast<int64_t>(op->audio_files_.size());
  return Status::OK();
}

Status TedliumOp::ComputeColMap() {
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
