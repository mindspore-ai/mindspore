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

#include "minddata/dataset/engine/datasetops/source/cmu_arctic_op.h"

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
const char kDataDirectory[] = "wav";
const char kLabelDirectory[] = "etc";
const char kLabelFileName[] = "txt.done.data";
const char kDataFilePrefix[] = "cmu_us_";
const char kDataFileSuffix[] = "_arctic";

CMUArcticOp::CMUArcticOp(const std::string &dataset_dir, const std::string &name, int32_t num_workers,
                         int32_t queue_size, std::unique_ptr<DataSchema> data_schema,
                         std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(dataset_dir),
      name_(name),
      data_schema_(std::move(data_schema)) {}

Status CMUArcticOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  const uint32_t sample_rate = 16000;
  const std::string wav_suffix = ".wav";
  size_t pos = label_pairs_[row_id].first.find_last_of('_');
  CHECK_FAIL_RETURN_UNEXPECTED(
    pos != std::string::npos && pos + 1 < label_pairs_[row_id].first.size(),
    "Invalid utterance id, please check if it is in valid format: " + label_pairs_[row_id].first);
  std::string utterance_id_t = label_pairs_[row_id].first.substr(pos + 1);
  std::string full_name_path = kDataFilePrefix + name_ + kDataFileSuffix;
  std::string file_name = label_pairs_[row_id].first + wav_suffix;
  Path root_folder(real_path_);
  Path wav_file_path = root_folder / full_name_path / kDataDirectory / file_name;
  std::shared_ptr<Tensor> waveform, rate, transcript, utterance_id;
  RETURN_IF_NOT_OK(ReadAudio(wav_file_path.ToString(), &waveform));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(sample_rate, &rate));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(label_pairs_[row_id].second, &transcript));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(utterance_id_t, &utterance_id));
  (*trow) = TensorRow(row_id, {std::move(waveform), std::move(rate), std::move(transcript), std::move(utterance_id)});
  Path label_dir = root_folder / full_name_path / kLabelDirectory / kLabelFileName;
  trow->setPath({wav_file_path.ToString(), wav_file_path.ToString(), label_dir.ToString(), label_dir.ToString()});
  return Status::OK();
}

void CMUArcticOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    ParallelOp::Print(out, show_all);
    out << "\n";
  } else {
    ParallelOp::Print(out, show_all);
    out << "\nNumber of rows: " << num_rows_ << "\nCMUArctic directory: " << folder_path_ << "\n\n";
  }
}

Status CMUArcticOp::CountTotalRows(const std::string &dir, const std::string &name, int64_t *count) {
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
  TensorShape scalar_utterance = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("transcript", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &scalar_utterance)));
  TensorShape scalar_utterance_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(
    ColDescriptor("utterance_id", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &scalar_utterance_id)));
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();

  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  auto op =
    std::make_shared<CMUArcticOp>(dir, name, num_workers, op_connect_size, std::move(schema), std::move(sampler));
  RETURN_IF_NOT_OK(op->PrepareData());
  *count = op->label_pairs_.size();
  return Status::OK();
}

Status CMUArcticOp::ComputeColMap() {
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status CMUArcticOp::ReadAudio(const std::string &audio_dir, std::shared_ptr<Tensor> *waveform) {
  RETURN_UNEXPECTED_IF_NULL(waveform);
  const int32_t kWavFileSampleRate = 16000;
  int32_t sample_rate = 0;
  std::vector<float> waveform_vec;
  RETURN_IF_NOT_OK(ReadWaveFile(audio_dir, &waveform_vec, &sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(
    sample_rate == kWavFileSampleRate,
    "Invalid file, sampling rate of CMUArctic wav file must be 16000, file path: " + audio_dir);
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(waveform_vec, waveform));
  RETURN_IF_NOT_OK((*waveform)->ExpandDim(0));
  return Status::OK();
}

Status CMUArcticOp::PrepareData() {
  auto realpath = FileUtils::GetRealPath(folder_path_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, CMUArctic Dataset dir: " << folder_path_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, CMUArctic Dataset dir: " + folder_path_ + " does not exist.");
  }
  real_path_ = realpath.value();
  Path dir(real_path_);
  std::string full_name_path = kDataFilePrefix + name_ + kDataFileSuffix;
  Path label_dir = dir / full_name_path / kLabelDirectory / kLabelFileName;
  CHECK_FAIL_RETURN_UNEXPECTED(label_dir.Exists() && !label_dir.IsDirectory(),
                               "Invalid file, failed to find label file: " + label_dir.ToString());
  std::ifstream label_reader(label_dir.ToString());
  CHECK_FAIL_RETURN_UNEXPECTED(label_reader.is_open(),
                               "Invalid file, failed to open label file: " + label_dir.ToString() +
                                 ", make sure file not damaged or permission denied.");
  std::string line = "";
  while (getline(label_reader, line)) {
    size_t quot_inx[2] = {0};
    size_t quot_num = 0;
    size_t quot_exact = 2;
    for (size_t i = 0; quot_num < quot_exact && i < line.size(); i++) {
      if (line[i] == '"') {
        quot_inx[quot_num++] = i;
      }
    }
    if (quot_num != quot_exact) {
      label_reader.close();
      RETURN_STATUS_UNEXPECTED("Invalid file, the file may not be a CMUArctic dataset file: " + label_dir.ToString());
    }
    label_pairs_.push_back(
      {line.substr(2, quot_inx[0] - 3), line.substr(quot_inx[0] + 1, quot_inx[1] - quot_inx[0] - 1)});
  }
  label_reader.close();
  num_rows_ = label_pairs_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows_ > 0, "Invalid data, no valid data found in path: " + folder_path_);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
