/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {

const size_t kWavHandSize=44;
const size_t kReadbufferSize=20480;
const std::string dataDirectory = "wav";
const std::string labelDirectory = "etc";
const std::string labelFileName = "txt.done.data";

const std::string pre="cmu_us_";
const std::string suf="_arctic";

CmuArcticOp::CmuArcticOp(const std::string &usage, int32_t num_workers, std::string folder_path, int32_t queue_size,
             std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      usage_(usage),
      folder_path_(folder_path),
      data_schema_(std::move(data_schema)) {
  io_block_queues_.Init(num_workers, queue_size);
}

Status CmuArcticOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  CmuArcticLabelTuple audio_tuple = audio_label_tuple_[row_id];
  std::shared_ptr <Tensor> waveform, rate, utterance, utterance_id;
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(audio_tuple.waveform, &waveform));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.sample_rate, &rate));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.utterance, &utterance));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(audio_tuple.utterance_id, &utterance_id));
  (*trow) = TensorRow(row_id, {std::move(waveform), std::move(rate), std::move(utterance), std::move(utterance_id)});
  trow->setPath({audio_names_[row_id].first});
  return Status::OK();
}

void CmuArcticOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  }
  else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nCmuArctic Directory: " << folder_path_ << "\n\n";
  }
}

// Derived from RandomAccessOp
Status CmuArcticOp::GetClassIds(std::map<std::string, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || audio_label_tuple_.empty()) {
    if (audio_label_tuple_.empty()) {
      RETURN_STATUS_UNEXPECTED("No audio found in dataset, please check if Op read audios successfully or not.");
    }
    else {
      RETURN_STATUS_UNEXPECTED(
          "Map for storaging audio-index pair is nullptr or has been set in other place,"
          "it must be empty before using GetClassIds.");
    }
  }
  for (size_t i = 0; i < audio_label_tuple_.size(); ++i) {
    (*cls_ids)[audio_label_tuple_[i].utterance_id].push_back(i);//
  }
  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}


Status CmuArcticOp::CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count) {
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  auto schema = std::make_unique<DataSchema>();

  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("waveform", DataType(DataType::DE_FLOAT64), TensorImpl::kCv, 1)));
  TensorShape scalar_rate = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("sample_rate", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0,
                      &scalar_rate)));
  TensorShape scalar_utterance = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("utterance", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0,
                      &scalar_utterance)));
  TensorShape scalar_utterance_id = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("utterance_id", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0,
                      &scalar_utterance_id)));
  std::shared_ptr <ConfigManager> cfg = GlobalContext::config_manager();

  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  auto op = std::make_shared<CmuArcticOp>(usage, num_workers, dir, op_connect_size, std::move(schema),
                      std::move(sampler));
  RETURN_IF_NOT_OK(op->WalkAllFiles());
  *count = op->audio_names_.size();
  return Status::OK();
}

Status CmuArcticOp::ComputeColMap() {
  // set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->column(i).name()] = i;
    }
  }
  else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status CmuArcticOp::ReadLabel() {
  char buffer[1024];
  for (std::string u:label_files_) {
    std::ifstream in(u);
    while (!in.eof()) {
      in.getline(buffer, 1024);
      if (buffer[0] != '(')
        break;
      int32_t blank[3] = {0};
      int32_t cur = 0;
      for (int32_t i = 0; cur < 2 && i < 1024; i++) {
        if (buffer[i] == '"')
          blank[cur++] = i;
      }
      if (cur != 2)
        RETURN_STATUS_UNEXPECTED("Label file error!");
      buffer[blank[0] - 1] = 0;
      buffer[blank[1]] = 0;
      label_pairs_.push_back({std::string(buffer + 2), std::string(buffer + blank[0] + 1)});
    }
  }
  if (audio_names_.size() != label_pairs_.size())
    RETURN_STATUS_UNEXPECTED("The number of files is different from the number of labels!");
  std::sort(audio_names_.begin(), audio_names_.end());
  std::sort(label_pairs_.begin(), label_pairs_.end());
  return Status::OK();
}

Status CmuArcticOp::ReadAudio() {
  char header[kWavHandSize];
  short buff[kReadbufferSize];
  const double mx = 32768.0;
  std::vector<double> tempArr;
  for (uint32_t i = 0; i < audio_names_.size(); i++) {
    if (audio_names_[i].first != label_pairs_[i].first + ".wav") {
      RETURN_STATUS_UNEXPECTED("An error occurred between the label and the file content!");
    }
    tempArr.clear();
    auto item = audio_names_[i];
    const char *dir = item.second.data();
    FILE *fp = fopen(dir, "rb");
    if (fp == NULL) {
      MS_LOG(WARNING) << "File missing . dir:" << dir;
      continue;
    }
    uint32_t s = fread(header, 1, kWavHandSize, fp);
    if (s != kWavHandSize)
      RETURN_STATUS_UNEXPECTED("Audio header error!");
    uint32_t rate = *(uint32_t * )(header + 0x18);
    uint32_t frame = *(uint32_t * )(header + 0x28) / 2;
    uint32_t surplus = frame;
    while (surplus) {
      uint32_t len = fread(buff, 2, kReadbufferSize, fp);
      for (uint32_t i = 0; i < len; i++) {
        tempArr.push_back(buff[i] / mx);
      }
      surplus -= len;
    }
    fclose(fp);
    std::shared_ptr <Tensor> audio;
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(tempArr, &audio));
    audio_label_tuple_.push_back({audio, rate, label_pairs_[i].second, label_pairs_[i].first});
  }
  num_rows_ = audio_names_.size();
  return Status::OK();
}

Status CmuArcticOp::WalkAllFiles() {
  Path dir(folder_path_);
  Path fullDir = (dir + pre + usage_ + suf) / dataDirectory;
  Path label = (dir + pre + usage_ + suf) / labelDirectory / labelFileName;
  label_files_.push_back(label.toString());
  auto dirIt = Path::DirIterator::OpenDirectory(&fullDir);
  if (dirIt != nullptr) {
    while (dirIt->hasNext()) {
      Path file = dirIt->next();
      std::string fileName = file.toString();
      auto pos = fileName.find_last_of('.');
      std::string ext = fileName.substr(pos);
      if (ext == ".wav") {
        audio_names_.push_back({file.Basename(), file.toString()});
      }
      else {
        MS_LOG(WARNING) << "File name format error :" << file.toString() << ".";
      }
    }
  }
  else {
    MS_LOG(WARNING) << "Unable to open directory " << fullDir.toString() << ".";
  }
  return Status::OK();
}

Status CmuArcticOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Pipeline init failed, Execution tree not set.");
  }
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
      tree_->LaunchWorkers(num_workers_, std::bind(&CmuArcticOp::WorkerEntry, this, std::placeholders::_1), "",
                 id()));
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(this->WalkAllFiles());
  RETURN_IF_NOT_OK(this->ReadLabel());
  RETURN_IF_NOT_OK(this->ReadAudio());
  RETURN_IF_NOT_OK(this->InitSampler());  // handle shake with sampler
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
