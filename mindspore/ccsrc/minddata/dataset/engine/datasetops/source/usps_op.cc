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
#include "minddata/dataset/engine/datasetops/source/usps_op.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <set>
#include <utility>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
constexpr int64_t kUSPSImageHeight = 16;
constexpr int64_t kUSPSImageWidth = 16;
constexpr int64_t kUSPSImageChannel = 1;
constexpr int64_t kUSPSImageSize = kUSPSImageHeight * kUSPSImageWidth * kUSPSImageChannel;

USPSOp::USPSOp(const std::string &dataset_dir, const std::string &usage, std::unique_ptr<DataSchema> data_schema,
               int32_t num_workers, int32_t worker_connector_size, int64_t num_samples, int32_t op_connector_size,
               bool shuffle_files, int32_t num_devices, int32_t device_id)
    : NonMappableLeafOp(num_workers, worker_connector_size, num_samples, op_connector_size, shuffle_files, num_devices,
                        device_id),
      usage_(usage),
      dataset_dir_(dataset_dir),
      data_schema_(std::move(data_schema)) {}

void USPSOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nRow count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nUSPS directory: " << dataset_dir_
        << "\nUSPS usage: " << usage_ << "\n\n";
    out << "\nData schema:\n";
    out << *data_schema_ << "\n\n";
  }
}

Status USPSOp::Init() {
  RETURN_IF_NOT_OK(this->GetFiles());
  RETURN_IF_NOT_OK(filename_index_->insert(data_files_list_));

  int32_t safe_queue_size = static_cast<int32_t>(std::ceil(data_files_list_.size() / num_workers_) + 1);
  io_block_queues_.Init(num_workers_, safe_queue_size);

  jagged_rows_connector_ = std::make_unique<JaggedConnector>(num_workers_, 1, worker_connector_size_);
  return Status::OK();
}

Status USPSOp::CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;

  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connector_size = cfg->op_connector_size();
  int32_t worker_connector_size = cfg->worker_connector_size();

  const int64_t num_samples = 0;
  const int32_t num_devices = 1;
  const int32_t device_id = 0;
  bool shuffle = false;

  auto op = std::make_shared<USPSOp>(dir, usage, std::move(schema), num_workers, worker_connector_size, num_samples,
                                     op_connector_size, shuffle, num_devices, device_id);
  RETURN_IF_NOT_OK(op->Init());
  // the logic of counting the number of samples
  for (auto data_file : op->FileNames()) {
    *count += op->CountRows(data_file);
  }
  return Status::OK();
}

int64_t USPSOp::CountRows(const std::string &data_file) const {
  std::ifstream data_file_reader;
  data_file_reader.open(data_file, std::ios::in);
  if (!data_file_reader.is_open()) {
    MS_LOG(ERROR) << "Invalid file, failed to open " << data_file << ": the file is permission denied.";
    return 0;
  }

  std::string line;
  int64_t count = 0;
  while (std::getline(data_file_reader, line)) {
    if (!line.empty()) {
      count++;
    }
  }
  data_file_reader.close();
  return count;
}

Status USPSOp::GetFiles() {
  auto real_dataset_dir = FileUtils::GetRealPath(dataset_dir_.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(real_dataset_dir.has_value(),
                               "Invalid file path, USPS dataset dir: " + dataset_dir_ + " does not exist.");
  Path root_dir(real_dataset_dir.value());

  const Path train_file_name("usps");
  const Path test_file_name("usps.t");

  bool use_train = false;
  bool use_test = false;

  if (usage_ == "train") {
    use_train = true;
  } else if (usage_ == "test") {
    use_test = true;
  } else if (usage_ == "all") {
    use_train = true;
    use_test = true;
  }

  if (use_train) {
    Path train_path = root_dir / train_file_name;
    CHECK_FAIL_RETURN_UNEXPECTED(
      train_path.Exists() && !train_path.IsDirectory(),
      "Invalid file, USPS dataset train file: " + train_path.ToString() + " does not exist or is a directory.");
    data_files_list_.emplace_back(train_path.ToString());
    MS_LOG(INFO) << "USPS operator found train data file " << train_path.ToString() << ".";
  }

  if (use_test) {
    Path test_path = root_dir / test_file_name;
    CHECK_FAIL_RETURN_UNEXPECTED(
      test_path.Exists() && !test_path.IsDirectory(),
      "Invalid file, USPS dataset test file: " + test_path.ToString() + " does not exist or is a directory.");
    data_files_list_.emplace_back(test_path.ToString());
    MS_LOG(INFO) << "USPS operator found test data file " << test_path.ToString() << ".";
  }
  return Status::OK();
}

Status USPSOp::LoadFile(const std::string &data_file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  std::ifstream data_file_reader(data_file);
  if (!data_file_reader.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open USPS dataset file: " + data_file +
                             ", the file is permission denied.");
  }

  int64_t rows_total = 0;
  std::string line;

  while (getline(data_file_reader, line)) {
    if (line.empty()) {
      continue;
    }
    // If read to the end offset of this file, break.
    if (rows_total >= end_offset) {
      break;
    }
    // Skip line before start offset.
    if (rows_total < start_offset) {
      rows_total++;
      continue;
    }

    TensorRow tRow(1, nullptr);
    tRow.setPath({data_file});
    Status rc = LoadTensor(&line, &tRow);
    if (rc.IsError()) {
      data_file_reader.close();
      return rc;
    }
    rc = jagged_rows_connector_->Add(worker_id, std::move(tRow));
    if (rc.IsError()) {
      data_file_reader.close();
      return rc;
    }

    rows_total++;
  }

  data_file_reader.close();
  return Status::OK();
}

Status USPSOp::LoadTensor(std::string *line, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(line);
  RETURN_UNEXPECTED_IF_NULL(trow);

  auto images_buffer = std::make_unique<unsigned char[]>(kUSPSImageSize);
  auto labels_buffer = std::make_unique<uint32_t[]>(1);
  if (images_buffer == nullptr || labels_buffer == nullptr) {
    MS_LOG(ERROR) << "[Internal ERROR] Failed to allocate memory for USPS buffer.";
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to allocate memory for USPS buffer.");
  }

  RETURN_IF_NOT_OK(this->ParseLine(line, images_buffer, labels_buffer));

  // create tensor
  std::shared_ptr<Tensor> image, label;
  TensorShape image_tensor_shape = TensorShape({kUSPSImageHeight, kUSPSImageWidth, kUSPSImageChannel});
  auto pixels = &images_buffer[0];
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(image_tensor_shape, data_schema_->Column(0).Type(),
                                            reinterpret_cast<unsigned char *>(pixels), &image));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(labels_buffer[0], &label));

  (*trow) = {std::move(image), std::move(label)};
  return Status::OK();
}

Status USPSOp::ParseLine(std::string *line, const std::unique_ptr<unsigned char[]> &images_buffer,
                         const std::unique_ptr<uint32_t[]> &labels_buffer) const {
  auto label = &labels_buffer[0];
  auto pixels = &images_buffer[0];

  size_t pos = 0;
  int32_t split_num = 0;
  while ((pos = line->find(" ")) != std::string::npos) {
    split_num += 1;
    std::string item = line->substr(0, pos);

    if (split_num == 1) {
      // the class label is 1~10 but we need 0~9
      *label = static_cast<uint32_t>(std::stoi(item)) - 1;
    } else {
      size_t split_pos = item.find(":");

      CHECK_FAIL_RETURN_UNEXPECTED(split_pos != std::string::npos,
                                   "Invalid data, split character ':' is missing in USPS data file.");
      // check pixel index
      CHECK_FAIL_RETURN_UNEXPECTED(std::stoi(item.substr(0, split_pos)) == (split_num - 1),
                                   "Invalid data, the character before ':' should be " + std::to_string(split_num - 1) +
                                     ", but got " + item.substr(0, split_pos) + ".");

      std::string pixel_str = item.substr(split_pos + 1, item.length() - split_pos);
      // transform the real pixel value from [-1, 1] to the integers within [0, 255]
      pixels[split_num - 2] = static_cast<uint8_t>((std::stof(pixel_str) + 1.0) / 2.0 * 255.0);
    }
    line->erase(0, pos + 1);
  }

  CHECK_FAIL_RETURN_UNEXPECTED(split_num == (kUSPSImageSize + 1),
                               "Invalid data, the number of split characters ':' in USPS data file is corrupted, "
                               "should be " +
                                 std::to_string(kUSPSImageSize + 1) + ", but got " + std::to_string(split_num) + ".");
  return Status::OK();
}

Status USPSOp::CalculateNumRowsPerShard() {
  for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
    int64_t count = CountRows(it.value());
    filename_numrows_[it.value()] = count;
    num_rows_ += count;
  }
  if (num_rows_ == 0) {
    std::stringstream ss;
    for (size_t i = 0; i < data_files_list_.size(); ++i) {
      ss << " " << data_files_list_[i];
    }
    std::string file_list = ss.str();
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, 'USPSDataset' API can't read the data file (interface mismatch or no data found). "
      "Check file: " +
      file_list);
  }

  num_rows_per_shard_ = static_cast<int64_t>(std::ceil(num_rows_ * 1.0 / num_devices_));
  MS_LOG(DEBUG) << "Number rows per shard is " << num_rows_per_shard_;
  return Status::OK();
}

Status USPSOp::FillIOBlockQueue(const std::vector<int64_t> &i_keys) {
  int32_t queue_index = 0;
  int64_t pre_count = 0;
  int64_t start_offset = 0;
  int64_t end_offset = 0;
  bool finish = false;
  while (!finish) {
    std::vector<std::pair<std::string, int64_t>> file_index;
    if (!i_keys.empty()) {
      for (auto it = i_keys.begin(); it != i_keys.end(); ++it) {
        {
          if (!GetLoadIoBlockQueue()) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>((*filename_index_)[*it], *it));
      }
    } else {
      for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
        {
          if (!GetLoadIoBlockQueue()) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>(it.value(), it.key()));
      }
    }
    for (auto file_info : file_index) {
      if (NeedPushFileToBlockQueue(file_info.first, &start_offset, &end_offset, pre_count)) {
        auto ioBlock =
          std::make_unique<FilenameBlock>(file_info.second, start_offset, end_offset, IOBlock::kDeIoBlockNone);
        RETURN_IF_NOT_OK(PushIoBlockQueue(queue_index, std::move(ioBlock)));
        queue_index = (queue_index + 1) % num_workers_;
      }

      pre_count += filename_numrows_[file_info.first];
    }

    if (pre_count < (static_cast<int64_t>(device_id_) + 1) * num_rows_per_shard_) {
      finish = false;
    } else {
      finish = true;
    }
  }

  RETURN_IF_NOT_OK(PostEndOfEpoch(queue_index));
  return Status::OK();
}

Status USPSOp::ComputeColMap() {
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
