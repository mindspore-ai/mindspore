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
#include "minddata/dataset/engine/datasetops/source/div2k_op.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <set>
#include <utility>

#include "utils/file_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
const std::map<std::string, std::string> DatasetPramMap = {{"train_hr", "DIV2K_train_HR"},
                                                           {"valid_hr", "DIV2K_valid_HR"},
                                                           {"train_bicubic_x2", "DIV2K_train_LR_bicubic"},
                                                           {"train_unknown_x2", "DIV2K_train_LR_unknown"},
                                                           {"valid_bicubic_x2", "DIV2K_valid_LR_bicubic"},
                                                           {"valid_unknown_x2", "DIV2K_valid_LR_unknown"},
                                                           {"train_bicubic_x3", "DIV2K_train_LR_bicubic"},
                                                           {"train_unknown_x3", "DIV2K_train_LR_unknown"},
                                                           {"valid_bicubic_x3", "DIV2K_valid_LR_bicubic"},
                                                           {"valid_unknown_x3", "DIV2K_valid_LR_unknown"},
                                                           {"train_bicubic_x4", "DIV2K_train_LR_bicubic"},
                                                           {"train_unknown_x4", "DIV2K_train_LR_unknown"},
                                                           {"valid_bicubic_x4", "DIV2K_valid_LR_bicubic"},
                                                           {"valid_unknown_x4", "DIV2K_valid_LR_unknown"},
                                                           {"train_bicubic_x8", "DIV2K_train_LR_x8"},
                                                           {"valid_bicubic_x8", "DIV2K_valid_LR_x8"},
                                                           {"train_mild_x4", "DIV2K_train_LR_mild"},
                                                           {"valid_mild_x4", "DIV2K_valid_LR_mild"},
                                                           {"train_difficult_x4", "DIV2K_train_LR_difficult"},
                                                           {"valid_difficult_x4", "DIV2K_valid_LR_difficult"},
                                                           {"train_wild_x4", "DIV2K_train_LR_wild"},
                                                           {"valid_wild_x4", "DIV2K_valid_LR_wild"}};

DIV2KOp::DIV2KOp(int32_t num_workers, const std::string &dataset_dir, const std::string &usage,
                 const std::string &downgrade, int32_t scale, bool decode, int32_t queue_size,
                 std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      downgrade_(downgrade),
      scale_(scale),
      decode_(decode),
      data_schema_(std::move(data_schema)) {
  io_block_queues_.Init(num_workers_, queue_size);
}

Status DIV2KOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Pipeline init failed, Execution tree not set.");
  }

  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&DIV2KOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  TaskManager::FindMe()->Post();
  // The order of the following 3 functions must not be changed!
  RETURN_IF_NOT_OK(ParseDIV2KData());    // Parse div2k data and get num rows, blocking
  RETURN_IF_NOT_OK(CountDatasetInfo());  // Count the total rows
  RETURN_IF_NOT_OK(InitSampler());       // Pass numRows to Sampler
  return Status::OK();
}

// Load 1 TensorRow (hr_image, lr_image) using 1 ImageLabelPair. 1 function call produces 1 TensorTow.
Status DIV2KOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::pair<std::string, std::string> data = image_hr_lr_pairs_[static_cast<size_t>(row_id)];
  std::shared_ptr<Tensor> hr_image;
  std::shared_ptr<Tensor> lr_image;
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(data.first, &hr_image));
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(data.second, &lr_image));

  if (decode_ == true) {
    Status hr_rc = Decode(hr_image, &hr_image);
    if (hr_rc.IsError()) {
      std::string err = "Invalid data, failed to decode image: " + data.first;
      RETURN_STATUS_UNEXPECTED(err);
    }

    Status lr_rc = Decode(lr_image, &lr_image);
    if (lr_rc.IsError()) {
      std::string err = "Invalid data, failed to decode image: " + data.second;
      RETURN_STATUS_UNEXPECTED(err);
    }
  }
  (*trow) = TensorRow(row_id, {std::move(hr_image), std::move(lr_image)});
  trow->setPath({data.first, data.second});
  return Status::OK();
}

void DIV2KOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nDIV2K DatasetDir: " << dataset_dir_ << "\nUsage: " << usage_
        << "\nScale: " << scale_ << "\nDowngrade: " << downgrade_ << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status DIV2KOp::ParseDIV2KData() {
  std::string hr_dir_key;
  std::string lr_dir_key;

  if (usage_ == "all") {
    std::vector<std::string> usage_all = {"train", "valid"};
    for (auto &item : usage_all) {
      hr_dir_key = item + "_hr";
      lr_dir_key = item + "_" + downgrade_ + "_x" + std::to_string(scale_);
      RETURN_IF_NOT_OK(GetDIV2KLRDirRealName(hr_dir_key, lr_dir_key));
      RETURN_IF_NOT_OK(GetDIV2KDataByUsage());
    }
  } else {
    hr_dir_key = usage_ + "_hr";
    lr_dir_key = usage_ + "_" + downgrade_ + "_x" + std::to_string(scale_);
    RETURN_IF_NOT_OK(GetDIV2KLRDirRealName(hr_dir_key, lr_dir_key));
    RETURN_IF_NOT_OK(GetDIV2KDataByUsage());
  }
  return Status::OK();
}

Status DIV2KOp::GetDIV2KLRDirRealName(const std::string &hr_dir_key, const std::string &lr_dir_key) {
  std::set<std::string> downgrade_2017 = {"bicubic", "unknown"};
  std::set<int32_t> scale_2017 = {2, 3, 4};

  hr_dir_real_name_ = DatasetPramMap.find(hr_dir_key)->second;
  auto lr_it = DatasetPramMap.find(lr_dir_key);
  if (lr_it == DatasetPramMap.end()) {
    std::string out_str = "{\n";
    std::for_each(DatasetPramMap.begin(), DatasetPramMap.end(),
                  [&out_str](std::pair<std::string, std::string> item) -> void {
                    out_str += ("\t" + item.first + ": " + item.second + ",\n");
                  });
    out_str += "\n}";
    RETURN_STATUS_UNEXPECTED("Invalid param, " + lr_dir_key + " not found in DatasetPramMap: \n" + out_str);
  }

  if (downgrade_2017.find(downgrade_) != downgrade_2017.end() && scale_2017.find(scale_) != scale_2017.end()) {
    Path ntire_2017(lr_it->second);
    lr_dir_real_name_ = (ntire_2017 / ("X" + std::to_string(scale_))).ToString();
  } else {
    lr_dir_real_name_ = lr_it->second;
  }
  return Status::OK();
}

Status DIV2KOp::GetDIV2KDataByUsage() {
  const std::string kExtension = ".png";

  auto real_dataset_dir = FileUtils::GetRealPath(dataset_dir_.data());
  if (!real_dataset_dir.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << dataset_dir_;
    RETURN_STATUS_UNEXPECTED("Get real path failed, path=" + dataset_dir_);
  }

  Path dataset_dir(real_dataset_dir.value());
  Path hr_images_dir = dataset_dir / hr_dir_real_name_;
  Path lr_images_dir = dataset_dir / lr_dir_real_name_;

  if (!hr_images_dir.IsDirectory()) {
    RETURN_STATUS_UNEXPECTED("Invalid path, " + hr_images_dir.ToString() + " is an invalid directory path.");
  }
  if (!lr_images_dir.IsDirectory()) {
    RETURN_STATUS_UNEXPECTED("Invalid path, " + lr_images_dir.ToString() + " is an invalid directory path.");
  }
  auto hr_it = Path::DirIterator::OpenDirectory(&hr_images_dir);
  if (hr_it == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid path, failed to open directory: " + hr_images_dir.ToString());
  }

  std::string image_name;
  std::string image_id_scale;
  std::string lr_image_file_path_;
  std::map<std::string, std::string> image_hr_lr_map_;
  std::map<std::string, std::string> downgrade_2018 = {{"mild", "m"}, {"difficult", "d"}, {"wild", "w"}};

  while (hr_it->HasNext()) {
    try {
      Path hr_img_file = hr_it->Next();
      if (hr_img_file.Extension() != kExtension) {
        continue;
      }

      image_name = hr_img_file.Basename();
      image_id_scale = image_name.substr(0, image_name.find_last_of(".")) + "x" + std::to_string(scale_);
      Path hr_image_file_path = hr_images_dir / image_name;
      auto lr_it = downgrade_2018.find(downgrade_);
      if (lr_it != downgrade_2018.end()) {
        lr_image_file_path_ = (lr_images_dir / (image_id_scale + lr_it->second + kExtension)).ToString();
      } else {
        lr_image_file_path_ = (lr_images_dir / (image_id_scale + kExtension)).ToString();
      }

      Path lr_image_file_path(lr_image_file_path_);
      if (!lr_image_file_path.Exists()) {
        RETURN_STATUS_UNEXPECTED("Invalid file, " + lr_image_file_path.ToString() + " not found.");
      }

      image_hr_lr_map_[hr_image_file_path.ToString()] = lr_image_file_path.ToString();
    } catch (const std::exception &err) {
      RETURN_STATUS_UNEXPECTED("Invalid path, failed to load DIV2K Dataset: " + dataset_dir_);
    }
  }
  for (auto item : image_hr_lr_map_) {
    image_hr_lr_pairs_.emplace_back(std::make_pair(item.first, item.second));
  }
  return Status::OK();
}

Status DIV2KOp::CountDatasetInfo() {
  num_rows_ = static_cast<int64_t>(image_hr_lr_pairs_.size());
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API DIV2KDataset. Please check file path or dataset API.");
  }
  return Status::OK();
}

Status DIV2KOp::CountTotalRows(const std::string &dir, const std::string &usage, const std::string &downgrade,
                               int32_t scale, int64_t *count) {
  // the logic of counting the number of samples is copied from ParseDIV2KData()
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto new_sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);

  // build a new unique schema object
  auto new_schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(new_schema->AddColumn(ColDescriptor("hr_image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    new_schema->AddColumn(ColDescriptor("lr_image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 0, &scalar)));

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  std::shared_ptr<DIV2KOp> op = std::make_shared<DIV2KOp>(
    num_workers, dir, usage, downgrade, scale, false, op_connect_size, std::move(new_schema), std::move(new_sampler));
  RETURN_IF_NOT_OK(op->ParseDIV2KData());
  *count = static_cast<int64_t>(op->image_hr_lr_pairs_.size());
  return Status::OK();
}

Status DIV2KOp::ComputeColMap() {
  // Set the column name map (base class field)
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
