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
#include "minddata/dataset/engine/datasetops/source/cityscapes_op.h"

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
constexpr char taskSuffix[] = "polygon";

CityscapesOp::CityscapesOp(int32_t num_workers, const std::string &dataset_dir, const std::string &usage,
                           const std::string &quality_mode, const std::string &task, bool decode, int32_t queue_size,
                           std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      quality_mode_(quality_mode),
      task_(task),
      decode_(decode),
      data_schema_(std::move(data_schema)) {
  io_block_queues_.Init(num_workers_, queue_size);
}

Status CityscapesOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Pipeline init failed, Execution tree not set.");
  }

  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&CityscapesOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  TaskManager::FindMe()->Post();
  // The order of the following 3 functions must not be changed!
  RETURN_IF_NOT_OK(ParseCityscapesData());  // Parse Cityscapes data and get num rows, blocking
  RETURN_IF_NOT_OK(CountDatasetInfo());     // Count the total rows
  RETURN_IF_NOT_OK(InitSampler());          // Pass numRows to Sampler
  return Status::OK();
}

// Load 1 TensorRow (image, task) using 1 ImageLabelPair. 1 function call produces 1 TensorTow
Status CityscapesOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::pair<std::string, std::string> data = image_task_pairs_[static_cast<size_t>(row_id)];
  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> task;
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(data.first, &image));

  if (task_ != taskSuffix) {
    RETURN_IF_NOT_OK(Tensor::CreateFromFile(data.second, &task));
  } else {
    std::ifstream file_handle(data.second);
    if (!file_handle.is_open()) {
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to open json file: " + data.second);
    }
    std::string contents((std::istreambuf_iterator<char>(file_handle)), std::istreambuf_iterator<char>());
    nlohmann::json contents_js = nlohmann::json::parse(contents);
    Status rc = Tensor::CreateScalar(contents_js.dump(), &task);
    if (rc.IsError()) {
      file_handle.close();
      return rc;
    }
    file_handle.close();
  }

  if (decode_ == true) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      std::string err = "Invalid data, failed to decode image: " + data.first;
      RETURN_STATUS_UNEXPECTED(err);
    }
    if (task_ != taskSuffix) {
      Status rc_t = Decode(task, &task);
      if (rc_t.IsError()) {
        std::string err_t = "Invalid data, failed to decode image: " + data.second;
        RETURN_STATUS_UNEXPECTED(err_t);
      }
    }
  }
  (*trow) = TensorRow(row_id, {std::move(image), std::move(task)});
  trow->setPath({data.first, data.second});
  return Status::OK();
}

void CityscapesOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nCityscapes DatasetDir: " << dataset_dir_ << "\nTask: " << task_
        << "\nQualityMode: " << quality_mode_ << "\nUsage: " << usage_ << "\nDecode: " << (decode_ ? "yes" : "no")
        << "\n\n";
  }
}

Status CityscapesOp::ParseCityscapesData() {
  auto real_dataset_dir = FileUtils::GetRealPath(dataset_dir_.data());
  if (!real_dataset_dir.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << dataset_dir_;
    RETURN_STATUS_UNEXPECTED("Get real path failed, path=" + dataset_dir_);
  }

  Path dataset_dir(real_dataset_dir.value());
  std::string real_quality_mode = quality_mode_ == "fine" ? "gtFine" : "gtCoarse";
  if (usage_ == "all" && quality_mode_ == "fine") {
    std::vector<std::string> all_usage_fine = {"train", "test", "val"};
    for (auto item : all_usage_fine) {
      std::string images_dir_fine = (dataset_dir / "leftImg8bit" / item).ToString();
      std::string task_dir_fine = (dataset_dir / real_quality_mode / item).ToString();
      RETURN_IF_NOT_OK(GetCityscapesDataByUsage(images_dir_fine, task_dir_fine, real_quality_mode));
    }
  } else if (usage_ == "all" && quality_mode_ == "coarse") {
    std::vector<std::string> all_usage_coarse = {"train", "train_extra", "val"};
    for (auto item : all_usage_coarse) {
      std::string images_dir_coarse = (dataset_dir / "leftImg8bit" / item).ToString();
      std::string task_dir_coarse = (dataset_dir / real_quality_mode / item).ToString();
      RETURN_IF_NOT_OK(GetCityscapesDataByUsage(images_dir_coarse, task_dir_coarse, real_quality_mode));
    }
  } else {
    std::string images_dir = (dataset_dir / "leftImg8bit" / usage_).ToString();
    std::string task_dir = (dataset_dir / real_quality_mode / usage_).ToString();
    RETURN_IF_NOT_OK(GetCityscapesDataByUsage(images_dir, task_dir, real_quality_mode));
  }
  return Status::OK();
}

Status CityscapesOp::GetCityscapesDataByUsage(const std::string &images_dir, const std::string &task_dir,
                                              const std::string &real_quality_mode) {
  const std::string kExtension = ".png";
  std::string img_file_name;
  std::map<std::string, std::string> image_task_map_;

  Path images_dir_p(images_dir);
  if (!images_dir_p.IsDirectory()) {
    RETURN_STATUS_UNEXPECTED("Invalid path, " + images_dir_p.ToString() + " is an invalid directory path.");
  }
  Path task_dir_p(task_dir);
  if (!task_dir_p.IsDirectory()) {
    RETURN_STATUS_UNEXPECTED("Invalid path, " + task_dir_p.ToString() + " is an invalid directory path.");
  }
  std::shared_ptr<Path::DirIterator> d_it = Path::DirIterator::OpenDirectory(&images_dir_p);
  if (d_it == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid path, failed to open directory: " + images_dir_p.ToString());
  }

  while (d_it->HasNext()) {
    try {
      Path city_dir = d_it->Next();
      if (!city_dir.IsDirectory()) {
        continue;
      }

      Path img_city_dir = images_dir_p / city_dir.Basename();
      Path task_city_dir = task_dir_p / city_dir.Basename();
      std::shared_ptr<Path::DirIterator> img_city_it = Path::DirIterator::OpenDirectory(&img_city_dir);
      if (img_city_it == nullptr) {
        RETURN_STATUS_UNEXPECTED("Invalid path, failed to open directory: " + img_city_dir.ToString());
      }

      while (img_city_it->HasNext()) {
        Path img_file = img_city_it->Next();
        if (img_file.Extension() != kExtension) {
          continue;
        }

        Path image_file_path = img_city_dir / img_file.Basename();
        img_file_name = img_file.Basename();
        Path task_file_path = task_city_dir / (img_file_name.substr(0, img_file_name.find("_leftImg8bit")) + "_" +
                                               GetTaskSuffix(task_, real_quality_mode));
        if (!task_file_path.Exists()) {
          RETURN_STATUS_UNEXPECTED("Invalid file, " + task_file_path.ToString() + " not found.");
        }

        image_task_map_[image_file_path.ToString()] = task_file_path.ToString();
      }
    } catch (const std::exception &err) {
      RETURN_STATUS_UNEXPECTED("Invalid path, failed to load Cityscapes Dataset: " + dataset_dir_);
    }
  }

  for (auto item : image_task_map_) {
    image_task_pairs_.emplace_back(std::make_pair(item.first, item.second));
  }
  return Status::OK();
}

std::string CityscapesOp::GetTaskSuffix(const std::string &task, const std::string &real_quality_mode) {
  std::string task_suffix;
  if (task == "instance") {
    task_suffix = real_quality_mode + "_instanceIds.png";
  } else if (task == "semantic") {
    task_suffix = real_quality_mode + "_labelIds.png";
  } else if (task == "color") {
    task_suffix = real_quality_mode + "_color.png";
  } else {
    task_suffix = real_quality_mode + "_polygons.json";
  }
  return task_suffix;
}

Status CityscapesOp::CountDatasetInfo() {
  num_rows_ = static_cast<int64_t>(image_task_pairs_.size());
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API CityscapesDataset. Please check file path or dataset API.");
  }
  return Status::OK();
}

Status CityscapesOp::CountTotalRows(const std::string &dir, const std::string &usage, const std::string &quality_mode,
                                    const std::string &task, int64_t *count) {
  // the logic of counting the number of samples is copied from ParseCityscapesData()
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto new_sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);

  // build a new unique schema object
  auto new_schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(new_schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  if (task == "polygon") {
    RETURN_IF_NOT_OK(
      new_schema->AddColumn(ColDescriptor("task", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &scalar)));
  } else {
    RETURN_IF_NOT_OK(
      new_schema->AddColumn(ColDescriptor("task", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 0, &scalar)));
  }

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  std::shared_ptr<CityscapesOp> op = std::make_shared<CityscapesOp>(
    num_workers, dir, usage, quality_mode, task, false, op_connect_size, std::move(new_schema), std::move(new_sampler));
  RETURN_IF_NOT_OK(op->ParseCityscapesData());
  *count = static_cast<int64_t>(op->image_task_pairs_.size());
  return Status::OK();
}

Status CityscapesOp::ComputeColMap() {
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
