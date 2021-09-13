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
#include "minddata/dataset/engine/datasetops/source/flickr_op.h"

#include <algorithm>
#include <fstream>
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
FlickrOp::FlickrOp(int32_t num_workers, const std::string &dataset_dir, const std::string &file_path, bool decode,
                   int32_t queue_size, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      dataset_dir_(dataset_dir),
      file_path_(file_path),
      decode_(decode),
      data_schema_(std::move(data_schema)) {
  io_block_queues_.Init(num_workers_, queue_size);
}

Status FlickrOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Pipeline init failed, Execution tree not set.");
  }

  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&FlickrOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  TaskManager::FindMe()->Post();
  // The order of the following 2 functions must not be changed!
  RETURN_IF_NOT_OK(ParseFlickrData());   // Parse Flickr data and get num rows, blocking
  RETURN_IF_NOT_OK(CountDatasetInfo());  // Count the total rows
  RETURN_IF_NOT_OK(InitSampler());       // Pass numRows to Sampler
  return Status::OK();
}

// Load 1 TensorRow (image, annotations) using 1 ImageLabelPair. 1 function call produces 1 TensorTow
Status FlickrOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  std::pair<std::string, std::vector<std::string>> data = image_annotation_pairs_[static_cast<size_t>(row_id)];
  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> annotations;
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(data.first, &image));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(data.second, &annotations));

  if (decode_ == true) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      std::string err = "Invalid data, failed to decode image: " + data.first;
      RETURN_STATUS_UNEXPECTED(err);
    }
  }

  (*trow) = TensorRow(row_id, {std::move(image), std::move(annotations)});
  trow->setPath({data.first, file_path_});
  return Status::OK();
}

void FlickrOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nFlickr DatasetDir: " << dataset_dir_
        << "\nAnnotationFile: " << file_path_ << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status FlickrOp::ParseFlickrData() {
  auto real_file_path = FileUtils::GetRealPath(file_path_.data());
  if (!real_file_path.has_value()) {
    MS_LOG(ERROR) << "Invalid file, get real path failed, path=" << file_path_;
    RETURN_STATUS_UNEXPECTED("Invalid file, get real path failed, path=" + file_path_);
  }

  std::ifstream file_handle(real_file_path.value());
  if (!file_handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open Flickr annotation file: " + file_path_);
  }

  std::string line;
  int32_t flag_idx;
  std::string sub_str_flag = "\t";
  std::string image_file_path;
  std::string image_name;
  std::map<std::string, std::vector<std::string>> image_annotation_map_;
  Path dataset_dir(dataset_dir_);
  while (getline(file_handle, line)) {
    try {
      if (line.empty()) {
        continue;
      }

      flag_idx = line.find_first_of(sub_str_flag);
      image_name = line.substr(0, flag_idx - 2);  // -2 because "#[0-4]\t"
      if (image_name.empty()) {
        file_handle.close();
        RETURN_STATUS_UNEXPECTED("Invalid data, image_name is not found in Flickr annotation file: " + file_path_ +
                                 "; line: " + line);
      }

      image_file_path = (dataset_dir / image_name).ToString();
      std::string annotation = line.substr(flag_idx + 1);
      if (annotation.empty()) {
        file_handle.close();
        RETURN_STATUS_UNEXPECTED("Invalid data, annotation is not found in Flickr annotation file: " + file_path_ +
                                 "; line: " + line);
      }

      bool valid = false;
      Status type_check = CheckImageType(image_file_path, &valid);
      if (type_check.IsError()) {
        file_handle.close();
        RETURN_IF_NOT_OK(type_check);
      }
      if (!valid) {
        continue;
      }

      image_annotation_map_[image_file_path].emplace_back(annotation);
    } catch (const std::exception &err) {
      file_handle.close();
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to open Flickr annotation file: " + file_path_);
    }
  }

  for (auto item : image_annotation_map_) {
    image_annotation_pairs_.emplace_back(std::make_pair(item.first, item.second));
  }

  file_handle.close();
  return Status::OK();
}

// Only support JPEG/PNG/GIF/BMP
// Optimization: Could take in a tensor
// This function does not return status because we want to just skip bad input, not crash
Status FlickrOp::CheckImageType(const std::string &file_name, bool *valid) {
  auto real_file_name = FileUtils::GetRealPath(file_name.data());
  if (!real_file_name.has_value()) {
    MS_LOG(ERROR) << "Invalid file, get real path failed, path=" << file_name;
    RETURN_STATUS_UNEXPECTED("Invalid file, get real path failed, path=" + file_name);
  }

  std::ifstream file_handle;
  constexpr int read_num = 3;
  *valid = false;
  file_handle.open(real_file_name.value(), std::ios::binary | std::ios::in);
  if (!file_handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open image file: " + file_name);
  }
  unsigned char file_type[read_num];
  (void)file_handle.read(reinterpret_cast<char *>(file_type), read_num);

  if (file_handle.fail()) {
    file_handle.close();
    RETURN_STATUS_UNEXPECTED("Invalid data, failed to read image file: " + file_name);
  }
  file_handle.close();
  if (file_type[0] == 0xff && file_type[1] == 0xd8 && file_type[2] == 0xff) {
    // Normal JPEGs start with \xff\xd8\xff\xe0
    // JPEG with EXIF stats with \xff\xd8\xff\xe1
    // Use \xff\xd8\xff to cover both.
    *valid = true;
  } else if (file_type[0] == 0x89 && file_type[1] == 0x50 && file_type[2] == 0x4e) {
    // It's a PNG
    *valid = true;
  } else if (file_type[0] == 0x47 && file_type[1] == 0x49 && file_type[2] == 0x46) {
    // It's a GIF
    *valid = true;
  } else if (file_type[0] == 0x42 && file_type[1] == 0x4d) {
    // It's a BMP
    *valid = true;
  }
  return Status::OK();
}

Status FlickrOp::CountDatasetInfo() {
  num_rows_ = static_cast<int64_t>(image_annotation_pairs_.size());
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API FlickrDataset. Please check file path or dataset API.");
  }
  return Status::OK();
}

Status FlickrOp::CountTotalRows(const std::string &dir, const std::string &file, int64_t *count) {
  // the logic of counting the number of samples is copied from ParseFlickrData()
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto new_sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);

  // build a new unique schema object
  auto new_schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    new_schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(new_schema->AddColumn(
    ColDescriptor("annotation", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  std::shared_ptr<FlickrOp> op = std::make_shared<FlickrOp>(num_workers, dir, file, false, op_connect_size,
                                                            std::move(new_schema), std::move(new_sampler));

  RETURN_IF_NOT_OK(op->ParseFlickrData());
  *count = static_cast<int64_t>(op->image_annotation_pairs_.size());
  return Status::OK();
}

Status FlickrOp::ComputeColMap() {
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
