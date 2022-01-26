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

#include "minddata/dataset/engine/datasetops/source/manifest_op.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <nlohmann/json.hpp>

#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
ManifestOp::ManifestOp(int32_t num_works, std::string file, int32_t queue_size, bool decode,
                       const std::map<std::string, int32_t> &class_index, std::unique_ptr<DataSchema> data_schema,
                       std::shared_ptr<SamplerRT> sampler, std::string usage)
    : MappableLeafOp(num_works, queue_size, std::move(sampler)),
      io_block_pushed_(0),
      sampler_ind_(0),
      data_schema_(std::move(data_schema)),
      file_(std::move(file)),
      class_index_(class_index),
      decode_(decode),
      usage_(usage) {
  (void)std::transform(usage_.begin(), usage_.end(), usage_.begin(), ::tolower);
}

// Load 1 TensorRow (image,label) using 1 ImageLabelPair. 1 function call produces 1 TensorTow
Status ManifestOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  std::pair<std::string, std::vector<std::string>> data = image_labelname_[static_cast<size_t>(row_id)];
  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> label;
  std::vector<int32_t> label_index(data.second.size());
  (void)std::transform(data.second.begin(), data.second.end(), label_index.begin(),
                       [this](const std::string &label_name) { return label_index_[label_name]; });
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(label_index, &label));
  if (label_index.size() == 1) {
    RETURN_IF_NOT_OK(label->Reshape(TensorShape({})));
  } else {
    RETURN_IF_NOT_OK(label->Reshape(TensorShape(std::vector<dsize_t>(1, label_index.size()))));
  }

  RETURN_IF_NOT_OK(Tensor::CreateFromFile(data.first, &image));
  if (decode_ == true) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      std::string err =
        "Invalid image, failed to decode: " + data.first + ", the image is damaged or permission denied.";
      RETURN_STATUS_UNEXPECTED(err);
    }
  }
  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  trow->setPath({data.first, file_});
  return Status::OK();
}

void ManifestOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nManifest file: " << file_ << "\nDecode: " << (decode_ ? "yes" : "no")
        << "\n\n";
  }
}

// Derived from RandomAccessOp
Status ManifestOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || image_labelname_.empty()) {
    if (image_labelname_.empty()) {
      RETURN_STATUS_UNEXPECTED("Invalid manifest file, image data is missing in " + file_);
    } else {
      RETURN_STATUS_UNEXPECTED(
        "[Internal ERROR] Map for containing image-index pair is nullptr or has been set in other place,"
        "it must be empty before using GetClassIds.");
    }
  }

  for (size_t i = 0; i < image_labelname_.size(); i++) {
    size_t image_index = i;
    for (size_t j = 0; j < image_labelname_[image_index].second.size(); j++) {
      std::string label_name = (image_labelname_[image_index].second)[j];
      int32_t label_index = label_index_.at(label_name);
      (*cls_ids)[label_index].emplace_back(image_index);
    }
  }

  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}

// Manifest file content
// {"source": "/path/to/image1.jpg", "usage":"train", annotation": ...}
// {"source": "/path/to/image2.jpg", "usage":"eval", "annotation": ...}
Status ManifestOp::PrepareData() {
  auto realpath = FileUtils::GetRealPath(file_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << file_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + file_ + " does not exist.");
  }

  std::ifstream file_handle(realpath.value());
  if (!file_handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open " + file_ +
                             ": manifest file is damaged or permission denied!");
  }
  std::string line;
  std::set<std::string> classes;
  uint64_t line_count = 1;
  while (getline(file_handle, line)) {
    try {
      nlohmann::json js = nlohmann::json::parse(line);
      std::string image_file_path = js.value("source", "");
      if (image_file_path == "") {
        file_handle.close();
        RETURN_STATUS_UNEXPECTED("Invalid manifest file, 'source' is missing in file: " + file_ + " at line " +
                                 std::to_string(line_count));
      }
      // If image is not JPEG/PNG/GIF/BMP, drop it
      bool valid = false;
      RETURN_IF_NOT_OK(CheckImageType(image_file_path, &valid));
      if (!valid) {
        continue;
      }
      std::string usage = js.value("usage", "");
      if (usage == "") {
        file_handle.close();
        RETURN_STATUS_UNEXPECTED("Invalid manifest file, 'usage' is missing in file: " + file_ + " at line " +
                                 std::to_string(line_count));
      }
      (void)std::transform(usage.begin(), usage.end(), usage.begin(), ::tolower);
      if (usage != usage_) {
        continue;
      }
      std::vector<std::string> labels;
      nlohmann::json annotations = js.at("annotation");
      for (nlohmann::json::iterator it = annotations.begin(); it != annotations.end(); ++it) {
        nlohmann::json annotation = it.value();
        std::string label_name = annotation.value("name", "");
        classes.insert(label_name);
        if (label_name == "") {
          file_handle.close();
          RETURN_STATUS_UNEXPECTED("Invalid manifest file, 'name' attribute of label is missing in file: " + file_ +
                                   " at line " + std::to_string(line_count));
        }
        if (class_index_.empty() || class_index_.find(label_name) != class_index_.end()) {
          if (label_index_.find(label_name) == label_index_.end()) {
            label_index_[label_name] = 0;
          }
          labels.emplace_back(label_name);
        }
      }
      if (!labels.empty()) {
        image_labelname_.emplace_back(std::make_pair(image_file_path, labels));
      }
      line_count++;
    } catch (const std::exception &err) {
      file_handle.close();
      RETURN_STATUS_UNEXPECTED("Invalid manifest file, parse ManiFest file: " + file_ + " failed, " +
                               std::string(err.what()));
    }
  }
  num_classes_ = classes.size();
  file_handle.close();
  RETURN_IF_NOT_OK(CountDatasetInfo());
  return Status::OK();
}

// Only support JPEG/PNG/GIF/BMP
Status ManifestOp::CheckImageType(const std::string &file_name, bool *valid) {
  auto realpath = FileUtils::GetRealPath(file_name.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << file_name << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + file_name + " does not exist.");
  }

  std::ifstream file_handle;
  constexpr int read_num = 3;
  *valid = false;
  file_handle.open(realpath.value(), std::ios::binary | std::ios::in);
  if (!file_handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid manifest file, failed to open " + file_name +
                             " : the manifest file is damaged or permission denied.");
  }
  unsigned char file_type[read_num];
  (void)file_handle.read(reinterpret_cast<char *>(file_type), read_num);

  if (file_handle.fail()) {
    file_handle.close();
    RETURN_STATUS_UNEXPECTED("Invalid manifest file, failed to read " + file_name +
                             " : the manifest file is damaged or permission denied.");
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

Status ManifestOp::CountDatasetInfo() {
  int32_t index = 0;
  for (auto &label : label_index_) {
    label.second = class_index_.empty() ? index : class_index_[label.first];
    index++;
  }

  num_rows_ = static_cast<int64_t>(image_labelname_.size());
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, ManifestDataset API can't read the data file (interface mismatch or no data found). "
      "Check file path: " +
      file_);
  }
  return Status::OK();
}

Status ManifestOp::CountTotalRows(int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;
  RETURN_IF_NOT_OK(PrepareData());
  *count = static_cast<int64_t>(image_labelname_.size());
  return Status::OK();
}

Status ManifestOp::ComputeColMap() {
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

// Get number of classes
Status ManifestOp::GetNumClasses(int64_t *num_classes) {
  RETURN_UNEXPECTED_IF_NULL(num_classes);
  if (num_classes_ > 0) {
    *num_classes = num_classes_;
    return Status::OK();
  }
  int64_t classes_count;
  RETURN_IF_NOT_OK(PrepareData());
  classes_count = static_cast<int64_t>(label_index_.size());
  *num_classes = classes_count;
  num_classes_ = classes_count;
  return Status::OK();
}

Status ManifestOp::GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) {
  RETURN_UNEXPECTED_IF_NULL(output_class_indexing);
  if ((*output_class_indexing).empty()) {
    RETURN_IF_NOT_OK(PrepareData());
    RETURN_IF_NOT_OK(CountDatasetInfo());
    int32_t count = 0;
    for (const auto &label : label_index_) {
      if (!class_index_.empty()) {
        (*output_class_indexing)
          .emplace_back(std::make_pair(label.first, std::vector<int32_t>(1, class_index_[label.first])));
      } else {
        (*output_class_indexing).emplace_back(std::make_pair(label.first, std::vector<int32_t>(1, count)));
      }
      count++;
    }
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
