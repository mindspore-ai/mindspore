/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/data_source/emnist_op.h"

#include <fstream>

#include "include/utils/common.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/data_source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
EMnistOp::EMnistOp(const std::string &name, const std::string &usage, int32_t num_workers,
                   const std::string &folder_path, int32_t queue_size, std::unique_ptr<DataSchema> data_schema,
                   std::shared_ptr<SamplerRT> sampler)
    : MnistOp(usage, num_workers, folder_path, queue_size, std::move(data_schema), std::move(sampler)), name_(name) {}

void EMnistOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nNumber of rows:" << num_rows_ << "\n"
        << DatasetName(true) << " directory: " << folder_path_ << "\nName: " << name_ << "\nUsage: " << usage_
        << "\n\n";
  }
}

Status EMnistOp::WalkAllFiles() {
  const std::string img_ext = "-images-idx3-ubyte";
  const std::string lbl_ext = "-labels-idx1-ubyte";
  const std::string train_prefix = "-train";
  const std::string test_prefix = "-test";
  auto realpath = FileUtils::GetRealPath(folder_path_.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(realpath.has_value(), "Invalid file path, " + folder_path_ + " does not exist.");
  Path dir(realpath.value());
  auto dir_it = Path::DirIterator::OpenDirectory(&dir);
  if (dir_it == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid path, failed to open emnist dataset dir: " + dir.ToString() +
                             ", the directory is not a directory or permission denied.");
  }
  std::string prefix;
  prefix = "emnist-" + name_;  // used to match usage == "all".
  if (usage_ == "train" || usage_ == "test") {
    prefix += (usage_ == "test" ? test_prefix : train_prefix);
  }
  if (dir_it != nullptr) {
    while (dir_it->HasNext()) {
      Path file = dir_it->Next();
      std::string fname = file.Basename();  // name of the emnist file.
      if ((fname.find(prefix) != std::string::npos) && (fname.find(img_ext) != std::string::npos)) {
        image_names_.push_back(file.ToString());
        MS_LOG(INFO) << DatasetName(true) << " operator found image file at " << fname << ".";
      } else if ((fname.find(prefix) != std::string::npos) && (fname.find(lbl_ext) != std::string::npos)) {
        label_names_.push_back(file.ToString());
        MS_LOG(INFO) << DatasetName(true) << " operator found label file at " << fname << ".";
      }
    }
  } else {
    MS_LOG(WARNING) << DatasetName(true) << " operator unable to open directory " << dir.ToString() << ".";
  }

  std::sort(image_names_.begin(), image_names_.end());
  std::sort(label_names_.begin(), label_names_.end());
  CHECK_FAIL_RETURN_UNEXPECTED(image_names_.size() == label_names_.size(),
                               "Invalid data, num of image files should be equal to num of label files under " +
                                 realpath.value() + ", but got num of images: " + std::to_string(image_names_.size()) +
                                 ", num of labels: " + std::to_string(label_names_.size()) + ".");

  return Status::OK();
}

Status EMnistOp::CountTotalRows(const std::string &dir, const std::string &name, const std::string &usage,
                                int64_t *count) {
  // the logic of counting the number of samples is copied from ParseEMnistData() and uses CheckReader().
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;

  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  auto op =
    std::make_shared<EMnistOp>(name, usage, num_workers, dir, op_connect_size, std::move(schema), std::move(sampler));

  RETURN_IF_NOT_OK(op->WalkAllFiles());

  for (size_t i = 0; i < op->image_names_.size(); ++i) {
    std::ifstream image_reader;
    image_reader.open(op->image_names_[i], std::ios::in | std::ios::binary);
    CHECK_FAIL_RETURN_UNEXPECTED(image_reader.is_open(), "Invalid file, failed to open " + op->image_names_[i] +
                                                           ": the image file is damaged or permission denied.");
    std::ifstream label_reader;
    label_reader.open(op->label_names_[i], std::ios::in | std::ios::binary);
    CHECK_FAIL_RETURN_UNEXPECTED(label_reader.is_open(), "Invalid file, failed to open " + op->label_names_[i] +
                                                           ": the label file is damaged or permission denied.");
    uint32_t num_images;
    Status s = op->CheckImage(op->image_names_[i], &image_reader, &num_images);
    image_reader.close();
    RETURN_IF_NOT_OK(s);

    uint32_t num_labels;
    s = op->CheckLabel(op->label_names_[i], &label_reader, &num_labels);
    label_reader.close();
    RETURN_IF_NOT_OK(s);

    CHECK_FAIL_RETURN_UNEXPECTED(
      (num_images == num_labels),
      "Invalid data, num of images should be equal to num of labels, but got num of images: " +
        std::to_string(num_images) + ", num of labels: " + std::to_string(num_labels) + ".");
    *count = *count + num_images;
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
