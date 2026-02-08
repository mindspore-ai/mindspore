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
#include "minddata/dataset/data_source/stl10_op.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <set>
#include <utility>

#include "include/utils/common.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/data_source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
constexpr uint32_t kSTLImageRows = 96;
constexpr uint32_t kSTLImageCols = 96;
constexpr uint32_t kSTLImageChannel = 3;
constexpr uint32_t kSTLImageSize = kSTLImageRows * kSTLImageCols * kSTLImageChannel;

STL10Op::STL10Op(const std::string &usage, int32_t num_workers, const std::string &folder_path, int32_t queue_size,
                 std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(folder_path),
      usage_(usage),
      data_schema_(std::move(data_schema)),
      image_path_({}),
      label_path_({}) {}

Status STL10Op::LoadTensorRow(row_id_type index, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::pair<std::shared_ptr<Tensor>, int32_t> stl10_pair = stl10_image_label_pairs_[index];
  std::shared_ptr<Tensor> image, label;
  // make a copy of cached tensor.
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(stl10_pair.first, &image));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(stl10_pair.second, &label));

  (*trow) = TensorRow(index, {std::move(image), std::move(label)});
  trow->setPath({image_path_[index], label_path_[index]});

  return Status::OK();
}

Status STL10Op::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || stl10_image_label_pairs_.empty()) {
    if (stl10_image_label_pairs_.empty()) {
      RETURN_STATUS_UNEXPECTED("No image found in dataset. Check if image was generated successfully.");
    } else {
      RETURN_STATUS_UNEXPECTED(
        "[Internal ERROR] Map for containing image-index pair is nullptr or has been set in other place, "
        "it must be empty before using GetClassIds.");
    }
  }
  for (size_t i = 0; i < stl10_image_label_pairs_.size(); ++i) {
    (*cls_ids)[stl10_image_label_pairs_[i].second].push_back(i);
  }
  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}

void STL10Op::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nNumber of rows: " << num_rows_ << "\nSTL10 directory: " << folder_path_ << "\n\n";
  }
}

Status STL10Op::WalkAllFiles() {
  auto real_dataset_dir = FileUtils::GetRealPath(folder_path_.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(real_dataset_dir.has_value(),
                               "Invalid file, get real path failed, path: " + folder_path_);
  Path root_dir(real_dataset_dir.value());

  const Path train_data_file("train_X.bin");
  const Path train_label_file("train_y.bin");
  const Path test_data_file("test_X.bin");
  const Path test_label_file("test_y.bin");
  const Path unlabeled_data_file("unlabeled_X.bin");

  bool use_train = false;
  bool use_test = false;
  bool use_unlabeled = false;

  if (usage_ == "train") {
    use_train = true;
  } else if (usage_ == "test") {
    use_test = true;
  } else if (usage_ == "unlabeled") {
    use_unlabeled = true;
  } else if (usage_ == "train+unlabeled") {
    use_train = true;
    use_unlabeled = true;
  } else if (usage_ == "all") {
    use_train = true;
    use_test = true;
    use_unlabeled = true;
  } else {
    RETURN_STATUS_UNEXPECTED(
      "Invalid parameter, usage should be \"train\", \"test\", \"unlabeled\", "
      "\"train+unlabeled\", \"all\", got " +
      usage_);
  }

  if (use_train) {
    Path train_data_path = root_dir / train_data_file;
    Path train_label_path = root_dir / train_label_file;
    CHECK_FAIL_RETURN_UNEXPECTED(
      train_data_path.Exists() && !train_data_path.IsDirectory(),
      "Invalid file, failed to find STL10 " + usage_ + " data file: " + train_data_path.ToString());
    CHECK_FAIL_RETURN_UNEXPECTED(
      train_label_path.Exists() && !train_label_path.IsDirectory(),
      "Invalid file, failed to find STL10 " + usage_ + " label file: " + train_label_path.ToString());
    image_names_.push_back(train_data_path.ToString());
    label_names_.push_back(train_label_path.ToString());
    MS_LOG(INFO) << "STL10 operator found train data file " << train_data_path.ToString() << ".";
    MS_LOG(INFO) << "STL10 operator found train label file " << train_label_path.ToString() << ".";
  }

  if (use_test) {
    Path test_data_path = root_dir / test_data_file;
    Path test_label_path = root_dir / test_label_file;
    CHECK_FAIL_RETURN_UNEXPECTED(
      test_data_path.Exists() && !test_data_path.IsDirectory(),
      "Invalid file, failed to find STL10 " + usage_ + " data file: " + test_data_path.ToString());
    CHECK_FAIL_RETURN_UNEXPECTED(
      test_label_path.Exists() && !test_label_path.IsDirectory(),
      "Invalid file, failed to find STL10 " + usage_ + " label file: " + test_label_path.ToString());
    image_names_.push_back(test_data_path.ToString());
    label_names_.push_back(test_label_path.ToString());
    MS_LOG(INFO) << "STL10 operator found test data file " << test_data_path.ToString() << ".";
    MS_LOG(INFO) << "STL10 operator found test label file " << test_label_path.ToString() << ".";
  }

  if (use_unlabeled) {
    Path unlabeled_data_path = root_dir / unlabeled_data_file;
    CHECK_FAIL_RETURN_UNEXPECTED(
      unlabeled_data_path.Exists() && !unlabeled_data_path.IsDirectory(),
      "Invalid file, failed to find STL10 " + usage_ + " data file: " + unlabeled_data_path.ToString());
    image_names_.push_back(unlabeled_data_path.ToString());
    MS_LOG(INFO) << "STL10 operator found unlabeled data file " << unlabeled_data_path.ToString() << ".";
  }

  std::sort(image_names_.begin(), image_names_.end());
  std::sort(label_names_.begin(), label_names_.end());

  return Status::OK();
}

Status STL10Op::ParseSTLData() {
  // STL10 contains 5 files, *_X.bin are image files, *_y.bin are labels.
  // training files contain 5k images and testing files contain 8K examples.
  // unlabeled file contain 10k images and they DO NOT have labels (i.e. no "unlabeled_y.bin" file).
  for (size_t i = 0; i < image_names_.size(); ++i) {
    std::ifstream image_reader, label_reader;
    if (image_names_[i].find("unlabeled") == std::string::npos) {
      image_reader.open(image_names_[i], std::ios::binary | std::ios::ate);
      label_reader.open(label_names_[i], std::ios::binary | std::ios::ate);

      Status s = ReadImageAndLabel(&image_reader, &label_reader, i);
      // Close the readers.
      image_reader.close();
      label_reader.close();

      RETURN_IF_NOT_OK(s);
    } else {  // unlabeled data -> no labels.
      image_reader.open(image_names_[i], std::ios::binary | std::ios::ate);

      Status s = ReadImageAndLabel(&image_reader, nullptr, i);
      // Close the readers.
      image_reader.close();

      RETURN_IF_NOT_OK(s);
    }
  }
  stl10_image_label_pairs_.shrink_to_fit();
  num_rows_ = stl10_image_label_pairs_.size();
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API STL10Dataset. Please check file path or dataset API.");
  }

  return Status::OK();
}

Status STL10Op::ReadImageAndLabel(std::ifstream *image_reader, std::ifstream *label_reader, size_t index) {
  RETURN_UNEXPECTED_IF_NULL(image_reader);

  Path image_path(image_names_[index]);
  bool has_label_file = image_path.Basename().find("unlabeled") == std::string::npos;

  std::streamsize image_size = image_reader->tellg();

  image_reader->seekg(0, std::ios::beg);
  auto images_buf = std::make_unique<char[]>(image_size);
  auto labels_buf = std::make_unique<char[]>(0);

  if (images_buf == nullptr) {
    std::string err_msg = "Failed to allocate memory for STL10 buffer.";
    MS_LOG(ERROR) << err_msg.c_str();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  uint64_t num_images = static_cast<uint64_t>(image_size / kSTLImageSize);
  (void)image_reader->read(images_buf.get(), image_size);
  if (image_reader->fail()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to read image: " + image_names_[index] +
                             ", size:" + std::to_string(kSTLImageSize * num_images));
  }

  if (has_label_file) {
    RETURN_UNEXPECTED_IF_NULL(label_reader);
    std::streamsize label_size = label_reader->tellg();
    if (static_cast<uint64_t>(label_size) != num_images) {
      RETURN_STATUS_UNEXPECTED("Invalid file, error in " + label_names_[index] +
                               ": the number of labels is not equal to the number of images in " + image_names_[index] +
                               "! Please check the file integrity!");
    }

    label_reader->seekg(0, std::ios::beg);
    labels_buf = std::make_unique<char[]>(label_size);
    if (labels_buf == nullptr) {
      std::string err_msg = "Failed to allocate memory for STL10 buffer.";
      MS_LOG(ERROR) << err_msg.c_str();
      RETURN_STATUS_UNEXPECTED(err_msg);
    }

    (void)label_reader->read(labels_buf.get(), label_size);
    if (label_reader->fail()) {
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to read label:" + label_names_[index] +
                               ", size: " + std::to_string(num_images));
    }
  }

  for (int64_t j = 0; j < num_images; ++j) {
    int32_t label = (has_label_file ? labels_buf[j] - 1 : -1);

    std::shared_ptr<Tensor> image_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({kSTLImageRows, kSTLImageCols, kSTLImageChannel}),
                                         data_schema_->Column(0).Type(), &image_tensor));

    auto iter = image_tensor->begin<uint8_t>();
    uint64_t total_pix = kSTLImageRows * kSTLImageCols;
    // stl10: Column major order.
    for (uint64_t count = 0, pix = 0; count < total_pix; count++) {
      if (count % kSTLImageRows == 0) {
        pix = count / kSTLImageRows;
      }

      for (int ch = 0; ch < kSTLImageChannel; ch++) {
        *iter = images_buf[j * kSTLImageSize + ch * total_pix + pix];
        iter++;
      }
      pix += kSTLImageRows;
    }

    (void)stl10_image_label_pairs_.emplace_back(std::make_pair(image_tensor, label));
    image_path_.push_back(image_names_[index]);
    label_path_.push_back(has_label_file ? label_names_[index] : "no label");
  }

  return Status::OK();
}

Status STL10Op::PrepareData() {
  RETURN_IF_NOT_OK(this->WalkAllFiles());
  RETURN_IF_NOT_OK(this->ParseSTLData());  // Parse stl10 data and get num rows, blocking.

  return Status::OK();
}

Status STL10Op::CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  // the logic of counting the number of samples is copied from ParseSTLData().
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_INT32), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  auto op = std::make_shared<STL10Op>(usage, num_workers, dir, op_connect_size, std::move(schema), std::move(sampler));

  RETURN_IF_NOT_OK(op->WalkAllFiles());

  bool use_train = false;
  bool use_test = false;
  bool use_unlabeled = false;

  if (usage == "train") {
    use_train = true;
  } else if (usage == "test") {
    use_test = true;
  } else if (usage == "unlabeled") {
    use_unlabeled = true;
  } else if (usage == "train+unlabeled") {
    use_train = true;
    use_unlabeled = true;
  } else if (usage == "all") {
    use_train = true;
    use_test = true;
    use_unlabeled = true;
  } else {
    RETURN_STATUS_UNEXPECTED(
      "Invalid parameter, usage should be \"train\", \"test\", \"unlabeled\", "
      "\"train+unlabeled\", \"all\", got " +
      usage);
  }

  *count = 0;
  uint64_t num_stl10_records = 0;
  uint64_t total_image_size = 0;

  if (use_train) {
    uint32_t index = (usage == "all" ? 1 : 0);
    Path train_image_path(op->image_names_[index]);
    CHECK_FAIL_RETURN_UNEXPECTED(train_image_path.Exists() && !train_image_path.IsDirectory(),
                                 "Invalid file, failed to open stl10 file: " + train_image_path.ToString());

    std::ifstream train_image_file(train_image_path.ToString(), std::ios::binary | std::ios::ate);
    CHECK_FAIL_RETURN_UNEXPECTED(train_image_file.is_open(),
                                 "Invalid file, failed to open stl10 file: " + train_image_path.ToString());
    total_image_size += static_cast<uint64_t>(train_image_file.tellg());

    train_image_file.close();
  }

  if (use_test) {
    uint32_t index = 0;
    Path test_image_path(op->image_names_[index]);
    CHECK_FAIL_RETURN_UNEXPECTED(test_image_path.Exists() && !test_image_path.IsDirectory(),
                                 "Invalid file, failed to open stl10 file: " + test_image_path.ToString());

    std::ifstream test_image_file(test_image_path.ToString(), std::ios::binary | std::ios::ate);
    CHECK_FAIL_RETURN_UNEXPECTED(test_image_file.is_open(),
                                 "Invalid file, failed to open stl10 file: " + test_image_path.ToString());
    total_image_size += static_cast<uint64_t>(test_image_file.tellg());

    test_image_file.close();
  }

  if (use_unlabeled) {
    uint32_t index = (usage == "unlabeled" ? 0 : (usage == "train+unlabeled" ? 1 : 2));
    Path unlabeled_image_path(op->image_names_[index]);
    CHECK_FAIL_RETURN_UNEXPECTED(unlabeled_image_path.Exists() && !unlabeled_image_path.IsDirectory(),
                                 "Invalid file, failed to open stl10 file: " + unlabeled_image_path.ToString());

    std::ifstream unlabeled_image_file(unlabeled_image_path.ToString(), std::ios::binary | std::ios::ate);
    CHECK_FAIL_RETURN_UNEXPECTED(unlabeled_image_file.is_open(),
                                 "Invalid file, failed to open stl10 file: " + unlabeled_image_path.ToString());
    total_image_size += static_cast<uint64_t>(unlabeled_image_file.tellg());

    unlabeled_image_file.close();
  }

  num_stl10_records = static_cast<uint64_t>(total_image_size / kSTLImageSize);

  *count = *count + num_stl10_records;

  return Status::OK();
}

Status STL10Op::ComputeColMap() {
  // set the column Name map (base class field).
  if (column_name_id_map_.empty()) {
    for (uint32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column Name map is already set!";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
