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
#include "minddata/dataset/data_source/qmnist_op.h"

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
const int32_t kQMnistLabelFileMagicNumber = 3074;
const size_t kQMnistImageRows = 28;
const size_t kQMnistImageCols = 28;
const size_t kQMnistLabelLength = 8;
const uint32_t kNum4 = 4;
const uint32_t kNum12 = 12;

QMnistOp::QMnistOp(const std::string &folder_path, const std::string &usage, bool compat,
                   std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler, int32_t num_workers,
                   int32_t queue_size)
    : MnistOp(usage, num_workers, folder_path, queue_size, std::move(data_schema), std::move(sampler)),
      compat_(compat) {}

void QMnistOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows: " << num_rows_ << "\n"
        << DatasetName(true) << " directory: " << folder_path_ << "\nUsage: " << usage_
        << "\nCompat: " << (compat_ ? "yes" : "no") << "\n\n";
  }
}

// Load 1 TensorRow (image, label) using 1 MnistLabelPair or QMnistImageInfoPair.
Status QMnistOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  size_t unsigned_row_id = static_cast<size_t>(row_id);
  std::shared_ptr<Tensor> image, label;
  if (compat_) {
    MnistLabelPair qmnist_pair = image_label_pairs_[unsigned_row_id];
    RETURN_IF_NOT_OK(Tensor::CreateFromTensor(qmnist_pair.first, &image));
    RETURN_IF_NOT_OK(Tensor::CreateScalar(qmnist_pair.second, &label));
  } else {
    QMnistImageInfoPair qmnist_pair = image_info_pairs_[unsigned_row_id];
    RETURN_IF_NOT_OK(Tensor::CreateFromTensor(qmnist_pair.first, &image));
    RETURN_IF_NOT_OK(Tensor::CreateFromTensor(qmnist_pair.second, &label));
  }
  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  trow->setPath({image_path_[unsigned_row_id], label_path_[unsigned_row_id]});
  return Status::OK();
}

Status QMnistOp::CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;

  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connector_size = cfg->op_connector_size();

  // compat does not affect the count result, so set it to true default.
  auto op =
    std::make_shared<QMnistOp>(dir, usage, true, std::move(schema), std::move(sampler), num_workers, op_connector_size);

  // the logic of counting the number of samples
  RETURN_IF_NOT_OK(op->WalkAllFiles());
  for (size_t i = 0; i < op->image_names_.size(); ++i) {
    std::ifstream image_reader;
    image_reader.open(op->image_names_[i], std::ios::in | std::ios::binary);
    std::ifstream label_reader;
    label_reader.open(op->label_names_[i], std::ios::in | std::ios::binary);

    uint32_t num_images;
    auto s = op->CheckImage(op->image_names_[i], &image_reader, &num_images);
    if (s != Status::OK()) {
      image_reader.close();
      label_reader.close();
      return s;
    }
    uint32_t num_labels;
    s = op->CheckLabel(op->label_names_[i], &label_reader, &num_labels);
    if (s != Status::OK()) {
      image_reader.close();
      label_reader.close();
      return s;
    }
    if (num_images != num_labels) {
      image_reader.close();
      label_reader.close();
      RETURN_STATUS_UNEXPECTED("Invalid data, num of images should be equal to num of labels loading from " + dir +
                               ", but got num of images: " + std::to_string(num_images) +
                               ", num of labels: " + std::to_string(num_labels) + ".");
    }

    if (usage == "test10k") {
      // only use the first 10k samples and drop the last 50k samples
      uint32_t first_10k = 10000;
      num_images = first_10k;
      num_labels = first_10k;
    } else if (usage == "test50k") {
      // only use the last 50k samples and drop the first 10k samples
      uint32_t last_50k = 50000;
      num_images = last_50k;
      num_labels = last_50k;
    }

    *count = *count + num_images;

    // Close the readers
    image_reader.close();
    label_reader.close();
  }

  return Status::OK();
}

Status QMnistOp::WalkAllFiles() {
  const std::string image_ext = "images-idx3-ubyte";
  const std::string label_ext = "labels-idx2-int";
  const std::string train_prefix = "qmnist-train";
  const std::string test_prefix = "qmnist-test";
  const std::string nist_prefix = "xnist";

  auto real_folder_path = FileUtils::GetRealPath(folder_path_.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(real_folder_path.has_value(),
                               "Invalid QMnist folder, " + folder_path_ + " does not exist or permission denied!");
  Path root_dir(real_folder_path.value());

  if (usage_ == "train") {
    image_names_.push_back((root_dir / Path(train_prefix + "-" + image_ext)).ToString());
    label_names_.push_back((root_dir / Path(train_prefix + "-" + label_ext)).ToString());
  } else if (usage_ == "test" || usage_ == "test10k" || usage_ == "test50k") {
    image_names_.push_back((root_dir / Path(test_prefix + "-" + image_ext)).ToString());
    label_names_.push_back((root_dir / Path(test_prefix + "-" + label_ext)).ToString());
  } else if (usage_ == "nist") {
    image_names_.push_back((root_dir / Path(nist_prefix + "-" + image_ext)).ToString());
    label_names_.push_back((root_dir / Path(nist_prefix + "-" + label_ext)).ToString());
  } else if (usage_ == "all") {
    image_names_.push_back((root_dir / Path(train_prefix + "-" + image_ext)).ToString());
    label_names_.push_back((root_dir / Path(train_prefix + "-" + label_ext)).ToString());
    image_names_.push_back((root_dir / Path(test_prefix + "-" + image_ext)).ToString());
    label_names_.push_back((root_dir / Path(test_prefix + "-" + label_ext)).ToString());
    image_names_.push_back((root_dir / Path(nist_prefix + "-" + image_ext)).ToString());
    label_names_.push_back((root_dir / Path(nist_prefix + "-" + label_ext)).ToString());
  } else {
    RETURN_STATUS_UNEXPECTED(
      "Invalid parameter, usage should be \"train\", \"test\", \"test10k\", \"test50k\", "
      "\"nist\" or \"all\", got " +
      usage_);
  }

  CHECK_FAIL_RETURN_UNEXPECTED(
    image_names_.size() == label_names_.size(),
    "Invalid data, num of Qmnist image files should be equal to num of Qmnist label files under directory:" +
      folder_path_ + ", but got num of image files: " + std::to_string(image_names_.size()) +
      ", num of label files: " + std::to_string(label_names_.size()) + ".");

  for (size_t i = 0; i < image_names_.size(); i++) {
    Path file_path(image_names_[i]);
    CHECK_FAIL_RETURN_UNEXPECTED(
      file_path.Exists() && !file_path.IsDirectory(),
      "Invalid file path, Qmnist data file: " + file_path.ToString() + " does not exist or is a directory.");
    MS_LOG(INFO) << DatasetName(true) << " operator found image file at " << file_path.ToString() << ".";
  }

  for (size_t i = 0; i < label_names_.size(); i++) {
    Path file_path(label_names_[i]);
    CHECK_FAIL_RETURN_UNEXPECTED(
      file_path.Exists() && !file_path.IsDirectory(),
      "Invalid file path, Qmnist data file: " + file_path.ToString() + " does not exist or is a directory.");
    MS_LOG(INFO) << DatasetName(true) << " operator found label file at " << file_path.ToString() << ".";
  }

  return Status::OK();
}

Status QMnistOp::ReadImageAndLabel(std::ifstream *image_reader, std::ifstream *label_reader, size_t index) {
  RETURN_UNEXPECTED_IF_NULL(image_reader);
  RETURN_UNEXPECTED_IF_NULL(label_reader);
  uint32_t num_images, num_labels;
  RETURN_IF_NOT_OK(CheckImage(image_names_[index], image_reader, &num_images));
  RETURN_IF_NOT_OK(CheckLabel(label_names_[index], label_reader, &num_labels));
  CHECK_FAIL_RETURN_UNEXPECTED((num_images == num_labels),
                               "Invalid data, num of images should be equal to num of labels loading from " +
                                 folder_path_ + ", but got num of images: " + std::to_string(num_images) +
                                 ", num of labels: " + std::to_string(num_labels) + ".");

  // The image size of the QMNIST dataset is fixed at [28,28]
  size_t image_size = kQMnistImageRows * kQMnistImageCols;
  size_t label_length = kQMnistLabelLength;

  uint32_t first_10k = 10000;
  if (usage_ == "test10k") {
    // only use the first 10k samples and drop the last 50k samples
    num_images = first_10k;
    num_labels = first_10k;
  } else if (usage_ == "test50k") {
    uint32_t last_50k = 50000;
    int num_bytes_for_unint32_t = 4;
    num_images = last_50k;
    num_labels = last_50k;
    // skip the first 10k samples for ifstream reader
    (void)image_reader->ignore(image_size * first_10k);
    (void)label_reader->ignore(label_length * first_10k * num_bytes_for_unint32_t);
  }

  auto images_buf = std::make_unique<char[]>(image_size * num_images);
  auto labels_buf = std::make_unique<uint32_t[]>(label_length * num_labels);
  if (images_buf == nullptr || labels_buf == nullptr) {
    std::string err_msg = "[Internal ERROR] Failed to allocate memory for " + DatasetName() + " buffer.";
    MS_LOG(ERROR) << err_msg.c_str();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  (void)image_reader->read(images_buf.get(), image_size * num_images);
  if (image_reader->fail()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to read " + std::to_string(image_size * num_images) +
                             " bytes from " + image_names_[index] +
                             ": the data file is damaged or the content is incomplete.");
  }
  // uint32_t use 4 bytes in memory
  (void)label_reader->read(reinterpret_cast<char *>(labels_buf.get()), label_length * num_labels * kNum4);
  if (label_reader->fail()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to read " + std::to_string(label_length * num_labels * kNum4) +
                             " bytes from " + label_names_[index] +
                             ": the data file is damaged or content is incomplete.");
  }
  TensorShape image_tensor_shape = TensorShape({kQMnistImageRows, kQMnistImageCols, 1});
  TensorShape label_tensor_shape = TensorShape({kQMnistLabelLength});
  for (size_t data_index = 0; data_index != num_images; data_index++) {
    auto image = &images_buf[data_index * image_size];
    std::shared_ptr<Tensor> image_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateFromMemory(image_tensor_shape, data_schema_->Column(0).Type(),
                                              reinterpret_cast<unsigned char *>(image), &image_tensor));

    auto label = &labels_buf[data_index * label_length];
    for (int64_t label_index = 0; label_index < static_cast<int64_t>(label_length); label_index++) {
      label[label_index] = SwapEndian(label[label_index]);
    }
    std::shared_ptr<Tensor> label_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateFromMemory(label_tensor_shape, data_schema_->Column(1).Type(),
                                              reinterpret_cast<unsigned char *>(label), &label_tensor));

    (void)image_info_pairs_.emplace_back(std::make_pair(image_tensor, label_tensor));
    (void)image_label_pairs_.emplace_back(std::make_pair(image_tensor, label[0]));
    image_path_.push_back(image_names_[index]);
    label_path_.push_back(label_names_[index]);
  }
  return Status::OK();
}

Status QMnistOp::CheckLabel(const std::string &file_name, std::ifstream *label_reader, uint32_t *num_labels) {
  RETURN_UNEXPECTED_IF_NULL(label_reader);
  RETURN_UNEXPECTED_IF_NULL(num_labels);
  CHECK_FAIL_RETURN_UNEXPECTED(label_reader->is_open(),
                               "Invalid file, failed to open " + file_name + ": the label file is permission denied.");
  int64_t label_len = label_reader->seekg(0, std::ios::end).tellg();
  (void)label_reader->seekg(0, std::ios::beg);
  // The first 12 bytes of the label file are type, number and length
  CHECK_FAIL_RETURN_UNEXPECTED(label_len >= kNum12,
                               "Invalid file, load " + file_name +
                                 " failed: the first 12 bytes of the label file should be type, number and length, " +
                                 "but got the first read bytes : " + std::to_string(label_len));
  uint32_t magic_number;
  RETURN_IF_NOT_OK(ReadFromReader(label_reader, &magic_number));
  CHECK_FAIL_RETURN_UNEXPECTED(magic_number == kQMnistLabelFileMagicNumber,
                               "Invalid label file, the number of labels loading from " + file_name + " should be " +
                                 std::to_string(kQMnistLabelFileMagicNumber) + ", but got " +
                                 std::to_string(magic_number) + ".");
  uint32_t num_items;
  RETURN_IF_NOT_OK(ReadFromReader(label_reader, &num_items));
  uint32_t length;
  RETURN_IF_NOT_OK(ReadFromReader(label_reader, &length));
  CHECK_FAIL_RETURN_UNEXPECTED(length == kQMnistLabelLength, "Invalid data, length of every label loading from " +
                                                               file_name + " should be equal to 8, but got " +
                                                               std::to_string(length) + ".");

  CHECK_FAIL_RETURN_UNEXPECTED((label_len - kNum12) == static_cast<int64_t>(num_items * kQMnistLabelLength * kNum4),
                               "Invalid data, the total bytes of labels loading from Qmnist label file: " + file_name +
                                 " should be " + std::to_string(label_len - kNum12) + ", but got " +
                                 std::to_string(num_items * kQMnistLabelLength * kNum4) + ".");
  *num_labels = num_items;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
