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
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"

#include <fstream>
#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
const int32_t kMnistImageFileMagicNumber = 2051;
const int32_t kMnistLabelFileMagicNumber = 2049;
const int32_t kMnistImageRows = 28;
const int32_t kMnistImageCols = 28;

MnistOp::MnistOp(std::string usage, int32_t num_workers, std::string folder_path, int32_t queue_size,
                 std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(std::move(folder_path)),
      usage_(std::move(usage)),
      data_schema_(std::move(data_schema)),
      image_path_({}),
      label_path_({}) {
  io_block_queues_.Init(num_workers, queue_size);
}

// Load 1 TensorRow (image,label) using 1 MnistLabelPair.
Status MnistOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  MnistLabelPair mnist_pair = image_label_pairs_[row_id];
  std::shared_ptr<Tensor> image, label;
  // make a copy of cached tensor
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(mnist_pair.first, &image));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(mnist_pair.second, &label));

  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  trow->setPath({image_path_[row_id], label_path_[row_id]});
  return Status::OK();
}

void MnistOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\n" << DatasetName(true) << " Directory: " << folder_path_ << "\n\n";
  }
}

// Derived from RandomAccessOp
Status MnistOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || image_label_pairs_.empty()) {
    if (image_label_pairs_.empty()) {
      RETURN_STATUS_UNEXPECTED("Invalid data, no image found in " + DatasetName() + " file.");
    } else {
      RETURN_STATUS_UNEXPECTED(
        "[Internal ERROR] Map for containing image-index pair is nullptr or has been set in other place,"
        "it must be empty before using GetClassIds.");
    }
  }
  for (size_t i = 0; i < image_label_pairs_.size(); ++i) {
    (*cls_ids)[image_label_pairs_[i].second].push_back(i);
  }
  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}

Status MnistOp::ReadFromReader(std::ifstream *reader, uint32_t *result) {
  uint32_t res = 0;
  reader->read(reinterpret_cast<char *>(&res), 4);
  CHECK_FAIL_RETURN_UNEXPECTED(!reader->fail(),
                               "Invalid data, failed to read 4 bytes from " + DatasetName() + " file.");
  *result = SwapEndian(res);
  return Status::OK();
}

uint32_t MnistOp::SwapEndian(uint32_t val) const {
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >> 16);
}

Status MnistOp::CheckImage(const std::string &file_name, std::ifstream *image_reader, uint32_t *num_images) {
  CHECK_FAIL_RETURN_UNEXPECTED(image_reader->is_open(),
                               "Invalid file, failed to open " + DatasetName() + " image file: " + file_name);
  int64_t image_len = image_reader->seekg(0, std::ios::end).tellg();
  (void)image_reader->seekg(0, std::ios::beg);
  // The first 16 bytes of the image file are type, number, row and column
  CHECK_FAIL_RETURN_UNEXPECTED(image_len >= 16, "Invalid file, " + DatasetName() + " file is corrupted: " + file_name);

  uint32_t magic_number;
  RETURN_IF_NOT_OK(ReadFromReader(image_reader, &magic_number));
  CHECK_FAIL_RETURN_UNEXPECTED(magic_number == kMnistImageFileMagicNumber,
                               "Invalid file, this is not the " + DatasetName() + " image file: " + file_name);

  uint32_t num_items;
  RETURN_IF_NOT_OK(ReadFromReader(image_reader, &num_items));
  uint32_t rows;
  RETURN_IF_NOT_OK(ReadFromReader(image_reader, &rows));
  uint32_t cols;
  RETURN_IF_NOT_OK(ReadFromReader(image_reader, &cols));
  // The image size of the Mnist dataset is fixed at [28,28]
  CHECK_FAIL_RETURN_UNEXPECTED((rows == kMnistImageRows) && (cols == kMnistImageCols),
                               "Invalid data, shape of image is not equal to (28, 28).");
  CHECK_FAIL_RETURN_UNEXPECTED((image_len - 16) == num_items * rows * cols,
                               "Invalid data, got truncated data len: " + std::to_string(image_len - 16) +
                                 ", which is not equal to real data len: " + std::to_string(num_items * rows * cols));
  *num_images = num_items;
  return Status::OK();
}

Status MnistOp::CheckLabel(const std::string &file_name, std::ifstream *label_reader, uint32_t *num_labels) {
  CHECK_FAIL_RETURN_UNEXPECTED(label_reader->is_open(),
                               "Invalid file, failed to open " + DatasetName() + " label file: " + file_name);
  int64_t label_len = label_reader->seekg(0, std::ios::end).tellg();
  (void)label_reader->seekg(0, std::ios::beg);
  // The first 8 bytes of the image file are type and number
  CHECK_FAIL_RETURN_UNEXPECTED(label_len >= 8, "Invalid file, " + DatasetName() + " file is corrupted: " + file_name);
  uint32_t magic_number;
  RETURN_IF_NOT_OK(ReadFromReader(label_reader, &magic_number));
  CHECK_FAIL_RETURN_UNEXPECTED(magic_number == kMnistLabelFileMagicNumber,
                               "Invalid file, this is not the " + DatasetName() + " label file: " + file_name);
  uint32_t num_items;
  RETURN_IF_NOT_OK(ReadFromReader(label_reader, &num_items));
  CHECK_FAIL_RETURN_UNEXPECTED((label_len - 8) == num_items, "Invalid data, number of labels is wrong.");
  *num_labels = num_items;
  return Status::OK();
}

Status MnistOp::ReadImageAndLabel(std::ifstream *image_reader, std::ifstream *label_reader, size_t index) {
  uint32_t num_images, num_labels;
  RETURN_IF_NOT_OK(CheckImage(image_names_[index], image_reader, &num_images));
  RETURN_IF_NOT_OK(CheckLabel(label_names_[index], label_reader, &num_labels));
  CHECK_FAIL_RETURN_UNEXPECTED((num_images == num_labels),
                               "Invalid data, num_images is not equal to num_labels. Ensure data file is not damaged.");
  // The image size of the Mnist dataset is fixed at [28,28]
  int64_t size = kMnistImageRows * kMnistImageCols;
  auto images_buf = std::make_unique<char[]>(size * num_images);
  auto labels_buf = std::make_unique<char[]>(num_images);
  if (images_buf == nullptr || labels_buf == nullptr) {
    std::string err_msg = "[Internal ERROR] Failed to allocate memory for " + DatasetName() + " buffer.";
    MS_LOG(ERROR) << err_msg.c_str();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  (void)image_reader->read(images_buf.get(), size * num_images);
  if (image_reader->fail()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to read " + DatasetName() + " image: " + image_names_[index] +
                             ", size:" + std::to_string(size * num_images) + ". Ensure data file is not damaged.");
  }
  (void)label_reader->read(labels_buf.get(), num_images);
  if (label_reader->fail()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to read " + DatasetName() + " label:" + label_names_[index] +
                             ", size: " + std::to_string(num_images) + ". Ensure data file is not damaged.");
  }
  TensorShape img_tensor_shape = TensorShape({kMnistImageRows, kMnistImageCols, 1});
  for (int64_t j = 0; j != num_images; ++j) {
    auto pixels = &images_buf[j * size];
    for (int64_t m = 0; m < size; ++m) {
      pixels[m] = (pixels[m] == 0) ? 0 : 255;
    }
    std::shared_ptr<Tensor> image;
    RETURN_IF_NOT_OK(Tensor::CreateFromMemory(img_tensor_shape, data_schema_->Column(0).Type(),
                                              reinterpret_cast<unsigned char *>(pixels), &image));
    image_label_pairs_.emplace_back(std::make_pair(image, labels_buf[j]));
    image_path_.push_back(image_names_[index]);
    label_path_.push_back(label_names_[index]);
  }
  return Status::OK();
}

Status MnistOp::ParseMnistData() {
  // MNIST contains 4 files, idx3 are image files, idx 1 are labels
  // training files contain 60K examples and testing files contain 10K examples
  // t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte  train-images-idx3-ubyte  train-labels-idx1-ubyte
  for (size_t i = 0; i < image_names_.size(); ++i) {
    std::ifstream image_reader, label_reader;
    image_reader.open(image_names_[i], std::ios::binary);
    label_reader.open(label_names_[i], std::ios::binary);

    Status s = ReadImageAndLabel(&image_reader, &label_reader, i);
    // Close the readers
    image_reader.close();
    label_reader.close();
    RETURN_IF_NOT_OK(s);
  }
  image_label_pairs_.shrink_to_fit();
  num_rows_ = image_label_pairs_.size();
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED("Invalid data, " + DatasetName(true) +
                             "Dataset API can't read the data file (interface mismatch or no data found). Check " +
                             DatasetName() + " file in directory: " + folder_path_);
  }
  return Status::OK();
}

Status MnistOp::WalkAllFiles() {
  const std::string img_ext = "idx3-ubyte";
  const std::string lbl_ext = "idx1-ubyte";
  const std::string train_prefix = "train";
  const std::string test_prefix = "t10k";

  std::string real_path{""};
  RETURN_IF_NOT_OK(Path::RealPath(folder_path_, real_path));
  Path dir(real_path);
  auto dir_it = Path::DirIterator::OpenDirectory(&dir);
  std::string prefix;  // empty string, used to match usage = "" (default) or usage == "all"
  if (usage_ == "train" || usage_ == "test") prefix = (usage_ == "test" ? test_prefix : train_prefix);
  if (dir_it != nullptr) {
    while (dir_it->HasNext()) {
      Path file = dir_it->Next();
      std::string fname = file.Basename();  // name of the mnist file
      if ((fname.find(prefix + "-images") != std::string::npos) && (fname.find(img_ext) != std::string::npos)) {
        image_names_.push_back(file.ToString());
        MS_LOG(INFO) << DatasetName(true) << " operator found image file at " << fname << ".";
      } else if ((fname.find(prefix + "-labels") != std::string::npos) && (fname.find(lbl_ext) != std::string::npos)) {
        label_names_.push_back(file.ToString());
        MS_LOG(INFO) << DatasetName(true) << " Operator found label file at " << fname << ".";
      }
    }
  } else {
    MS_LOG(WARNING) << DatasetName(true) << " operator unable to open directory " << dir.ToString() << ".";
  }

  std::sort(image_names_.begin(), image_names_.end());
  std::sort(label_names_.begin(), label_names_.end());

  CHECK_FAIL_RETURN_UNEXPECTED(image_names_.size() == label_names_.size(),
                               "Invalid data, num of images does not equal to num of labels.");

  return Status::OK();
}

Status MnistOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Pipeline init failed, Execution tree not set.");
  }
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&MnistOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(this->WalkAllFiles());
  RETURN_IF_NOT_OK(this->ParseMnistData());
  RETURN_IF_NOT_OK(this->InitSampler());  // handle shake with sampler
  return Status::OK();
}

Status MnistOp::CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count) {
  // the logic of counting the number of samples is copied from ParseMnistData() and uses CheckReader()
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
  auto op = std::make_shared<MnistOp>(usage, num_workers, dir, op_connect_size, std::move(schema), std::move(sampler));
  RETURN_IF_NOT_OK(op->WalkAllFiles());

  for (size_t i = 0; i < op->image_names_.size(); ++i) {
    std::ifstream image_reader;
    image_reader.open(op->image_names_[i], std::ios::binary);
    std::ifstream label_reader;
    label_reader.open(op->label_names_[i], std::ios::binary);

    uint32_t num_images;
    RETURN_IF_NOT_OK(op->CheckImage(op->image_names_[i], &image_reader, &num_images));
    uint32_t num_labels;
    RETURN_IF_NOT_OK(op->CheckLabel(op->label_names_[i], &label_reader, &num_labels));
    CHECK_FAIL_RETURN_UNEXPECTED((num_images == num_labels),
                                 "Invalid data, num of images is not equal to num of labels.");
    *count = *count + num_images;

    // Close the readers
    image_reader.close();
    label_reader.close();
  }

  return Status::OK();
}

Status MnistOp::ComputeColMap() {
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
