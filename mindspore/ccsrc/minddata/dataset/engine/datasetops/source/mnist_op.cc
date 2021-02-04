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
const int32_t kMnistImageFileMagicNumber = 2051;
const int32_t kMnistLabelFileMagicNumber = 2049;
const int32_t kMnistImageRows = 28;
const int32_t kMnistImageCols = 28;

MnistOp::Builder::Builder() : builder_sampler_(nullptr), builder_usage_("") {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_rows_per_buffer_ = cfg->rows_per_buffer();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status MnistOp::Builder::Build(std::shared_ptr<MnistOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  if (builder_sampler_ == nullptr) {
    const int64_t num_samples = 0;
    const int64_t start_index = 0;
    builder_sampler_ = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  }
  builder_schema_ = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(
    builder_schema_->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(builder_schema_->AddColumn(
    ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  *ptr = std::make_shared<MnistOp>(builder_usage_, builder_num_workers_, builder_rows_per_buffer_, builder_dir_,
                                   builder_op_connector_size_, std::move(builder_schema_), std::move(builder_sampler_));
  return Status::OK();
}

Status MnistOp::Builder::SanityCheck() {
  const std::set<std::string> valid = {"test", "train", "all", ""};
  Path dir(builder_dir_);
  std::string err_msg;
  err_msg += dir.IsDirectory() == false
               ? "Invalid parameter, MNIST path is invalid or not set, path: " + builder_dir_ + ".\n"
               : "";
  err_msg += builder_num_workers_ <= 0 ? "Invalid parameter, num_parallel_workers must be greater than 0, but got " +
                                           std::to_string(builder_num_workers_) + ".\n"
                                       : "";
  err_msg += valid.find(builder_usage_) == valid.end()
               ? "Invalid parameter, usage must be 'train','test' or 'all', but got " + builder_usage_ + ".\n"
               : "";
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, err_msg);
}

MnistOp::MnistOp(const std::string &usage, int32_t num_workers, int32_t rows_per_buffer, std::string folder_path,
                 int32_t queue_size, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : ParallelOp(num_workers, queue_size, std::move(sampler)),
      usage_(usage),
      buf_cnt_(0),
      row_cnt_(0),
      folder_path_(folder_path),
      rows_per_buffer_(rows_per_buffer),
      image_path_({}),
      label_path_({}),
      data_schema_(std::move(data_schema)) {
  io_block_queues_.Init(num_workers, queue_size);
}

Status MnistOp::TraversalSampleIds(const std::shared_ptr<Tensor> &sample_ids, std::vector<int64_t> *keys) {
  for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); ++itr) {
    if ((*itr) >= num_rows_) continue;  // index out of bound, skipping
    keys->push_back(*itr);
    row_cnt_++;
    if (row_cnt_ % rows_per_buffer_ == 0) {
      RETURN_IF_NOT_OK(io_block_queues_[buf_cnt_++ % num_workers_]->Add(
        std::make_unique<IOBlock>(IOBlock(*keys, IOBlock::kDeIoBlockNone))));
      keys->clear();
    }
  }
  return Status::OK();
}

// functor that contains the main logic of MNIST op
Status MnistOp::operator()() {
  RETURN_IF_NOT_OK(LaunchThreadsAndInitOp());
  std::unique_ptr<DataBuffer> sampler_buffer;
  RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
  while (true) {  // each iterator is 1 epoch
    std::vector<int64_t> keys;
    keys.reserve(rows_per_buffer_);
    while (sampler_buffer->eoe() == false) {
      std::shared_ptr<Tensor> sample_ids;
      RETURN_IF_NOT_OK(sampler_buffer->GetTensor(&sample_ids, 0, 0));
      if (sample_ids->type() != DataType(DataType::DE_INT64)) {
        RETURN_STATUS_UNEXPECTED("Invalid parameter, data type of Sampler Tensor isn't int64, got " +
                                 sample_ids->type().ToString());
      }
      RETURN_IF_NOT_OK(TraversalSampleIds(sample_ids, &keys));
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    }
    if (keys.empty() == false) {
      RETURN_IF_NOT_OK(io_block_queues_[(buf_cnt_++) % num_workers_]->Add(
        std::make_unique<IOBlock>(IOBlock(keys, IOBlock::kDeIoBlockNone))));
    }
    if (IsLastIteration()) {
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof)));
      for (int32_t i = 0; i < num_workers_; ++i) {
        RETURN_IF_NOT_OK(
          io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
      }
      return Status::OK();
    } else {
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe)));
    }

    if (epoch_sync_flag_) {
      // If epoch_sync_flag_ is set, then master thread sleeps until all the worker threads have finished their job for
      // the current epoch.
      RETURN_IF_NOT_OK(WaitForWorkers());
    }
    // If not the last repeat, self-reset and go to loop again.
    if (!IsLastIteration()) {
      RETURN_IF_NOT_OK(Reset());
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    }
    UpdateRepeatAndEpochCounter();
  }
}

// contains the logic of pulling a IOBlock from IOBlockQueue, load a buffer and push the buffer to out_connector_
Status MnistOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  int64_t buffer_id = worker_id;
  std::unique_ptr<IOBlock> iOBlock;
  RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&iOBlock));
  while (iOBlock != nullptr) {
    if (iOBlock->wait() == true) {
      // Sync io_block is a signal that master thread wants us to pause and sync with other workers.
      // The last guy who comes to this sync point should reset the counter and wake up the master thread.
      if (++num_workers_paused_ == num_workers_) {
        wait_for_workers_post_.Set();
      }
    } else if (iOBlock->eoe() == true) {
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE)));
      buffer_id = worker_id;
    } else if (iOBlock->eof() == true) {
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF)));
    } else {
      std::vector<int64_t> keys;
      RETURN_IF_NOT_OK(iOBlock->GetKeys(&keys));
      if (keys.empty() == true) return Status::OK();  // empty key is a quit signal for workers
      std::unique_ptr<DataBuffer> db = std::make_unique<DataBuffer>(buffer_id, DataBuffer::kDeBFlagNone);
      RETURN_IF_NOT_OK(LoadBuffer(keys, &db));
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::move(db)));
      buffer_id += num_workers_;
    }
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&iOBlock));
  }
  RETURN_STATUS_UNEXPECTED("Unexpected nullptr received in worker.");
}

// Load 1 TensorRow (image,label) using 1 MnistLabelPair.
Status MnistOp::LoadTensorRow(row_id_type row_id, const MnistLabelPair &mnist_pair, TensorRow *trow) {
  std::shared_ptr<Tensor> image, label;
  // make a copy of cached tensor
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(mnist_pair.first, &image));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(mnist_pair.second, &label));

  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  trow->setPath({image_path_[row_id], label_path_[row_id]});
  return Status::OK();
}

// Looping over LoadTensorRow to make 1 DataBuffer. 1 function call produces 1 buffer
Status MnistOp::LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db) {
  std::unique_ptr<TensorQTable> deq = std::make_unique<TensorQTable>();
  TensorRow trow;
  for (const int64_t &key : keys) {
    RETURN_IF_NOT_OK(this->LoadTensorRow(key, image_label_pairs_[key], &trow));
    deq->push_back(std::move(trow));
  }
  (*db)->set_tensor_table(std::move(deq));
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
    out << "\nNumber of rows:" << num_rows_ << "\nMNIST Directory: " << folder_path_ << "\n\n";
  }
}

// Reset Sampler and wakeup Master thread (functor)
Status MnistOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  RETURN_IF_NOT_OK(sampler_->ResetSampler());
  row_cnt_ = 0;
  return Status::OK();
}

// hand shake with Sampler, allow Sampler to call RandomAccessOp's functions to get NumRows
Status MnistOp::InitSampler() {
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
  return Status::OK();
}

// Derived from RandomAccessOp
Status MnistOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || image_label_pairs_.empty()) {
    if (image_label_pairs_.empty()) {
      RETURN_STATUS_UNEXPECTED("No image found in dataset, please check if Op read images successfully or not.");
    } else {
      RETURN_STATUS_UNEXPECTED(
        "Map for storaging image-index pair is nullptr or has been set in other place,"
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
  CHECK_FAIL_RETURN_UNEXPECTED(!reader->fail(), "Invalid data, failed to read 4 bytes from file.");
  *result = SwapEndian(res);
  return Status::OK();
}

uint32_t MnistOp::SwapEndian(uint32_t val) const {
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | (val >> 16);
}

Status MnistOp::CheckImage(const std::string &file_name, std::ifstream *image_reader, uint32_t *num_images) {
  CHECK_FAIL_RETURN_UNEXPECTED(image_reader->is_open(), "Invalid file, failed to open mnist image file: " + file_name);
  int64_t image_len = image_reader->seekg(0, std::ios::end).tellg();
  (void)image_reader->seekg(0, std::ios::beg);
  // The first 16 bytes of the image file are type, number, row and column
  CHECK_FAIL_RETURN_UNEXPECTED(image_len >= 16, "Invalid file, Mnist file is corrupted: " + file_name);

  uint32_t magic_number;
  RETURN_IF_NOT_OK(ReadFromReader(image_reader, &magic_number));
  CHECK_FAIL_RETURN_UNEXPECTED(magic_number == kMnistImageFileMagicNumber,
                               "Invalid file, this is not the mnist image file: " + file_name);

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
  CHECK_FAIL_RETURN_UNEXPECTED(label_reader->is_open(), "Invalid file, failed to open mnist label file: " + file_name);
  int64_t label_len = label_reader->seekg(0, std::ios::end).tellg();
  (void)label_reader->seekg(0, std::ios::beg);
  // The first 8 bytes of the image file are type and number
  CHECK_FAIL_RETURN_UNEXPECTED(label_len >= 8, "Invalid file, Mnist file is corrupted: " + file_name);
  uint32_t magic_number;
  RETURN_IF_NOT_OK(ReadFromReader(label_reader, &magic_number));
  CHECK_FAIL_RETURN_UNEXPECTED(magic_number == kMnistLabelFileMagicNumber,
                               "Invalid file, this is not the mnist label file: " + file_name);
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
  CHECK_FAIL_RETURN_UNEXPECTED((num_images == num_labels), "Invalid data, num_images is not equal to num_labels.");
  // The image size of the Mnist dataset is fixed at [28,28]
  int64_t size = kMnistImageRows * kMnistImageCols;
  auto images_buf = std::make_unique<char[]>(size * num_images);
  auto labels_buf = std::make_unique<char[]>(num_images);
  if (images_buf == nullptr || labels_buf == nullptr) {
    std::string err_msg = "Failed to allocate memory for MNIST buffer.";
    MS_LOG(ERROR) << err_msg.c_str();
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  (void)image_reader->read(images_buf.get(), size * num_images);
  if (image_reader->fail()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to read image: " + image_names_[index] +
                             ", size:" + std::to_string(size * num_images));
  }
  (void)label_reader->read(labels_buf.get(), num_images);
  if (label_reader->fail()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to read label:" + label_names_[index] +
                             ", size: " + std::to_string(num_images));
  }
  TensorShape img_tensor_shape = TensorShape({kMnistImageRows, kMnistImageCols, 1});
  for (int64_t j = 0; j != num_images; ++j) {
    auto pixels = &images_buf[j * size];
    for (int64_t m = 0; m < size; ++m) {
      pixels[m] = (pixels[m] == 0) ? 0 : 255;
    }
    std::shared_ptr<Tensor> image;
    RETURN_IF_NOT_OK(Tensor::CreateFromMemory(img_tensor_shape, data_schema_->column(0).type(),
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
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API MnistDataset. Please check file path or dataset API.");
  }
  return Status::OK();
}

Status MnistOp::WalkAllFiles() {
  const std::string img_ext = "idx3-ubyte";
  const std::string lbl_ext = "idx1-ubyte";
  const std::string train_prefix = "train";
  const std::string test_prefix = "t10k";

  Path dir(folder_path_);
  auto dir_it = Path::DirIterator::OpenDirectory(&dir);
  std::string prefix;  // empty string, used to match usage = "" (default) or usage == "all"
  if (usage_ == "train" || usage_ == "test") prefix = (usage_ == "test" ? test_prefix : train_prefix);
  if (dir_it != nullptr) {
    while (dir_it->hasNext()) {
      Path file = dir_it->next();
      std::string fname = file.Basename();  // name of the mnist file
      if ((fname.find(prefix + "-images") != std::string::npos) && (fname.find(img_ext) != std::string::npos)) {
        image_names_.push_back(file.toString());
        MS_LOG(INFO) << "Mnist operator found image file at " << fname << ".";
      } else if ((fname.find(prefix + "-labels") != std::string::npos) && (fname.find(lbl_ext) != std::string::npos)) {
        label_names_.push_back(file.toString());
        MS_LOG(INFO) << "Mnist Operator found label file at " << fname << ".";
      }
    }
  } else {
    MS_LOG(WARNING) << "Mnist operator unable to open directory " << dir.toString() << ".";
  }

  std::sort(image_names_.begin(), image_names_.end());
  std::sort(label_names_.begin(), label_names_.end());

  CHECK_FAIL_RETURN_UNEXPECTED(image_names_.size() == label_names_.size(),
                               "Invalid data, num of images does not equal to num of labels.");

  return Status::OK();
}

Status MnistOp::LaunchThreadsAndInitOp() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Pipeline init failed, Execution tree not set.");
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
  std::shared_ptr<MnistOp> op;
  *count = 0;
  RETURN_IF_NOT_OK(Builder().SetDir(dir).SetUsage(usage).Build(&op));

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
      column_name_id_map_[data_schema_->column(i).name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
