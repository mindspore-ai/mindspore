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
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include <fstream>
#include <unordered_set>
#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
ImageFolderOp::Builder::Builder() : builder_decode_(false), builder_recursive_(false), builder_sampler_(nullptr) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_rows_per_buffer_ = cfg->rows_per_buffer();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status ImageFolderOp::Builder::Build(std::shared_ptr<ImageFolderOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  if (builder_sampler_ == nullptr) {
    const int64_t num_samples = 0;  // default num samples of 0 means to sample entire set of data
    const int64_t start_index = 0;
    builder_sampler_ = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  }
  builder_schema_ = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    builder_schema_->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(builder_schema_->AddColumn(
    ColDescriptor("label", DataType(DataType::DE_INT32), TensorImpl::kFlexible, 0, &scalar)));
  *ptr = std::make_shared<ImageFolderOp>(builder_num_workers_, builder_rows_per_buffer_, builder_dir_,
                                         builder_op_connector_size_, builder_recursive_, builder_decode_,
                                         builder_extensions_, builder_labels_to_read_, std::move(builder_schema_),
                                         std::move(builder_sampler_));
  return Status::OK();
}

Status ImageFolderOp::Builder::SanityCheck() {
  Path dir(builder_dir_);
  std::string err_msg;
  err_msg += dir.IsDirectory() == false
               ? "Invalid parameter, ImageFolder path is invalid or not set, path: " + builder_dir_ + ".\n"
               : "";
  err_msg += builder_num_workers_ <= 0 ? "Invalid parameter, num_parallel_workers must be greater than 0, but got " +
                                           std::to_string(builder_num_workers_) + ".\n"
                                       : "";
  return err_msg.empty() ? Status::OK() : Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, err_msg);
}

ImageFolderOp::ImageFolderOp(int32_t num_wkrs, int32_t rows_per_buffer, std::string file_dir, int32_t queue_size,
                             bool recursive, bool do_decode, const std::set<std::string> &exts,
                             const std::map<std::string, int32_t> &map, std::unique_ptr<DataSchema> data_schema,
                             std::shared_ptr<SamplerRT> sampler)
    : ParallelOp(num_wkrs, queue_size, std::move(sampler)),
      rows_per_buffer_(rows_per_buffer),
      folder_path_(file_dir),
      recursive_(recursive),
      decode_(do_decode),
      extensions_(exts),
      class_index_(map),
      data_schema_(std::move(data_schema)),
      row_cnt_(0),
      buf_cnt_(0),
      sampler_ind_(0),
      dirname_offset_(0) {
  folder_name_queue_ = std::make_unique<Queue<std::string>>(num_wkrs * queue_size);
  image_name_queue_ = std::make_unique<Queue<FolderImagesPair>>(num_wkrs * queue_size);
  io_block_queues_.Init(num_workers_, queue_size);
}

// Master thread that pulls the prescan worker's results.
// Keep collecting results until all prescan workers quit
// Then consolidate 2 level shuffles together into 1 giant vector
// calculate numRows then return
Status ImageFolderOp::PrescanMasterEntry(const std::string &filedir) {
  std::vector<FolderImagesPair> v;
  int64_t cnt = 0;
  while (cnt != num_workers_) {  // count number of end signals
    FolderImagesPair p;
    RETURN_IF_NOT_OK(image_name_queue_->PopFront(&p));
    if (p == nullptr) {
      cnt++;
    } else {
      v.push_back(p);
    }
  }
  std::sort(v.begin(), v.end(),
            [](const FolderImagesPair &lhs, const FolderImagesPair &rhs) { return lhs->first < rhs->first; });
  // following loop puts the 2 level of shuffles together into 1 vector
  for (size_t ind = 0; ind < v.size(); ++ind) {
    while (v[ind]->second.empty() == false) {
      MS_ASSERT(!(v[ind]->first.empty()));  // make sure that v[ind]->first.substr(1) is not out of bound
      v[ind]->second.front()->second = class_index_.empty() ? ind : class_index_[v[ind]->first.substr(1)];
      image_label_pairs_.push_back(v[ind]->second.front());
      v[ind]->second.pop();
    }
  }
  image_label_pairs_.shrink_to_fit();
  num_rows_ = image_label_pairs_.size();
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, no valid data matching the dataset API ImageFolderDataset. "
      "Please check file path or dataset API.");
  }
  // free memory of two queues used for pre-scan
  folder_name_queue_->Reset();
  image_name_queue_->Reset();
  return Status::OK();
}

// Main logic, Register Queue with TaskGroup, launch all threads and do the functor's work
Status ImageFolderOp::operator()() {
  RETURN_IF_NOT_OK(LaunchThreadsAndInitOp());
  std::unique_ptr<DataBuffer> sampler_buffer;
  RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
  while (true) {  // each iterator is 1 epoch
    std::vector<int64_t> keys;
    keys.reserve(rows_per_buffer_);
    while (sampler_buffer->eoe() == false) {
      TensorRow sample_row;
      RETURN_IF_NOT_OK(sampler_buffer->PopRow(&sample_row));
      std::shared_ptr<Tensor> sample_ids = sample_row[0];
      for (auto itr = sample_ids->begin<int64_t>(); itr != sample_ids->end<int64_t>(); ++itr) {
        if ((*itr) >= num_rows_) continue;  // index out of bound, skipping
        keys.push_back(*itr);
        row_cnt_++;
        if (row_cnt_ % rows_per_buffer_ == 0) {
          RETURN_IF_NOT_OK(
            io_block_queues_[buf_cnt_++ % num_workers_]->Add(std::make_unique<IOBlock>(keys, IOBlock::kDeIoBlockNone)));
          keys.clear();
        }
      }
      RETURN_IF_NOT_OK(sampler_->GetNextSample(&sampler_buffer));
    }
    if (keys.empty() == false) {
      RETURN_IF_NOT_OK(
        io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::make_unique<IOBlock>(keys, IOBlock::kDeIoBlockNone)));
    }
    if (IsLastIteration()) {
      std::unique_ptr<IOBlock> eoe_block = std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEoe);
      std::unique_ptr<IOBlock> eof_block = std::make_unique<IOBlock>(IOBlock::kDeIoBlockFlagEof);
      RETURN_IF_NOT_OK(io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::move(eoe_block)));
      RETURN_IF_NOT_OK(io_block_queues_[(buf_cnt_++) % num_workers_]->Add(std::move(eof_block)));
      for (int32_t i = 0; i < num_workers_; ++i) {
        RETURN_IF_NOT_OK(
          io_block_queues_[i]->Add(std::make_unique<IOBlock>(std::vector<int64_t>(), IOBlock::kDeIoBlockNone)));
      }
      return Status::OK();
    } else {  // not the last repeat.
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

// contains the main logic of pulling a IOBlock from IOBlockQueue, load a buffer and push the buffer to out_connector_
// IMPORTANT: 1 IOBlock produces 1 DataBuffer
Status ImageFolderOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  int64_t buffer_id = worker_id;
  std::unique_ptr<IOBlock> io_block;
  RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  while (io_block != nullptr) {
    if (io_block->wait() == true) {
      // Sync io_block is a signal that master thread wants us to pause and sync with other workers.
      // The last guy who comes to this sync point should reset the counter and wake up the master thread.
      if (++num_workers_paused_ == num_workers_) {
        wait_for_workers_post_.Set();
      }
    } else if (io_block->eoe() == true) {
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE)));
      buffer_id = worker_id;
    } else if (io_block->eof() == true) {
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF)));
    } else {
      std::vector<int64_t> keys;
      RETURN_IF_NOT_OK(io_block->GetKeys(&keys));
      if (keys.empty() == true) return Status::OK();  // empty key is a quit signal for workers
      std::unique_ptr<DataBuffer> db = std::make_unique<DataBuffer>(buffer_id, DataBuffer::kDeBFlagNone);
      RETURN_IF_NOT_OK(LoadBuffer(keys, &db));
      RETURN_IF_NOT_OK(out_connector_->Add(worker_id, std::move(db)));
      buffer_id += num_workers_;
    }
    RETURN_IF_NOT_OK(io_block_queues_[worker_id]->PopFront(&io_block));
  }
  RETURN_STATUS_UNEXPECTED("Unexpected nullptr received in worker");
}

// Load 1 TensorRow (image,label) using 1 ImageLabelPair. 1 function call produces 1 TensorTow in a DataBuffer
Status ImageFolderOp::LoadTensorRow(row_id_type row_id, ImageLabelPair pairPtr, TensorRow *trow) {
  std::shared_ptr<Tensor> image, label;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(pairPtr->second, &label));
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(folder_path_ + (pairPtr->first), &image));

  if (decode_ == true) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      std::string err = "Invalid data, failed to decode image: " + folder_path_ + (pairPtr->first);
      RETURN_STATUS_UNEXPECTED(err);
    }
  }
  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  trow->setPath({folder_path_ + (pairPtr->first), std::string("")});
  return Status::OK();
}

// Looping over LoadTensorRow to make 1 DataBuffer. 1 function call produces 1 buffer
Status ImageFolderOp::LoadBuffer(const std::vector<int64_t> &keys, std::unique_ptr<DataBuffer> *db) {
  std::unique_ptr<TensorQTable> deq = std::make_unique<TensorQTable>();
  TensorRow trow;
  for (const int64_t &key : keys) {
    RETURN_IF_NOT_OK(this->LoadTensorRow(key, image_label_pairs_[key], &trow));
    deq->push_back(std::move(trow));
  }
  (*db)->set_tensor_table(std::move(deq));
  return Status::OK();
}

void ImageFolderOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nImageFolder directory: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

// Reset Sampler and wakeup Master thread (functor)
Status ImageFolderOp::Reset() {
  MS_LOG(DEBUG) << Name() << " performing a self-reset.";
  RETURN_IF_NOT_OK(sampler_->ResetSampler());
  row_cnt_ = 0;
  return Status::OK();
}

// hand shake with Sampler, allow Sampler to call RandomAccessOp's functions to get NumRows
Status ImageFolderOp::InitSampler() {
  RETURN_IF_NOT_OK(sampler_->HandshakeRandomAccessOp(this));
  return Status::OK();
}

// Derived from RandomAccessOp
Status ImageFolderOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || image_label_pairs_.empty()) {
    if (image_label_pairs_.empty()) {
      RETURN_STATUS_UNEXPECTED("No images found in dataset, please check if Op read images successfully or not.");
    } else {
      RETURN_STATUS_UNEXPECTED(
        "Map containing image-index pair is nullptr or has been set in other place,"
        "it must be empty before using GetClassIds.");
    }
  }
  for (size_t i = 0; i < image_label_pairs_.size(); ++i) {
    (*cls_ids)[image_label_pairs_[i]->second].push_back(i);
  }
  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}

// Worker Entry for pre-scanning all the folders and do the 1st level shuffle
// Worker pull a file name from folder_name_queue_ (which is a Queue), walks all the images under that foldername
// After walking is complete, sort all the file names (relative path to all jpeg files under the same directory )
// (Sort is automatically conducted using a set which is implemented using a Red-Black Tree)
// Add the sorted filenames in to a queue. The make a pair (foldername, queue<filenames>*),
// foldername is used for 2nd level sorting.
// FYI: 1st level sorting: sort all images under the same directory.
// FYI: 2nd level sorting: sort all folder names
// push this pair to image_name_queue (which is again a Queue)
Status ImageFolderOp::PrescanWorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::string folder_name;
  RETURN_IF_NOT_OK(folder_name_queue_->PopFront(&folder_name));
  while (folder_name.empty() == false) {
    Path folder(folder_path_ + folder_name);
    std::shared_ptr<Path::DirIterator> dirItr = Path::DirIterator::OpenDirectory(&folder);
    if (folder.Exists() == false || dirItr == nullptr) {
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to open folder: " + folder_name);
    }
    std::set<std::string> imgs;  // use this for ordering
    while (dirItr->hasNext()) {
      Path file = dirItr->next();
      if (extensions_.empty() || extensions_.find(file.Extension()) != extensions_.end()) {
        (void)imgs.insert(file.toString().substr(dirname_offset_));
      } else {
        MS_LOG(WARNING) << "Image folder operator unsupported file found: " << file.toString()
                        << ", extension: " << file.Extension() << ".";
      }
    }
    FolderImagesPair p = std::make_shared<std::pair<std::string, std::queue<ImageLabelPair>>>();
    p->first = folder_name;
    for (const std::string &img : imgs) {
      p->second.push(std::make_shared<std::pair<std::string, int32_t>>(img, 0));
    }
    RETURN_IF_NOT_OK(image_name_queue_->EmplaceBack(p));
    RETURN_IF_NOT_OK(folder_name_queue_->PopFront(&folder_name));
  }
  RETURN_IF_NOT_OK(image_name_queue_->EmplaceBack(nullptr));  // end signal
  return Status::OK();
}

// This helper function recursively walks all folder_paths, and send each foldername to folder_name_queue_
// if mRecursive == false, don't go into folder of folders
Status ImageFolderOp::RecursiveWalkFolder(Path *dir) {
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(dir);
  RETURN_UNEXPECTED_IF_NULL(dir_itr);
  while (dir_itr->hasNext()) {
    Path subdir = dir_itr->next();
    if (subdir.IsDirectory()) {
      if (class_index_.empty() ||
          class_index_.find(subdir.toString().substr(dirname_offset_ + 1)) != class_index_.end()) {
        RETURN_IF_NOT_OK(folder_name_queue_->EmplaceBack(subdir.toString().substr(dirname_offset_)));
      }
      if (recursive_ == true) {
        MS_LOG(ERROR) << "RecursiveWalkFolder(&subdir) functionality is disabled permanently. No recursive walk of "
                      << "directory will be performed.";
      }
    }
  }
  return Status::OK();
}

// A thread that calls RecursiveWalkFolder
Status ImageFolderOp::StartAsyncWalk() {
  TaskManager::FindMe()->Post();
  Path dir(folder_path_);
  if (dir.Exists() == false || dir.IsDirectory() == false) {
    RETURN_STATUS_UNEXPECTED("Invalid parameter, failed to open image folder: " + folder_path_);
  }
  dirname_offset_ = folder_path_.length();
  RETURN_IF_NOT_OK(RecursiveWalkFolder(&dir));
  // send out num_workers_ end signal to folder_name_queue_, 1 for each worker.
  // Upon receiving end Signal, worker quits and set another end Signal to image_name_queue.
  for (int32_t ind = 0; ind < num_workers_; ++ind) {
    RETURN_IF_NOT_OK(folder_name_queue_->EmplaceBack(""));  // end signal
  }
  return Status::OK();
}

Status ImageFolderOp::LaunchThreadsAndInitOp() {
  RETURN_UNEXPECTED_IF_NULL(tree_);
  // Registers QueueList and individual Queues for interrupt services
  RETURN_IF_NOT_OK(io_block_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(folder_name_queue_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(image_name_queue_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(wait_for_workers_post_.Register(tree_->AllTasks()));
  // The following code launch 3 threads group
  // 1) A thread that walks all folders and push the folder names to a util:Queue folder_name_queue_.
  // 2) Workers that pull foldername from folder_name_queue_, walk it and return the sorted images to image_name_queue
  // 3) Launch main workers that load DataBuffers by reading all images
  RETURN_IF_NOT_OK(
    tree_->AllTasks()->CreateAsyncTask("walk dir", std::bind(&ImageFolderOp::StartAsyncWalk, this), nullptr, id()));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_,
                                        std::bind(&ImageFolderOp::PrescanWorkerEntry, this, std::placeholders::_1),
                                        Name() + "::PrescanWorkerEntry", id()));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(
    num_workers_, std::bind(&ImageFolderOp::WorkerEntry, this, std::placeholders::_1), Name() + "::WorkerEntry", id()));
  TaskManager::FindMe()->Post();
  // The order of the following 2 functions must not be changed!
  RETURN_IF_NOT_OK(this->PrescanMasterEntry(folder_path_));  // Master thread of pre-scan workers, blocking
  RETURN_IF_NOT_OK(this->InitSampler());                     // pass numRows to Sampler
  return Status::OK();
}

Status ImageFolderOp::CountRowsAndClasses(const std::string &path, const std::set<std::string> &exts, int64_t *num_rows,
                                          int64_t *num_classes, std::map<std::string, int32_t> class_index) {
  Path dir(path);
  std::string err_msg = "";
  int64_t row_cnt = 0;
  err_msg += (dir.Exists() == false || dir.IsDirectory() == false)
               ? "Invalid parameter, image folder path is invalid or not set, path: " + path
               : "";
  err_msg +=
    (num_classes == nullptr && num_rows == nullptr) ? "Invalid parameter, num_class and num_rows are null.\n" : "";
  if (err_msg.empty() == false) {
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::queue<std::string> folder_paths;
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(&dir);
  std::unordered_set<std::string> folder_names;
  while (dir_itr->hasNext()) {
    Path subdir = dir_itr->next();
    if (subdir.IsDirectory()) {
      folder_paths.push(subdir.toString());
      if (!class_index.empty()) folder_names.insert(subdir.Basename());
    }
  }
  if (num_classes != nullptr) {
    // if class index is empty, get everything on disk
    if (class_index.empty()) {
      *num_classes = folder_paths.size();
    } else {
      for (const auto &p : class_index) {
        CHECK_FAIL_RETURN_UNEXPECTED(folder_names.find(p.first) != folder_names.end(),
                                     "folder: " + p.first + " doesn't exist in " + path + " .");
      }
      (*num_classes) = class_index.size();
    }
  }
  // return here if only num_class is needed
  RETURN_OK_IF_TRUE(num_rows == nullptr);
  while (folder_paths.empty() == false) {
    Path subdir(folder_paths.front());
    dir_itr = Path::DirIterator::OpenDirectory(&subdir);
    while (dir_itr->hasNext()) {
      if (exts.empty() || exts.find(subdir.Extension()) != exts.end()) {
        ++row_cnt;
      }
    }
    folder_paths.pop();
  }
  (*num_rows) = row_cnt;
  return Status::OK();
}

Status ImageFolderOp::ComputeColMap() {
  // Set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->column(i).name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

// Get number of classes
Status ImageFolderOp::GetNumClasses(int64_t *num_classes) {
  if (num_classes_ > 0) {
    *num_classes = num_classes_;
    return Status::OK();
  }
  RETURN_IF_NOT_OK(CountRowsAndClasses(folder_path_, extensions_, nullptr, num_classes, class_index_));
  num_classes_ = *num_classes;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
