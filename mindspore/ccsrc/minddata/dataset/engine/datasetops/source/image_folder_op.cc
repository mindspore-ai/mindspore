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
#include "utils/ms_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
ImageFolderOp::ImageFolderOp(int32_t num_wkrs, std::string file_dir, int32_t queue_size, bool recursive, bool do_decode,
                             const std::set<std::string> &exts, const std::map<std::string, int32_t> &map,
                             std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_wkrs, queue_size, std::move(sampler)),
      folder_path_(std::move(file_dir)),
      recursive_(recursive),
      decode_(do_decode),
      extensions_(exts),
      class_index_(map),
      data_schema_(std::move(data_schema)),
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
    RETURN_STATUS_UNEXPECTED("Invalid data, " + DatasetName(true) +
                             "Dataset API can't read the data file (interface mismatch or no data found). Check " +
                             DatasetName() + " file path: " + folder_path_);
  }
  // free memory of two queues used for pre-scan
  folder_name_queue_->Reset();
  image_name_queue_->Reset();
  return Status::OK();
}

// Load 1 TensorRow (image,label) using 1 ImageLabelPair. 1 function call produces 1 TensorTow
Status ImageFolderOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  ImageLabelPair pair_ptr = image_label_pairs_[row_id];
  std::shared_ptr<Tensor> image, label;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(pair_ptr->second, &label));
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(folder_path_ + (pair_ptr->first), &image));

  if (decode_ == true) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      std::string err = "Invalid data, failed to decode image: " + folder_path_ + (pair_ptr->first);
      RETURN_STATUS_UNEXPECTED(err);
    }
  }
  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  trow->setPath({folder_path_ + (pair_ptr->first), std::string("")});
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
    out << "\nNumber of rows:" << num_rows_ << "\n"
        << DatasetName(true) << " directory: " << folder_path_ << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

// Derived from RandomAccessOp
Status ImageFolderOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || image_label_pairs_.empty()) {
    if (image_label_pairs_.empty()) {
      RETURN_STATUS_UNEXPECTED("Invalid data, " + DatasetName(true) +
                               "Dataset API can't read the data file(interface mismatch or no data found). Check " +
                               DatasetName() + " file path: " + folder_path_);
    } else {
      RETURN_STATUS_UNEXPECTED(
        "[Internal ERROR], Map containing image-index pair is nullptr or has been set in other place,"
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
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to open " + DatasetName() + ": " + folder_name);
    }
    std::set<std::string> imgs;  // use this for ordering
    while (dirItr->HasNext()) {
      Path file = dirItr->Next();
      if (extensions_.empty() || extensions_.find(file.Extension()) != extensions_.end()) {
        (void)imgs.insert(file.ToString().substr(dirname_offset_));
      } else {
        MS_LOG(WARNING) << DatasetName(true) << " operator unsupported file found: " << file.ToString()
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
  while (dir_itr->HasNext()) {
    Path subdir = dir_itr->Next();
    if (subdir.IsDirectory()) {
      if (class_index_.empty() ||
          class_index_.find(subdir.ToString().substr(dirname_offset_ + 1)) != class_index_.end()) {
        RETURN_IF_NOT_OK(folder_name_queue_->EmplaceBack(subdir.ToString().substr(dirname_offset_)));
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
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open " + DatasetName() + ": " + folder_path_);
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
  // 3) Launch main workers that load TensorRows by reading all images
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
               ? "Invalid parameter, input path is invalid or not set, path: " + path
               : "";
  err_msg +=
    (num_classes == nullptr && num_rows == nullptr) ? "Invalid parameter, num_class and num_rows are null.\n" : "";
  if (err_msg.empty() == false) {
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::queue<std::string> folder_paths;
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(&dir);
  std::unordered_set<std::string> folder_names;
  while (dir_itr->HasNext()) {
    Path subdir = dir_itr->Next();
    if (subdir.IsDirectory()) {
      folder_paths.push(subdir.ToString());
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
                                     "Invalid parameter, folder: " + p.first + " doesn't exist in " + path + " .");
      }
      (*num_classes) = class_index.size();
    }
  }
  // return here if only num_class is needed
  RETURN_OK_IF_TRUE(num_rows == nullptr);
  while (folder_paths.empty() == false) {
    Path subdir(folder_paths.front());
    dir_itr = Path::DirIterator::OpenDirectory(&subdir);
    if (subdir.Exists() == false || dir_itr == nullptr) {
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to open folder: " + subdir.ToString());
    }
    while (dir_itr->HasNext()) {
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
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
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
