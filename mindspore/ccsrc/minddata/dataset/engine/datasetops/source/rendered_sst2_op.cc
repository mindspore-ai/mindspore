/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/source/rendered_sst2_op.h"

#include <unordered_set>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
RenderedSST2Op::RenderedSST2Op(int32_t num_wkrs, const std::string &file_dir, const std::string &usage,
                               int32_t queue_size, bool do_decode, const std::set<std::string> &exts,
                               const std::map<std::string, uint32_t> &map, std::unique_ptr<DataSchema> data_schema,
                               std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_wkrs, queue_size, std::move(sampler)),
      folder_path_(file_dir),
      usage_(usage),
      decode_(do_decode),
      extensions_(exts),
      class_index_(map),
      data_schema_(std::move(data_schema)),
      sampler_ind_(0) {
  folder_path_queue_ = std::make_unique<Queue<std::string>>(num_wkrs * queue_size);
  folder_classId_queue_ = std::make_unique<Queue<uint32_t>>(num_wkrs * queue_size);
  image_name_queue_ = std::make_unique<Queue<FolderImagesPair>>(num_wkrs * queue_size);
}

// Master thread that pulls the prescan worker's results.
// Keep collecting results until all prescan workers quit
// Then consolidate 2 level shuffles together into 1 giant vector
// calculate numRows then return
Status RenderedSST2Op::PrepareData() {
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
    while (!v[ind]->second.empty()) {
      image_label_pairs_.push_back(v[ind]->second.front());
      image_prefix_.push_back(v[ind]->first);
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
  folder_path_queue_->Reset();
  folder_classId_queue_->Reset();
  image_name_queue_->Reset();
  return Status::OK();
}

// Load 1 TensorRow (image,label) using 1 ImageLabelPair. 1 function call produces 1 TensorTow
Status RenderedSST2Op::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  ImageLabelPair pair_ptr = image_label_pairs_[row_id];
  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> label;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(pair_ptr->second, &label));
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(image_prefix_[row_id] + pair_ptr->first, &image));

  if (decode_) {
    Status rc = Decode(image, &image);
    if (rc.IsError()) {
      std::string err = "Invalid image, " + folder_path_ + (pair_ptr->first) +
                        " decode failed, the image is broken or permission denied.";
      RETURN_STATUS_UNEXPECTED(err);
    }
  }
  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  trow->setPath({folder_path_ + (pair_ptr->first), std::string("")});
  return Status::OK();
}

void RenderedSST2Op::Print(std::ostream &out, bool show_all) const {
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
        << DatasetName(true) << " directory: " << folder_path_ << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

// Derived from RandomAccessOp
Status RenderedSST2Op::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || image_label_pairs_.empty()) {
    if (image_label_pairs_.empty()) {
      RETURN_STATUS_UNEXPECTED("Invalid dataset_dir, " + DatasetName(true) +
                               "Dataset API can't read the data file (interface mismatch or no data found). Check " +
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
// Worker pull a file path from folder_path_queue_ (which is a Queue), walks all the images under that folderpath
// After walking is complete, sort all the file names (relative path to all png files under the same directory )
// (Sort is automatically conducted using a set which is implemented using a Red-Black Tree)
// Add the sorted filenames in to a queue. The make a pair (folderpath, queue<filenames>*),
// folderpath is used for 2nd level sorting.
// FYI: 1st level sorting: sort all images under the same directory.
// FYI: 2nd level sorting: sort all folder names
// push this pair to image_name_queue (which is again a Queue)
Status RenderedSST2Op::PrescanWorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  std::string folder_;
  uint32_t current_class_id;
  RETURN_IF_NOT_OK(folder_path_queue_->PopFront(&folder_));
  RETURN_IF_NOT_OK(folder_classId_queue_->PopFront(&current_class_id));
  while (!folder_.empty()) {
    Path folder(folder_);
    std::shared_ptr<Path::DirIterator> dirItr = Path::DirIterator::OpenDirectory(&folder);
    if (!folder.Exists() || dirItr == nullptr) {
      RETURN_STATUS_UNEXPECTED("Invalid dataset_dir, " + folder_ + " does not exist or permission denied.");
    }
    auto offset_ = folder_.size();
    std::set<std::string> imgs;  // use this for ordering
    while (dirItr->HasNext()) {
      Path file = dirItr->Next();
      if (extensions_.empty() || extensions_.find(file.Extension()) != extensions_.end()) {
        (void)imgs.insert(file.ToString().substr(offset_));
      } else {
        MS_LOG(WARNING) << DatasetName(true) << " operator unsupported file found: " << file.ToString()
                        << ", extension: " << file.Extension() << ".";
      }
    }
    FolderImagesPair p = std::make_shared<std::pair<std::string, std::queue<ImageLabelPair>>>();
    p->first = folder_;
    for (const std::string &img : imgs) {
      p->second.push(std::make_shared<std::pair<std::string, uint32_t>>(img, current_class_id));
    }
    RETURN_IF_NOT_OK(image_name_queue_->EmplaceBack(p));
    RETURN_IF_NOT_OK(folder_path_queue_->PopFront(&folder_));
    RETURN_IF_NOT_OK(folder_classId_queue_->PopFront(&current_class_id));
  }
  RETURN_IF_NOT_OK(image_name_queue_->EmplaceBack(nullptr));  // end signal
  return Status::OK();
}

// This helper function walks all folder_paths, and send each folderpath to folder_path_queue_
Status RenderedSST2Op::WalkFolder(Path *dir) {
  RETURN_UNEXPECTED_IF_NULL(dir);
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(dir);
  RETURN_UNEXPECTED_IF_NULL(dir_itr);
  auto offset_ = dir->ToString().size();
  std::string current_class;
  while (dir_itr->HasNext()) {
    Path subdir = dir_itr->Next();
    if (subdir.IsDirectory()) {
      RETURN_IF_NOT_OK(folder_path_queue_->EmplaceBack(subdir.ToString()));
      current_class = subdir.ToString().substr(offset_ + 1);
      if (class_index_.find(current_class) == class_index_.end()) {
        class_index_[current_class] = class_index_.size();
      }
      RETURN_IF_NOT_OK(folder_classId_queue_->EmplaceBack(class_index_[current_class]));
    }
  }
  return Status::OK();
}

// A thread that calls WalkFolder
Status RenderedSST2Op::StartAsyncWalk() {
  TaskManager::FindMe()->Post();
  Path dir(folder_path_);

  if (!dir.Exists() || !dir.IsDirectory()) {
    RETURN_STATUS_UNEXPECTED("Invalid path, " + folder_path_ + " may not exist or is not a directory.");
  }

  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(&dir);
  RETURN_UNEXPECTED_IF_NULL(dir_itr);
  auto offset_ = folder_path_.length();
  while (dir_itr->HasNext()) {
    Path subdir = dir_itr->Next();
    if (subdir.IsDirectory()) {
      std::string name = subdir.ToString().substr(offset_ + 1);
      if (usage_ == name) {
        RETURN_IF_NOT_OK(WalkFolder(&subdir));
      } else if (usage_ == "val" && name == "valid") {
        RETURN_IF_NOT_OK(WalkFolder(&subdir));
      } else if (usage_ == "all" && (name == "train" || name == "test" || name == "valid")) {
        RETURN_IF_NOT_OK(WalkFolder(&subdir));
      }
    }
  }
  // send out num_workers_ end signal to folder_path_queue_, 1 for each worker.
  // Upon receiving end Signal, worker quits and set another end Signal to image_name_queue.
  for (int32_t ind = 0; ind < num_workers_; ++ind) {
    RETURN_IF_NOT_OK(folder_path_queue_->EmplaceBack(""));  // end signal
    RETURN_IF_NOT_OK(folder_classId_queue_->EmplaceBack(0));
  }
  return Status::OK();
}

Status RenderedSST2Op::RegisterAndLaunchThreads() {
  RETURN_IF_NOT_OK(ParallelOp::RegisterAndLaunchThreads());
  RETURN_IF_NOT_OK(folder_path_queue_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(folder_classId_queue_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(image_name_queue_->Register(tree_->AllTasks()));

  // The following code launch 3 threads group
  // 1) A thread that walks all folders and push the folder names to a util:Queue folder_path_queue_.
  // 2) Workers that pull foldername from folder_path_queue_, walk it and return the sorted images to image_name_queue
  // 3) Launch main workers that load TensorRows by reading all images
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask(Name() + "::WalkDir",
                                                      std::bind(&RenderedSST2Op::StartAsyncWalk, this), nullptr, id()));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_,
                                        std::bind(&RenderedSST2Op::PrescanWorkerEntry, this, std::placeholders::_1),
                                        Name() + "::PrescanWorkerEntry", id()));

  return Status::OK();
}

Status RenderedSST2Op::WalkFolderForCountRows(Path *dir, std::queue<std::string> *folder_paths,
                                              std::map<std::string, uint32_t> *class_index) {
  RETURN_UNEXPECTED_IF_NULL(dir);
  RETURN_UNEXPECTED_IF_NULL(folder_paths);
  RETURN_UNEXPECTED_IF_NULL(class_index);
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(dir);
  RETURN_UNEXPECTED_IF_NULL(dir_itr);
  std::string current_class;
  auto offset_ = dir->ToString().size();
  while (dir_itr->HasNext()) {
    Path subdir = dir_itr->Next();
    if (subdir.IsDirectory()) {
      folder_paths->push(subdir.ToString());
      current_class = subdir.ToString().substr(offset_ + 1);
      if (class_index->find(current_class) == class_index->end()) {
        (*class_index)[current_class] = class_index->size();
      }
    }
  }
  return Status::OK();
}

Status RenderedSST2Op::CountRows(std::queue<std::string> *folder_paths, int64_t *num_rows,
                                 const std::set<std::string> &exts) {
  RETURN_UNEXPECTED_IF_NULL(folder_paths);
  RETURN_UNEXPECTED_IF_NULL(num_rows);
  int64_t row_cnt = 0;
  while (!folder_paths->empty()) {
    Path subdir(folder_paths->front());
    std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(&subdir);
    if (!subdir.Exists() || dir_itr == nullptr) {
      RETURN_STATUS_UNEXPECTED("Invalid subdirectory, RenderedSST2 Dataset subdirectory: " + subdir.ToString() +
                               " does not exist or permission denied");
    }
    while (dir_itr->HasNext()) {
      if (exts.empty() || exts.find(dir_itr->Next().Extension()) != exts.end()) {
        ++row_cnt;
      }
    }
    folder_paths->pop();
  }
  (*num_rows) = row_cnt;
  return Status::OK();
}

Status RenderedSST2Op::CountRowsAndClasses(const std::string &path, const std::string &usage,
                                           const std::set<std::string> &exts, int64_t *num_rows, int64_t *num_classes) {
  Path dir(path);
  std::string err_msg = "";
  err_msg += (!dir.Exists() || !dir.IsDirectory())
               ? "Invalid dataset_dir, " + path + " does not exist or the path is not a directory. "
               : "";
  err_msg += (num_classes == nullptr && num_rows == nullptr) ? "[Internal ERROR] num_class and num_rows are null." : "";
  if (!err_msg.empty()) {
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::queue<std::string> folder_paths;
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(&dir);
  std::map<std::string, uint32_t> class_index;
  auto offset_ = path.size();
  RETURN_UNEXPECTED_IF_NULL(dir_itr);

  while (dir_itr->HasNext()) {
    Path subdir = dir_itr->Next();
    if (subdir.IsDirectory()) {
      std::string name = subdir.ToString().substr(offset_ + 1);
      name = name == "valid" ? "val" : name;
      if (usage == name) {
        RETURN_IF_NOT_OK(WalkFolderForCountRows(&subdir, &folder_paths, &class_index));
      } else if (usage == "all" && (name == "train" || name == "test" || name == "val")) {
        RETURN_IF_NOT_OK(WalkFolderForCountRows(&subdir, &folder_paths, &class_index));
      }
    }
  }

  if (num_classes != nullptr) {
    *num_classes = class_index.size();
  }
  // return here if only num_class is needed
  RETURN_OK_IF_TRUE(num_rows == nullptr);

  RETURN_IF_NOT_OK(CountRows(&folder_paths, num_rows, exts));
  return Status::OK();
}

Status RenderedSST2Op::ComputeColMap() {
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
Status RenderedSST2Op::GetNumClasses(int64_t *num_classes) {
  RETURN_UNEXPECTED_IF_NULL(num_classes);
  if (num_classes_ > 0) {
    *num_classes = num_classes_;
    return Status::OK();
  }
  RETURN_IF_NOT_OK(CountRowsAndClasses(folder_path_, usage_, extensions_, nullptr, num_classes));
  num_classes_ = *num_classes;
  return Status::OK();
}

Status RenderedSST2Op::InitPullMode() {
  // to avoid the concurrent and multi end signal in StartAsyncWalk, explicitly set num_workers_ to 1
  num_workers_ = 1;
  RETURN_IF_NOT_OK(folder_path_queue_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(image_name_queue_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask(Name() + "::WalkDir",
                                                      std::bind(&RenderedSST2Op::StartAsyncWalk, this), nullptr, id()));
  RETURN_IF_NOT_OK(tree_->LaunchWorkers(num_workers_,
                                        std::bind(&RenderedSST2Op::PrescanWorkerEntry, this, std::placeholders::_1),
                                        Name() + "::PrescanWorkerEntry", id()));
  return PrepareData();
}
}  // namespace dataset
}  // namespace mindspore
