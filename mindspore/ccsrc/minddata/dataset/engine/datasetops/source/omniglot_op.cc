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

#include "minddata/dataset/engine/datasetops/source/omniglot_op.h"

#include <unordered_set>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
OmniglotOp::OmniglotOp(int32_t num_wkrs, const std::string &file_dir, int32_t queue_size, bool background,
                       bool do_decode, std::unique_ptr<DataSchema> data_schema,
                       const std::shared_ptr<SamplerRT> &sampler)
    : ImageFolderOp(num_wkrs, file_dir, queue_size, false, do_decode, {}, {}, std::move(data_schema),
                    std::move(sampler)) {
  Path dir(file_dir);
  if (background) {
    folder_path_ = (dir / "images_background").ToString();
  } else {
    folder_path_ = (dir / "images_evaluation").ToString();
  }
}

void OmniglotOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nNumber of rows: " << num_rows_ << "\nOmniglot directory: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

// This helper function walks all folder_paths, and send each foldername to folder_name_queue_.
Status OmniglotOp::RecursiveWalkFolder(Path *dir) {
  RETURN_UNEXPECTED_IF_NULL(dir);
  std::queue<std::string> folder_paths;
  return WalkDir(dir, &folder_paths, folder_name_queue_.get(), dirname_offset_, false);
}

Status OmniglotOp::WalkDir(Path *dir, std::queue<std::string> *folder_paths, Queue<std::string> *folder_name_queue,
                           uint64_t dirname_offset, bool std_queue) {
  RETURN_UNEXPECTED_IF_NULL(dir);
  RETURN_UNEXPECTED_IF_NULL(folder_paths);
  RETURN_UNEXPECTED_IF_NULL(folder_name_queue);
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(dir);
  CHECK_FAIL_RETURN_UNEXPECTED(dir_itr != nullptr, "Invalid path, failed to open omniglot image dir: " +
                                                     (*dir).ToString() + ", permission denied.");
  while (dir_itr->HasNext()) {
    Path subdir = dir_itr->Next();
    if (subdir.IsDirectory()) {
      std::shared_ptr<Path::DirIterator> dir_itr_sec = Path::DirIterator::OpenDirectory(&subdir);
      CHECK_FAIL_RETURN_UNEXPECTED(dir_itr_sec != nullptr, "Invalid path, failed to open omniglot image dir: " +
                                                             subdir.ToString() + ", permission denied.");
      while (dir_itr_sec->HasNext()) {
        Path subsubdir = dir_itr_sec->Next();
        if (subsubdir.IsDirectory()) {
          if (std_queue) {
            folder_paths->push(subsubdir.ToString());
          } else {
            RETURN_IF_NOT_OK(folder_name_queue->EmplaceBack(subsubdir.ToString().substr(dirname_offset)));
          }
        }
      }
    }
  }
  return Status::OK();
}

Status OmniglotOp::CountRowsAndClasses(const std::string &path, int64_t *num_rows, int64_t *num_classes) {
  Path dir(path);
  CHECK_FAIL_RETURN_UNEXPECTED(dir.Exists() && dir.IsDirectory(),
                               "Invalid parameter, input path is invalid or not set, path: " + path);
  CHECK_FAIL_RETURN_UNEXPECTED(num_classes != nullptr || num_rows != nullptr,
                               "[Internal ERROR] num_class and num_rows are null.");
  int64_t row_cnt = 0;
  std::queue<std::string> folder_paths;
  Queue<std::string> tmp_queue = Queue<std::string>(1);
  RETURN_IF_NOT_OK(WalkDir(&dir, &folder_paths, &tmp_queue, 0, true));
  if (num_classes != nullptr) {
    *num_classes = folder_paths.size();
  }
  RETURN_OK_IF_TRUE(num_rows == nullptr);
  while (!folder_paths.empty()) {
    Path subdir(folder_paths.front());
    auto dir_itr = Path::DirIterator::OpenDirectory(&subdir);
    RETURN_UNEXPECTED_IF_NULL(dir_itr);
    while (dir_itr->HasNext()) {
      ++row_cnt;
    }
    folder_paths.pop();
  }
  (*num_rows) = row_cnt;
  return Status::OK();
}

// Get number of classes
Status OmniglotOp::GetNumClasses(int64_t *num_classes) {
  RETURN_UNEXPECTED_IF_NULL(num_classes);
  if (num_classes_ > 0) {
    *num_classes = num_classes_;
    return Status::OK();
  }
  RETURN_IF_NOT_OK(CountRowsAndClasses(folder_path_, nullptr, num_classes));
  num_classes_ = *num_classes;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
