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

#include "minddata/dataset/engine/datasetops/source/lsun_op.h"

#include <fstream>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
LSUNOp::LSUNOp(int32_t num_wkrs, const std::string &file_dir, int32_t queue_size, const std::string &usage,
               const std::vector<std::string> &classes, bool do_decode, std::unique_ptr<DataSchema> data_schema,
               std::shared_ptr<SamplerRT> sampler)
    : ImageFolderOp(num_wkrs, file_dir, queue_size, false, do_decode, {}, {}, std::move(data_schema),
                    std::move(sampler)),
      usage_(std::move(usage)),
      classes_(std::move(classes)) {}

void LSUNOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows: " << num_rows_ << "\nLSUN directory: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status LSUNOp::WalkDir(Path *dir, const std::string &usage, const std::vector<std::string> &classes,
                       const std::unique_ptr<Queue<std::string>> &folder_name_queue, int64_t *num_class) {
  RETURN_UNEXPECTED_IF_NULL(dir);
  std::vector<std::string> split;
  if (usage == "train" || usage == "all") {
    split.push_back("_train");
  }
  if (usage == "valid" || usage == "all") {
    split.push_back("_val");
  }
  uint64_t dirname_offset = dir->ToString().length();
  std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(dir);
  CHECK_FAIL_RETURN_UNEXPECTED(dir_itr != nullptr,
                               "Invalid path, failed to open image dir: " + dir->ToString() + ", permission denied.");
  std::set<std::string> classes_set;
  std::vector<std::string> valid_classes = classes;
  if (classes.empty()) {
    valid_classes = {"bedroom",     "bridge",  "church_outdoor", "classroom",  "conference_room",
                     "dining_room", "kitchen", "living_room",    "restaurant", "tower"};
  }
  while (dir_itr->HasNext()) {
    std::string subdir = dir_itr->Next().ToString();
    for (auto str : split) {
      std::string name = subdir.substr(dirname_offset);
      for (auto class_name : valid_classes) {
        if (name.find(class_name + str) != std::string::npos) {
          RETURN_IF_NOT_OK(folder_name_queue->EmplaceBack(name));
          classes_set.insert(class_name);
        }
      }
    }
  }
  if (num_class != nullptr) {
    *num_class = classes_set.size();
  }
  return Status::OK();
}

// A thread that calls WalkFolder
Status LSUNOp::RecursiveWalkFolder(Path *dir) {
  RETURN_UNEXPECTED_IF_NULL(dir);
  if (usage_ == "test") {
    Path folder(folder_path_);
    folder = folder / "test";

    RETURN_IF_NOT_OK(folder_name_queue_->EmplaceBack(folder.ToString().substr(dirname_offset_)));
    return Status::OK();
  }

  RETURN_IF_NOT_OK(WalkDir(dir, usage_, classes_, folder_name_queue_, nullptr));
  return Status::OK();
}

Status LSUNOp::CountRowsAndClasses(const std::string &path, const std::string &usage,
                                   const std::vector<std::string> &classes, int64_t *num_rows, int64_t *num_classes) {
  Path dir(path);
  int64_t row_cnt = 0;
  CHECK_FAIL_RETURN_UNEXPECTED(dir.Exists() && dir.IsDirectory(), "Invalid parameter, input dataset path " + path +
                                                                    " does not exist or is not a directory.");
  CHECK_FAIL_RETURN_UNEXPECTED(num_classes != nullptr || num_rows != nullptr,
                               "[Internal ERROR] num_class and num_rows are null.");

  int32_t queue_size = 1024;
  auto folder_name_queue = std::make_unique<Queue<std::string>>(queue_size);
  RETURN_IF_NOT_OK(WalkDir(&dir, usage, classes, folder_name_queue, num_classes));

  // return here if only num_class is needed
  RETURN_OK_IF_TRUE(num_rows == nullptr);

  while (!folder_name_queue->empty()) {
    std::string name;
    RETURN_IF_NOT_OK(folder_name_queue->PopFront(&name));
    Path subdir(path + name);
    std::shared_ptr<Path::DirIterator> dir_itr = Path::DirIterator::OpenDirectory(&subdir);
    while (dir_itr->HasNext()) {
      ++row_cnt;
      Path subdir_pic = dir_itr->Next();
    }
  }
  *num_rows = row_cnt;
  return Status::OK();
}

Status LSUNOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  ImageLabelPair pair_ptr = image_label_pairs_[row_id];
  std::shared_ptr<Tensor> image, label;
  uint32_t label_num = static_cast<uint32_t>(pair_ptr->second);
  RETURN_IF_NOT_OK(Tensor::CreateScalar(label_num, &label));
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(folder_path_ + (pair_ptr->first), &image));

  if (decode_ == true) {
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

// Get number of classes
Status LSUNOp::GetNumClasses(int64_t *num_classes) {
  RETURN_UNEXPECTED_IF_NULL(num_classes);
  if (num_classes_ > 0) {
    *num_classes = num_classes_;
    return Status::OK();
  }

  RETURN_IF_NOT_OK(CountRowsAndClasses(folder_path_, usage_, classes_, nullptr, num_classes));
  num_classes_ = *num_classes;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
