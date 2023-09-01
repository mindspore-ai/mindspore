/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include "minddata/dataset/engine/datasetops/source/food101_op.h"

#include <algorithm>
#include <iomanip>
#include <regex>
#include <utility>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/path.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
Food101Op::Food101Op(const std::string &folder_path, const std::string &usage, int32_t num_workers, int32_t queue_size,
                     bool decode, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(folder_path),
      decode_(decode),
      usage_(usage),
      data_schema_(std::move(data_schema)) {}

Status Food101Op::PrepareData() {
  auto realpath = FileUtils::GetRealPath(folder_path_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, Food101 dataset dir: " << folder_path_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, Food101 dataset dir: " + folder_path_ + " does not exist.");
  }

  std::string image_root_path = (Path(realpath.value()) / Path("images")).ToString();
  std::string train_list_txt_ = (Path(realpath.value()) / Path("meta") / Path("train.txt")).ToString();
  std::string test_list_txt_ = (Path(realpath.value()) / Path("meta") / Path("test.txt")).ToString();

  Path img_folder(image_root_path);
  CHECK_FAIL_RETURN_UNEXPECTED(img_folder.Exists(),
                               "Invalid path, Food101 image path: " + image_root_path + " does not exist.");
  CHECK_FAIL_RETURN_UNEXPECTED(img_folder.IsDirectory(),
                               "Invalid path, Food101 image path: " + image_root_path + " is not a folder.");
  std::shared_ptr<Path::DirIterator> img_folder_itr = Path::DirIterator::OpenDirectory(&img_folder);

  RETURN_UNEXPECTED_IF_NULL(img_folder_itr);
  int32_t dirname_offset_ = img_folder.ToString().length() + 1;

  while (img_folder_itr->HasNext()) {
    Path sub_dir = img_folder_itr->Next();
    if (sub_dir.IsDirectory()) {
      classes_.insert(sub_dir.ToString().substr(dirname_offset_));
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!classes_.empty(),
                               "Invalid path, no subfolder found under path: " + img_folder.ToString());

  if (usage_ == "test") {
    RETURN_IF_NOT_OK(GetAllImageList(test_list_txt_));
  } else if (usage_ == "train") {
    RETURN_IF_NOT_OK(GetAllImageList(train_list_txt_));
  } else {
    RETURN_IF_NOT_OK(GetAllImageList(train_list_txt_));
    RETURN_IF_NOT_OK(GetAllImageList(test_list_txt_));
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!all_img_lists_.empty(),
                               "No valid train.txt or test.txt file under path: " + image_root_path);
  all_img_lists_.shrink_to_fit();
  num_rows_ = all_img_lists_.size();

  return Status::OK();
}

void Food101Op::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    ParallelOp::Print(out, show_all);
    out << "\n";
  } else {
    ParallelOp::Print(out, show_all);
    out << "\nNumber of rows: " << num_rows_ << "\nFood101 dataset dir: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status Food101Op::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::string img_name = (Path(folder_path_) / Path("images") / Path(all_img_lists_[row_id])).ToString() + ".jpg";
  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> label;
  std::string label_str;
  for (auto it : classes_) {
    if (all_img_lists_[row_id].find(it) != all_img_lists_[row_id].npos) {
      label_str = it;
      break;
    }
  }
  RETURN_IF_NOT_OK(Tensor::CreateScalar(label_str, &label));
  RETURN_IF_NOT_OK(ReadImageToTensor(img_name, &image));
  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  trow->setPath({img_name, std::string("")});
  return Status::OK();
}

Status Food101Op::ReadImageToTensor(const std::string &image_path, std::shared_ptr<Tensor> *tensor) {
  RETURN_UNEXPECTED_IF_NULL(tensor);
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(image_path, tensor));
  if (decode_) {
    Status rc = Decode(*tensor, tensor);
    CHECK_FAIL_RETURN_UNEXPECTED(
      rc.IsOk(), "Invalid file, failed to decode image, the image may be broken or permission denied: " + image_path);
  }
  return Status::OK();
}

// Get dataset size.
Status Food101Op::CountTotalRows(int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  if (all_img_lists_.size() == 0) {
    RETURN_IF_NOT_OK(PrepareData());
  }
  *count = static_cast<int64_t>(all_img_lists_.size());
  return Status::OK();
}

Status Food101Op::GetAllImageList(const std::string &file_path) {
  std::ifstream handle(file_path, std::ios::in);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open text:" + file_path +
                             ", the file is damaged or permission denied.");
  }

  std::string line;
  while (getline(handle, line)) {
    if (!line.empty()) {
      all_img_lists_.push_back(line);
    }
  }
  handle.close();
  return Status::OK();
}

Status Food101Op::ComputeColMap() {
  // Set the column name map (base class field).
  if (column_name_id_map_.empty()) {
    for (int32_t index = 0; index < data_schema_->NumColumns(); index++) {
      column_name_id_map_[data_schema_->Column(index).Name()] = index;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
