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
#include "minddata/dataset/data_source/photo_tour_op.h"

#include <fstream>
#include <iomanip>
#include <regex>
#include <set>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/data_source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "include/utils/common.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
constexpr uint32_t kPatchNumPerRow = 16;
constexpr uint32_t kPatchNumPerCol = 16;
constexpr uint32_t kColPerPatch = 64;
constexpr uint32_t kRowPerPatch = 64;

const std::map<std::string, int> kLens = {
  {"notredame", 468159},      {"yosemite", 633587},        {"liberty", 450092},
  {"liberty_harris", 379587}, {"yosemite_harris", 450912}, {"notredame_harris", 325295},
};
constexpr char kImageExt[] = "bmp";
constexpr char kInfoFile[] = "info.txt";
constexpr char kMatchesFiles[] = "m50_100000_100000_0.txt";
const std::map<std::string, bool> kTrain = {{"train", true}, {"test", false}};

PhotoTourOp::PhotoTourOp(const std::string &dataset_dir, const std::string &name, const std::string &usage,
                         int32_t num_workers, int32_t queue_size, std::unique_ptr<DataSchema> data_schema,
                         std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      dataset_dir_(dataset_dir),
      name_(name),
      usage_(usage),
      buf_cnt_(0),
      data_schema_(std::move(data_schema)),
      train_(true),
      image_names_({}),
      image_bmps_({}),
      matches_({}),
      labels_({}) {}

Status PhotoTourOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  if (train_) {
    std::shared_ptr<Tensor> image;
    // make a copy of cached tensor
    RETURN_IF_NOT_OK(GetPhotoTourDataTensor(row_id, &image));
    (*trow) = TensorRow(row_id, {std::move(image)});
    trow->setPath({image_names_[row_id / (kPatchNumPerRow * kPatchNumPerCol)],
                   std::to_string(row_id % (kPatchNumPerRow * kPatchNumPerCol))});
  } else {
    std::shared_ptr<Tensor> image1, image2, matches;
    // make a copy of cached tensor
    uint32_t row1 = std::get<0>(matches_[row_id]);
    uint32_t row2 = std::get<1>(matches_[row_id]);

    RETURN_IF_NOT_OK(GetPhotoTourDataTensor(row1, &image1));
    RETURN_IF_NOT_OK(GetPhotoTourDataTensor(row2, &image2));
    RETURN_IF_NOT_OK(Tensor::CreateScalar(std::get<2>(matches_[row_id]), &matches));
    (*trow) = TensorRow(row_id, {std::move(image1), std::move(image2), std::move(matches)});
    trow->setPath({image_names_[row1 / (kPatchNumPerRow * kPatchNumPerCol)],
                   std::to_string(row1 % (kPatchNumPerRow * kPatchNumPerCol)),
                   image_names_[row2 / (kPatchNumPerRow * kPatchNumPerCol)],
                   std::to_string(row2 % (kPatchNumPerRow * kPatchNumPerCol))});
  }

  return Status::OK();
}

void PhotoTourOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows: " << num_rows_ << "\nPhotoTour directory: " << dataset_dir_ << "\nName: " << name_
        << "\nUsage: " << usage_ << "\n\n";
  }
}

// Derived from RandomAccessOp.
Status PhotoTourOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || labels_.empty()) {
    if (labels_.empty()) {
      RETURN_STATUS_UNEXPECTED("No image found in dataset, please check if image was read successfully.");
    } else {
      RETURN_STATUS_UNEXPECTED(
        "[Internal ERROR] Map for containing image-index pair is nullptr or has been set in other place, "
        "it must be empty before using GetClassIds.");
    }
  }
  if (train_) {
    for (size_t i = 0; i < labels_.size(); ++i) {
      (*cls_ids)[labels_[i]].push_back(i);
    }
    for (auto &pair : (*cls_ids)) {
      pair.second.shrink_to_fit();
    }
  } else {
    for (size_t i = 0; i < matches_.size(); ++i) {
      (*cls_ids)[std::get<2>(matches_[i])].push_back(i);
    }
    for (auto &pair : (*cls_ids)) {
      pair.second.shrink_to_fit();
    }
  }

  return Status::OK();
}

bool PhotoTourOp::EndsWith(const std::string &s, const std::string &sub) {
  return s.rfind(sub) == (s.length() - sub.length()) ? true : false;
}

Status PhotoTourOp::GetFileContent(const std::string &info_file, std::string *ans) {
  RETURN_UNEXPECTED_IF_NULL(ans);
  std::ifstream reader;
  reader.open(info_file, std::ios::in);
  CHECK_FAIL_RETURN_UNEXPECTED(!reader.fail(), "Invalid file, failed to open " + info_file +
                                                 ": PhotoTour info file is damaged or permission denied.");
  (void)reader.seekg(0, std::ios::end);
  std::size_t size = reader.tellg();
  (void)reader.seekg(0, std::ios::beg);

  CHECK_FAIL_RETURN_UNEXPECTED(size > 0, "Invalid file, the file size of " + info_file + " is unexpected, got size 0.");
  std::string buffer(size, ' ');
  reader.read(&buffer[0], size);
  reader.close();

  // remove \n character in the buffer
  std::regex pattern("([\\s\\n]+)");
  std::string fmt = " ";
  std::string s = std::regex_replace(buffer, pattern, fmt);

  // remove the head and tail whiteblanks of the s
  (void)s.erase(0, s.find_first_not_of(" "));
  (void)s.erase(s.find_last_not_of(" ") + 1);
  // append one whiteblanks to the end of s
  s += " ";
  *ans = s;
  return Status::OK();
}

Status PhotoTourOp::ReadInfoFile(const std::string &data_dir, const std::string &info_file) {
  std::vector<uint32_t> tmp;
  labels_.swap(tmp);
  std::string info_file_path = (Path(data_dir) / Path(info_file)).ToString();
  std::string s;
  RETURN_IF_NOT_OK(GetFileContent(info_file_path, &s));
  auto get_splited_str = [&s](std::size_t pos) {
    std::string item = s.substr(0, pos);
    s = s.substr(pos + 1);
    return item;
  };
  enum ColType { ID_3DPOINT, UNKNOWN };
  std::size_t pos = 0;
  ColType col_idx = ID_3DPOINT;
  while ((pos = s.find(" ")) != std::string::npos) {
    switch (col_idx) {
      case ID_3DPOINT: {
        std::string item = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!item.empty(),
                                     "Invalid data, reading PhotoTour info file failed: " + info_file_path +
                                       " at line: " + std::to_string(pos) + ", the content should not be empty.");
        int id_3dpoint = std::atoi(item.c_str());
        labels_.push_back(id_3dpoint);
        col_idx = UNKNOWN;
        break;
      }
      case UNKNOWN: {
        std::string item2 = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(
          !item2.empty(), "Invalid data, Reading PhotoTour info file failed: " + info_file_path +
                            " at line: " + std::to_string(pos) + ", the content in file should not be empty.");
        col_idx = ID_3DPOINT;
        break;
      }
      default: {
        break;
      }
    }
  }

  return Status::OK();
}

Status PhotoTourOp::ReadMatchedFile(const std::string &data_dir, const std::string &matches_file) {
  std::vector<MatchTuple> tmp;
  matches_.swap(tmp);
  std::string info_file_path = (Path(data_dir) / Path(matches_file)).ToString();

  std::string s;
  RETURN_IF_NOT_OK(GetFileContent(info_file_path, &s));

  auto get_splited_str = [&s](std::size_t pos) {
    std::string item = s.substr(0, pos);
    s = s.substr(pos + 1);
    return item;
  };
  enum ColType { PATCH_ID1, LABEL1, UNUSED1, PATCH_ID2, LABEL2, UNUSED2, UNUSED3 };
  uint32_t patch_id1, label1, patch_id2, label2;
  std::size_t pos = 0;
  ColType col_idx = PATCH_ID1;
  while ((pos = s.find(" ")) != std::string::npos) {
    switch (col_idx) {
      case PATCH_ID1: {
        std::string item = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!item.empty(),
                                     "Invalid data，Reading PhotoTour matched file failed: " + info_file_path +
                                       " at line: " + std::to_string(pos) + ", the content should not be empty.");
        patch_id1 = std::atoi(item.c_str());
        col_idx = LABEL1;
        break;
      }
      case LABEL1: {
        std::string item = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!item.empty(),
                                     "Invalid data, Reading PhotoTour matched file failed: " + info_file_path +
                                       " at line: " + std::to_string(pos) + ", the content should not be empty.");
        label1 = std::atoi(item.c_str());
        col_idx = UNUSED1;
        break;
      }
      case UNUSED1: {
        std::string item = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!item.empty(),
                                     "Invalid data, Reading PhotoTour matched file failed: " + info_file_path +
                                       " at line: " + std::to_string(pos) + ", the content should not be empty.");
        col_idx = PATCH_ID2;
        break;
      }
      case PATCH_ID2: {
        std::string item = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!item.empty(),
                                     "Invalid data, Reading PhotoTour matched file failed: " + info_file_path +
                                       " at line: " + std::to_string(pos) + ", the content should not be empty.");
        patch_id2 = std::atoi(item.c_str());
        col_idx = LABEL2;
        break;
      }
      case LABEL2: {
        std::string item = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!item.empty(),
                                     "Invalid data, Reading PhotoTour matched file failed: " + info_file_path +
                                       " at line: " + std::to_string(pos) + ", the content should not be empty.");
        label2 = std::atoi(item.c_str());
        col_idx = UNUSED2;
        matches_.push_back(std::make_tuple(patch_id1, patch_id2, uint32_t(label1 == label2)));
        break;
      }
      case UNUSED2: {
        std::string item = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!item.empty(),
                                     "Invalid data, Reading PhotoTour matched file failed: " + info_file_path +
                                       " at line: " + std::to_string(pos) + ", the content should not be empty.");
        col_idx = UNUSED3;
        break;
      }
      case UNUSED3: {
        std::string item2 = get_splited_str(pos);
        CHECK_FAIL_RETURN_UNEXPECTED(!item2.empty(),
                                     "Invalid data, Reading PhotoTour matched file failed: " + info_file_path +
                                       " at line: " + std::to_string(pos) + ", the content should not be empty.");
        col_idx = PATCH_ID1;
        break;
      }
      default: {
        break;
      }
    }
  }

  return Status::OK();
}

Status PhotoTourOp::GetPhotoTourDataTensor(uint32_t index, std::shared_ptr<Tensor> *image_tensor) {
  RETURN_UNEXPECTED_IF_NULL(image_tensor);
  CHECK_FAIL_RETURN_UNEXPECTED(
    index < kLens.at(name_),
    "[Internal ERROR] Index exceeds the maximum count of image, got: " + std::to_string(index));

  int image_id = index / (kPatchNumPerRow * kPatchNumPerCol);
  int row_in_image = (index % (kPatchNumPerRow * kPatchNumPerCol)) / kPatchNumPerRow;
  int col_in_image = (index % (kPatchNumPerRow * kPatchNumPerCol)) % kPatchNumPerRow;
  {
    std::unique_lock<std::mutex> lock(access_mutex_);
    if (image_bmps_[image_id].empty()) {
      image_bmps_[image_id] = cv::imread(image_names_[image_id], 0);
    }
  }

  uint32_t x = col_in_image * kColPerPatch;
  uint32_t y = row_in_image * kRowPerPatch;

  cv::Rect myROI(x, y, kColPerPatch, kRowPerPatch);

  // Crop the full image to that image contained by the rectangle myROI
  // Note that this doesn't copy the data
  cv::Mat croppedRef(image_bmps_[image_id], myROI);
  cv::Mat cropped;
  // Copy the data into new matrix
  croppedRef.copyTo(cropped);

  uchar *uc_img = cropped.data;
  TensorShape img_tensor_shape = TensorShape({kRowPerPatch, kColPerPatch, 1});

  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(img_tensor_shape, data_schema_->Column(0).Type(), uc_img, image_tensor));

  return Status::OK();
}

// Read all files in the directory.
// @return Status The status code returned.
Status PhotoTourOp::PrepareData() {
  chosen_dataset_folder_path_ = (Path(dataset_dir_) / Path(name_)).ToString();
  train_ = kTrain.at(usage_);
  auto real_folder_path = FileUtils::GetRealPath(chosen_dataset_folder_path_.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(real_folder_path.has_value(), chosen_dataset_folder_path_ + " does not exist.");

  std::vector<cv::String> file_names;
  cv::glob(real_folder_path.value(), file_names);
  image_names_.clear();
  image_bmps_.clear();
  for (auto &&file_name : file_names) {
    if (EndsWith(file_name, kImageExt)) {
      image_names_.push_back(file_name);
    }
  }
  std::sort(image_names_.begin(), image_names_.end());
  image_bmps_.resize(image_names_.size());
  RETURN_IF_NOT_OK(ReadInfoFile(real_folder_path.value(), kInfoFile));
  RETURN_IF_NOT_OK(ReadMatchedFile(real_folder_path.value(), kMatchesFiles));
  if (train_) {
    num_rows_ = labels_.size();
  } else {
    num_rows_ = matches_.size();
  }
  if (num_rows_ == 0) {
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, data file may not suitable to read with PhotoTourDataset API."
      "Check file in directory: " +
      chosen_dataset_folder_path_);
  }
  return Status::OK();
}

Status PhotoTourOp::CountTotalRows(const std::string &dir, const std::string &name, const std::string &usage,
                                   int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  auto schema = std::make_unique<DataSchema>();
  if (usage == "train") {
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  } else {
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image1", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image2", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
    TensorShape scalar = TensorShape::CreateScalar();
    RETURN_IF_NOT_OK(
      schema->AddColumn(ColDescriptor("matches", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  }

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  auto op = std::make_shared<PhotoTourOp>(dir, name, usage, num_workers, op_connect_size, std::move(schema),
                                          std::move(sampler));
  RETURN_IF_NOT_OK(op->PrepareData());

  if (usage == "train") {
    *count = op->labels_.size();
  } else {
    *count = op->matches_.size();
  }

  return Status::OK();
}

Status PhotoTourOp::ComputeColMap() {
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
