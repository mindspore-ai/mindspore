/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/wider_face_op.h"

#include <algorithm>
#include <iomanip>
#include <regex>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/path.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
constexpr char kSplitPath[] = "wider_face_split";
constexpr char kTrainAnno[] = "wider_face_train_bbx_gt.txt";
constexpr char kValAnno[] = "wider_face_val_bbx_gt.txt";
constexpr char kTestAnno[] = "wider_face_test_filelist.txt";
constexpr char kTrainBase[] = "WIDER_train";
constexpr char kValBase[] = "WIDER_val";
constexpr char kTestBase[] = "WIDER_test";
constexpr char kImage[] = "images";
constexpr char kExtension[] = ".jpg";
constexpr int kDataLen = 10;           // Length of each data row.
constexpr int kBboxLen = 4;            // Length of bbox in annotation vector.
constexpr int kBlurIndex = 4;          // Index of blur in annotation vector.
constexpr int kExpressionIndex = 5;    // Index of expression in annotation vector.
constexpr int kIlluminationIndex = 6;  // Index of illumination in annotation vector.
constexpr int kOcclusionIndex = 7;     // Index of occlusion in annotation vector.
constexpr int kPoseIndex = 8;          // Index of pose in annotation vector.
constexpr int kInvalidIndex = 9;       // Index of invalid in annotation vector.

WIDERFaceOp::WIDERFaceOp(const std::string &folder_path, const std::string &usage, int32_t num_workers,
                         int32_t queue_size, bool decode, std::unique_ptr<DataSchema> data_schema,
                         std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(folder_path),
      decode_(decode),
      usage_(usage),
      data_schema_(std::move(data_schema)) {}

Status WIDERFaceOp::PrepareData() {
  auto realpath = FileUtils::GetRealPath(folder_path_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, WIDERFace dataset dir: " << folder_path_ << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, WIDERFace dataset dir: " + folder_path_ + " does not exist.");
  }
  std::string train_folder = (Path(realpath.value()) / Path(kTrainBase) / Path(kImage)).ToString();
  std::string val_folder = (Path(realpath.value()) / Path(kValBase) / Path(kImage)).ToString();
  std::string test_folder = (Path(realpath.value()) / Path(kTestBase) / Path(kImage)).ToString();
  std::string train_anno_dir = (Path(realpath.value()) / Path(kSplitPath) / Path(kTrainAnno)).ToString();
  std::string val_anno_dir = (Path(realpath.value()) / Path(kSplitPath) / Path(kValAnno)).ToString();

  if (usage_ == "train") {
    RETURN_IF_NOT_OK(WalkFolders(train_folder));
    RETURN_IF_NOT_OK(GetTraValAnno(train_anno_dir, train_folder));
  } else if (usage_ == "valid") {
    RETURN_IF_NOT_OK(WalkFolders(val_folder));
    RETURN_IF_NOT_OK(GetTraValAnno(val_anno_dir, val_folder));
  } else if (usage_ == "test") {
    RETURN_IF_NOT_OK(WalkFolders(test_folder));
  } else {
    RETURN_IF_NOT_OK(WalkFolders(train_folder));
    RETURN_IF_NOT_OK(WalkFolders(val_folder));
    RETURN_IF_NOT_OK(GetTraValAnno(train_anno_dir, train_folder));
    RETURN_IF_NOT_OK(GetTraValAnno(val_anno_dir, val_folder));
  }
  all_img_names_.shrink_to_fit();
  num_rows_ = all_img_names_.size();
  return Status::OK();
}

void WIDERFaceOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    ParallelOp::Print(out, show_all);
    out << "\n";
  } else {
    ParallelOp::Print(out, show_all);
    out << "\nNumber of rows: " << num_rows_ << "\nWIDERFace dataset dir: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

Status WIDERFaceOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::string img_name = all_img_names_[row_id];
  std::shared_ptr<Tensor> image;
  std::vector<std::string> path_list;
  RETURN_IF_NOT_OK(ReadImageToTensor(img_name, &image));
  trow->setId(row_id);
  trow->push_back(std::move(image));
  if (usage_ == "test") {
    path_list = {img_name};
  } else if (usage_ == "all" || usage_ == "train" || usage_ == "valid") {
    TensorRow annotation;
    RETURN_IF_NOT_OK(ParseAnnotations(img_name, &annotation));
    std::string train_path = (Path(folder_path_) / Path(kSplitPath) / Path(kTrainAnno)).ToString();
    std::string val_path = (Path(folder_path_) / Path(kSplitPath) / Path(kValAnno)).ToString();
    trow->insert(trow->end(), annotation.begin(), annotation.end());
    if (img_name.find("train") != std::string::npos) {
      path_list = {img_name, train_path, train_path, train_path, train_path, train_path, train_path, train_path};
    } else {
      path_list = {img_name, val_path, val_path, val_path, val_path, val_path, val_path, val_path};
    }
  } else {
    RETURN_STATUS_UNEXPECTED("Invalid parameter, usage should be \"train\", \"test\", \"valid\" or \"all\", got " +
                             usage_);
  }
  trow->setPath(path_list);
  return Status::OK();
}

Status WIDERFaceOp::ReadImageToTensor(const std::string &image_path, std::shared_ptr<Tensor> *tensor) {
  RETURN_UNEXPECTED_IF_NULL(tensor);
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(image_path, tensor));
  if (decode_) {
    Status rc = Decode(*tensor, tensor);
    CHECK_FAIL_RETURN_UNEXPECTED(
      rc.IsOk(), "Invalid file, failed to decode image, the image may be broken or permission denied: " + image_path);
  }
  return Status::OK();
}

// Get annotations of usage of train or valid.
Status WIDERFaceOp::GetTraValAnno(const std::string &list_path, const std::string &image_folder_path) {
  std::string line;
  bool file_name_line = true;
  bool num_boxes_line = false;
  bool box_annotation_line = false;
  int32_t num_boxes = 0, box_counter = 0;
  std::string image_path;
  std::vector<int32_t> image_labels;
  std::ifstream file_reader(list_path);
  while (getline(file_reader, line)) {
    if (file_name_line) {
      box_counter = 0;
      image_labels.clear();
      image_path = (Path(image_folder_path) / Path(line)).ToString();
      file_name_line = false;
      num_boxes_line = true;
    } else if (num_boxes_line) {
      try {
        num_boxes = std::stoi(line);
      } catch (const std::exception &e) {
        file_reader.close();
        RETURN_STATUS_UNEXPECTED("Invalid data, failed to read the number of bbox: " + line);
      }
      num_boxes_line = false;
      box_annotation_line = true;
    } else if (box_annotation_line) {
      box_counter += 1;
      std::vector<int32_t> labels;
      RETURN_IF_NOT_OK(Split(line, &labels));
      image_labels.insert(image_labels.end(), labels.begin(), labels.end());
      if (box_counter >= num_boxes) {
        box_annotation_line = false;
        file_name_line = true;
        annotation_map_[image_path] = image_labels;
      }
    }
  }
  file_reader.close();
  return Status::OK();
}

// Parse annotations to tensors.
Status WIDERFaceOp::ParseAnnotations(const std::string &path, TensorRow *tensor) {
  RETURN_UNEXPECTED_IF_NULL(tensor);
  std::vector<int32_t> annotation = annotation_map_[path];
  CHECK_FAIL_RETURN_UNEXPECTED(
    static_cast<int>(annotation.size()) % kDataLen == 0,
    "Invalid parameter, only annotation with size multiple of eight are accepted, but got: " +
      std::to_string(annotation.size()));
  std::vector<int32_t> bboxes_vec, blur_vec, expression_vec, illumination_vec, occlusion_vec, pose_vec, invalid_vec;
  std::vector<int32_t> label_vec;
  std::shared_ptr<Tensor> bbox, blur, expression, illumination, occlusion, pose, invalid;
  int32_t bbox_num = static_cast<int>(annotation.size()) / kDataLen;
  for (int32_t index = 0; index < bbox_num; index++) {
    label_vec.clear();
    for (int32_t inner_index = 0; inner_index < kDataLen; inner_index++) {
      label_vec.emplace_back(annotation[index * kDataLen + inner_index]);
    }
    bboxes_vec.insert(bboxes_vec.end(), label_vec.begin(), label_vec.begin() + kBboxLen);
    blur_vec.emplace_back(label_vec[kBlurIndex]);
    expression_vec.emplace_back(label_vec[kExpressionIndex]);
    illumination_vec.emplace_back(label_vec[kIlluminationIndex]);
    occlusion_vec.emplace_back(label_vec[kOcclusionIndex]);
    pose_vec.emplace_back(label_vec[kPoseIndex]);
    invalid_vec.emplace_back(label_vec[kInvalidIndex]);
  }

  RETURN_IF_NOT_OK(Tensor::CreateFromVector(bboxes_vec, TensorShape({bbox_num, 4}), &bbox));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(blur_vec, &blur));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(expression_vec, &expression));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(illumination_vec, &illumination));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(occlusion_vec, &occlusion));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(pose_vec, &pose));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(invalid_vec, &invalid));
  (*tensor) = TensorRow({std::move(bbox), std::move(blur), std::move(expression), std::move(illumination),
                         std::move(occlusion), std::move(pose), std::move(invalid)});
  return Status::OK();
}

// Convert annotation line to int32_t vector.
Status WIDERFaceOp::Split(const std::string &line, std::vector<int32_t> *split_num) {
  RETURN_UNEXPECTED_IF_NULL(split_num);
  std::string str = line;
  std::string::size_type pos;
  std::vector<std::string> split;
  int size = str.size();
  for (int index = 0; index < size;) {
    pos = str.find(" ", index);
    if (pos != index) {
      std::string s = str.substr(index, pos - index);
      split.push_back(s);
    }
    index = pos + 1;
  }
  int i = 0;
  try {
    for (; i < split.size(); i++) {
      split_num->emplace_back(stoi(split[i]));
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Invalid data, failed to parse the annotation " << i << ": " << split[i];
    RETURN_STATUS_UNEXPECTED("Invalid data, failed to parse the annotation " + std::to_string(i) + ": " + split[i]);
  }
  return Status::OK();
}

// Get dataset size.
Status WIDERFaceOp::CountTotalRows(int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  if (all_img_names_.size() == 0) {
    RETURN_IF_NOT_OK(PrepareData());
  }
  *count = static_cast<int64_t>(all_img_names_.size());
  return Status::OK();
}

Status WIDERFaceOp::WalkFolders(const std::string &wf_path) {
  Path img_folder(wf_path);
  CHECK_FAIL_RETURN_UNEXPECTED(img_folder.Exists() && img_folder.IsDirectory(),
                               "Invalid path, failed to open WIDERFace folder: " + wf_path);
  std::shared_ptr<Path::DirIterator> img_folder_itr = Path::DirIterator::OpenDirectory(&img_folder);

  RETURN_UNEXPECTED_IF_NULL(img_folder_itr);
  int32_t dirname_offset_ = img_folder.ToString().length() + 1;

  while (img_folder_itr->HasNext()) {
    Path sub_dir = img_folder_itr->Next();
    if (sub_dir.IsDirectory()) {
      folder_names_.insert(sub_dir.ToString().substr(dirname_offset_));
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!folder_names_.empty(),
                               "Invalid file, no subfolder found under path: " + img_folder.ToString());
  for (std::set<std::string>::iterator it = folder_names_.begin(); it != folder_names_.end(); ++it) {
    Path folder_dir(img_folder / (*it));
    auto folder_it = Path::DirIterator::OpenDirectory(&folder_dir);
    while (folder_it->HasNext()) {
      Path file = folder_it->Next();
      if (file.Extension() == kExtension) {
        all_img_names_.emplace_back(file.ToString());
      }
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!all_img_names_.empty(),
                               "Invalid file, no " + std::string(kExtension) + " file found under path: " + wf_path);
  return Status::OK();
}

Status WIDERFaceOp::ComputeColMap() {
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
