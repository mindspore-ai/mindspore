/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/sbu_op.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <set>
#include <utility>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/ms_utils.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
SBUOp::SBUOp(const std::string &folder_path, bool decode, std::unique_ptr<DataSchema> data_schema,
             std::shared_ptr<SamplerRT> sampler, int32_t num_workers, int32_t queue_size)
    : MappableLeafOp(num_workers, queue_size, std::move(sampler)),
      folder_path_(folder_path),
      decode_(decode),
      url_path_(""),
      caption_path_(""),
      image_folder_(""),
      data_schema_(std::move(data_schema)) {}

void SBUOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows: " << num_rows_ << "\nSBU directory: " << folder_path_
        << "\nDecode: " << (decode_ ? "yes" : "no") << "\n\n";
  }
}

// Load 1 TensorRow (image, caption) using 1 SBUImageCaptionPair.
Status SBUOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);

  SBUImageCaptionPair image_caption_pair = image_caption_pairs_[row_id];
  Path path = image_caption_pair.first;

  std::shared_ptr<Tensor> image, caption;
  RETURN_IF_NOT_OK(ReadImageToTensor(path.ToString(), &image));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(image_caption_pair.second, &caption));

  (*trow) = TensorRow(row_id, {std::move(image), std::move(caption)});
  trow->setPath({path.ToString()});
  return Status::OK();
}

Status SBUOp::ReadImageToTensor(const std::string &path, std::shared_ptr<Tensor> *tensor) const {
  RETURN_IF_NOT_OK(Tensor::CreateFromFile(path, tensor));
  if (decode_ == true) {
    Status rc = Decode(*tensor, tensor);
    if (rc.IsError()) {
      RETURN_STATUS_UNEXPECTED("Invalid image, failed to decode image:" + path +
                               ", the image is damaged or permission denied.");
    }
  }
  return Status::OK();
}

Status SBUOp::ComputeColMap() {
  // set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < static_cast<int32_t>(data_schema_->NumColumns()); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status SBUOp::CountTotalRows(const std::string &dir, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;

  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("caption", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));

  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connector_size = cfg->op_connector_size();

  // compat does not affect the count result, so set it to true default.
  auto op = std::make_shared<SBUOp>(dir, true, std::move(schema), std::move(sampler), num_workers, op_connector_size);

  // the logic of counting the number of samples
  RETURN_IF_NOT_OK(op->PrepareData());
  *count = op->image_caption_pairs_.size();

  return Status::OK();
}

Status SBUOp::PrepareData() {
  const Path url_file_name("SBU_captioned_photo_dataset_urls.txt");
  const Path caption_file_name("SBU_captioned_photo_dataset_captions.txt");
  const Path image_folder_name("sbu_images");
  auto real_folder_path = FileUtils::GetRealPath(common::SafeCStr(folder_path_));
  CHECK_FAIL_RETURN_UNEXPECTED(real_folder_path.has_value(), "Get real path failed: " + folder_path_);
  Path root_dir(real_folder_path.value());

  url_path_ = root_dir / url_file_name;
  CHECK_FAIL_RETURN_UNEXPECTED(
    url_path_.Exists() && !url_path_.IsDirectory(),
    "Invalid file, SBU url file: " + url_path_.ToString() + " does not exist or is a directory.");
  MS_LOG(INFO) << "SBU operator found url file " << url_path_.ToString() << ".";

  caption_path_ = root_dir / caption_file_name;
  CHECK_FAIL_RETURN_UNEXPECTED(
    caption_path_.Exists() && !caption_path_.IsDirectory(),
    "Invalid file, SBU caption file: " + caption_path_.ToString() + " does not exist or is a directory.");
  MS_LOG(INFO) << "SBU operator found caption file " << caption_path_.ToString() << ".";

  image_folder_ = root_dir / image_folder_name;
  CHECK_FAIL_RETURN_UNEXPECTED(
    image_folder_.Exists() && image_folder_.IsDirectory(),
    "Invalid folder, SBU image folder:" + image_folder_.ToString() + " does not exist or is not a directory.");
  MS_LOG(INFO) << "SBU operator found image folder " << image_folder_.ToString() << ".";

  std::ifstream url_file_reader;
  std::ifstream caption_file_reader;

  url_file_reader.open(url_path_.ToString(), std::ios::in);
  caption_file_reader.open(caption_path_.ToString(), std::ios::in);

  CHECK_FAIL_RETURN_UNEXPECTED(url_file_reader.is_open(), "Invalid file, failed to open " + url_path_.ToString() +
                                                            ": the SBU url file is permission denied.");
  CHECK_FAIL_RETURN_UNEXPECTED(
    caption_file_reader.is_open(),
    "Invalid file, failed to open " + caption_path_.ToString() + ": the SBU caption file is permission denied.");

  Status rc = GetAvailablePairs(url_file_reader, caption_file_reader);
  url_file_reader.close();
  caption_file_reader.close();
  if (rc.IsError()) {
    return rc;
  }

  return Status::OK();
}

Status SBUOp::GetAvailablePairs(std::ifstream &url_file_reader, std::ifstream &caption_file_reader) {
  std::string url_line;
  std::string caption_line;
  int64_t line_num = 0;

  while (std::getline(url_file_reader, url_line) && std::getline(caption_file_reader, caption_line)) {
    CHECK_FAIL_RETURN_UNEXPECTED(
      (url_line.empty() && caption_line.empty()) || (!url_line.empty() && !caption_line.empty()),
      "Invalid data, SBU url: " + url_path_.ToString() + " and caption file: " + caption_path_.ToString() +
        " load empty data at line: " + std::to_string(line_num) + ".");
    if (!url_line.empty() && !caption_line.empty()) {
      line_num++;
      RETURN_IF_NOT_OK(this->ParsePair(url_line, caption_line));
    }
  }

  image_caption_pairs_.shrink_to_fit();

  CHECK_FAIL_RETURN_UNEXPECTED(image_caption_pairs_.size() > 0,
                               "Invalid data, no valid images in " + image_folder_.ToString() + ", check SBU dataset.");

  // base field of RandomAccessOp
  num_rows_ = static_cast<int64_t>(image_caption_pairs_.size());

  return Status::OK();
}

Status SBUOp::ParsePair(const std::string &url, const std::string &caption) {
  constexpr int64_t max_url_length = 23;
  CHECK_FAIL_RETURN_UNEXPECTED(url.length() > max_url_length, "Invalid url in " + url_path_.ToString() + ": " + url);
  std::string image_name = url.substr(23);
  RETURN_IF_NOT_OK(this->ReplaceAll(&image_name, "/", "_"));

  Path image_path = image_folder_ / Path(image_name);
  if (image_path.Exists() && !image_path.IsDirectory()) {
    // rstrip caption
    image_caption_pairs_.emplace_back(std::make_pair(image_path, caption.substr(0, caption.find_last_not_of(" ") + 1)));
  }

  return Status::OK();
}

Status SBUOp::ReplaceAll(std::string *str, const std::string &from, const std::string &to) const {
  size_t pos = 0;
  while ((pos = str->find(from, pos)) != std::string::npos) {
    str->replace(pos, from.length(), to);
    pos += to.length();
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
