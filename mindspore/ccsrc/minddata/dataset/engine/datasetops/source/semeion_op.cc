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
#include "minddata/dataset/engine/datasetops/source/semeion_op.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
constexpr uint32_t kSemeionImageSize = 256;
constexpr uint32_t kSemeionLabelSize = 10;

SemeionOp::SemeionOp(const std::string &dataset_dir, int32_t num_parallel_workers,
                     std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler, int32_t queue_size)
    : MappableLeafOp(num_parallel_workers, queue_size, std::move(sampler)),
      dataset_dir_(dataset_dir),
      data_schema_(std::move(data_schema)),
      semeionline_rows_({}) {}

void SemeionOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nNumber of rows: " << num_rows_ << "\nSemeionOp directory: " << dataset_dir_;
  }
}

Status SemeionOp::PrepareData() {
  auto real_path = FileUtils::GetRealPath(dataset_dir_.c_str());
  if (!real_path.has_value()) {
    RETURN_STATUS_UNEXPECTED("Invalid file path, Semeion Dataset folder: " + dataset_dir_ + " does not exist.");
  }
  Path data_folder(real_path.value());

  Path file_path = data_folder / "semeion.data";
  CHECK_FAIL_RETURN_UNEXPECTED(file_path.Exists() && !file_path.IsDirectory(),
                               "Invalid file, failed to find semeion file: " + data_folder.ToString());

  MS_LOG(INFO) << "Semeion file found: " << file_path << ".";

  std::ifstream handle(file_path.ToString());

  CHECK_FAIL_RETURN_UNEXPECTED(handle.is_open(), "Invalid file, failed to open file: " + file_path.ToString());

  std::string line;
  while (getline(handle, line)) {
    semeionline_rows_.push_back(line);
  }
  handle.close();

  num_rows_ = semeionline_rows_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows_ > 0,
                               "Invalid data, SemeionDataset API can't read the data file (interface mismatch or no "
                               "data found). Check file path: " +
                                 dataset_dir_);
  return Status::OK();
}

Status SemeionOp::TransRowIdResult(row_id_type index, std::shared_ptr<Tensor> *img_tensor,
                                   std::shared_ptr<Tensor> *label_tensor) {
  RETURN_UNEXPECTED_IF_NULL(img_tensor);
  RETURN_UNEXPECTED_IF_NULL(label_tensor);
  std::vector<uint8_t> img;
  uint32_t label;
  std::string line = semeionline_rows_[index];
  uint32_t i = 0;
  while (i < kSemeionImageSize) {
    auto pos = line.find(" ");
    CHECK_FAIL_RETURN_UNEXPECTED(pos != std::string::npos, "Invalid data, file content does not match SemeionDataset.");
    std::string s = line.substr(0, pos);
    uint8_t value_img;
    try {
      value_img = std::stoi(s);
    } catch (std::exception &e) {
      RETURN_STATUS_UNEXPECTED("Invalid data, image data in file should be in type of uint8, but got: " + s + ".");
    }
    img.push_back(value_img);
    line.erase(0, pos + 1);  // to dedele space
    ++i;
  }
  i = 0;
  while (i < kSemeionLabelSize) {
    auto pos = line.find(" ");
    CHECK_FAIL_RETURN_UNEXPECTED(pos != std::string::npos, "Invalid data, file content does not match SemeionDataset.");
    std::string s = line.substr(0, pos);
    line.erase(0, pos + 1);
    uint8_t value_label;
    try {
      value_label = std::stoi(s);
    } catch (std::exception &e) {
      RETURN_STATUS_UNEXPECTED("Invalid data, label data in file should be in type of uint8, but got: " + s + ".");
    }
    if (value_label != 0) {
      label = i;
      break;
    }
    ++i;
  }
  RETURN_IF_NOT_OK(Tensor::CreateScalar(label, label_tensor));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(img, img_tensor));
  RETURN_IF_NOT_OK((*img_tensor)->Reshape(TensorShape{16, 16}));
  return Status::OK();
}

Status SemeionOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  Path dir_path(dataset_dir_);
  std::shared_ptr<Tensor> img_tensor, label_tensor;
  RETURN_IF_NOT_OK(TransRowIdResult(row_id, &img_tensor, &label_tensor));
  (*trow) = TensorRow(row_id, {img_tensor, label_tensor});
  trow->setPath({dir_path.ToString(), dir_path.ToString()});

  return Status::OK();
}

Status SemeionOp::CountTotalRows(const std::string &dataset_dir, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto new_sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);

  // build a new unique schema object
  auto new_schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(new_schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape label_scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(new_schema->AddColumn(
    ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &label_scalar)));

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  auto op = std::make_shared<SemeionOp>(dataset_dir, num_workers, std::move(new_schema), std::move(new_sampler),
                                        op_connect_size);
  RETURN_IF_NOT_OK(op->PrepareData());
  *count = static_cast<int64_t>(op->semeionline_rows_.size());
  return Status::OK();
}

Status SemeionOp::ComputeColMap() {
  // set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (uint32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
