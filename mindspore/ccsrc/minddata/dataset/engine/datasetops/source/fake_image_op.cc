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
#include "minddata/dataset/engine/datasetops/source/fake_image_op.h"

#include <iomanip>
#include <set>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {
FakeImageOp::FakeImageOp(int32_t num_images, const std::vector<int32_t> &image_size, int32_t num_classes,
                         int32_t base_seed, int32_t num_workers, int32_t op_connector_size,
                         std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_workers, op_connector_size, std::move(sampler)),
      num_images_(num_images),
      image_size_(image_size),
      num_classes_(num_classes),
      base_seed_(base_seed),
      image_tensor_({}),
      data_schema_(std::move(data_schema)) {}

// Load 1 TensorRow (image, label) using 1 trow.
Status FakeImageOp::LoadTensorRow(row_id_type row_id, TensorRow *trow) {
  RETURN_UNEXPECTED_IF_NULL(trow);
  std::shared_ptr<Tensor> image, label;

  auto images_buf = std::make_unique<double[]>(image_total_size_);
  auto pixels = &images_buf[0];
  {
    std::unique_lock<std::mutex> lock(access_mutex_);
    if (image_tensor_[row_id] == nullptr) {
      rand_gen_.seed(base_seed_ + row_id);  // set seed for random generator.
      std::normal_distribution<double> distribution(0.0, 1.0);
      for (int i = 0; i < image_total_size_; ++i) {
        pixels[i] = distribution(rand_gen_);  // generate the Gaussian distribution pixel.
        if (pixels[i] < 0) {
          pixels[i] = 0;
        }
        if (pixels[i] > 255) {
          pixels[i] = 255;
        }
      }
      TensorShape img_tensor_shape = TensorShape({image_size_[0], image_size_[1], image_size_[2]});
      RETURN_IF_NOT_OK(Tensor::CreateFromMemory(img_tensor_shape, data_schema_->Column(0).Type(),
                                                reinterpret_cast<unsigned char *>(pixels), &image));
      RETURN_IF_NOT_OK(Tensor::CreateFromTensor(image, &image_tensor_[row_id]));
    } else {
      RETURN_IF_NOT_OK(Tensor::CreateFromTensor(image_tensor_[row_id], &image));
    }
  }
  RETURN_IF_NOT_OK(Tensor::CreateScalar(label_list_[row_id], &label));
  (*trow) = TensorRow(row_id, {std::move(image), std::move(label)});
  return Status::OK();
}

// A print method typically used for debugging.
void FakeImageOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nNumber of images: " << num_images_ << "\nNumber of classes: " << num_classes_
        << "\nBase seed: " << base_seed_ << "\n\n";
  }
}

// Derived from RandomAccessOp.
Status FakeImageOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty() || label_list_.empty()) {
    if (label_list_.empty()) {
      RETURN_STATUS_UNEXPECTED(
        "[Internal ERROR] No image found in dataset. Check if image was generated successfully.");
    } else {
      RETURN_STATUS_UNEXPECTED(
        "[Internal ERROR] Map for storing image-index pair is nullptr or has been set in other place, "
        "it must be empty before using GetClassIds.");
    }
  }
  for (size_t i = 0; i < label_list_.size(); ++i) {
    (*cls_ids)[label_list_[i]].push_back(i);
  }
  for (auto &pr : (*cls_ids)) {
    pr.second.shrink_to_fit();
  }
  return Status::OK();
}

Status FakeImageOp::GetItem(int32_t index) {
  // generate one target label according to index and save it in label_list_.
  rand_gen_.seed(base_seed_ + index);  // set seed for random generator.
  std::uniform_int_distribution<int32_t> dist(0, num_classes_ - 1);
  uint32_t target = dist(rand_gen_);  // generate the target.
  label_list_.emplace_back(target);

  return Status::OK();
}

Status FakeImageOp::PrepareData() {
  // FakeImage generate image with Gaussian distribution.
  image_total_size_ = image_size_[0] * image_size_[1] * image_size_[2];

  for (size_t i = 0; i < num_images_; ++i) {
    RETURN_IF_NOT_OK(GetItem(i));
  }

  label_list_.shrink_to_fit();
  num_rows_ = label_list_.size();
  CHECK_FAIL_RETURN_UNEXPECTED(num_rows_ > 0, "Invalid data, generate fake data failed, please check dataset API.");
  image_tensor_.clear();
  image_tensor_.resize(num_rows_);
  return Status::OK();
}

Status FakeImageOp::ComputeColMap() {
  // Extract the column name mapping from the schema and save it in the class.
  if (column_name_id_map_.empty()) {
    RETURN_IF_NOT_OK(data_schema_->GetColumnNameMap(&(column_name_id_map_)));
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
