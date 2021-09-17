/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <string>
#include <utility>
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/cutmix_batch_op.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

constexpr size_t kMinLabelShapeSize = 2;
constexpr size_t kMaxLabelShapeSize = 3;
constexpr size_t kExpectedImageShapeSize = 4;
constexpr size_t kDimensionOne = 1;
constexpr size_t kDimensionTwo = 2;
constexpr size_t kDimensionThree = 3;
constexpr int64_t kValueOne = 1;
constexpr int64_t kValueThree = 3;

CutMixBatchOp::CutMixBatchOp(ImageBatchFormat image_batch_format, float alpha, float prob)
    : image_batch_format_(image_batch_format), alpha_(alpha), prob_(prob) {
  rnd_.seed(GetSeed());
}

void CutMixBatchOp::GetCropBox(int height, int width, float lam, int *x, int *y, int *crop_width, int *crop_height) {
  const float cut_ratio = 1 - lam;
  int cut_w = static_cast<int>(width * cut_ratio);
  int cut_h = static_cast<int>(height * cut_ratio);
  std::uniform_int_distribution<int> width_uniform_distribution(0, width);
  std::uniform_int_distribution<int> height_uniform_distribution(0, height);
  int cx = width_uniform_distribution(rnd_);
  int x2, y2;
  int cy = height_uniform_distribution(rnd_);
  constexpr int cut_half = 2;
  *x = std::clamp(cx - cut_w / cut_half, 0, width - 1);   // horizontal coordinate of left side of crop box
  *y = std::clamp(cy - cut_h / cut_half, 0, height - 1);  // vertical coordinate of the top side of crop box
  x2 = std::clamp(cx + cut_w / cut_half, 0, width - 1);   // horizontal coordinate of right side of crop box
  y2 = std::clamp(cy + cut_h / cut_half, 0, height - 1);  // vertical coordinate of the bottom side of crop box
  *crop_width = std::clamp(x2 - *x, 1, width - 1);
  *crop_height = std::clamp(y2 - *y, 1, height - 1);
}

Status CutMixBatchOp::ValidateCutMixBatch(const TensorRow &input) {
  if (input.size() < kMinLabelShapeSize) {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: invalid input, size of input should be 2 (including image and label), but got: " +
      std::to_string(input.size()));
  }
  std::vector<int64_t> image_shape = input.at(0)->shape().AsVector();
  std::vector<int64_t> label_shape = input.at(1)->shape().AsVector();

  // Check inputs
  if (image_shape.size() != kExpectedImageShapeSize || image_shape[0] != label_shape[0]) {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: please make sure images are <H,W,C> or <C,H,W> format, and batched before calling CutMixBatch.");
  }
  if (!input.at(1)->type().IsInt()) {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: Wrong labels type. The second column (labels) must only include int types, but got:" +
      input.at(1)->type().ToString());
  }
  if (label_shape.size() != kMinLabelShapeSize && label_shape.size() != kMaxLabelShapeSize) {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: wrong labels shape. "
      "The second column (labels) must have a shape of NC or NLC where N is the batch size, "
      "L is the number of labels in each row, and C is the number of classes. "
      "labels must be in one-hot format and in a batch, but got rank: " +
      std::to_string(label_shape.size()));
  }
  std::string shape_info = "(";
  for (auto i : image_shape) {
    shape_info = shape_info + std::to_string(i) + ", ";
  }
  shape_info.replace(shape_info.end() - 1, shape_info.end(), ")");
  if ((image_shape[kDimensionOne] != kValueOne && image_shape[kDimensionOne] != kValueThree) &&
      image_batch_format_ == ImageBatchFormat::kNCHW) {
    RETURN_STATUS_UNEXPECTED("CutMixBatch: image doesn't match the <N,C,H,W> format, got shape: " + shape_info);
  }
  if ((image_shape[kDimensionThree] != kValueOne && image_shape[kDimensionThree] != kValueThree) &&
      image_batch_format_ == ImageBatchFormat::kNHWC) {
    RETURN_STATUS_UNEXPECTED("CutMixBatch: image doesn't match the <N,H,W,C> format, got shape: " + shape_info);
  }

  return Status::OK();
}

Status CutMixBatchOp::ComputeImage(const TensorRow &input, const int64_t rand_indx_i, const float lam, float *label_lam,
                                   std::shared_ptr<Tensor> *image_i) {
  std::vector<int64_t> image_shape = input.at(0)->shape().AsVector();
  int x, y, crop_width, crop_height;
  // Get a random image
  TensorShape remaining({-1});
  uchar *start_addr_of_index = nullptr;
  std::shared_ptr<Tensor> rand_image;

  RETURN_IF_NOT_OK(input.at(0)->StartAddrOfIndex({rand_indx_i, 0, 0, 0}, &start_addr_of_index, &remaining));
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(
    TensorShape({image_shape[kDimensionOne], image_shape[kDimensionTwo], image_shape[kDimensionThree]}),
    input.at(0)->type(), start_addr_of_index, &rand_image));

  // Compute image
  if (image_batch_format_ == ImageBatchFormat::kNHWC) {
    // NHWC Format
    GetCropBox(static_cast<int32_t>(image_shape[kDimensionOne]), static_cast<int32_t>(image_shape[kDimensionTwo]), lam,
               &x, &y, &crop_width, &crop_height);
    std::shared_ptr<Tensor> cropped;
    RETURN_IF_NOT_OK(Crop(rand_image, &cropped, x, y, crop_width, crop_height));
    RETURN_IF_NOT_OK(MaskWithTensor(cropped, image_i, x, y, crop_width, crop_height, ImageFormat::HWC));
    *label_lam = kValueOne - (crop_width * crop_height /
                              static_cast<float>(image_shape[kDimensionOne] * image_shape[kDimensionTwo]));
  } else {
    // NCHW Format
    GetCropBox(static_cast<int32_t>(image_shape[kDimensionTwo]), static_cast<int32_t>(image_shape[kDimensionThree]),
               lam, &x, &y, &crop_width, &crop_height);
    std::vector<std::shared_ptr<Tensor>> channels;          // A vector holding channels of the CHW image
    std::vector<std::shared_ptr<Tensor>> cropped_channels;  // A vector holding the channels of the cropped CHW
    RETURN_IF_NOT_OK(BatchTensorToTensorVector(rand_image, &channels));
    for (auto channel : channels) {
      // Call crop for each single channel
      std::shared_ptr<Tensor> cropped_channel;
      RETURN_IF_NOT_OK(Crop(channel, &cropped_channel, x, y, crop_width, crop_height));
      cropped_channels.push_back(cropped_channel);
    }
    std::shared_ptr<Tensor> cropped;
    // Merge channels to a single tensor
    RETURN_IF_NOT_OK(TensorVectorToBatchTensor(cropped_channels, &cropped));

    RETURN_IF_NOT_OK(MaskWithTensor(cropped, image_i, x, y, crop_width, crop_height, ImageFormat::CHW));
    *label_lam = kValueOne - (crop_width * crop_height /
                              static_cast<float>(image_shape[kDimensionTwo] * image_shape[kDimensionThree]));
  }

  return Status::OK();
}

Status CutMixBatchOp::ComputeLabel(const TensorRow &input, const int64_t rand_indx_i, const int64_t index_i,
                                   const int64_t row_labels, const int64_t num_classes,
                                   const std::size_t label_shape_size, const float label_lam,
                                   std::shared_ptr<Tensor> *out_labels) {
  // Compute labels
  for (int64_t j = 0; j < row_labels; j++) {
    for (int64_t k = 0; k < num_classes; k++) {
      std::vector<int64_t> first_index =
        label_shape_size == kMaxLabelShapeSize ? std::vector{index_i, j, k} : std::vector{index_i, k};
      std::vector<int64_t> second_index =
        label_shape_size == kMaxLabelShapeSize ? std::vector{rand_indx_i, j, k} : std::vector{rand_indx_i, k};
      if (input.at(1)->type().IsSignedInt()) {
        int64_t first_value, second_value;
        RETURN_IF_NOT_OK(input.at(1)->GetItemAt(&first_value, first_index));
        RETURN_IF_NOT_OK(input.at(1)->GetItemAt(&second_value, second_index));
        RETURN_IF_NOT_OK(
          (*out_labels)->SetItemAt(first_index, label_lam * first_value + (1 - label_lam) * second_value));
      } else {
        uint64_t first_value, second_value;
        RETURN_IF_NOT_OK(input.at(1)->GetItemAt(&first_value, first_index));
        RETURN_IF_NOT_OK(input.at(1)->GetItemAt(&second_value, second_index));
        RETURN_IF_NOT_OK(
          (*out_labels)->SetItemAt(first_index, label_lam * first_value + (1 - label_lam) * second_value));
      }
    }
  }

  return Status::OK();
}

Status CutMixBatchOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  RETURN_IF_NOT_OK(ValidateCutMixBatch(input));
  std::vector<int64_t> image_shape = input.at(0)->shape().AsVector();
  std::vector<int64_t> label_shape = input.at(1)->shape().AsVector();

  // Move images into a vector of Tensors
  std::vector<std::shared_ptr<Tensor>> images;
  RETURN_IF_NOT_OK(BatchTensorToTensorVector(input.at(0), &images));

  // Calculate random labels
  std::vector<int64_t> rand_indx;
  CHECK_FAIL_RETURN_UNEXPECTED(
    images.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
    "The size of \"images\" must not be more than \"INT64_MAX\", but got: " + std::to_string(images.size()));
  for (int64_t idx = 0; idx < static_cast<int64_t>(images.size()); idx++) rand_indx.push_back(idx);
  std::shuffle(rand_indx.begin(), rand_indx.end(), rnd_);
  std::gamma_distribution<float> gamma_distribution(alpha_, 1);
  std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

  // Tensor holding the output labels
  std::shared_ptr<Tensor> out_labels;
  RETURN_IF_NOT_OK(TypeCast(std::move(input.at(1)), &out_labels, DataType(DataType::DE_FLOAT32)));
  int64_t row_labels = label_shape.size() == kValueThree ? label_shape[kDimensionOne] : kValueOne;
  int64_t num_classes = label_shape.size() == kValueThree ? label_shape[kDimensionTwo] : label_shape[kDimensionOne];

  // Compute labels and images
  for (size_t i = 0; i < static_cast<size_t>(image_shape[0]); i++) {
    // Calculating lambda
    // If x1 is a random variable from Gamma(a1, 1) and x2 is a random variable from Gamma(a2, 1)
    // then x = x1 / (x1+x2) is a random variable from Beta(a1, a2)
    float x1 = gamma_distribution(rnd_);
    float x2 = gamma_distribution(rnd_);
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() - x1) > x2,
                                 "CutMixBatchOp: gamma_distribution x1 and x2 are too large, got x1: " +
                                   std::to_string(x1) + ", x2:" + std::to_string(x2));
    float lam = x1 / (x1 + x2);
    double random_number = uniform_distribution(rnd_);
    if (random_number < prob_) {
      float label_lam;  // lambda used for labels
      // Compute image
      RETURN_IF_NOT_OK(ComputeImage(input, rand_indx[i], lam, &label_lam, &images[i]));
      // Compute labels
      RETURN_IF_NOT_OK(ComputeLabel(input, rand_indx[i], static_cast<int64_t>(i), row_labels, num_classes,
                                    label_shape.size(), label_lam, &out_labels));
    }
  }

  std::shared_ptr<Tensor> out_images;
  RETURN_IF_NOT_OK(TensorVectorToBatchTensor(images, &out_images));

  // Move the output into a TensorRow
  output->push_back(out_images);
  output->push_back(out_labels);

  return Status::OK();
}

void CutMixBatchOp::Print(std::ostream &out) const {
  out << "CutMixBatchOp: "
      << "\n";
}
}  // namespace dataset
}  // namespace mindspore
