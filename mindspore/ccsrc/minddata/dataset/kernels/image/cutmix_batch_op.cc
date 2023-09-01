/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/image/cutmix_batch_op.h"

#include <cmath>
#include <limits>
#include <string>

#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
constexpr size_t kInputColumnSize = 2;
constexpr size_t kMinLabelShapeSize = 2;
constexpr size_t kMaxLabelShapeSize = 3;
constexpr size_t kImageShapeSize = 4;
constexpr size_t kDimensionOne = 1;
constexpr size_t kDimensionTwo = 2;
constexpr size_t kDimensionThree = 3;

CutMixBatchOp::CutMixBatchOp(ImageBatchFormat image_batch_format, float alpha, float prob)
    : image_batch_format_(image_batch_format), alpha_(alpha), prob_(prob) {
  rnd_.seed(GetSeed());
}

void CutMixBatchOp::GetCropBox(int height, int width, float lam, int *x, int *y, int *crop_width, int *crop_height) {
  const float cut_ratio = std::sqrt(1.F - lam);
  auto cut_w = static_cast<int>(static_cast<float>(width) * cut_ratio);
  auto cut_h = static_cast<int>(static_cast<float>(height) * cut_ratio);
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
  // check column size
  if (input.size() < kInputColumnSize) {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: invalid input size, input should have 2 columns (image and label), but got: " +
      std::to_string(input.size()));
  }

  std::shared_ptr<Tensor> image = input.at(0);
  std::shared_ptr<Tensor> label = input.at(1);
  TensorShape image_shape = image->shape();
  TensorShape label_shape = label->shape();

  // check image shape
  if (image_shape.Size() != kImageShapeSize) {
    std::string err_msg = "CutMixBatch: input image is not in shape of <B,H,W,C> or <B,C,H,W>, but got dimension: " +
                          std::to_string(image_shape.Size());
    if (image_shape.Size() == kDefaultImageRank) {
      err_msg += ". You may need to perform Batch first.";
    }
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // check image channel
  dsize_t channel_index;
  if (image_batch_format_ == ImageBatchFormat::kNCHW) {
    channel_index = kChannelIndexCHW + 1;
  } else {  // ImageBatchFormat::kNHWC
    channel_index = kChannelIndexHWC + 1;
  }
  if (image_shape[channel_index] != kMinImageChannel && image_shape[channel_index] != kDefaultImageChannel) {
    RETURN_STATUS_UNEXPECTED("CutMixBatch: input image is not in channel of 1 or 3, but got channel: " +
                             std::to_string(image_shape[channel_index]));
  }

  // check label dtype
  CHECK_FAIL_RETURN_UNEXPECTED(label->type().IsNumeric(),
                               "CutMixBatch: invalid label type, label must be in a numeric type, but got: " +
                                 label->type().ToString() + ". You may need to perform OneHot first.");

  // check label size
  if (label_shape.Size() != kMinLabelShapeSize && label_shape.Size() != kMaxLabelShapeSize) {
    RETURN_STATUS_UNEXPECTED(
      "CutMixBatch: input label is not in shape of <Batch,Class> or <Batch,Row,Class>, but got dimension: " +
      std::to_string(label_shape.Size()) + ". You may need to perform OneHot and Batch first.");
  }

  // check batch size equal
  if (image_shape[0] != label_shape[0]) {
    RETURN_STATUS_UNEXPECTED("CutMixBatch: batch sizes of image and label must be the same, but got image size: " +
                             std::to_string(image_shape[0]) + " and label size: " + std::to_string(label_shape[0]));
  }

  return Status::OK();
}

Status CutMixBatchOp::ComputeImage(const std::shared_ptr<Tensor> &image, int64_t rand_indx_i, float lam,
                                   float *label_lam, std::shared_ptr<Tensor> *image_i) {
  TensorShape image_shape = image->shape();
  int x, y, crop_width, crop_height;
  // Get a random image
  TensorShape remaining({-1});
  uchar *start_addr_of_index = nullptr;
  std::shared_ptr<Tensor> rand_image;

  RETURN_IF_NOT_OK(image->StartAddrOfIndex({rand_indx_i, 0, 0, 0}, &start_addr_of_index, &remaining));
  RETURN_IF_NOT_OK(Tensor::CreateFromMemory(
    TensorShape({image_shape[kDimensionOne], image_shape[kDimensionTwo], image_shape[kDimensionThree]}), image->type(),
    start_addr_of_index, &rand_image));

  // Compute image
  if (image_batch_format_ == ImageBatchFormat::kNHWC) {
    // NHWC Format
    GetCropBox(static_cast<int32_t>(image_shape[kDimensionOne]), static_cast<int32_t>(image_shape[kDimensionTwo]), lam,
               &x, &y, &crop_width, &crop_height);
    std::shared_ptr<CVTensor> crop_from = CVTensor::AsCVTensor(rand_image);
    std::shared_ptr<CVTensor> mix_to = CVTensor::AsCVTensor(*image_i);
    cv::Rect roi(x, y, crop_width, crop_height);
    (crop_from->mat())(roi).copyTo((mix_to->mat())(roi));
    *image_i = std::static_pointer_cast<Tensor>(mix_to);
    *label_lam = 1.F - (static_cast<float>(crop_width * crop_height) /
                        static_cast<float>(image_shape[kDimensionOne] * image_shape[kDimensionTwo]));
  } else {
    // NCHW Format
    GetCropBox(static_cast<int32_t>(image_shape[kDimensionTwo]), static_cast<int32_t>(image_shape[kDimensionThree]),
               lam, &x, &y, &crop_width, &crop_height);
    // Divide a multi-channel array into several single-channel arrays
    std::vector<std::shared_ptr<Tensor>> rand_image_channels;
    std::vector<std::shared_ptr<Tensor>> image_i_channels;
    RETURN_IF_NOT_OK(BatchTensorToTensorVector(rand_image, &rand_image_channels));
    RETURN_IF_NOT_OK(BatchTensorToTensorVector(*image_i, &image_i_channels));
    std::vector<std::shared_ptr<Tensor>> mix_channels;
    for (auto i = 0; i < rand_image_channels.size() && i < image_i_channels.size(); ++i) {
      std::shared_ptr<CVTensor> crop_from = CVTensor::AsCVTensor(rand_image_channels[i]);
      std::shared_ptr<CVTensor> mix_to = CVTensor::AsCVTensor(image_i_channels[i]);
      cv::Rect roi(x, y, crop_width, crop_height);
      (crop_from->mat())(roi).copyTo((mix_to->mat())(roi));
      mix_channels.push_back(std::static_pointer_cast<Tensor>(mix_to));
    }
    RETURN_IF_NOT_OK(TensorVectorToBatchTensor(mix_channels, image_i));
    *label_lam = 1.F - (static_cast<float>(crop_width * crop_height) /
                        static_cast<float>(image_shape[kDimensionTwo] * image_shape[kDimensionThree]));
  }

  return Status::OK();
}

Status CutMixBatchOp::ComputeLabel(const std::shared_ptr<Tensor> &label, int64_t rand_indx_i, int64_t index_i,
                                   int64_t row_labels, int64_t num_classes, std::size_t label_shape_size,
                                   float label_lam, std::shared_ptr<Tensor> *out_labels) {
  // Compute labels
  std::shared_ptr<Tensor> float_label;
  RETURN_IF_NOT_OK(TypeCast(label, &float_label, DataType(DataType::DE_FLOAT32)));
  for (int64_t j = 0; j < row_labels; j++) {
    for (int64_t k = 0; k < num_classes; k++) {
      std::vector<int64_t> first_index =
        label_shape_size == kMaxLabelShapeSize ? std::vector{index_i, j, k} : std::vector{index_i, k};
      std::vector<int64_t> second_index =
        label_shape_size == kMaxLabelShapeSize ? std::vector{rand_indx_i, j, k} : std::vector{rand_indx_i, k};
      float first_value;
      float second_value;
      RETURN_IF_NOT_OK(float_label->GetItemAt(&first_value, first_index));
      RETURN_IF_NOT_OK(float_label->GetItemAt(&second_value, second_index));
      RETURN_IF_NOT_OK((*out_labels)->SetItemAt(first_index, label_lam * first_value + (1 - label_lam) * second_value));
    }
  }

  return Status::OK();
}

Status CutMixBatchOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  RETURN_IF_NOT_OK(ValidateCutMixBatch(input));
  TensorShape image_shape = input.at(0)->shape();
  TensorShape label_shape = input.at(1)->shape();

  // Move images into a vector of Tensors
  std::vector<std::shared_ptr<Tensor>> images;
  RETURN_IF_NOT_OK(BatchTensorToTensorVector(input.at(0), &images));

  // Calculate random labels
  std::vector<int64_t> rand_indx;
  CHECK_FAIL_RETURN_UNEXPECTED(
    images.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
    "The size of \"images\" must not be more than \"INT64_MAX\", but got: " + std::to_string(images.size()));
  for (auto idx = 0; idx < images.size(); idx++) {
    rand_indx.push_back(idx);
  }
  std::shuffle(rand_indx.begin(), rand_indx.end(), rnd_);
  std::gamma_distribution<float> gamma_alpha(alpha_, 1.0);
  std::gamma_distribution<float> gamma_beta(alpha_, 1.0);
  std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

  // Tensor holding the output labels
  std::shared_ptr<Tensor> out_labels;
  RETURN_IF_NOT_OK(TypeCast(input.at(1), &out_labels, DataType(DataType::DE_FLOAT32)));
  int64_t row_labels = label_shape.Size() == kMaxLabelShapeSize ? label_shape[kDimensionOne] : 1;
  int64_t num_classes = label_shape[-1];

  // Compute labels and images
  for (auto i = 0; i < image_shape[0]; i++) {
    // Calculating lambda
    // If x1 is a random variable from Gamma(a1, 1) and x2 is a random variable from Gamma(a2, 1)
    // then x = x1 / (x1+x2) is a random variable from Beta(a1, a2)
    float x1 = gamma_alpha(rnd_);
    float x2 = gamma_beta(rnd_);
    CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() - x1) > x2,
                                 "CutMixBatchOp: gamma_distribution x1 and x2 are too large, got x1: " +
                                   std::to_string(x1) + ", x2:" + std::to_string(x2));
    float lam = x1 / (x1 + x2);
    double random_number = uniform_distribution(rnd_);
    if (random_number < prob_) {
      float label_lam;  // lambda used for labels
      // Compute image
      RETURN_IF_NOT_OK(ComputeImage(input.at(0), rand_indx[i], lam, &label_lam, &images[i]));
      // Compute labels
      RETURN_IF_NOT_OK(ComputeLabel(input.at(1), rand_indx[i], static_cast<int64_t>(i), row_labels, num_classes,
                                    label_shape.Size(), label_lam, &out_labels));
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
