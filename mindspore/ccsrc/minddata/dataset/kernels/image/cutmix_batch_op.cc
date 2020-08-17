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

CutMixBatchOp::CutMixBatchOp(ImageBatchFormat image_batch_format, float alpha, float prob)
    : image_batch_format_(image_batch_format), alpha_(alpha), prob_(prob) {
  rnd_.seed(GetSeed());
}

void CutMixBatchOp::GetCropBox(int height, int width, float lam, int *x, int *y, int *crop_width, int *crop_height) {
  float cut_ratio = 1 - lam;
  int cut_w = static_cast<int>(width * cut_ratio);
  int cut_h = static_cast<int>(height * cut_ratio);
  std::uniform_int_distribution<int> width_uniform_distribution(0, width);
  std::uniform_int_distribution<int> height_uniform_distribution(0, height);
  int cx = width_uniform_distribution(rnd_);
  int x2, y2;
  int cy = height_uniform_distribution(rnd_);
  *x = std::clamp(cx - cut_w / 2, 0, width - 1);   // horizontal coordinate of left side of crop box
  *y = std::clamp(cy - cut_h / 2, 0, height - 1);  // vertical coordinate of the top side of crop box
  x2 = std::clamp(cx + cut_w / 2, 0, width - 1);   // horizontal coordinate of right side of crop box
  y2 = std::clamp(cy + cut_h / 2, 0, height - 1);  // vertical coordinate of the bottom side of crop box
  *crop_width = std::clamp(x2 - *x, 1, width - 1);
  *crop_height = std::clamp(y2 - *y, 1, height - 1);
}

Status CutMixBatchOp::Compute(const TensorRow &input, TensorRow *output) {
  if (input.size() < 2) {
    RETURN_STATUS_UNEXPECTED("Both images and labels columns are required for this operation");
  }

  std::vector<std::shared_ptr<Tensor>> images;
  std::vector<int64_t> image_shape = input.at(0)->shape().AsVector();
  std::vector<int64_t> label_shape = input.at(1)->shape().AsVector();

  // Check inputs
  if (image_shape.size() != 4 || image_shape[0] != label_shape[0]) {
    RETURN_STATUS_UNEXPECTED("You must batch before calling CutMixBatch.");
  }
  if (label_shape.size() != 2) {
    RETURN_STATUS_UNEXPECTED("CutMixBatch: Label's must be in one-hot format and in a batch");
  }
  if ((image_shape[1] != 1 && image_shape[1] != 3) && image_batch_format_ == ImageBatchFormat::kNCHW) {
    RETURN_STATUS_UNEXPECTED("CutMixBatch: Image doesn't match the given image format.");
  }
  if ((image_shape[3] != 1 && image_shape[3] != 3) && image_batch_format_ == ImageBatchFormat::kNHWC) {
    RETURN_STATUS_UNEXPECTED("CutMixBatch: Image doesn't match the given image format.");
  }

  // Move images into a vector of Tensors
  RETURN_IF_NOT_OK(BatchTensorToTensorVector(input.at(0), &images));

  // Calculate random labels
  std::vector<int64_t> rand_indx;
  for (int64_t i = 0; i < images.size(); i++) rand_indx.push_back(i);
  std::shuffle(rand_indx.begin(), rand_indx.end(), rnd_);

  std::gamma_distribution<float> gamma_distribution(alpha_, 1);
  std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

  // Tensor holding the output labels
  std::shared_ptr<Tensor> out_labels;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape(label_shape), DataType(DataType::DE_FLOAT32), &out_labels));

  // Compute labels and images
  for (int i = 0; i < image_shape[0]; i++) {
    // Calculating lambda
    // If x1 is a random variable from Gamma(a1, 1) and x2 is a random variable from Gamma(a2, 1)
    // then x = x1 / (x1+x2) is a random variable from Beta(a1, a2)
    float x1 = gamma_distribution(rnd_);
    float x2 = gamma_distribution(rnd_);
    float lam = x1 / (x1 + x2);
    double random_number = uniform_distribution(rnd_);
    if (random_number < prob_) {
      int x, y, crop_width, crop_height;
      float label_lam;  // lambda used for labels

      // Get a random image
      TensorShape remaining({-1});
      uchar *start_addr_of_index = nullptr;
      std::shared_ptr<Tensor> rand_image;
      RETURN_IF_NOT_OK(input.at(0)->StartAddrOfIndex({rand_indx[i], 0, 0, 0}, &start_addr_of_index, &remaining));
      RETURN_IF_NOT_OK(Tensor::CreateFromMemory(TensorShape({image_shape[1], image_shape[2], image_shape[3]}),
                                                input.at(0)->type(), start_addr_of_index, &rand_image));

      // Compute image
      if (image_batch_format_ == ImageBatchFormat::kNHWC) {
        // NHWC Format
        GetCropBox(static_cast<int32_t>(image_shape[1]), static_cast<int32_t>(image_shape[2]), lam, &x, &y, &crop_width,
                   &crop_height);
        std::shared_ptr<Tensor> cropped;
        RETURN_IF_NOT_OK(Crop(rand_image, &cropped, x, y, crop_width, crop_height));
        RETURN_IF_NOT_OK(MaskWithTensor(cropped, &images[i], x, y, crop_width, crop_height, ImageFormat::HWC));
        label_lam = 1 - (crop_width * crop_height / static_cast<float>(image_shape[1] * image_shape[2]));

      } else {
        // NCHW Format
        GetCropBox(static_cast<int32_t>(image_shape[2]), static_cast<int32_t>(image_shape[3]), lam, &x, &y, &crop_width,
                   &crop_height);
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

        RETURN_IF_NOT_OK(MaskWithTensor(cropped, &images[i], x, y, crop_width, crop_height, ImageFormat::CHW));
        label_lam = 1 - (crop_width * crop_height / static_cast<float>(image_shape[2] * image_shape[3]));
      }

      // Compute labels
      for (int j = 0; j < label_shape[1]; j++) {
        uint64_t first_value, second_value;
        RETURN_IF_NOT_OK(input.at(1)->GetItemAt(&first_value, {i, j}));
        RETURN_IF_NOT_OK(input.at(1)->GetItemAt(&second_value, {rand_indx[i] % label_shape[0], j}));
        RETURN_IF_NOT_OK(out_labels->SetItemAt({i, j}, label_lam * first_value + (1 - label_lam) * second_value));
      }
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
      << "image_batch_format: " << image_batch_format_ << "alpha: " << alpha_ << ", probability: " << prob_ << "\n";
}
}  // namespace dataset
}  // namespace mindspore
