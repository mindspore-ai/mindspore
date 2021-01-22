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
#include "minddata/dataset/kernels/image/mixup_batch_op.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

MixUpBatchOp::MixUpBatchOp(float alpha) : alpha_(alpha) { rnd_.seed(GetSeed()); }

Status MixUpBatchOp::Compute(const TensorRow &input, TensorRow *output) {
  if (input.size() < 2) {
    RETURN_STATUS_UNEXPECTED("MixUpBatch: input lack of images or labels");
  }

  std::vector<std::shared_ptr<CVTensor>> images;
  std::vector<int64_t> image_shape = input.at(0)->shape().AsVector();
  std::vector<int64_t> label_shape = input.at(1)->shape().AsVector();

  // Check inputs
  if (image_shape.size() != 4 || image_shape[0] != label_shape[0]) {
    RETURN_STATUS_UNEXPECTED(
      "MixUpBatch: "
      "please make sure images are HWC or CHW and batched before calling MixUpBatch.");
  }
  if (!input.at(1)->type().IsInt()) {
    RETURN_STATUS_UNEXPECTED(
      "MixUpBatch: wrong labels type. "
      "The second column (labels) must only include int types.");
  }
  if (label_shape.size() != 2 && label_shape.size() != 3) {
    RETURN_STATUS_UNEXPECTED(
      "MixUpBatch: wrong labels shape. "
      "The second column (labels) must have a shape of NC or NLC where N is the batch size, "
      "L is the number of labels in each row, and C is the number of classes. "
      "labels must be in one-hot format and in a batch.");
  }
  if ((image_shape[1] != 1 && image_shape[1] != 3) && (image_shape[3] != 1 && image_shape[3] != 3)) {
    RETURN_STATUS_UNEXPECTED("MixUpBatch: images must be in the shape of HWC or CHW.");
  }

  // Move images into a vector of CVTensors
  RETURN_IF_NOT_OK(BatchTensorToCVTensorVector(input.at(0), &images));

  // Calculating lambda
  // If x1 is a random variable from Gamma(a1, 1) and x2 is a random variable from Gamma(a2, 1)
  // then x = x1 / (x1+x2) is a random variable from Beta(a1, a2)
  std::gamma_distribution<float> distribution(alpha_, 1);
  float x1 = distribution(rnd_);
  float x2 = distribution(rnd_);
  float lam = x1 / (x1 + x2);

  // Calculate random labels
  std::vector<int64_t> rand_indx;
  for (int64_t i = 0; i < images.size(); i++) rand_indx.push_back(i);
  std::shuffle(rand_indx.begin(), rand_indx.end(), rnd_);

  // Compute labels
  std::shared_ptr<Tensor> out_labels;
  RETURN_IF_NOT_OK(TypeCast(std::move(input.at(1)), &out_labels, DataType(DataType::DE_FLOAT32)));

  int64_t row_labels = label_shape.size() == 3 ? label_shape[1] : 1;
  int64_t num_classes = label_shape.size() == 3 ? label_shape[2] : label_shape[1];

  for (int64_t i = 0; i < label_shape[0]; i++) {
    for (int64_t j = 0; j < row_labels; j++) {
      for (int64_t k = 0; k < num_classes; k++) {
        std::vector<int64_t> first_index = label_shape.size() == 3 ? std::vector{i, j, k} : std::vector{i, k};
        std::vector<int64_t> second_index =
          label_shape.size() == 3 ? std::vector{rand_indx[i], j, k} : std::vector{rand_indx[i], k};
        if (input.at(1)->type().IsSignedInt()) {
          int64_t first_value, second_value;
          RETURN_IF_NOT_OK(input.at(1)->GetItemAt(&first_value, first_index));
          RETURN_IF_NOT_OK(input.at(1)->GetItemAt(&second_value, second_index));
          RETURN_IF_NOT_OK(out_labels->SetItemAt(first_index, lam * first_value + (1 - lam) * second_value));
        } else {
          uint64_t first_value, second_value;
          RETURN_IF_NOT_OK(input.at(1)->GetItemAt(&first_value, first_index));
          RETURN_IF_NOT_OK(input.at(1)->GetItemAt(&second_value, second_index));
          RETURN_IF_NOT_OK(out_labels->SetItemAt(first_index, lam * first_value + (1 - lam) * second_value));
        }
      }
    }
  }
  // Compute images
  for (int64_t i = 0; i < images.size(); i++) {
    TensorShape remaining({-1});
    uchar *start_addr_of_index = nullptr;
    std::shared_ptr<Tensor> out;
    RETURN_IF_NOT_OK(input.at(0)->StartAddrOfIndex({rand_indx[i], 0, 0, 0}, &start_addr_of_index, &remaining));
    RETURN_IF_NOT_OK(input.at(0)->CreateFromMemory(TensorShape({image_shape[1], image_shape[2], image_shape[3]}),
                                                   input.at(0)->type(), start_addr_of_index, &out));
    std::shared_ptr<CVTensor> rand_image = CVTensor::AsCVTensor(std::move(out));
    if (!rand_image->mat().data) {
      RETURN_STATUS_UNEXPECTED("MixUpBatch: allocate memory failed.");
    }
    images[i]->mat() = lam * images[i]->mat() + (1 - lam) * rand_image->mat();
  }

  // Move the output into a TensorRow
  std::shared_ptr<Tensor> output_image;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input.at(0)->shape(), input.at(0)->type(), &output_image));
  for (int64_t i = 0; i < images.size(); i++) {
    RETURN_IF_NOT_OK(output_image->InsertTensor({i}, images[i]));
  }
  output->push_back(output_image);
  output->push_back(out_labels);

  return Status::OK();
}

void MixUpBatchOp::Print(std::ostream &out) const {
  out << "MixUpBatchOp: "
      << "alpha: " << alpha_ << "\n";
}
}  // namespace dataset
}  // namespace mindspore
