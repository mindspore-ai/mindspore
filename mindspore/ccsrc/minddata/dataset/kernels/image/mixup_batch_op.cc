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

#include "minddata/dataset/kernels/image/mixup_batch_op.h"

#include <limits>
#include <string>
#include <utility>

#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

constexpr size_t kExpectedImageShapeSize = 4;
constexpr size_t kMaxLabelShapeSize = 3;
constexpr size_t kMinLabelShapeSize = 2;
constexpr size_t dimension_one = 1;
constexpr size_t dimension_two = 2;
constexpr size_t dimension_three = 3;
constexpr int64_t value_one = 1;
constexpr int64_t value_three = 3;

MixUpBatchOp::MixUpBatchOp(float alpha) : alpha_(alpha) { rnd_.seed(GetSeed()); }

Status MixUpBatchOp::ComputeLabels(const std::shared_ptr<Tensor> &label, std::shared_ptr<Tensor> *out_labels,
                                   std::vector<int64_t> *rand_indx, const std::vector<int64_t> &label_shape, float lam,
                                   size_t images_size) {
  CHECK_FAIL_RETURN_UNEXPECTED(
    images_size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
    "The \'images_size\' must not be more than \'INT64_MAX\', but got: " + std::to_string(images_size));
  for (int64_t i = 0; i < static_cast<int64_t>(images_size); i++) {
    rand_indx->push_back(i);
  }
  std::shuffle(rand_indx->begin(), rand_indx->end(), rnd_);

  std::shared_ptr<Tensor> float_label;
  RETURN_IF_NOT_OK(TypeCast(label, &float_label, DataType(DataType::DE_FLOAT32)));
  RETURN_IF_NOT_OK(TypeCast(label, out_labels, DataType(DataType::DE_FLOAT32)));

  int64_t row_labels = label_shape.size() == kMaxLabelShapeSize ? label_shape[1] : 1;
  int64_t num_classes = label_shape.size() == kMaxLabelShapeSize ? label_shape[dimension_two] : label_shape[1];

  for (int64_t i = 0; i < label_shape[0]; i++) {
    for (int64_t j = 0; j < row_labels; j++) {
      for (int64_t k = 0; k < num_classes; k++) {
        std::vector<int64_t> first_index =
          label_shape.size() == kMaxLabelShapeSize ? std::vector{i, j, k} : std::vector{i, k};
        std::vector<int64_t> second_index = label_shape.size() == kMaxLabelShapeSize
                                              ? std::vector{(*rand_indx)[static_cast<size_t>(i)], j, k}
                                              : std::vector{(*rand_indx)[static_cast<size_t>(i)], k};
        float first_value, second_value;
        RETURN_IF_NOT_OK(float_label->GetItemAt(&first_value, first_index));
        RETURN_IF_NOT_OK(float_label->GetItemAt(&second_value, second_index));
        RETURN_IF_NOT_OK((*out_labels)->SetItemAt(first_index, lam * first_value + (1 - lam) * second_value));
      }
    }
  }
  return Status::OK();
}

Status MixUpBatchOp::Compute(const TensorRow &input, TensorRow *output) {
  constexpr int64_t input_size = 2;
  if (input.size() < input_size) {
    RETURN_STATUS_UNEXPECTED("MixUpBatch: size of input data should be 2 (including images or labels), but got: " +
                             std::to_string(input.size()) + ", check 'input_columns' when call this operator.");
  }

  std::vector<std::shared_ptr<CVTensor>> images;
  std::vector<int64_t> image_shape = input.at(0)->shape().AsVector();
  std::vector<int64_t> label_shape = input.at(1)->shape().AsVector();

  // Check inputs
  if (image_shape.size() != kExpectedImageShapeSize || image_shape[0] != label_shape[0]) {
    RETURN_STATUS_UNEXPECTED("MixUpBatch: rank of image shape should be: " + std::to_string(kExpectedImageShapeSize) +
                             ", but got: " + std::to_string(image_shape.size()) +
                             ", make sure image shape are <H,W,C> or <C,H,W> and batched before calling MixUpBatch.");
  }

  CHECK_FAIL_RETURN_UNEXPECTED(input.at(1)->type().IsNumeric(),
                               "MixUpBatch: invalid label type, label must be in a numeric type, but got: " +
                                 input.at(1)->type().ToString() + ". You may need to perform OneHot first.");
  if (label_shape.size() != kMinLabelShapeSize && label_shape.size() != kMaxLabelShapeSize) {
    RETURN_STATUS_UNEXPECTED(
      "MixUpBatch: wrong labels shape. "
      "The second column (labels) must have a shape of NC or NLC where N is the batch size, "
      "L is the number of labels in each row, and C is the number of classes. "
      "labels must be in one-hot format and in a batch, but got rank: " +
      std::to_string(label_shape.size()));
  }
  if ((image_shape[dimension_one] != value_one && image_shape[dimension_one] != value_three) &&
      (image_shape[dimension_three] != value_one && image_shape[dimension_three] != value_three)) {
    RETURN_STATUS_UNEXPECTED("MixUpBatch: images shape should in <N,H,W,C> or <N,C,H,W>, got shape:" +
                             input.at(0)->shape().ToString());
  }

  // Move images into a vector of CVTensors
  RETURN_IF_NOT_OK(BatchTensorToCVTensorVector(input.at(0), &images));

  // Calculating lambda
  // If x1 is a random variable from Gamma(a1, 1) and x2 is a random variable from Gamma(a2, 1)
  // then x = x1 / (x1+x2) is a random variable from Beta(a1, a2)
  std::gamma_distribution<float> distribution(alpha_, 1);
  float x1 = distribution(rnd_);
  float x2 = distribution(rnd_);
  CHECK_FAIL_RETURN_UNEXPECTED((std::numeric_limits<float_t>::max() - x1) > x2,
                               "multiplication out of bounds, with multipliers: " + std::to_string(x1) + " and " +
                                 std::to_string(x2) +
                                 ", which result in the out of bounds product:" + std::to_string(x1 * x2));
  CHECK_FAIL_RETURN_UNEXPECTED(x1 + x2 != 0.0, "addition of variable(x1 and x2) of Gamma should not be 0.");
  float lam = x1 / (x1 + x2);

  // Calculate random labels
  std::vector<int64_t> rand_indx;
  std::shared_ptr<Tensor> out_labels;

  // Compute labels
  RETURN_IF_NOT_OK(ComputeLabels(input.at(1), &out_labels, &rand_indx, label_shape, lam, images.size()));

  // Compute images
  for (int64_t i = 0; i < images.size(); i++) {
    TensorShape remaining({-1});
    uchar *start_addr_of_index = nullptr;
    std::shared_ptr<Tensor> out;
    RETURN_IF_NOT_OK(input.at(0)->StartAddrOfIndex({rand_indx[i], 0, 0, 0}, &start_addr_of_index, &remaining));
    RETURN_IF_NOT_OK(input.at(0)->CreateFromMemory(
      TensorShape({image_shape[dimension_one], image_shape[dimension_two], image_shape[dimension_three]}),
      input.at(0)->type(), start_addr_of_index, &out));
    std::shared_ptr<CVTensor> rand_image = CVTensor::AsCVTensor(std::move(out));
    if (!rand_image->mat().data) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] MixUpBatch: allocate memory failed.");
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
