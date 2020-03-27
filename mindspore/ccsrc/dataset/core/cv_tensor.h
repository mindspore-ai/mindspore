/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_CORE_CV_TENSOR_H_
#define DATASET_CORE_CV_TENSOR_H_

#include <memory>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "./securec.h"

#include "dataset/core/constants.h"
#include "dataset/core/data_type.h"
#include "dataset/core/tensor.h"

namespace mindspore {
namespace dataset {
class CVTensor : public Tensor {
 public:
  // Create an empty CVTensor of shape `shape` and type `type`.
  // @note The shape and type information should be known and valid.
  // @param shape TensorShape
  // @param type DataType
  CVTensor(const TensorShape &shape, const DataType &type);

  // Create a CVTensor from a given buffer, shape and type.
  // @note This constructor allocates a new space in the memory and copies the buffer into it.
  // @note The buffer should be valid and the shape and type information should be known and valid.
  // @param shape TensorShape
  // @param type DataType
  // @param data unsigned char*, pointer to the data.
  CVTensor(const TensorShape &shape, const DataType &type, const uchar *data);

  // Create a CVTensor from a given CV::Mat.
  // @note This constructor allocates a new space in the memory and copies the CV::Mat buffer into it.
  // @param mat CV::Mat
  explicit CVTensor(const cv::Mat &mat)
      : CVTensor(TensorShape(mat.size, mat.type()), DataType::FromCVType(mat.type()), mat.data) {}

  ~CVTensor() = default;

  // Static function to cast a given Tensor as CVTensor. If the input tensor is already of type CVTensor,
  // this function would be treated as a no-op. Fot other tensor types, a new CVTensor is created based on the data
  // provided. The Passed Tensor will be invalidated.
  // @note there is no memory copying here, the buffer will be assigned to the constructed tensor.
  // @param tensor
  // @return CVTensor
  static std::shared_ptr<CVTensor> AsCVTensor(std::shared_ptr<Tensor> tensor);

  // Create a CVTensor from a given tensor. The input tensor will be invalidated (i.e., the shape and type will be
  // set to unknown and the data buffer will point to null.
  // @note there is no memory copying here, the buffer will be assigned to the constructed tensor.
  // @param tensor
  explicit CVTensor(std::shared_ptr<Tensor> tensor);

  // Getter function for the CV::Mat
  // @return
  cv::Mat mat() const { return mat_; }

  // Static function to check if the passed information (shape and type) can be treated as a valid description
  // of an image in OpenCV. Moreover, it returns OpenCV shape and type
  // For example, if the shape is <512,512,3> and type is DE_UINT8, the output would be [512,512] and CV_8UC3.
  // In case of invalid shape or type, the function will return pair<null,0>
  // @param shape TensorShape
  // @param type DataType
  // @return std::pair of OpenCV shape and type
  std::pair<std::array<int, 2>, int> IsValidImage(const TensorShape &shape, const DataType &type);

  Status Reshape(const TensorShape &shape) override;

  Status ExpandDim(const dsize_t &axis) override;

  void Squeeze() override;

  Status Mat(const std::vector<dsize_t> &index, cv::Mat *mat) {
    uchar *start = nullptr;
    TensorShape remaining({-1});
    RETURN_IF_NOT_OK(this->StartAddrOfIndex(index, &start, &remaining));
    RETURN_IF_NOT_OK(this->MatInit(start, remaining, type_, mat));
    return Status::OK();
  }

 private:
  cv::Mat mat_;

  // Initialize CV::Mat with the data_, shape_ and type_
  Status MatInit(uchar *data, const TensorShape &shape, const DataType &type, cv::Mat *mat);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_CORE_CV_TENSOR_H_
