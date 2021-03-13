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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CV_TENSOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CV_TENSOR_H_

#include <memory>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "./securec.h"

#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"

namespace mindspore {
namespace dataset {
using CVTensorPtr = std::shared_ptr<CVTensor>;
class CVTensor : public Tensor {
 public:
  // Inherit Tensor's constructors
  using Tensor::Tensor;

  /// Create a CVTensor from a given tensor. This constructor should not be used directly, use Create* instead.
  /// The input tensor will be invalidated (i.e., the shape and type will be
  /// set to unknown and the data buffer will point to null.
  /// \note there is no memory copying here, the buffer will be assigned to the constructed tensor.
  /// \param tensor
  explicit CVTensor(std::shared_ptr<Tensor> tensor);

  /// Create CV tensor with type and shape. Items of the tensor would be uninitialized.
  /// \param shape [in] shape of the output tensor
  /// \param type [in] type of the output tensor
  /// \param out [out] Generated tensor
  /// \return Status code
  static Status CreateEmpty(const TensorShape &shape, DataType type, CVTensorPtr *out);

  /// Create CV tensor from cv::Mat
  /// \note This constructor allocates a new space in the memory and copies the CV::Mat buffer into it.
  /// \param mat [in] cv::Mat to be copied into the new tensor.
  /// \param out [out] Generated tensor
  /// \return Status code
  static Status CreateFromMat(const cv::Mat &mat, CVTensorPtr *out);

  ~CVTensor() override = default;

  /// Static function to cast a given Tensor as CVTensor. If the input tensor is already of type CVTensor,
  /// this function would be treated as a no-op. Fot other tensor types, a new CVTensor is created based on the data
  /// provided. The Passed Tensor will be invalidated.
  /// \note the input tensor will be invalidated.
  /// \note there is no memory copying here, the buffer will be assigned to the constructed tensor.
  /// \param tensor [in]
  /// \return CVTensor
  static std::shared_ptr<CVTensor> AsCVTensor(std::shared_ptr<Tensor> tensor);

  /// Get a reference to the CV::Mat
  /// \return a reference to the internal CV::Mat
  cv::Mat &mat() { return mat_; }

  /// Get a copy of the CV::Mat
  /// \return a copy of internal CV::Mat
  cv::Mat matCopy() const { return mat_.clone(); }

  /// Static function to check if the passed information (shape and type) can be treated as a valid description
  /// of an image in OpenCV. Moreover, it returns OpenCV shape and type
  /// For example, if the shape is <512,512,3> and type is DE_UINT8, the output would be [512,512] and CV_8UC3.
  /// In case of invalid shape or type, the function will return pair<null,0>
  /// \param shape [in] TensorShape
  /// \param type [in] DataType
  /// \return std::pair of OpenCV shape and type
  static std::pair<std::array<int, 2>, int> IsValidImage(const TensorShape &shape, const DataType &type);

  Status Reshape(const TensorShape &shape) override;

  Status ExpandDim(const dsize_t &axis) override;

  void Squeeze() override;

  Status MatAtIndex(const std::vector<dsize_t> &index, cv::Mat *mat);

 private:
  /// Opencv Mat object wrapping the raw data of the tensor.
  /// Modifying the content of the matrix, modifies the tensor.
  cv::Mat mat_;

  /// Create cv::Mat from data, TensorShape and DataType
  /// \param data [in] Pointer to the data in memory.
  /// \param shape [in] Shape of the tensor.
  /// \param type [in] Type of the tensor.
  /// \param mat [out] cv::Mat initialized with the provided data.
  /// \return Status code
  Status MatInit(uchar *data, const TensorShape &shape, const DataType &type, cv::Mat *mat);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CV_TENSOR_H_
