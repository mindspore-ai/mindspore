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
#include "minddata/dataset/core/cv_tensor.h"

#include <memory>
#include <vector>

#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/tensor.h"

namespace mindspore {
namespace dataset {

CVTensor::CVTensor(std::shared_ptr<Tensor> tensor) : Tensor(std::move(*tensor)) {
  (void)this->MatInit(GetMutableBuffer(), shape_, type_, &mat_);
}

Status CVTensor::CreateEmpty(const TensorShape &shape, DataType type, CVTensorPtr *out) {
  const CVTensorAlloc *alloc = GlobalContext::Instance()->cv_tensor_allocator();
  *out = std::allocate_shared<CVTensor>(*alloc, shape, type);
  int64_t byte_size = (*out)->SizeInBytes();
  // Don't allocate if we have a tensor with no elements.
  if (byte_size != 0) {
    RETURN_IF_NOT_OK((*out)->AllocateBuffer(byte_size));
  }

  return (*out)->MatInit((*out)->GetMutableBuffer(), (*out)->shape_, (*out)->type_, &(*out)->mat_);
}

Status CVTensor::CreateFromMat(const cv::Mat &mat, CVTensorPtr *out) {
  TensorPtr out_tensor;
  cv::Mat mat_local = mat;
  // if the input Mat's memory is not continuous, copy it to one block of memory
  if (!mat.isContinuous()) mat_local = mat.clone();
  TensorShape shape(mat.size, mat_local.type());
  DataType type = DataType::FromCVType(mat_local.type());
  RETURN_IF_NOT_OK(CreateFromMemory(shape, type, mat_local.data, &out_tensor));
  *out = AsCVTensor(out_tensor);
  return Status::OK();
}

std::pair<std::array<int, 2>, int> CVTensor::IsValidImage(const TensorShape &shape, const DataType &type) {
  std::array<int, 2> size = {1, 1};
  if (shape.Rank() <= 2 || (shape.Rank() == 3 && shape[2] <= CV_CN_MAX)) {
    uint8_t ch = 1;
    if (shape.Rank() == 3) {
      ch = static_cast<uint8_t>(shape[2]);
    }
    if (shape.Rank() > 0) size[0] = static_cast<int>(shape[0]);
    if (shape.Rank() > 1) size[1] = static_cast<int>(shape[1]);
    if (type.AsCVType() == kCVInvalidType) return std::make_pair(size, -1);

    int cv_type = CV_MAKETYPE(type.AsCVType(), ch);
    return std::make_pair(size, cv_type);
  }
  return std::make_pair(size, -1);
}

std::shared_ptr<CVTensor> CVTensor::AsCVTensor(std::shared_ptr<Tensor> t) {
  std::shared_ptr<CVTensor> cv_t = std::dynamic_pointer_cast<CVTensor>(t);
  if (cv_t != nullptr) {
    return cv_t;
  } else {
    const CVTensorAlloc *alloc = GlobalContext::Instance()->cv_tensor_allocator();
    return std::allocate_shared<CVTensor>(*alloc, t);
  }
}

Status CVTensor::MatInit(uchar *data, const TensorShape &shape, const DataType &type, cv::Mat *mat) {
  std::pair<std::array<int, 2>, int> cv_shape_type = IsValidImage(shape, type);
  if (cv_shape_type.second == -1) {
    std::vector<dsize_t> sizes = shape.AsVector();
    std::vector<int> sizes32(sizes.begin(), sizes.end());  // convert long to int for usage with OpenCV
    if (static_cast<int>(shape.Rank()) != shape.Rank()) {
      RETURN_STATUS_UNEXPECTED("Error in creating CV mat. Wrong shape.");
    }

    uint8_t cv_type = type.AsCVType();
    if (cv_type == kCVInvalidType) {
      RETURN_STATUS_UNEXPECTED("Error in creating CV mat. Invalid type.");
    }
    *mat = cv::Mat(static_cast<int>(shape.Rank()), &sizes32[0], cv_type, data);
  } else {
    *mat = cv::Mat(2, &(cv_shape_type.first[0]), cv_shape_type.second, data);
  }
  return Status::OK();
}

Status CVTensor::Reshape(const TensorShape &shape) {
  RETURN_IF_NOT_OK(Tensor::Reshape(shape));
  RETURN_IF_NOT_OK(this->MatInit(GetMutableBuffer(), shape_, type_, &mat_));
  return Status::OK();
}

Status CVTensor::ExpandDim(const dsize_t &axis) {
  RETURN_IF_NOT_OK(Tensor::ExpandDim(axis));
  RETURN_IF_NOT_OK(this->MatInit(GetMutableBuffer(), shape_, type_, &mat_));
  return Status::OK();
}

void CVTensor::Squeeze() {
  Tensor::Squeeze();
  (void)this->MatInit(GetMutableBuffer(), shape_, type_, &mat_);
}

Status CVTensor::MatAtIndex(const std::vector<dsize_t> &index, cv::Mat *mat) {
  uchar *start = nullptr;
  TensorShape remaining({-1});
  RETURN_IF_NOT_OK(this->StartAddrOfIndex(index, &start, &remaining));
  RETURN_IF_NOT_OK(this->MatInit(start, remaining, type_, mat));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
