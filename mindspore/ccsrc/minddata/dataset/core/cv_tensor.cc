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

#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/core/tensor.h"

namespace mindspore {
namespace dataset {

CVTensor::CVTensor(std::shared_ptr<Tensor> tensor) : Tensor(std::move(*tensor)) {
  (void)this->MatInit(GetMutableBuffer(), shape_, type_, &mat_);
}

Status CVTensor::CreateEmpty(const TensorShape &shape, DataType type, CVTensorPtr *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  const CVTensorAlloc *alloc = GlobalContext::Instance()->cv_tensor_allocator();
  *out = std::allocate_shared<CVTensor>(*alloc, shape, type);
  RETURN_UNEXPECTED_IF_NULL(out);
  int64_t byte_size = (*out)->SizeInBytes();
  // Don't allocate if we have a tensor with no elements.
  if (byte_size != 0) {
    RETURN_IF_NOT_OK((*out)->AllocateBuffer(byte_size));
  }

  return (*out)->MatInit((*out)->GetMutableBuffer(), (*out)->shape_, (*out)->type_, &(*out)->mat_);
}

Status CVTensor::CreateFromMat(const cv::Mat &mat, const dsize_t rank, CVTensorPtr *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  TensorPtr out_tensor;
  cv::Mat mat_local = mat;
  // if the input Mat's memory is not continuous, copy it to one block of memory
  if (!mat.isContinuous()) {
    mat_local = mat.clone();
  }
  TensorShape shape({});
  if (mat.dims == 2 && rank == 2) {
    shape = TensorShape({mat.rows, mat.cols});
  } else if (mat.dims == 2 && rank == 3) {
    shape = TensorShape({mat.rows, mat.cols, mat.channels()});
  } else {
    // the info of <C, H, W> tensor is: dims = 3, size = (C, H, W), channels = 1
    RETURN_STATUS_UNEXPECTED("CreateFromMat: tensor should be in shape of <H,W,C> or <H,W>.");
  }
  DataType type = DataType::FromCVType(mat_local.type());
  RETURN_IF_NOT_OK(CreateFromMemory(shape, type, mat_local.data, &out_tensor));
  *out = AsCVTensor(out_tensor);
  return Status::OK();
}

std::pair<std::array<int, 2>, int> CVTensor::IsValidImage(const TensorShape &shape, const DataType &type) {
  constexpr int64_t array_size = 2;
  constexpr int64_t rank_two = 2;
  constexpr int64_t rank_three = 3;
  constexpr int64_t index = 2;
  std::array<int, array_size> size = {1, 1};
  if (shape.Rank() <= rank_two || (shape.Rank() == rank_three && shape[index] <= CV_CN_MAX)) {
    uint16_t ch = 1;
    if (shape.Rank() == rank_three) {
      ch = static_cast<uint16_t>(shape[2]);
    }
    if (shape.Rank() > 0) {
      size[0] = static_cast<int>(shape[0]);
    }
    if (shape.Rank() > 1) {
      size[1] = static_cast<int>(shape[1]);
    }
    if (type.AsCVType() == kCVInvalidType) {
      return std::make_pair(size, -1);
    }
    int cv_type = CV_MAKETYPE(type.AsCVType(), ch);
    return std::make_pair(size, cv_type);
  }
  return std::make_pair(size, -1);
}

std::shared_ptr<CVTensor> CVTensor::AsCVTensor(std::shared_ptr<Tensor> t) {
  if (t == nullptr) {
    return nullptr;
  }
  std::shared_ptr<CVTensor> cv_t = std::dynamic_pointer_cast<CVTensor>(t);
  if (cv_t != nullptr) {
    return cv_t;
  } else {
    const CVTensorAlloc *alloc = GlobalContext::Instance()->cv_tensor_allocator();
    return std::allocate_shared<CVTensor>(*alloc, t);
  }
}

Status CVTensor::MatInit(uchar *data, const TensorShape &shape, const DataType &type, cv::Mat *mat) {
  RETURN_UNEXPECTED_IF_NULL(data);
  RETURN_UNEXPECTED_IF_NULL(mat);
  const int kShapeAsDefault = 2;
  std::pair<std::array<int, kShapeAsDefault>, int> cv_shape_type = IsValidImage(shape, type);
  if (cv_shape_type.second == -1) {
    std::vector<dsize_t> sizes = shape.AsVector();
    std::vector<int> sizes32(sizes.begin(), sizes.end());  // convert long to int for usage with OpenCV

    uint8_t cv_type = type.AsCVType();
    if (cv_type == kCVInvalidType) {
      RETURN_STATUS_UNEXPECTED("Error in creating CV mat. Invalid type.");
    }
    *mat = cv::Mat(static_cast<int>(shape.Rank()), &sizes32[0], cv_type, data);
  } else {
    *mat = cv::Mat(kShapeAsDefault, &(cv_shape_type.first[0]), cv_shape_type.second, data);
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
  Status rc = this->MatInit(GetMutableBuffer(), shape_, type_, &mat_);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Squeeze failed, error details is " << rc;
  }
}

Status CVTensor::MatAtIndex(const std::vector<dsize_t> &index, cv::Mat *mat) {
  RETURN_UNEXPECTED_IF_NULL(mat);
  uchar *start = nullptr;
  TensorShape remaining({-1});
  RETURN_IF_NOT_OK(this->StartAddrOfIndex(index, &start, &remaining));
  RETURN_IF_NOT_OK(this->MatInit(start, remaining, type_, mat));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
