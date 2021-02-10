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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_ASCEND_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_ASCEND_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/api/status.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/include/transforms.h"

namespace mindspore {
namespace dataset {

// Transform operations for performing computer vision.
namespace vision {

// Char arrays storing name of corresponding classes (in alphabetical order)
constexpr char kDvppCropJpegOperation[] = "DvppCropJpeg";
constexpr char kDvppDecodeResizeOperation[] = "DvppDecodeResize";
constexpr char kDvppDecodeResizeCropOperation[] = "DvppDecodeResizeCrop";
constexpr char kDvppDecodeJpegOperation[] = "DvppDecodeJpeg";
constexpr char kDvppDecodePngOperation[] = "DvppDecodePng";
constexpr char kDvppResizeJpegOperation[] = "DvppResizeJpeg";

class DvppCropJpegOperation;
class DvppDecodeResizeOperation;
class DvppDecodeResizeCropOperation;
class DvppDecodeJpegOperation;
class DvppDecodePngOperation;
class DvppResizeJpegOperation;

/// \brief Function to create a DvppCropJpeg TensorOperation.
/// \notes Tensor operation to crop JPEG image using the simulation algorithm of Ascend series
///     chip DVPP module. It is recommended to use this algorithm in the following scenarios:
///     When training, the DVPP of the Ascend chip is not used,
///     and the DVPP of the Ascend chip is used during inference,
///     and the accuracy of inference is lower than the accuracy of training;
///     and the input image size should be in range [32*32, 2048*2048].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
/// \param[in] crop vector representing the output size of the final crop image.
/// \param[in] size A vector representing the output size of the intermediate resized image.
///     If size is a single value, the shape will be a square. If size has 2 values, it should be (height, width).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<DvppCropJpegOperation> DvppCropJpeg(std::vector<uint32_t> crop = {256, 256});

/// \brief Function to create a DvppDecodeResizeJpeg TensorOperation.
/// \notes Tensor operation to decode and resize JPEG image using the simulation algorithm of Ascend series
///     chip DVPP module. It is recommended to use this algorithm in the following scenarios:
///     When training, the DVPP of the Ascend chip is not used,
///     and the DVPP of the Ascend chip is used during inference,
///     and the accuracy of inference is lower than the accuracy of training;
///     and the input image size should be in range [32*32, 2048*2048].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
/// \param[in] crop vector representing the output size of the final crop image.
/// \param[in] size A vector representing the output size of the intermediate resized image.
///     If size is a single value, smaller edge of the image will be resized to this value with
///     the same image aspect ratio. If size has 2 values, it should be (height, width).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<DvppDecodeResizeOperation> DvppDecodeResizeJpeg(std::vector<uint32_t> resize = {256, 256});

/// \brief Function to create a DvppDecodeResizeCropJpeg TensorOperation.
/// \notes Tensor operation to decode and resize JPEG image using the simulation algorithm of Ascend series
///     chip DVPP module. It is recommended to use this algorithm in the following scenarios:
///     When training, the DVPP of the Ascend chip is not used,
///     and the DVPP of the Ascend chip is used during inference,
///     and the accuracy of inference is lower than the accuracy of training;
///     and the input image size should be in range [32*32, 2048*2048].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
/// \param[in] crop vector representing the output size of the final crop image.
/// \param[in] Resize vector representing the output size of the intermediate resized image.
///     If size is a single value, smaller edge of the image will be resized to the value with
///     the same image aspect ratio. If size has 2 values, it should be (height, width).
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<DvppDecodeResizeCropOperation> DvppDecodeResizeCropJpeg(std::vector<uint32_t> crop = {224, 224},
                                                                        std::vector<uint32_t> resize = {256, 256});

/// \brief Function to create a DvppDecodeJpeg TensorOperation.
/// \notes Tensor operation to decode JPEG image using the simulation algorithm of Ascend series
///     chip DVPP module. It is recommended to use this algorithm in the following scenarios:
///     When training, the DVPP of the Ascend chip is not used,
///     and the DVPP of the Ascend chip is used during inference,
///     and the accuracy of inference is lower than the accuracy of training;
///     and the input image size should be in range [32*32, 2048*2048].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<DvppDecodeJpegOperation> DvppDecodeJpeg();

/// \brief Function to create a DvppDecodePng TensorOperation.
/// \notes Tensor operation to decode PNG image using the simulation algorithm of Ascend series
///     chip DVPP module. It is recommended to use this algorithm in the following scenarios:
///     When training, the DVPP of the Ascend chip is not used,
///     and the DVPP of the Ascend chip is used during inference,
///     and the accuracy of inference is lower than the accuracy of training;
///     and the input image size should be in range [32*32, 2048*2048].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<DvppDecodePngOperation> DvppDecodePng();

/// \brief Function to create a DvppResizeJpeg TensorOperation.
/// \notes Tensor operation to resize JPEG image using Ascend series chip DVPP module.
///     It is recommended to use this algorithm in the following scenarios:
///     When training, the DVPP of the Ascend chip is not used,
///     and the DVPP of the Ascend chip is used during inference,
///     and the accuracy of inference is lower than the accuracy of training;
///     and the input image size should be in range [32*32, 2048*2048].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
/// \param[in] resize vector represents the shape of image after resize.
/// \return Shared pointer to the current TensorOperation.
std::shared_ptr<DvppResizeJpegOperation> DvppResizeJpeg(std::vector<uint32_t> resize = {256, 256});

class DvppCropJpegOperation : public TensorOperation {
 public:
  explicit DvppCropJpegOperation(const std::vector<uint32_t> &resize);

  ~DvppCropJpegOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDvppCropJpegOperation; }

 private:
  std::vector<uint32_t> crop_;
};

class DvppDecodeResizeOperation : public TensorOperation {
 public:
  explicit DvppDecodeResizeOperation(const std::vector<uint32_t> &resize);

  ~DvppDecodeResizeOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDvppDecodeResizeOperation; }

 private:
  std::vector<uint32_t> resize_;
};

class DvppDecodeResizeCropOperation : public TensorOperation {
 public:
  explicit DvppDecodeResizeCropOperation(const std::vector<uint32_t> &crop, const std::vector<uint32_t> &resize);

  ~DvppDecodeResizeCropOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDvppDecodeResizeCropOperation; }

 private:
  std::vector<uint32_t> crop_;
  std::vector<uint32_t> resize_;
};

class DvppDecodeJpegOperation : public TensorOperation {
 public:
  ~DvppDecodeJpegOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDvppDecodeJpegOperation; }
};

class DvppDecodePngOperation : public TensorOperation {
 public:
  ~DvppDecodePngOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDvppDecodePngOperation; }
};

class DvppResizeJpegOperation : public TensorOperation {
 public:
  explicit DvppResizeJpegOperation(const std::vector<uint32_t> &resize);

  ~DvppResizeJpegOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDvppResizeJpegOperation; }

 private:
  std::vector<uint32_t> resize_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_VISION_ASCEND_H_
