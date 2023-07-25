/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_ASCEND_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_ASCEND_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"

namespace mindspore {
namespace dataset {
// Transform operations for performing computer vision.
namespace vision {
/* ##################################### API class ###########################################*/

/// \brief Decode and resize JPEG image using the hardware algorithm of
///     Ascend series chip DVPP module.
class DATASET_API DvppDecodeResizeJpeg final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] resize Parameter vector of two integers for each dimension, with respect to H,W order.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto dvpp_op = vision::DvppDecodeResizeJpeg({255, 255});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({dvpp_op},   // operations
  ///                            {"image"});  // input columns
  /// \endcode
  explicit DvppDecodeResizeJpeg(const std::vector<uint32_t> &resize);

  /// \brief Destructor.
  ~DvppDecodeResizeJpeg() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Decode, resize and crop JPEG image using the hardware algorithm of
///     Ascend series chip DVPP module.
class DATASET_API DvppDecodeResizeCropJpeg final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] crop Parameter vector of two integers for each dimension after final crop, with respect to H,W order.
  /// \param[in] resize Parameter vector of two integers for each dimension after resize, with respect to H,W order.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto dvpp_op = vision::DvppDecodeResizeCropJpeg({50, 50}, {100, 100});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({dvpp_op},   // operations
  ///                            {"image"});  // input columns
  /// \endcode
  DvppDecodeResizeCropJpeg(const std::vector<uint32_t> &crop, const std::vector<uint32_t> &resize);

  /// \brief Destructor.
  ~DvppDecodeResizeCropJpeg() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Decode H264/H265 video using the hardware algorithm of
///     DVPP module on Ascend series chip.
class DATASET_API DvppDecodeVideo final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size Parameter vector of two integers for each dimension of input video frames, with respect to H,W
  ///     order.
  /// \param[in] type An enum for the coding protocol of video.
  ///   - VdecStreamFormat::kH265MainLevel, video coding protocol is H265-main level.
  ///   - VdecStreamFormat::kH264BaselineLevel, video coding protocol is H264-baseline level.
  ///   - VdecStreamFormat::kH264MainLevel, video coding protocol is H264-main level.
  ///   - VdecStreamFormat::kH264HighLevel, video coding protocol is H264-high level.
  /// \param[in] out_format An enum for the format of output image (default=VdecOutputFormat::kYUV_SEMIPLANAR_420).
  ///   - VdecOutputFormat::kYUV_SEMIPLANAR_420, format of output image is YUV420SP NV12 8bit.
  ///   - VdecOutputFormat::kYVU_SEMIPLANAR_420, format of output image is YUV420SP NV21 8bit.
  /// \param[in] output The output path of the decoded images corresponds to video frames.
  /// \par Example
  /// \code
  ///     namespace ds = mindspore::dataset;
  ///
  ///     /* Define operations */
  ///     std::shared_ptr<ds::TensorTransform> dvpp_decode(new ds::vision::DvppDecodeVideo({1080, 1920},
  ///                                                      ds::VdecStreamFormat::kH265MainLevel));
  ///
  ///     /* define preprocessor */
  ///     ds::Execute preprocessor({dvpp_decode}, ds::MapTargetDevice::kCpu, 0);
  ///
  /// \endcode

  DvppDecodeVideo(const std::vector<uint32_t> &size, VdecStreamFormat type,
                  VdecOutputFormat out_format = VdecOutputFormat::kYuvSemiplanar420,
                  const std::string &output = "./output")
      : DvppDecodeVideo(size, type, out_format, StringToChar(output)) {}

  /// \brief Constructor.
  /// \param[in] size Parameter vector of two integers for each dimension of input video frames, with respect to H,W
  ///     order.
  /// \param[in] type An enum for the coding protocol of video.
  ///   - VdecStreamFormat::kH265MainLevel, video coding protocol is H265-main level.
  ///   - VdecStreamFormat::kH264BaselineLevel, video coding protocol is H264-baseline level.
  ///   - VdecStreamFormat::kH264MainLevel, video coding protocol is H264-main level.
  ///   - VdecStreamFormat::kH264HighLevel, video coding protocol is H264-high level.
  /// \param[in] output The output path of the decoded images corresponds to video frames.
  /// \par Example
  /// \code
  ///     namespace ds = mindspore::dataset;
  ///
  ///     /* Define operations */
  ///     std::shared_ptr<ds::TensorTransform> dvpp_decode(new ds::vision::DvppDecodeVideo({1080, 1920}));
  ///
  ///     /* define preprocessor */
  ///     ds::Execute preprocessor({dvpp_decode}, ds::MapTargetDevice::kCpu, 0);
  ///
  /// \endcode

  DvppDecodeVideo(const std::vector<uint32_t> &size, VdecStreamFormat type, const std::string &output = "./output")
      : DvppDecodeVideo(size, type, VdecOutputFormat::kYuvSemiplanar420, StringToChar(output)) {}

  DvppDecodeVideo(const std::vector<uint32_t> &size, VdecStreamFormat type, VdecOutputFormat out_format,
                  const std::vector<char> &output);

  /// \brief Destructor.
  ~DvppDecodeVideo() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Decode PNG image using the hardware algorithm of
///     Ascend series chip DVPP module.
class DATASET_API DvppDecodePng final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto dvpp_op = vision::DvppDecodePng();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({dvpp_op},   // operations
  ///                            {"image"});  // input columns
  /// \endcode
  DvppDecodePng();

  /// \brief Destructor.
  ~DvppDecodePng() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

  std::shared_ptr<TensorOperation> Parse(const MapTargetDevice &env) override;
};
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_ASCEND_H_
