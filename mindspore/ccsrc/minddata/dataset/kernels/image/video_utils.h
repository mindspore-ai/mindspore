/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_VIDEO_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_VIDEO_UTILS_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
/// \brief Decode the raw input video bytes. Supported video formats are AVI, H264, H265, MOV, MP4 and WMV.
/// \param input: CVTensor containing the not decoded video 1D bytes.
/// \param output: Decoded visual Tensor and audio Tensor. For visual tensor, the shape is <T,H,W,C>, the type is
///     DE_UINT8. Pixel order is RGB. For audio tensor, the shape is <C, L>.
Status DecodeVideo(const TensorRow &input, TensorRow *output);

/// \brief Read the video, audio, metadata from a video file. It supports AVI, H264, H265, MOV, MP4, WMV files.
/// \param[in] filename The path to the videoe file to be read.
/// \param[out] video_output The video frames of the video file.
/// \param[out] audio_output The audio frames of the video file.
/// \param[out] metadata_output The metadata contains video_fps, audio_fps.
/// \param[in] start_pts The start presentation timestamp of the video.
/// \param[in] end_pts The end presentation timestamp of the video.
/// \param[in] pts_unit The unit for the timestamps, can be one of ["pts", "sec"].
/// \return The status code.
Status ReadVideo(const std::string &filename, std::shared_ptr<Tensor> *video_output,
                 std::shared_ptr<Tensor> *audio_output, std::map<std::string, std::string> *metadata_output,
                 float start_pts, float end_pts, const std::string &pts_unit);

/// \brief Read the timestamps and frame rate of a video file. It supports AVI, H264, H265, MOV, MP4, WMV files.
/// \param[in] filename The path to the video file to be read.
/// \param[out] pts_int64_vector The pts vector of the video file.
/// \param[out] video_fps The video frame rate of the video file.
/// \param[out] time_base The time base for the pts_int64_vector.
/// \param[in] pts_unit The unit for the timestamps, can be one of ["pts", "sec"].
/// \return The status code.
Status ReadVideoTimestamps(const std::string &filename, std::vector<int64_t> *pts_int64_vector, float *video_fps,
                           float *time_base, const std::string &pts_unit);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_VIDEO_UTILS_H_
