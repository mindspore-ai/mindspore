/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "utils/dual_abi_helper.h"
#include "utils/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision_lite.h"
#include "include/dataset/ms_tensor.h"
#include "minddata/dataset/util/visible.h"

namespace mindspore::dataset {
class TensorOperation;

// Transform operations for performing computer vision.
namespace vision {

/// \brief AdjustGamma TensorTransform.
/// \note Apply gamma correction on input image.
class DATASET_API AdjustGamma final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] gamma Non negative real number, which makes the output image pixel value
  ///     exponential in relation to the input image pixel value.
  /// \param[in] gain The constant multiplier. Default: 1.0.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto adjust_gamma_op = vision::AdjustGamma(10.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, adjust_gamma_op},  // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit AdjustGamma(float gamma, float gain = 1.0);

  /// \brief Destructor.
  ~AdjustGamma() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply AutoAugment data augmentation method.
class DATASET_API AutoAugment final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] policy An enum for the data auto augmentation policy (default=AutoAugmentPolicy::kImageNet).
  ///     - AutoAugmentPolicy::kImageNet, AutoAugment policy learned on the ImageNet dataset.
  ///     - AutoAugmentPolicy::kCifar10, AutoAugment policy learned on the Cifar10 dataset.
  ///     - AutoAugmentPolicy::kSVHN, AutoAugment policy learned on the SVHN dataset.
  /// \param[in] interpolation An enum for the mode of interpolation (default=InterpolationMode::kNearestNeighbour).
  ///     - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///     - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///     - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///     - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders (default={0, 0, 0}).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto auto_augment_op = vision::AutoAugment(AutoAugmentPolicy::kImageNet,
  ///                                                InterpolationMode::kNearestNeighbour, {0, 0, 0});
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, auto_augment_op}, // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit AutoAugment(AutoAugmentPolicy policy = AutoAugmentPolicy::kImageNet,
                       InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                       const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~AutoAugment() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief BoundingBoxAugment TensorTransform.
/// \note  Apply a given image transform on a random selection of bounding box regions of a given image.
class DATASET_API BoundingBoxAugment final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transform Raw pointer to the TensorTransform operation.
  /// \param[in] ratio Ratio of bounding boxes to apply augmentation on. Range: [0, 1] (default=0.3).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     TensorTransform *rotate_op = new vision::RandomRotation({-180, 180});
  ///     auto bbox_aug_op = vision::BoundingBoxAugment(rotate_op, 0.5);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({bbox_aug_op},       // operations
  ///                            {"image", "bbox"});  // input columns
  /// \endcode
  explicit BoundingBoxAugment(TensorTransform *transform, float ratio = 0.3);

  /// \brief Constructor.
  /// \param[in] transform Smart pointer to the TensorTransform operation.
  /// \param[in] ratio Ratio of bounding boxes where augmentation is applied to. Range: [0, 1] (default=0.3).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     std::shared_ptr<TensorTransform> flip_op = std::make_shared<vision::RandomHorizontalFlip>(0.5);
  ///     std::shared_ptr<TensorTransform> bbox_aug_op = std::make_shared<vision::BoundingBoxAugment>(flip_op, 0.1);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({bbox_aug_op},       // operations
  ///                            {"image", "bbox"});  // input columns
  /// \endcode
  explicit BoundingBoxAugment(const std::shared_ptr<TensorTransform> &transform, float ratio = 0.3);

  /// \brief Constructor.
  /// \param[in] transform Object pointer to the TensorTransform operation.
  /// \param[in] ratio Ratio of bounding boxes where augmentation is applied to. Range: [0, 1] (default=0.3).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     vision::RandomColor random_color_op = vision::RandomColor(0.5, 1.0);
  ///     vision::BoundingBoxAugment bbox_aug_op = vision::BoundingBoxAugment(random_color_op, 0.8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({bbox_aug_op},       // operations
  ///                            {"image", "bbox"});  // input columns
  /// \endcode
  explicit BoundingBoxAugment(const std::reference_wrapper<TensorTransform> &transform, float ratio = 0.3);

  /// \brief Destructor.
  ~BoundingBoxAugment() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Mask a random section of each image with the corresponding part of another randomly
///     selected image in that batch.
class DATASET_API CutMixBatch final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] image_batch_format The format of the batch.
  /// \param[in] alpha The hyperparameter of beta distribution (default = 1.0).
  /// \param[in] prob The probability by which CutMix is applied to each image (default = 1.0).
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Batch(5);
  ///     dataset = dataset->Map({std::make_shared<vision::CutMixBatch>(ImageBatchFormat::kNHWC)}, // operations
  ///                            {"image", "label"});                                             // input columns
  /// \endcode
  explicit CutMixBatch(ImageBatchFormat image_batch_format, float alpha = 1.0, float prob = 1.0);

  /// \brief Destructor.
  ~CutMixBatch() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

#ifdef ENABLE_FFMPEG
/// \brief Decode the input video.
class DATASET_API DecodeVideo final : public TensorTransform {
 public:
  /// \brief Constructor. It will decode a vector containing a raw video tensor into a vector containing two tensors.
  ///     The raw video tensor in the input vector should be 1D array of UINT8.
  ///     The first tensor in the output vector is a visual tensor, the shape is <T,H,W,C>, the type is DE_UINT8. Pixel
  ///     order is RGB. The second tensor in the output vector is an audio tensor, the shape is <C, L>.
  /// \par Example
  /// \code
  ///     /* Read video file into tensor */
  ///     mindspore::MSTensor video;
  ///     ASSERT_OK(mindspore::dataset::vision::ReadFile("/path/to/video/file", &video));
  ///     std::vector<mindspore::MSTensor> input_tensor;
  ///     std::vector<mindspore::MSTensor> output_tensor;
  ///     input_tensor.push_back(video);
  ///     auto decode_video = vision::DecodeVideo();
  ///     auto transform = Execute(decode_video);
  ///     Status rc = transform(input_tensor, &output_tensor);
  /// \endcode
  DecodeVideo();

  /// \brief Destructor.
  ~DecodeVideo() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};
#endif

/// \brief Encode the image as JPEG data.
/// \param[in] image The image to be encoded.
/// \param[out] output The Tensor data.
/// \param[in] quality The quality for the output tensor, in range of [1, 100]. Default: 75.
/// \return The status code.
Status DATASET_API EncodeJpeg(const mindspore::MSTensor &image, mindspore::MSTensor *output, int quality = 75);

/// \brief Encode the image as PNG data.
/// \param[in] image The image to be encoded.
/// \param[out] output The Tensor data.
/// \param[in] compression_level The compression_level for encoding, in range of [0, 9]. Default: 6.
/// \return The status code.
Status DATASET_API EncodePng(const mindspore::MSTensor &image, mindspore::MSTensor *output, int compression_level = 6);

/// \brief Erase the input image with given value.
class DATASET_API Erase final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] top Vertical ordinate of the upper left corner of erased region.
  /// \param[in] left Horizontal ordinate of the upper left corner of erased region.
  /// \param[in] height Height of erased region.
  /// \param[in] width Width of erased region.
  /// \param[in] value Pixel value used to pad the erased area.
  ///     If a single integer is provided, it will be used for all RGB channels.
  ///     If a sequence of length 3 is provided, it will be used for R, G, B channels respectively. Default: 0.
  /// \param[in] inplace Whether to erase inplace. Default: False.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::Erase>(10, 10, 10, 10)}, // operations
  ///                            {"image"});                                        // input columns
  /// \endcode
  Erase(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<float> &value = {0., 0., 0.},
        bool inplace = false);

  /// \brief Destructor.
  ~Erase() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Get the number of input image channels.
/// \param[in] image Tensor of the image.
/// \param[out] channels Channels of the image.
/// \return The status code.
Status DATASET_API GetImageNumChannels(const mindspore::MSTensor &image, dsize_t *channels);

/// \brief Get the size of input image.
/// \param[in] image Tensor of the image.
/// \param[out] size Size of the image as [height, width].
/// \return The status code.
Status DATASET_API GetImageSize(const mindspore::MSTensor &image, std::vector<dsize_t> *size);

/// \brief Apply MixUp transformation on an input batch of images and labels. The labels must be in
///     one-hot format and Batch must be called before calling this function.
class DATASET_API MixUpBatch final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] alpha hyperparameter of beta distribution (default = 1.0).
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Batch(5);
  ///     dataset = dataset->Map({std::make_shared<vision::MixUpBatch>()}, // operations
  ///                            {"image"});                               // input columns
  /// \endcode
  explicit MixUpBatch(float alpha = 1.0);

  /// \brief Destructor.
  ~MixUpBatch() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Normalize the input image with respect to mean and standard deviation and pads an extra
///     channel with value zero.
class DATASET_API NormalizePad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] mean A vector of mean values for each channel, with respect to channel order.
  ///     The mean values must be in range [0.0, 255.0].
  /// \param[in] std A vector of standard deviations for each channel, with respect to channel order.
  ///     The standard deviation values must be in range (0.0, 255.0].
  /// \param[in] dtype The output datatype of Tensor.
  ///     The standard deviation values must be "float32" or "float16"（default = "float32"）.
  /// \param[in] is_hwc A boolean to indicate whether the input image is in HWC format (true) or CHW
  ///     format (false) (default = true).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto normalize_pad_op = vision::NormalizePad({121.0, 115.0, 100.0}, {70.0, 68.0, 71.0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, normalize_pad_op},  // operations
  ///                            {"image"});                     // input columns
  /// \endcode
  NormalizePad(const std::vector<float> &mean, const std::vector<float> &std, const std::string &dtype = "float32",
               bool is_hwc = true)
      : NormalizePad(mean, std, StringToChar(dtype), is_hwc) {}

  NormalizePad(const std::vector<float> &mean, const std::vector<float> &std, const std::vector<char> &dtype,
               bool is_hwc = true);

  /// \brief Destructor.
  ~NormalizePad() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Perform perspective transform on the image.
class DATASET_API Perspective final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] start_points List containing four lists of two integers corresponding to four
  ///     corners [top-left, top-right, bottom-right, bottom-left] of the original image.
  /// \param[in] end_points List containing four lists of two integers corresponding to four
  ///     corners [top-left, top-right, bottom-right, bottom-left] of the transformed image.
  /// \param[in] interpolation An enum for the mode of interpolation. Default: InterpolationMode::kLinear.
  ///     - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///     - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///     - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///     - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///     - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     std::vector<std::vector<int32_t>> start_points = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  ///     std::vector<std::vector<int32_t>> end_points = {{0, 2}, {2, 0}, {2, 2}, {0, 2}};
  ///     auto perspective_op = vision::Perspective(start_points, end_points, InterpolationMode::kLinear);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, perspective_op},  // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  Perspective(const std::vector<std::vector<int32_t>> &start_points,
              const std::vector<std::vector<int32_t>> &end_points, InterpolationMode interpolation);

  /// \brief Destructor.
  ~Perspective() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Posterize an image by reducing the number of bits for each color channel.
class DATASET_API Posterize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] bits The number of bits to keep for each channel,
  ///     should be in range of [0, 8].
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto posterize_op = vision::Posterize(8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, posterize_op},  // operations
  ///                            {"image"});                 // input columns
  /// \endcode
  explicit Posterize(uint8_t bits);

  /// \brief Destructor.
  ~Posterize() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply RandAugment data augmentation method.
class DATASET_API RandAugment final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] num_ops Number of augmentation transformations to apply sequentially. Default: 2.
  /// \param[in] magnitude Magnitude for all the transformations. Default: 9.
  /// \param[in] num_magnitude_bins The number of different magnitude values. Default: 31.
  /// \param[in] interpolation An enum for the mode of interpolation. Default: InterpolationMode::kNearestNeighbour.
  ///     - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///     - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///     - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders. Default: {0, 0, 0}.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rand_augment_op = vision::RandAugment();
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rand_augment_op}, // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit RandAugment(int32_t num_ops = 2, int32_t magnitude = 9, int32_t num_magnitude_bins = 31,
                       InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                       const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~RandAugment() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly adjust the brightness, contrast, saturation, and hue of the input image.
class DATASET_API RandomColorAdjust final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] brightness Brightness adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values it needs to be in the form of [min, max] (Default={1, 1}).
  /// \param[in] contrast Contrast adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values, it needs to be in the form of [min, max] (Default={1, 1}).
  /// \param[in] saturation Saturation adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values, it needs to be in the form of [min, max] (Default={1, 1}).
  /// \param[in] hue Hue adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values, it must be in the form of [min, max] where -0.5 <= min <= max <= 0.5
  ///     (Default={0, 0}).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_color_adjust_op = vision::RandomColorAdjust({1.0, 5.0}, {10.0, 20.0}, {40.0, 40.0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_color_adjust_op},  // operations
  ///                            {"image"});                           // input columns
  /// \endcode
  explicit RandomColorAdjust(const std::vector<float> &brightness = {1.0, 1.0},
                             const std::vector<float> &contrast = {1.0, 1.0},
                             const std::vector<float> &saturation = {1.0, 1.0},
                             const std::vector<float> &hue = {0.0, 0.0});

  /// \brief Destructor.
  ~RandomColorAdjust() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image at a random location.
class DATASET_API RandomCrop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] padding A vector representing the number of pixels to pad the image.
  ///    If the vector has one value, it pads all sides of the image with that value.
  ///    If the vector has two values, it pads left and right with the first and
  ///    top and bottom with the second value.
  ///    If the vector has four values, it pads left, top, right, and bottom with
  ///    those values respectively.
  /// \param[in] pad_if_needed A boolean indicating that whether to pad the image
  ///    if either side is smaller than the given output size.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders if the padding_mode is
  ///     BorderType.kConstant. If 1 value is provided, it is used for all RGB channels.
  ///     If 3 values are provided, it is used to fill R, G, B channels respectively.
  /// \param[in] padding_mode The method of padding (default=BorderType::kConstant).It can be any of
  ///     [BorderType::kConstant, BorderType::kEdge, BorderType::kReflect, BorderType::kSymmetric].
  ///   - BorderType::kConstant, Fill the border with constant values.
  ///   - BorderType::kEdge, Fill the border with the last value on the edge.
  ///   - BorderType::kReflect, Reflect the values on the edge omitting the last value of edge.
  ///   - BorderType::kSymmetric, Reflect the values on the edge repeating the last value of edge.
  /// \note If the input image is more than one, then make sure that the image size is the same.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_crop_op = vision::RandomCrop({255, 255}, {10, 10, 10, 10});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_crop_op},  // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit RandomCrop(const std::vector<int32_t> &size, const std::vector<int32_t> &padding = {0, 0, 0, 0},
                      bool pad_if_needed = false, const std::vector<uint8_t> &fill_value = {0, 0, 0},
                      BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~RandomCrop() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Equivalent to RandomResizedCrop TensorTransform, but crop the image before decoding.
class DATASET_API RandomCropDecodeResize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///               If the size is a single value, a squared crop of size (size, size) is returned.
  ///               If the size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the
  ///               original size to be cropped (default=(0.08, 1.0)).
  /// \param[in] ratio Range [min, max) of aspect ratio to be
  ///               cropped (default=(3. / 4., 4. / 3.)).
  /// \param[in] interpolation An enum for the mode of interpolation.
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \param[in] max_attempts The maximum number of attempts to propose a valid crop_area (default=10).
  ///               If exceeded, fall back to use center_crop instead.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomCropDecodeResize({255, 255}, {0.1, 0.5});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomCropDecodeResize(const std::vector<int32_t> &size, const std::vector<float> &scale = {0.08, 1.0},
                                  const std::vector<float> &ratio = {3. / 4., 4. / 3.},
                                  InterpolationMode interpolation = InterpolationMode::kLinear,
                                  int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomCropDecodeResize() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image at a random location and adjust bounding boxes accordingly.
///        If the cropped area is out of bbox, the returned bbox will be empty.
class DATASET_API RandomCropWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] padding A vector representing the number of pixels to pad the image
  ///    If the vector has one value, it pads all sides of the image with that value.
  ///    If the vector has two values, it pads left and right with the first and
  ///    top and bottom with the second value.
  ///    If the vector has four values, it pads left, top, right, and bottom with
  ///    those values respectively.
  /// \param[in] pad_if_needed A boolean indicating that whether to pad the image
  ///    if either side is smaller than the given output size.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders. Only valid
  ///    if the padding_mode is BorderType.kConstant. If 1 value is provided, it is used for all
  ///    RGB channels. If 3 values are provided, it is used to fill R, G, B channels respectively.
  /// \param[in] padding_mode The method of padding (default=BorderType::kConstant).It can be any of
  ///     [BorderType::kConstant, BorderType::kEdge, BorderType::kReflect, BorderType::kSymmetric].
  ///   - BorderType::kConstant, Fill the border with constant values.
  ///   - BorderType::kEdge, Fill the border with the last value on the edge.
  ///   - BorderType::kReflect, Reflect the values on the edge omitting the last value of edge.
  ///   - BorderType::kSymmetric, Reflect the values on the edge repeating the last value of edge.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomCropWithBBox({224, 224}, {0, 0, 0, 0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit RandomCropWithBBox(const std::vector<int32_t> &size, const std::vector<int32_t> &padding = {0, 0, 0, 0},
                              bool pad_if_needed = false, const std::vector<uint8_t> &fill_value = {0, 0, 0},
                              BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~RandomCropWithBBox() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly flip the input image horizontally with a given probability.
class DATASET_API RandomHorizontalFlip final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomHorizontalFlip(0.8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomHorizontalFlip(float prob = 0.5);

  /// \brief Destructor.
  ~RandomHorizontalFlip() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly flip the input image horizontally with a given probability and adjust bounding boxes accordingly.
class DATASET_API RandomHorizontalFlipWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomHorizontalFlipWithBBox(1.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit RandomHorizontalFlipWithBBox(float prob = 0.5);

  /// \brief Destructor.
  ~RandomHorizontalFlipWithBBox() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Add AlexNet-style PCA-based noise to an image.
class DATASET_API RandomLighting final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] alpha A float representing the intensity of the image (default=0.05).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomLighting(0.1);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomLighting(float alpha = 0.05);

  /// \brief Destructor.
  ~RandomLighting() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Reduce the number of bits for each color channel randomly.
class DATASET_API RandomPosterize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] bit_range Range of random posterize to compress image.
  ///     uint8_t vector representing the minimum and maximum bit in range of [1,8] (Default={4, 8}).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomPosterize({4, 8});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomPosterize(const std::vector<uint8_t> &bit_range = {4, 8});

  /// \brief Destructor.
  ~RandomPosterize() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Resize the input image using a randomly selected interpolation mode and adjust
///     bounding boxes accordingly.
class DATASET_API RandomResizeWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If the size is a single value, the smaller edge of the image will be resized to this value with
  ///      the same image aspect ratio. If the size has 2 values, it should be (height, width).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomResizeWithBBox({50, 50});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit RandomResizeWithBBox(const std::vector<int32_t> &size);

  /// \brief Destructor.
  ~RandomResizeWithBBox() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image to a random size and aspect ratio.
class DATASET_API RandomResizedCrop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the original
  ///     size to be cropped (default=(0.08, 1.0)).
  /// \param[in] ratio Range [min, max) of aspect ratio to be cropped
  ///     (default=(3. / 4., 4. / 3.)).
  /// \param[in] interpolation Image interpolation mode (default=InterpolationMode::kLinear).
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \param[in] max_attempts The maximum number of attempts to propose a valid.
  ///     crop_area (default=10). If exceeded, fall back to use center_crop instead.
  /// \note If the input image is more than one, then make sure that the image size is the same.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomResizedCrop({32, 32}, {0.08, 1.0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomResizedCrop(const std::vector<int32_t> &size, const std::vector<float> &scale = {0.08, 1.0},
                             const std::vector<float> &ratio = {3. / 4., 4. / 3.},
                             InterpolationMode interpolation = InterpolationMode::kLinear, int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomResizedCrop() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image to a random size and aspect ratio.
///        If cropped area is out of bbox, the return bbox will be empty.
class DATASET_API RandomResizedCropWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the original
  ///     size to be cropped (default=(0.08, 1.0)).
  /// \param[in] ratio Range [min, max) of aspect ratio to be cropped
  ///     (default=(3. / 4., 4. / 3.)).
  /// \param[in] interpolation Image interpolation mode (default=InterpolationMode::kLinear).
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \param[in] max_attempts The maximum number of attempts to propose a valid
  ///     crop_area (default=10). If exceeded, fall back to use center_crop instead.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomResizedCropWithBBox({50, 50}, {0.05, 0.5}, {0.2, 0.4},
  ///                                                        InterpolationMode::kCubic);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit RandomResizedCropWithBBox(const std::vector<int32_t> &size, const std::vector<float> &scale = {0.08, 1.0},
                                     const std::vector<float> &ratio = {3. / 4., 4. / 3.},
                                     InterpolationMode interpolation = InterpolationMode::kLinear,
                                     int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomResizedCropWithBBox() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Invert pixels randomly within a specified range.
class DATASET_API RandomSolarize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] threshold A vector with two elements specifying the pixel range to invert.
  ///     Threshold values should always be in (min, max) format.
  ///     If min=max, it will to invert all pixels above min(max).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomSharpness({0, 255});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomSolarize(const std::vector<uint8_t> &threshold = {0, 255});

  /// \brief Destructor.
  ~RandomSolarize() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly flip the input image vertically with a given probability.
class DATASET_API RandomVerticalFlip final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomVerticalFlip();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomVerticalFlip(float prob = 0.5);

  /// \brief Destructor.
  ~RandomVerticalFlip() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly flip the input image vertically with a given probability and adjust bounding boxes accordingly.
class DATASET_API RandomVerticalFlipWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomVerticalFlipWithBBox();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit RandomVerticalFlipWithBBox(float prob = 0.5);

  /// \brief Destructor.
  ~RandomVerticalFlipWithBBox() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Reads a file in binary mode.
/// \param[in] filename The path to the file to be read.
/// \param[out] output The binary data.
/// \return The status code.
Status DATASET_API ReadFile(const std::string &filename, mindspore::MSTensor *output);

/// \brief Read a image file and decode it into one or three channels data.
/// \param[in] filename The path to the file to be read.
/// \param[out] output The Tensor data.
/// \param[in] mode The read mode used for optionally converting the image, can be one of
///    [ImageReadMode::kUNCHANGED, ImageReadMode::kGRAYSCALE, ImageReadMode::kCOLOR]. Default:
///    ImageReadMode::kUNCHANGED.
///    - ImageReadMode::kUNCHANGED, remain the output in the original format.
///    - ImageReadMode::kGRAYSCALE, convert the output into one channel grayscale data.
///    - ImageReadMode::kCOLOR, convert the output into three channels RGB color data.
/// \return The status code.
Status DATASET_API ReadImage(const std::string &filename, mindspore::MSTensor *output,
                             ImageReadMode mode = ImageReadMode::kUNCHANGED);

#ifdef ENABLE_FFMPEG
/// \brief Read the video, audio, metadata from a video file. It supports AVI, H264, H265, MOV, MP4, WMV file formats.
/// \param[in] filename The path to the videoe file to be read.
/// \param[out] video_output The video frames of the video file.
/// \param[out] audio_output The audio frames of the video file.
/// \param[out] metadata_output The metadata contains video_fps, audio_fps.
/// \param[in] start_pts The start presentation timestamp of the video. Default: 0.0.
/// \param[in] end_pts The end presentation timestamp of the video. Default: 2147483647.0.
/// \param[in] pts_unit The unit for the timestamps, can be one of ["pts", "sec"]. Default: "pts".
/// \return The status code.
Status DATASET_API ReadVideo(const std::string &filename, mindspore::MSTensor *video_output,
                             mindspore::MSTensor *audio_output, std::map<std::string, std::string> *metadata_output,
                             float start_pts = 0.0, float end_pts = 2147483647.0, const std::string &pts_unit = "pts");

/// \brief Read the timestamps and frame rate of a video file. It supports AVI, H264, H265, MOV, MP4, WMV files.
/// \param[in] filename The path to the videoe file to be read.
/// \param[out] output The tuple(video_timestamps, video_fps) of the video.
/// \param[in] pts_unit The unit for the timestamps, can be one of ["pts", "sec"]. Default: "pts".
/// \return The status code.
Status DATASET_API ReadVideoTimestamps(const std::string &filename, std::tuple<std::vector<float>, float> *output,
                                       const std::string &pts_unit = "pts");
#endif

/// \brief Resize the input image to the given size and adjust bounding boxes accordingly.
class DATASET_API ResizeWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size The output size of the resized image.
  ///     If the size is an integer, smaller edge of the image will be resized to this value with the same image aspect
  ///     ratio. If the size is a sequence of length 2, it should be (height, width).
  /// \param[in] interpolation An enum for the mode of interpolation (default=InterpolationMode::kLinear).
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::ResizeWithBBox({100, 100}, InterpolationMode::kNearestNeighbour);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit ResizeWithBBox(const std::vector<int32_t> &size,
                          InterpolationMode interpolation = InterpolationMode::kLinear);

  /// \brief Destructor.
  ~ResizeWithBBox() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Change the format of input tensor from 4-channel RGBA to 3-channel BGR.
class DATASET_API RGBA2BGR final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rgb2bgr_op = vision::RGBA2BGR();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rgb2bgr_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  RGBA2BGR();

  /// \brief Destructor.
  ~RGBA2BGR() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Change the input 4 channel RGBA tensor to 3 channel RGB.
class DATASET_API RGBA2RGB final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rgba2rgb_op = vision::RGBA2RGB();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rgba2rgb_op},  // operations
  ///                            {"image"});                // input columns
  /// \endcode
  RGBA2RGB();

  /// \brief Destructor.
  ~RGBA2RGB() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Invert pixels within a specified range.
class DATASET_API Solarize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] threshold A vector with two elements specifying the pixel range to invert.
  ///     Threshold values should always be in (min, max) format.
  ///     If min=max, it will to invert all pixels above min(max).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto solarize_op = vision::Solarize({0, 255});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, solarize_op},  // operations
  ///                            {"image"});                // input columns
  /// \endcode
  explicit Solarize(const std::vector<float> &threshold);

  /// \brief Destructor.
  ~Solarize() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Divide the pixel values by 255 and convert from HWC format to CHW format with required datatype.
class DATASET_API ToTensor final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] output_type The type of the output tensor of type mindspore::DataType or String
  ///   (default=mindspore::DataType::kNumberTypeFloat32).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto to_tensor_op = vision::ToTensor();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({to_tensor_op},  // operations
  ///                            {"image"});  // input columns
  /// \endcode
  ToTensor();
  explicit ToTensor(std::string output_type);
  explicit ToTensor(mindspore::DataType output_type);

  /// \brief Destructor.
  ~ToTensor() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Dataset-independent data-augmentation with TrivialAugment Wide.
class DATASET_API TrivialAugmentWide final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] num_magnitude_bins The number of different magnitude values. Default: 31.
  /// \param[in] interpolation An enum for the mode of interpolation. Default: InterpolationMode::kNearestNeighbour.
  ///     - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///     - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///     - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///     - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders. Default: {0, 0, 0}.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto trivial_augment_wide_op = vision::TrivialAugmentWide();
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, trivial_augment_wide_op}, // operations
  ///                            {"image"});                           // input columns
  /// \endcode
  explicit TrivialAugmentWide(int32_t num_magnitude_bins = 31,
                              InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                              const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~TrivialAugmentWide() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Write the one dimension uint8 data into a file using binary mode.
/// \param[in] filename The path to the file to be written.
/// \param[in] data The tensor data.
/// \return The status code.
Status DATASET_API WriteFile(const std::string &filename, const mindspore::MSTensor &data);

/// \brief Write the image data into a JPEG file.
/// \param[in] filename The path to the file to be written.
/// \param[in] image The data tensor.
/// \param[in] quality The quality for JPEG file, in range of [1, 100]. Default: 75.
/// \return The status code.
Status DATASET_API WriteJpeg(const std::string &filename, const mindspore::MSTensor &image, int quality = 75);

/// \brief Write the image into a PNG file.
/// \param[in] filename The path to the file to be written.
/// \param[in] image The data tensor.
/// \param[in] compression_level The compression level for PNG file, in range of [0, 9]. Default: 6.
/// \return The status code.
Status DATASET_API WritePng(const std::string &filename, const mindspore::MSTensor &image, int compression_level = 6);
}  // namespace vision
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_H_
