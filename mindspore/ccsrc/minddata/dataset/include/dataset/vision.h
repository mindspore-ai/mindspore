/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include <utility>
#include <vector>

#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision_lite.h"

namespace mindspore {
namespace dataset {
class TensorOperation;

// Transform operations for performing computer vision.
namespace vision {
/// \brief Apply brightness adjustment on input image.
class DATASET_API AdjustBrightness final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] brightness_factor Adjusts image brightness, non negative real number.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto adjust_brightness_op = vision::AdjustBrightness(2.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, adjust_brightness_op},  // operations
  ///                            {"image"});                         // input columns
  /// \endcode
  explicit AdjustBrightness(float brightness_factor);

  /// \brief Destructor.
  ~AdjustBrightness() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply contrast adjustment on input image.
class DATASET_API AdjustContrast final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] contrast_factor Adjusts image contrast, non negative real number.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto adjust_contrast_op = vision::AdjustContrast(10.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, adjust_contrast_op},  // operations
  ///                            {"image"});                       // input columns
  /// \endcode
  explicit AdjustContrast(float contrast_factor);

  /// \brief Destructor.
  ~AdjustContrast() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

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

/// \note Apply hue adjustment on input image.
class DATASET_API AdjustHue final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] hue_factor How much to shift the hue channel, must be in the interval [-0.5, 0.5].
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto adjust_hue_op = vision::AdjustHue(0.2);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, adjust_contrast_op},  // operations
  ///                            {"image"});                       // input columns
  /// \endcode
  explicit AdjustHue(float hue_factor);

  /// \brief Destructor.
  ~AdjustHue() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Adjust the color saturation of the input image.
class DATASET_API AdjustSaturation final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] saturation_factor Adjust image saturation, non negative real number.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto adjust_saturation_op = vision::AdjustSaturation(2.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, adjust_saturation_op},  // operations
  ///                            {"image"});                         // input columns
  /// \endcode
  explicit AdjustSaturation(float saturation_factor);

  /// \brief Destructor.
  ~AdjustSaturation() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply adjust sharpness on input image. Input image is expected to be in [H, W, C] or [H, W] format.
class DATASET_API AdjustSharpness final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] sharpness_factor How much to adjust the sharpness. Can be any Non negative real number.
  ///     0 gives a blurred image, 1 gives the original image while 2 increases the Sharpness by a factor of 2.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto adjust_sharpness_op = vision::AdjustSharpness(2.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, adjust_sharpness_op},   // operations
  ///                            {"image"});                         // input columns
  /// \endcode
  explicit AdjustSharpness(float sharpness_factor);

  /// \brief Destructor.
  ~AdjustSharpness() override = default;

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

/// \brief Apply automatic contrast on the input image.
class DATASET_API AutoContrast final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] cutoff Percent of pixels to cut off from the histogram, the valid range of cutoff value is 0 to 50.
  /// \param[in] ignore Pixel values to ignore.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto autocontrast_op = vision::AutoContrast(10.0, {10, 20});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, autocontrast_op},  // operations
  ///                            {"image"});                    // input columns
  /// \endcode
  explicit AutoContrast(float cutoff = 0.0, const std::vector<uint32_t> &ignore = {});

  /// \brief Destructor.
  ~AutoContrast() override = default;

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

/// \brief Change the color space of the image.
class DATASET_API ConvertColor final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] convert_mode The mode of image channel conversion.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::ConvertColor>(ConvertMode::COLOR_BGR2RGB)}, // operations
  ///                            {"image"});                                                           // input columns
  /// \endcode
  explicit ConvertColor(ConvertMode convert_mode);

  /// \brief Destructor.
  ~ConvertColor() override = default;

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

/// \brief Randomly cut (mask) out a given number of square patches from the input image.
class DATASET_API CutOut final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] length Integer representing the side length of each square patch.
  /// \param[in] num_patches Integer representing the number of patches to be cut out of an image.
  /// \param[in] is_hwc A boolean to indicate whether the input image is in HWC format (true) or CHW
  ///     format (false) (default = true).
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::CutOut>(1, 4, true)}, // operations
  ///                            {"image"});                                     // input columns
  /// \endcode
  explicit CutOut(int32_t length, int32_t num_patches = 1, bool is_hwc = true);

  /// \brief Destructor.
  ~CutOut() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

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

/// \brief Apply histogram equalization on the input image.
class DATASET_API Equalize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::Equalize>()}, // operations
  ///                            {"image"});                             // input columns
  /// \endcode
  Equalize();

  /// \brief Destructor.
  ~Equalize() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

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
  Erase(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<uint8_t> &value = {0, 0, 0},
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

/// \brief Flip the input image horizontally.
class DATASET_API HorizontalFlip final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::HorizontalFlip>()}, // operations
  ///                            {"image"});                                   // input columns
  /// \endcode
  HorizontalFlip();

  /// \brief Destructor.
  ~HorizontalFlip() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Apply invert on the input image in RGB mode.
class DATASET_API Invert final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::Invert>()}, // operations
  ///                            {"image"});                           // input columns
  /// \endcode
  Invert();

  /// \brief Destructor.
  ~Invert() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

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

/// \brief Pad the image to a fixed size.
class DATASET_API PadToSize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A two element vector representing the target size to pad, in order of [height, width].
  /// \param[in] offset A two element vector representing the lengths to pad on the top and left,
  ///    in order of [top, left]. Default: {}, means to pad symmetrically, keeping the original image in center.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders. Only valid if the
  ///    padding_mode is BorderType.kConstant. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G, B channels respectively. Default: {0}.
  /// \param[in] padding_mode The method of padding, which can be one of BorderType.kConstant, BorderType.kEdge,
  ///    BorderType.kReflect or BorderType.kSymmetric. Default: BorderType.kConstant.
  ///    - BorderType.kConstant, pads with a constant value.
  ///    - BorderType.kEdge, pads with the last value at the edge of the image.
  ///    - BorderType.kReflect, pads with reflection of the image omitting the last value on the edge.
  ///    - BorderType.kSymmetric, pads with reflection of the image repeating the last value on the edge.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto pad_to_size_op = vision::PadToSize({256, 256}, {10, 20}, {255, 255, 255});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, pad_to_size_op},  // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit PadToSize(const std::vector<int32_t> &size, const std::vector<int32_t> &offset = {},
                     const std::vector<uint8_t> &fill_value = {0}, BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~PadToSize() override = default;

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

/// \brief Automatically adjust the contrast of the image with a given probability.
class DATASET_API RandomAutoContrast final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] cutoff Percent of the lightest and darkest pixels to be cut off from
  ///     the histogram of the input image. The value must be in range of [0.0, 50.0) (default=0.0).
  /// \param[in] ignore The background pixel values to be ignored, each of which must be
  ///     in range of [0, 255] (default={}).
  /// \param[in] prob A float representing the probability of AutoContrast, which must be
  ///     in range of [0, 1] (default=0.5).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_auto_contrast_op = vision::RandomAutoContrast(5.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_auto_contrast_op},  // operations
  ///                            {"image"});                            // input columns
  /// \endcode
  explicit RandomAutoContrast(float cutoff = 0.0, const std::vector<uint32_t> &ignore = {}, float prob = 0.5);

  /// \brief Destructor.
  ~RandomAutoContrast() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly adjust the sharpness of the input image with a given probability.
class DATASET_API RandomAdjustSharpness final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degree A float representing sharpness adjustment degree, which must be non negative.
  /// \param[in] prob A float representing the probability of the image being sharpness adjusted, which
  ///     must in range of [0, 1] (default=0.5).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_adjust_sharpness_op = vision::RandomAdjustSharpness(30.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_adjust_sharpness_op},  // operations
  ///                            {"image"});                               // input columns
  /// \endcode
  explicit RandomAdjustSharpness(float degree, float prob = 0.5);

  /// \brief Destructor.
  ~RandomAdjustSharpness() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Blend an image with its grayscale version with random weights
///        t and 1 - t generated from a given range. If the range is trivial
///        then the weights are determinate and t equals to the bound of the interval.
class DATASET_API RandomColor final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] t_lb Lower bound random weights.
  /// \param[in] t_ub Upper bound random weights.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_color_op = vision::RandomColor(5.0, 50.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_color_op},  // operations
  ///                            {"image"});                    // input columns
  /// \endcode
  RandomColor(float t_lb, float t_ub);

  /// \brief Destructor.
  ~RandomColor() override = default;

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

/// \brief Randomly apply histogram equalization on the input image with a given probability.
class DATASET_API RandomEqualize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of equalization, which
  ///     must be in range of [0, 1] (default=0.5).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomEqualize(0.5);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomEqualize(float prob = 0.5);

  /// \brief Destructor.
  ~RandomEqualize() override = default;

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

/// \brief Randomly invert the input image with a given probability.
class DATASET_API RandomInvert final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of the image being inverted, which
  ///     must be in range of [0, 1] (default=0.5).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomInvert(0.8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomInvert(float prob = 0.5);

  /// \brief Destructor.
  ~RandomInvert() override = default;

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

/// \brief Resize the input image using a randomly selected interpolation mode.
class DATASET_API RandomResize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If the size is a single value, the smaller edge of the image will be resized to this value with
  ///      the same image aspect ratio. If the size has 2 values, it should be (height, width).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomResize({32, 32});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomResize(const std::vector<int32_t> &size);

  /// \brief Destructor.
  ~RandomResize() override = default;

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

/// \brief Rotate the image according to parameters.
class DATASET_API RandomRotation final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degrees A float vector of size 2, representing the starting and ending degrees.
  /// \param[in] resample An enum for the mode of interpolation.
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \param[in] expand A boolean representing whether the image is expanded after rotation.
  /// \param[in] center A float vector of size 2 or empty, representing the x and y center of rotation
  ///     or the center of the image.
  /// \param[in] fill_value A vector representing the value to fill the area outside the transform
  ///    in the output image. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G, B channels respectively.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomRotation({30, 60}, InterpolationMode::kNearestNeighbour);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomRotation(const std::vector<float> &degrees,
                          InterpolationMode resample = InterpolationMode::kNearestNeighbour, bool expand = false,
                          const std::vector<float> &center = {}, const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~RandomRotation() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Choose a random sub-policy from a list to be applied on the input image. A sub-policy is a list of tuples
///     (operation, prob), where operation is a TensorTransform operation and prob is the probability that this
///     operation will be applied. Once a sub-policy is selected, each operation within the sub-policy with be
///     applied in sequence according to its probability.
class DATASET_API RandomSelectSubpolicy final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] policy Vector of sub-policies to choose from, in which the TensorTransform objects are raw pointers.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto invert_op(new vision::Invert());
  ///     auto equalize_op(new vision::Equalize());
  ///
  ///     std::vector<std::pair<TensorTransform *, double>> policy = {{invert_op, 0.5}, {equalize_op, 0.4}};
  ///     vision::RandomSelectSubpolicy random_select_subpolicy_op = vision::RandomSelectSubpolicy({policy});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_select_subpolicy_op},    // operations
  ///                            {"image"});                      // input columns
  /// \endcode
  explicit RandomSelectSubpolicy(const std::vector<std::vector<std::pair<TensorTransform *, double>>> &policy);

  /// \brief Constructor.
  /// \param[in] policy Vector of sub-policies to choose from, in which the TensorTransform objects are shared pointers.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     std::shared_ptr<TensorTransform> invert_op(new vision::Invert());
  ///     std::shared_ptr<TensorTransform> equalize_op(new vision::Equalize());
  ///     std::shared_ptr<TensorTransform> resize_op(new vision::Resize({15, 15}));
  ///
  ///     auto random_select_subpolicy_op = vision::RandomSelectSubpolicy({
  ///                                          {{invert_op, 0.5}, {equalize_op, 0.4}},
  ///                                          {{resize_op, 0.1}}
  ///                                       });
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_select_subpolicy_op},    // operations
  ///                            {"image"});                      // input columns
  /// \endcode
  explicit RandomSelectSubpolicy(
    const std::vector<std::vector<std::pair<std::shared_ptr<TensorTransform>, double>>> &policy);

  /// \brief Constructor.
  /// \param[in] policy Vector of sub-policies to choose from, in which the TensorTransform objects are object pointers.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     vision::Invert invert_op = vision::Invert();
  ///     vision::Equalize equalize_op = vision::Equalize();
  ///     vision::Resize resize_op = vision::Resize({15, 15});
  ///
  ///     auto random_select_subpolicy_op = vision::RandomSelectSubpolicy({
  ///                                          {{invert_op, 0.5}, {equalize_op, 0.4}},
  ///                                          {{resize_op, 0.1}}
  ///                                       });
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_select_subpolicy_op},    // operations
  ///                            {"image"});                      // input columns
  /// \endcode
  explicit RandomSelectSubpolicy(
    const std::vector<std::vector<std::pair<std::reference_wrapper<TensorTransform>, double>>> &policy);

  /// \brief Destructor.
  ~RandomSelectSubpolicy() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Adjust the sharpness of the input image by a fixed or random degree.
class DATASET_API RandomSharpness final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degrees A float vector of size 2, representing the range of random sharpness
  ///     adjustment degrees. It should be in (min, max) format. If min=max, then it is a
  ///     single fixed magnitude operation (default = (0.1, 1.9)).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomSharpness({0.1, 1.5});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomSharpness(const std::vector<float> &degrees = {0.1, 1.9});

  /// \brief Destructor.
  ~RandomSharpness() override = default;

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

/// \brief Crop the given image and zoom to the specified size.
class DATASET_API ResizedCrop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] top Horizontal ordinate of the upper left corner of the crop image.
  /// \param[in] left Vertical ordinate of the upper left corner of the crop image.
  /// \param[in] height Height of cropped image.
  /// \param[in] width Width of cropped image.
  /// \param[in] size A vector representing the output size of the image.
  ///     If the size is a single value, a squared resized of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] interpolation Image interpolation mode. Default: InterpolationMode::kLinear.
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \note If the input image is more than one, then make sure that the image size is the same.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto resized_crop_op = vision::ResizedCrop(128, 128, 256, 256, {128, 128});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, resized_crop_op},  // operations
  ///                            {"image"});                    // input columns
  /// \endcode
  ResizedCrop(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<int32_t> &size,
              InterpolationMode interpolation = InterpolationMode::kLinear);

  /// \brief Destructor.
  ~ResizedCrop() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

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

/// \note Slice the tensor to multiple patches in horizontal and vertical directions.
class DATASET_API SlicePatches final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] num_height The number of patches in vertical direction (default=1).
  /// \param[in] num_width The number of patches in horizontal direction (default=1).
  /// \param[in] slice_mode An enum for the mode of slice (default=SliceMode::kPad).
  /// \param[in] fill_value A value representing the pixel to fill the padding area in right and
  ///     bottom border if slice_mode is kPad. Then padded tensor could be just sliced to multiple patches (default=0).
  /// \note The usage scenerio is suitable to tensor with large height and width. The tensor will keep the same
  ///     if set both num_height and num_width to 1. And the number of output tensors is equal to num_height*num_width.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto slice_patch_op = vision::SlicePatches(255, 255);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, slice_patch_op},  // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit SlicePatches(int32_t num_height = 1, int32_t num_width = 1, SliceMode slice_mode = SliceMode::kPad,
                        uint8_t fill_value = 0);

  /// \brief Destructor.
  ~SlicePatches() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
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

/// \brief Randomly perform transformations, as selected from input transform list, on the input tensor.
class DATASET_API UniformAugment final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transforms Raw pointer to vector of TensorTransform operations.
  /// \param[in] num_ops An integer representing the number of operations to be selected and applied.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto resize_op(new vision::Resize({30, 30}));
  ///     auto random_crop_op(new vision::RandomCrop({28, 28}));
  ///     auto center_crop_op(new vision::CenterCrop({16, 16}));
  ///     auto uniform_op(new vision::UniformAugment({random_crop_op, center_crop_op}, 2));
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({resize_op, uniform_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  explicit UniformAugment(const std::vector<TensorTransform *> &transforms, int32_t num_ops = 2);

  /// \brief Constructor.
  /// \param[in] transforms Smart pointer to vector of TensorTransform operations.
  /// \param[in] num_ops An integer representing the number of operations to be selected and applied.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     std::shared_ptr<TensorTransform> resize_op(new vision::Resize({30, 30}));
  ///     std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({28, 28}));
  ///     std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));
  ///     std::shared_ptr<TensorTransform> uniform_op(new vision::UniformAugment({random_crop_op, center_crop_op}, 2));
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({resize_op, uniform_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  explicit UniformAugment(const std::vector<std::shared_ptr<TensorTransform>> &transforms, int32_t num_ops = 2);

  /// \brief Constructor.
  /// \param[in] transforms Object pointer to vector of TensorTransform operations.
  /// \param[in] num_ops An integer representing the number of operations to be selected and applied.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     vision::Resize resize_op = vision::Resize({30, 30});
  ///     vision::RandomCrop random_crop_op = vision::RandomCrop({28, 28});
  ///     vision::CenterCrop center_crop_op = vision::CenterCrop({16, 16});
  ///     vision::UniformAugment uniform_op = vision::UniformAugment({random_crop_op, center_crop_op}, 2);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({resize_op, uniform_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  explicit UniformAugment(const std::vector<std::reference_wrapper<TensorTransform>> &transforms, int32_t num_ops = 2);

  /// \brief Destructor.
  ~UniformAugment() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Flip the input image vertically.
class DATASET_API VerticalFlip final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto flip_op = vision::VerticalFlip();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, flip_op},  // operations
  ///                            {"image"});            // input columns
  /// \endcode
  VerticalFlip();

  /// \brief Destructor.
  ~VerticalFlip() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
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
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_H_
