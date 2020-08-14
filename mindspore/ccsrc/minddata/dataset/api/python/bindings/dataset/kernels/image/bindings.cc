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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/kernels/py_func_op.h"
#include "minddata/dataset/kernels/image/auto_contrast_op.h"
#include "minddata/dataset/kernels/image/bounding_box_augment_op.h"
#include "minddata/dataset/kernels/image/center_crop_op.h"
#include "minddata/dataset/kernels/image/cut_out_op.h"
#include "minddata/dataset/kernels/image/decode_op.h"
#include "minddata/dataset/kernels/image/equalize_op.h"
#include "minddata/dataset/kernels/image/hwc_to_chw_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/kernels/image/invert_op.h"
#include "minddata/dataset/kernels/image/mixup_batch_op.h"
#include "minddata/dataset/kernels/image/normalize_op.h"
#include "minddata/dataset/kernels/image/pad_op.h"
#include "minddata/dataset/kernels/image/random_affine_op.h"
#include "minddata/dataset/kernels/image/random_color_op.h"
#include "minddata/dataset/kernels/image/random_color_adjust_op.h"
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include "minddata/dataset/kernels/image/random_crop_and_resize_with_bbox_op.h"
#include "minddata/dataset/kernels/image/random_crop_decode_resize_op.h"
#include "minddata/dataset/kernels/image/random_crop_op.h"
#include "minddata/dataset/kernels/image/random_crop_with_bbox_op.h"
#include "minddata/dataset/kernels/image/random_horizontal_flip_op.h"
#include "minddata/dataset/kernels/image/random_horizontal_flip_with_bbox_op.h"
#include "minddata/dataset/kernels/image/random_resize_op.h"
#include "minddata/dataset/kernels/image/random_resize_with_bbox_op.h"
#include "minddata/dataset/kernels/image/random_rotation_op.h"
#include "minddata/dataset/kernels/image/random_sharpness_op.h"
#include "minddata/dataset/kernels/image/random_select_subpolicy_op.h"
#include "minddata/dataset/kernels/image/random_solarize_op.h"
#include "minddata/dataset/kernels/image/random_vertical_flip_op.h"
#include "minddata/dataset/kernels/image/random_vertical_flip_with_bbox_op.h"
#include "minddata/dataset/kernels/image/rescale_op.h"
#include "minddata/dataset/kernels/image/resize_bilinear_op.h"
#include "minddata/dataset/kernels/image/resize_op.h"
#include "minddata/dataset/kernels/image/resize_with_bbox_op.h"
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_random_crop_resize_jpeg_op.h"
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_resize_jpeg_op.h"
#include "minddata/dataset/kernels/image/uniform_aug_op.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(AutoContrastOp, 1, ([](const py::module *m) {
                  (void)py::class_<AutoContrastOp, TensorOp, std::shared_ptr<AutoContrastOp>>(
                    *m, "AutoContrastOp", "Tensor operation to apply autocontrast on an image.")
                    .def(py::init<float, std::vector<uint32_t>>(), py::arg("cutoff") = AutoContrastOp::kCutOff,
                         py::arg("ignore") = AutoContrastOp::kIgnore);
                }));

PYBIND_REGISTER(NormalizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<NormalizeOp, TensorOp, std::shared_ptr<NormalizeOp>>(
                    *m, "NormalizeOp", "Tensor operation to normalize an image. Takes mean and std.")
                    .def(py::init<float, float, float, float, float, float>(), py::arg("meanR"), py::arg("meanG"),
                         py::arg("meanB"), py::arg("stdR"), py::arg("stdG"), py::arg("stdB"));
                }));

PYBIND_REGISTER(EqualizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<EqualizeOp, TensorOp, std::shared_ptr<EqualizeOp>>(
                    *m, "EqualizeOp", "Tensor operation to apply histogram equalization on images.")
                    .def(py::init<>());
                }));

PYBIND_REGISTER(InvertOp, 1, ([](const py::module *m) {
                  (void)py::class_<InvertOp, TensorOp, std::shared_ptr<InvertOp>>(
                    *m, "InvertOp", "Tensor operation to apply invert on RGB images.")
                    .def(py::init<>());
                }));

PYBIND_REGISTER(RescaleOp, 1, ([](const py::module *m) {
                  (void)py::class_<RescaleOp, TensorOp, std::shared_ptr<RescaleOp>>(
                    *m, "RescaleOp", "Tensor operation to rescale an image. Takes scale and shift.")
                    .def(py::init<float, float>(), py::arg("rescale"), py::arg("shift"));
                }));

PYBIND_REGISTER(CenterCropOp, 1, ([](const py::module *m) {
                  (void)py::class_<CenterCropOp, TensorOp, std::shared_ptr<CenterCropOp>>(
                    *m, "CenterCropOp",
                    "Tensor operation to crop and image in the middle. Takes height and width (optional)")
                    .def(py::init<int32_t, int32_t>(), py::arg("height"), py::arg("width") = CenterCropOp::kDefWidth);
                }));

PYBIND_REGISTER(MixUpBatchOp, 1, ([](const py::module *m) {
                  (void)py::class_<MixUpBatchOp, TensorOp, std::shared_ptr<MixUpBatchOp>>(
                    *m, "MixUpBatchOp", "Tensor operation to mixup a batch of images")
                    .def(py::init<float>(), py::arg("alpha"));
                }));

PYBIND_REGISTER(ResizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<ResizeOp, TensorOp, std::shared_ptr<ResizeOp>>(
                    *m, "ResizeOp", "Tensor operation to resize an image. Takes height, width and mode")
                    .def(py::init<int32_t, int32_t, InterpolationMode>(), py::arg("targetHeight"),
                         py::arg("targetWidth") = ResizeOp::kDefWidth,
                         py::arg("interpolation") = ResizeOp::kDefInterpolation);
                }));

PYBIND_REGISTER(ResizeWithBBoxOp, 1, ([](const py::module *m) {
                  (void)py::class_<ResizeWithBBoxOp, TensorOp, std::shared_ptr<ResizeWithBBoxOp>>(
                    *m, "ResizeWithBBoxOp", "Tensor operation to resize an image. Takes height, width and mode.")
                    .def(py::init<int32_t, int32_t, InterpolationMode>(), py::arg("targetHeight"),
                         py::arg("targetWidth") = ResizeWithBBoxOp::kDefWidth,
                         py::arg("interpolation") = ResizeWithBBoxOp::kDefInterpolation);
                }));

PYBIND_REGISTER(RandomAffineOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomAffineOp, TensorOp, std::shared_ptr<RandomAffineOp>>(
                    *m, "RandomAffineOp", "Tensor operation to apply random affine transformations on an image.")
                    .def(py::init<std::vector<float_t>, std::vector<float_t>, std::vector<float_t>,
                                  std::vector<float_t>, InterpolationMode, std::vector<uint8_t>>(),
                         py::arg("degrees") = RandomAffineOp::kDegreesRange,
                         py::arg("translate_range") = RandomAffineOp::kTranslationPercentages,
                         py::arg("scale_range") = RandomAffineOp::kScaleRange,
                         py::arg("shear_ranges") = RandomAffineOp::kShearRanges,
                         py::arg("interpolation") = RandomAffineOp::kDefInterpolation,
                         py::arg("fill_value") = RandomAffineOp::kFillValue);
                }));

PYBIND_REGISTER(
  RandomResizeWithBBoxOp, 1, ([](const py::module *m) {
    (void)py::class_<RandomResizeWithBBoxOp, TensorOp, std::shared_ptr<RandomResizeWithBBoxOp>>(
      *m, "RandomResizeWithBBoxOp",
      "Tensor operation to resize an image using a randomly selected interpolation. Takes height and width.")
      .def(py::init<int32_t, int32_t>(), py::arg("targetHeight"),
           py::arg("targetWidth") = RandomResizeWithBBoxOp::kDefTargetWidth);
  }));
PYBIND_REGISTER(UniformAugOp, 1, ([](const py::module *m) {
                  (void)py::class_<UniformAugOp, TensorOp, std::shared_ptr<UniformAugOp>>(
                    *m, "UniformAugOp", "Tensor operation to apply random augmentation(s).")
                    .def(py::init<std::vector<std::shared_ptr<TensorOp>>, int32_t>(), py::arg("transforms"),
                         py::arg("NumOps") = UniformAugOp::kDefNumOps);
                }));
PYBIND_REGISTER(BoundingBoxAugmentOp, 1, ([](const py::module *m) {
                  (void)py::class_<BoundingBoxAugmentOp, TensorOp, std::shared_ptr<BoundingBoxAugmentOp>>(
                    *m, "BoundingBoxAugmentOp",
                    "Tensor operation to apply a transformation on a random choice of bounding boxes.")
                    .def(py::init<std::shared_ptr<TensorOp>, float>(), py::arg("transform"),
                         py::arg("ratio") = BoundingBoxAugmentOp::kDefRatio);
                }));
PYBIND_REGISTER(ResizeBilinearOp, 1, ([](const py::module *m) {
                  (void)py::class_<ResizeBilinearOp, TensorOp, std::shared_ptr<ResizeBilinearOp>>(
                    *m, "ResizeBilinearOp",
                    "Tensor operation to resize an image using "
                    "Bilinear mode. Takes height and width.")
                    .def(py::init<int32_t, int32_t>(), py::arg("targetHeight"),
                         py::arg("targetWidth") = ResizeBilinearOp::kDefWidth);
                }));

PYBIND_REGISTER(DecodeOp, 1, ([](const py::module *m) {
                  (void)py::class_<DecodeOp, TensorOp, std::shared_ptr<DecodeOp>>(
                    *m, "DecodeOp", "Tensor operation to decode a jpg image")
                    .def(py::init<>())
                    .def(py::init<bool>(), py::arg("rgb_format") = DecodeOp::kDefRgbFormat);
                }));

PYBIND_REGISTER(RandomHorizontalFlipOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomHorizontalFlipOp, TensorOp, std::shared_ptr<RandomHorizontalFlipOp>>(
                    *m, "RandomHorizontalFlipOp", "Tensor operation to randomly flip an image horizontally.")
                    .def(py::init<float>(), py::arg("probability") = RandomHorizontalFlipOp::kDefProbability);
                }));

PYBIND_REGISTER(
  RandomHorizontalFlipWithBBoxOp, 1, ([](const py::module *m) {
    (void)py::class_<RandomHorizontalFlipWithBBoxOp, TensorOp, std::shared_ptr<RandomHorizontalFlipWithBBoxOp>>(
      *m, "RandomHorizontalFlipWithBBoxOp",
      "Tensor operation to randomly flip an image horizontally, while flipping bounding boxes.")
      .def(py::init<float>(), py::arg("probability") = RandomHorizontalFlipWithBBoxOp::kDefProbability);
  }));
PYBIND_REGISTER(RandomVerticalFlipOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomVerticalFlipOp, TensorOp, std::shared_ptr<RandomVerticalFlipOp>>(
                    *m, "RandomVerticalFlipOp", "Tensor operation to randomly flip an image vertically.")
                    .def(py::init<float>(), py::arg("probability") = RandomVerticalFlipOp::kDefProbability);
                }));
PYBIND_REGISTER(RandomVerticalFlipWithBBoxOp, 1, ([](const py::module *m) {
                  (void)
                    py::class_<RandomVerticalFlipWithBBoxOp, TensorOp, std::shared_ptr<RandomVerticalFlipWithBBoxOp>>(
                      *m, "RandomVerticalFlipWithBBoxOp",
                      "Tensor operation to randomly flip an image vertically"
                      " and adjust bounding boxes.")
                      .def(py::init<float>(), py::arg("probability") = RandomVerticalFlipWithBBoxOp::kDefProbability);
                }));
PYBIND_REGISTER(
  RandomCropOp, 1, ([](const py::module *m) {
    (void)py::class_<RandomCropOp, TensorOp, std::shared_ptr<RandomCropOp>>(*m, "RandomCropOp",
                                                                            "Gives random crop of specified size "
                                                                            "Takes crop size")
      .def(
        py::init<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, BorderType, bool, uint8_t, uint8_t, uint8_t>(),
        py::arg("cropHeight"), py::arg("cropWidth"), py::arg("padTop") = RandomCropOp::kDefPadTop,
        py::arg("padBottom") = RandomCropOp::kDefPadBottom, py::arg("padLeft") = RandomCropOp::kDefPadLeft,
        py::arg("padRight") = RandomCropOp::kDefPadRight, py::arg("borderType") = RandomCropOp::kDefBorderType,
        py::arg("padIfNeeded") = RandomCropOp::kDefPadIfNeeded, py::arg("fillR") = RandomCropOp::kDefFillR,
        py::arg("fillG") = RandomCropOp::kDefFillG, py::arg("fillB") = RandomCropOp::kDefFillB);
  }));
PYBIND_REGISTER(
  HwcToChwOp, 1, ([](const py::module *m) {
    (void)py::class_<HwcToChwOp, TensorOp, std::shared_ptr<HwcToChwOp>>(*m, "ChannelSwapOp").def(py::init<>());
  }));
PYBIND_REGISTER(
  RandomCropWithBBoxOp, 1, ([](const py::module *m) {
    (void)py::class_<RandomCropWithBBoxOp, TensorOp, std::shared_ptr<RandomCropWithBBoxOp>>(
      *m, "RandomCropWithBBoxOp",
      "Gives random crop of given "
      "size + adjusts bboxes "
      "Takes crop size")
      .def(
        py::init<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, BorderType, bool, uint8_t, uint8_t, uint8_t>(),
        py::arg("cropHeight"), py::arg("cropWidth"), py::arg("padTop") = RandomCropWithBBoxOp::kDefPadTop,
        py::arg("padBottom") = RandomCropWithBBoxOp::kDefPadBottom,
        py::arg("padLeft") = RandomCropWithBBoxOp::kDefPadLeft,
        py::arg("padRight") = RandomCropWithBBoxOp::kDefPadRight,
        py::arg("borderType") = RandomCropWithBBoxOp::kDefBorderType,
        py::arg("padIfNeeded") = RandomCropWithBBoxOp::kDefPadIfNeeded,
        py::arg("fillR") = RandomCropWithBBoxOp::kDefFillR, py::arg("fillG") = RandomCropWithBBoxOp::kDefFillG,
        py::arg("fillB") = RandomCropWithBBoxOp::kDefFillB);
  }));
PYBIND_REGISTER(CutOutOp, 1, ([](const py::module *m) {
                  (void)py::class_<CutOutOp, TensorOp, std::shared_ptr<CutOutOp>>(
                    *m, "CutOutOp",
                    "Tensor operation to randomly erase a portion of the image. Takes height and width.")
                    .def(py::init<int32_t, int32_t, int32_t, bool, uint8_t, uint8_t, uint8_t>(), py::arg("boxHeight"),
                         py::arg("boxWidth"), py::arg("numPatches"), py::arg("randomColor") = CutOutOp::kDefRandomColor,
                         py::arg("fillR") = CutOutOp::kDefFillR, py::arg("fillG") = CutOutOp::kDefFillG,
                         py::arg("fillB") = CutOutOp::kDefFillB);
                }));
PYBIND_REGISTER(PadOp, 1, ([](const py::module *m) {
                  (void)py::class_<PadOp, TensorOp, std::shared_ptr<PadOp>>(
                    *m, "PadOp",
                    "Pads image with specified color, default black, "
                    "Takes amount to pad for top, bottom, left, right of image, boarder type and color")
                    .def(py::init<int32_t, int32_t, int32_t, int32_t, BorderType, uint8_t, uint8_t, uint8_t>(),
                         py::arg("padTop"), py::arg("padBottom"), py::arg("padLeft"), py::arg("padRight"),
                         py::arg("borderTypes") = PadOp::kDefBorderType, py::arg("fillR") = PadOp::kDefFillR,
                         py::arg("fillG") = PadOp::kDefFillG, py::arg("fillB") = PadOp::kDefFillB);
                }));

PYBIND_REGISTER(RandomCropDecodeResizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomCropDecodeResizeOp, TensorOp, std::shared_ptr<RandomCropDecodeResizeOp>>(
                    *m, "RandomCropDecodeResizeOp", "equivalent to RandomCropAndResize but crops before decoding")
                    .def(py::init<int32_t, int32_t, float, float, float, float, InterpolationMode, int32_t>(),
                         py::arg("targetHeight"), py::arg("targetWidth"),
                         py::arg("scaleLb") = RandomCropDecodeResizeOp::kDefScaleLb,
                         py::arg("scaleUb") = RandomCropDecodeResizeOp::kDefScaleUb,
                         py::arg("aspectLb") = RandomCropDecodeResizeOp::kDefAspectLb,
                         py::arg("aspectUb") = RandomCropDecodeResizeOp::kDefAspectUb,
                         py::arg("interpolation") = RandomCropDecodeResizeOp::kDefInterpolation,
                         py::arg("maxIter") = RandomCropDecodeResizeOp::kDefMaxIter);
                }));

PYBIND_REGISTER(
  RandomResizeOp, 1, ([](const py::module *m) {
    (void)py::class_<RandomResizeOp, TensorOp, std::shared_ptr<RandomResizeOp>>(
      *m, "RandomResizeOp",
      "Tensor operation to resize an image using a randomly selected interpolation. Takes height and width.")
      .def(py::init<int32_t, int32_t>(), py::arg("targetHeight"),
           py::arg("targetWidth") = RandomResizeOp::kDefTargetWidth);
  }));

PYBIND_REGISTER(RandomColorOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomColorOp, TensorOp, std::shared_ptr<RandomColorOp>>(
                    *m, "RandomColorOp",
                    "Tensor operation to blend an image with its grayscale version with random weights"
                    "Takes min and max for the range of random weights")
                    .def(py::init<float, float>(), py::arg("min"), py::arg("max"));
                }));

PYBIND_REGISTER(RandomColorAdjustOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomColorAdjustOp, TensorOp, std::shared_ptr<RandomColorAdjustOp>>(
                    *m, "RandomColorAdjustOp",
                    "Tensor operation to adjust an image's color randomly."
                    "Takes range for brightness, contrast, saturation, hue and")
                    .def(py::init<float, float, float, float, float, float, float, float>(),
                         py::arg("bright_factor_start"), py::arg("bright_factor_end"), py::arg("contrast_factor_start"),
                         py::arg("contrast_factor_end"), py::arg("saturation_factor_start"),
                         py::arg("saturation_factor_end"), py::arg("hue_factor_start"), py::arg("hue_factor_end"));
                }));

PYBIND_REGISTER(RandomCropAndResizeWithBBoxOp, 1, ([](const py::module *m) {
                  (void)
                    py::class_<RandomCropAndResizeWithBBoxOp, TensorOp, std::shared_ptr<RandomCropAndResizeWithBBoxOp>>(
                      *m, "RandomCropAndResizeWithBBoxOp",
                      "Tensor operation to randomly crop an image (with BBoxes) and resize to a given size."
                      "Takes output height and width and"
                      "optional parameters for lower and upper bound for aspect ratio (h/w) and scale,"
                      "interpolation mode, and max attempts to crop")
                      .def(py::init<int32_t, int32_t, float, float, float, float, InterpolationMode, int32_t>(),
                           py::arg("targetHeight"), py::arg("targetWidth"),
                           py::arg("scaleLb") = RandomCropAndResizeWithBBoxOp::kDefScaleLb,
                           py::arg("scaleUb") = RandomCropAndResizeWithBBoxOp::kDefScaleUb,
                           py::arg("aspectLb") = RandomCropAndResizeWithBBoxOp::kDefAspectLb,
                           py::arg("aspectUb") = RandomCropAndResizeWithBBoxOp::kDefAspectUb,
                           py::arg("interpolation") = RandomCropAndResizeWithBBoxOp::kDefInterpolation,
                           py::arg("maxIter") = RandomCropAndResizeWithBBoxOp::kDefMaxIter);
                }));

PYBIND_REGISTER(RandomCropAndResizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomCropAndResizeOp, TensorOp, std::shared_ptr<RandomCropAndResizeOp>>(
                    *m, "RandomCropAndResizeOp",
                    "Tensor operation to randomly crop an image and resize to a given size."
                    "Takes output height and width and"
                    "optional parameters for lower and upper bound for aspect ratio (h/w) and scale,"
                    "interpolation mode, and max attempts to crop")
                    .def(py::init<int32_t, int32_t, float, float, float, float, InterpolationMode, int32_t>(),
                         py::arg("targetHeight"), py::arg("targetWidth"),
                         py::arg("scaleLb") = RandomCropAndResizeOp::kDefScaleLb,
                         py::arg("scaleUb") = RandomCropAndResizeOp::kDefScaleUb,
                         py::arg("aspectLb") = RandomCropAndResizeOp::kDefAspectLb,
                         py::arg("aspectUb") = RandomCropAndResizeOp::kDefAspectUb,
                         py::arg("interpolation") = RandomCropAndResizeOp::kDefInterpolation,
                         py::arg("maxIter") = RandomCropAndResizeOp::kDefMaxIter);
                }));

PYBIND_REGISTER(RandomRotationOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomRotationOp, TensorOp, std::shared_ptr<RandomRotationOp>>(
                    *m, "RandomRotationOp",
                    "Tensor operation to apply RandomRotation."
                    "Takes a range for degrees and "
                    "optional parameters for rotation center and image expand")
                    .def(
                      py::init<float, float, float, float, InterpolationMode, bool, uint8_t, uint8_t, uint8_t>(),
                      py::arg("startDegree"), py::arg("endDegree"), py::arg("centerX") = RandomRotationOp::kDefCenterX,
                      py::arg("centerY") = RandomRotationOp::kDefCenterY,
                      py::arg("interpolation") = RandomRotationOp::kDefInterpolation,
                      py::arg("expand") = RandomRotationOp::kDefExpand, py::arg("fillR") = RandomRotationOp::kDefFillR,
                      py::arg("fillG") = RandomRotationOp::kDefFillG, py::arg("fillB") = RandomRotationOp::kDefFillB);
                }));

PYBIND_REGISTER(RandomSharpnessOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomSharpnessOp, TensorOp, std::shared_ptr<RandomSharpnessOp>>(
                    *m, "RandomSharpnessOp",
                    "Tensor operation to apply RandomSharpness."
                    "Takes a range for degrees")
                    .def(py::init<float, float>(), py::arg("startDegree") = RandomSharpnessOp::kDefStartDegree,
                         py::arg("endDegree") = RandomSharpnessOp::kDefEndDegree);
                }));

PYBIND_REGISTER(RandomSelectSubpolicyOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomSelectSubpolicyOp, TensorOp, std::shared_ptr<RandomSelectSubpolicyOp>>(
                    *m, "RandomSelectSubpolicyOp")
                    .def(py::init([](const py::list &py_policy) {
                      std::vector<Subpolicy> cpp_policy;
                      for (auto &py_sub : py_policy) {
                        cpp_policy.push_back({});
                        for (auto handle : py_sub.cast<py::list>()) {
                          py::tuple tp = handle.cast<py::tuple>();
                          if (tp.is_none() || tp.size() != 2) {
                            THROW_IF_ERROR(
                              Status(StatusCode::kUnexpectedError, "Each tuple in subpolicy should be (op, prob)."));
                          }
                          std::shared_ptr<TensorOp> t_op;
                          if (py::isinstance<TensorOp>(tp[0])) {
                            t_op = (tp[0]).cast<std::shared_ptr<TensorOp>>();
                          } else if (py::isinstance<py::function>(tp[0])) {
                            t_op = std::make_shared<PyFuncOp>((tp[0]).cast<py::function>());
                          } else {
                            THROW_IF_ERROR(
                              Status(StatusCode::kUnexpectedError, "op is neither a tensorOp nor a pyfunc."));
                          }
                          double prob = (tp[1]).cast<py::float_>();
                          if (prob < 0 || prob > 1) {
                            THROW_IF_ERROR(Status(StatusCode::kUnexpectedError, "prob needs to be with [0,1]."));
                          }
                          cpp_policy.back().emplace_back(std::make_pair(t_op, prob));
                        }
                      }
                      return std::make_shared<RandomSelectSubpolicyOp>(cpp_policy);
                    }));
                }));
PYBIND_REGISTER(SoftDvppDecodeResizeJpegOp, 1, ([](const py::module *m) {
                  (void)py::class_<SoftDvppDecodeResizeJpegOp, TensorOp, std::shared_ptr<SoftDvppDecodeResizeJpegOp>>(
                    *m, "SoftDvppDecodeResizeJpegOp", "TensorOp to use soft dvpp decode and resize jpeg image.")
                    .def(py::init<int32_t, int32_t>(), py::arg("targetHeight"), py::arg("targetWidth"));
                }));
PYBIND_REGISTER(
  SoftDvppDecodeRandomCropResizeJpegOp, 1, ([](const py::module *m) {
    (void)
      py::class_<SoftDvppDecodeRandomCropResizeJpegOp, TensorOp, std::shared_ptr<SoftDvppDecodeRandomCropResizeJpegOp>>(
        *m, "SoftDvppDecodeRandomCropResizeJpegOp",
        "TensorOp to use soft dvpp decode, random crop and resize jepg image.")
        .def(py::init<int32_t, int32_t, float, float, float, float, int32_t>(), py::arg("targetHeight"),
             py::arg("targetWidth"), py::arg("scaleLb") = RandomCropDecodeResizeOp::kDefScaleLb,
             py::arg("scaleUb") = RandomCropDecodeResizeOp::kDefScaleUb,
             py::arg("aspectLb") = RandomCropDecodeResizeOp::kDefAspectLb,
             py::arg("aspectUb") = RandomCropDecodeResizeOp::kDefAspectUb,
             py::arg("maxIter") = RandomCropDecodeResizeOp::kDefMaxIter);
  }));

PYBIND_REGISTER(RandomSolarizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomSolarizeOp, TensorOp, std::shared_ptr<RandomSolarizeOp>>(*m,
                                                                                                  "RandomSolarizeOp")
                    .def(py::init<uint8_t, uint8_t>());
                }));

}  // namespace dataset
}  // namespace mindspore
