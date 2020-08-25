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
#include "minddata/dataset/kernels/image/cutmix_batch_op.h"
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
#include "minddata/dataset/kernels/image/random_posterize_op.h"
#include "minddata/dataset/kernels/image/random_resize_op.h"
#include "minddata/dataset/kernels/image/random_resize_with_bbox_op.h"
#include "minddata/dataset/kernels/image/random_rotation_op.h"
#include "minddata/dataset/kernels/image/random_sharpness_op.h"
#include "minddata/dataset/kernels/image/random_select_subpolicy_op.h"
#include "minddata/dataset/kernels/image/random_solarize_op.h"
#include "minddata/dataset/kernels/image/random_vertical_flip_op.h"
#include "minddata/dataset/kernels/image/random_vertical_flip_with_bbox_op.h"
#include "minddata/dataset/kernels/image/rescale_op.h"
#include "minddata/dataset/kernels/image/resize_op.h"
#include "minddata/dataset/kernels/image/resize_with_bbox_op.h"
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_random_crop_resize_jpeg_op.h"
#include "minddata/dataset/kernels/image/soft_dvpp/soft_dvpp_decode_resize_jpeg_op.h"
#include "minddata/dataset/kernels/image/uniform_aug_op.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(AutoContrastOp, 1, ([](const py::module *m) {
                  (void)py::class_<AutoContrastOp, TensorOp, std::shared_ptr<AutoContrastOp>>(*m, "AutoContrastOp")
                    .def(py::init<float, std::vector<uint32_t>>());
                }));

PYBIND_REGISTER(NormalizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<NormalizeOp, TensorOp, std::shared_ptr<NormalizeOp>>(*m, "NormalizeOp")
                    .def(py::init<float, float, float, float, float, float>());
                }));

PYBIND_REGISTER(
  EqualizeOp, 1, ([](const py::module *m) {
    (void)py::class_<EqualizeOp, TensorOp, std::shared_ptr<EqualizeOp>>(*m, "EqualizeOp").def(py::init<>());
  }));

PYBIND_REGISTER(InvertOp, 1, ([](const py::module *m) {
                  (void)py::class_<InvertOp, TensorOp, std::shared_ptr<InvertOp>>(*m, "InvertOp").def(py::init<>());
                }));

PYBIND_REGISTER(
  RescaleOp, 1, ([](const py::module *m) {
    (void)py::class_<RescaleOp, TensorOp, std::shared_ptr<RescaleOp>>(*m, "RescaleOp").def(py::init<float, float>());
  }));

PYBIND_REGISTER(CenterCropOp, 1, ([](const py::module *m) {
                  (void)py::class_<CenterCropOp, TensorOp, std::shared_ptr<CenterCropOp>>(
                    *m, "CenterCropOp",
                    "Tensor operation to crop and image in the middle. Takes height and width (optional)")
                    .def(py::init<int32_t, int32_t>());
                }));

PYBIND_REGISTER(
  MixUpBatchOp, 1, ([](const py::module *m) {
    (void)py::class_<MixUpBatchOp, TensorOp, std::shared_ptr<MixUpBatchOp>>(*m, "MixUpBatchOp").def(py::init<float>());
  }));

PYBIND_REGISTER(CutMixBatchOp, 1, ([](const py::module *m) {
                  (void)py::class_<CutMixBatchOp, TensorOp, std::shared_ptr<CutMixBatchOp>>(
                    *m, "CutMixBatchOp", "Tensor operation to cutmix a batch of images")
                    .def(py::init<ImageBatchFormat, float, float>());
                }));

PYBIND_REGISTER(ResizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<ResizeOp, TensorOp, std::shared_ptr<ResizeOp>>(*m, "ResizeOp")
                    .def(py::init<int32_t, int32_t, InterpolationMode>());
                }));

PYBIND_REGISTER(ResizeWithBBoxOp, 1, ([](const py::module *m) {
                  (void)py::class_<ResizeWithBBoxOp, TensorOp, std::shared_ptr<ResizeWithBBoxOp>>(*m,
                                                                                                  "ResizeWithBBoxOp")
                    .def(py::init<int32_t, int32_t, InterpolationMode>());
                }));

//####
PYBIND_REGISTER(RandomAffineOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomAffineOp, TensorOp, std::shared_ptr<RandomAffineOp>>(
                    *m, "RandomAffineOp", "Tensor operation to apply random affine transformations on an image.")
                    .def(py::init<std::vector<float_t>, std::vector<float_t>, std::vector<float_t>,
                                  std::vector<float_t>, InterpolationMode, std::vector<uint8_t>>());
                }));

PYBIND_REGISTER(RandomResizeWithBBoxOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomResizeWithBBoxOp, TensorOp, std::shared_ptr<RandomResizeWithBBoxOp>>(
                    *m, "RandomResizeWithBBoxOp")
                    .def(py::init<int32_t, int32_t>());
                }));

PYBIND_REGISTER(RandomPosterizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomPosterizeOp, TensorOp, std::shared_ptr<RandomPosterizeOp>>(*m,
                                                                                                    "RandomPosterizeOp")
                    .def(py::init<std::vector<uint8_t>>());
                }));

PYBIND_REGISTER(UniformAugOp, 1, ([](const py::module *m) {
                  (void)py::class_<UniformAugOp, TensorOp, std::shared_ptr<UniformAugOp>>(*m, "UniformAugOp")
                    .def(py::init<std::vector<std::shared_ptr<TensorOp>>, int32_t>());
                }));

PYBIND_REGISTER(BoundingBoxAugmentOp, 1, ([](const py::module *m) {
                  (void)py::class_<BoundingBoxAugmentOp, TensorOp, std::shared_ptr<BoundingBoxAugmentOp>>(
                    *m, "BoundingBoxAugmentOp")
                    .def(py::init<std::shared_ptr<TensorOp>, float>());
                }));

PYBIND_REGISTER(DecodeOp, 1, ([](const py::module *m) {
                  (void)py::class_<DecodeOp, TensorOp, std::shared_ptr<DecodeOp>>(*m, "DecodeOp")
                    .def(py::init<>())
                    .def(py::init<bool>());
                }));

PYBIND_REGISTER(RandomHorizontalFlipOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomHorizontalFlipOp, TensorOp, std::shared_ptr<RandomHorizontalFlipOp>>(
                    *m, "RandomHorizontalFlipOp")
                    .def(py::init<float>());
                }));

PYBIND_REGISTER(
  RandomHorizontalFlipWithBBoxOp, 1, ([](const py::module *m) {
    (void)py::class_<RandomHorizontalFlipWithBBoxOp, TensorOp, std::shared_ptr<RandomHorizontalFlipWithBBoxOp>>(
      *m, "RandomHorizontalFlipWithBBoxOp")
      .def(py::init<float>());
  }));
PYBIND_REGISTER(RandomVerticalFlipOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomVerticalFlipOp, TensorOp, std::shared_ptr<RandomVerticalFlipOp>>(
                    *m, "RandomVerticalFlipOp")
                    .def(py::init<float>());
                }));
PYBIND_REGISTER(RandomVerticalFlipWithBBoxOp, 1, ([](const py::module *m) {
                  (void)
                    py::class_<RandomVerticalFlipWithBBoxOp, TensorOp, std::shared_ptr<RandomVerticalFlipWithBBoxOp>>(
                      *m, "RandomVerticalFlipWithBBoxOp")
                      .def(py::init<float>());
                }));
PYBIND_REGISTER(
  RandomCropOp, 1, ([](const py::module *m) {
    (void)py::class_<RandomCropOp, TensorOp, std::shared_ptr<RandomCropOp>>(*m, "RandomCropOp")
      .def(
        py::init<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, BorderType, bool, uint8_t, uint8_t, uint8_t>());
  }));

PYBIND_REGISTER(
  HwcToChwOp, 1, ([](const py::module *m) {
    (void)py::class_<HwcToChwOp, TensorOp, std::shared_ptr<HwcToChwOp>>(*m, "ChannelSwapOp").def(py::init<>());
  }));

PYBIND_REGISTER(
  RandomCropWithBBoxOp, 1, ([](const py::module *m) {
    (void)py::class_<RandomCropWithBBoxOp, TensorOp, std::shared_ptr<RandomCropWithBBoxOp>>(*m, "RandomCropWithBBoxOp")
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
                    .def(py::init<int32_t, int32_t, int32_t, bool, uint8_t, uint8_t, uint8_t>());
                }));

PYBIND_REGISTER(PadOp, 1, ([](const py::module *m) {
                  (void)py::class_<PadOp, TensorOp, std::shared_ptr<PadOp>>(*m, "PadOp")
                    .def(py::init<int32_t, int32_t, int32_t, int32_t, BorderType, uint8_t, uint8_t, uint8_t>());
                }));

PYBIND_REGISTER(RandomCropDecodeResizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomCropDecodeResizeOp, TensorOp, std::shared_ptr<RandomCropDecodeResizeOp>>(
                    *m, "RandomCropDecodeResizeOp")
                    .def(py::init<int32_t, int32_t, float, float, float, float, InterpolationMode, int32_t>());
                }));

PYBIND_REGISTER(RandomResizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomResizeOp, TensorOp, std::shared_ptr<RandomResizeOp>>(*m, "RandomResizeOp")
                    .def(py::init<int32_t, int32_t>());
                }));

PYBIND_REGISTER(RandomColorOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomColorOp, TensorOp, std::shared_ptr<RandomColorOp>>(*m, "RandomColorOp")
                    .def(py::init<float, float>());
                }));

PYBIND_REGISTER(RandomColorAdjustOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomColorAdjustOp, TensorOp, std::shared_ptr<RandomColorAdjustOp>>(
                    *m, "RandomColorAdjustOp")
                    .def(py::init<float, float, float, float, float, float, float, float>());
                }));

PYBIND_REGISTER(RandomCropAndResizeWithBBoxOp, 1, ([](const py::module *m) {
                  (void)
                    py::class_<RandomCropAndResizeWithBBoxOp, TensorOp, std::shared_ptr<RandomCropAndResizeWithBBoxOp>>(
                      *m, "RandomCropAndResizeWithBBoxOp")
                      .def(py::init<int32_t, int32_t, float, float, float, float, InterpolationMode, int32_t>());
                }));

PYBIND_REGISTER(RandomCropAndResizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomCropAndResizeOp, TensorOp, std::shared_ptr<RandomCropAndResizeOp>>(
                    *m, "RandomCropAndResizeOp")
                    .def(py::init<int32_t, int32_t, float, float, float, float, InterpolationMode, int32_t>());
                }));

PYBIND_REGISTER(RandomRotationOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomRotationOp, TensorOp, std::shared_ptr<RandomRotationOp>>(*m,
                                                                                                  "RandomRotationOp")
                    .def(py::init<float, float, float, float, InterpolationMode, bool, uint8_t, uint8_t, uint8_t>());
                }));

PYBIND_REGISTER(RandomSharpnessOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomSharpnessOp, TensorOp, std::shared_ptr<RandomSharpnessOp>>(*m,
                                                                                                    "RandomSharpnessOp")
                    .def(py::init<float, float>());
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
                    *m, "SoftDvppDecodeResizeJpegOp")
                    .def(py::init<int32_t, int32_t>());
                }));

PYBIND_REGISTER(
  SoftDvppDecodeRandomCropResizeJpegOp, 1, ([](const py::module *m) {
    (void)
      py::class_<SoftDvppDecodeRandomCropResizeJpegOp, TensorOp, std::shared_ptr<SoftDvppDecodeRandomCropResizeJpegOp>>(
        *m, "SoftDvppDecodeRandomCropResizeJpegOp")
        .def(py::init<int32_t, int32_t, float, float, float, float, int32_t>());
  }));

PYBIND_REGISTER(RandomSolarizeOp, 1, ([](const py::module *m) {
                  (void)py::class_<RandomSolarizeOp, TensorOp, std::shared_ptr<RandomSolarizeOp>>(*m,
                                                                                                  "RandomSolarizeOp")
                    .def(py::init<std::vector<uint8_t>>());
                }));

}  // namespace dataset
}  // namespace mindspore
