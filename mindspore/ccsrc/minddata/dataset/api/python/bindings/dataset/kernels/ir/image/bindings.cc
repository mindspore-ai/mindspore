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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/kernels/ir/vision/vision_ir.h"

namespace mindspore {
namespace dataset {

PYBIND_REGISTER(
  AutoContrastOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::AutoContrastOperation, TensorOperation, std::shared_ptr<vision::AutoContrastOperation>>(
      *m, "AutoContrastOperation")
      .def(py::init([](float cutoff, std::vector<uint32_t> ignore) {
        auto auto_contrast = std::make_shared<vision::AutoContrastOperation>(cutoff, ignore);
        THROW_IF_ERROR(auto_contrast->ValidateParams());
        return auto_contrast;
      }));
  }));

PYBIND_REGISTER(BoundingBoxAugmentOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::BoundingBoxAugmentOperation, TensorOperation,
                                   std::shared_ptr<vision::BoundingBoxAugmentOperation>>(*m,
                                                                                         "BoundingBoxAugmentOperation")
                    .def(py::init([](py::object transform, float ratio) {
                      auto bounding_box_augment = std::make_shared<vision::BoundingBoxAugmentOperation>(
                        std::move(toTensorOperation(transform)), ratio);
                      THROW_IF_ERROR(bounding_box_augment->ValidateParams());
                      return bounding_box_augment;
                    }));
                }));

PYBIND_REGISTER(
  CenterCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::CenterCropOperation, TensorOperation, std::shared_ptr<vision::CenterCropOperation>>(
      *m, "CenterCropOperation", "Tensor operation to crop and image in the middle. Takes height and width (optional)")
      .def(py::init([](std::vector<int32_t> size) {
        auto center_crop = std::make_shared<vision::CenterCropOperation>(size);
        THROW_IF_ERROR(center_crop->ValidateParams());
        return center_crop;
      }));
  }));

PYBIND_REGISTER(
  CutMixBatchOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::CutMixBatchOperation, TensorOperation, std::shared_ptr<vision::CutMixBatchOperation>>(
      *m, "CutMixBatchOperation", "Tensor operation to cutmix a batch of images")
      .def(py::init([](ImageBatchFormat image_batch_format, float alpha, float prob) {
        auto cut_mix_batch = std::make_shared<vision::CutMixBatchOperation>(image_batch_format, alpha, prob);
        THROW_IF_ERROR(cut_mix_batch->ValidateParams());
        return cut_mix_batch;
      }));
  }));

PYBIND_REGISTER(CutOutOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::CutOutOperation, TensorOperation, std::shared_ptr<vision::CutOutOperation>>(
                    *m, "CutOutOperation",
                    "Tensor operation to randomly erase a portion of the image. Takes height and width.")
                    .def(py::init([](int32_t length, int32_t num_patches) {
                      auto cut_out = std::make_shared<vision::CutOutOperation>(length, num_patches);
                      THROW_IF_ERROR(cut_out->ValidateParams());
                      return cut_out;
                    }));
                }));

PYBIND_REGISTER(DecodeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::DecodeOperation, TensorOperation, std::shared_ptr<vision::DecodeOperation>>(
                    *m, "DecodeOperation")
                    .def(py::init([](bool rgb) {
                      auto decode = std::make_shared<vision::DecodeOperation>(rgb);
                      THROW_IF_ERROR(decode->ValidateParams());
                      return decode;
                    }))
                    .def(py::init([](bool rgb) {
                      auto decode = std::make_shared<vision::DecodeOperation>(rgb);
                      THROW_IF_ERROR(decode->ValidateParams());
                      return decode;
                    }));
                }));

PYBIND_REGISTER(EqualizeOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::EqualizeOperation, TensorOperation, std::shared_ptr<vision::EqualizeOperation>>(
                      *m, "EqualizeOperation")
                      .def(py::init([]() {
                        auto equalize = std::make_shared<vision::EqualizeOperation>();
                        THROW_IF_ERROR(equalize->ValidateParams());
                        return equalize;
                      }));
                }));

PYBIND_REGISTER(HwcToChwOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::HwcToChwOperation, TensorOperation, std::shared_ptr<vision::HwcToChwOperation>>(
                      *m, "HwcToChwOperation")
                      .def(py::init([]() {
                        auto hwc_to_chw = std::make_shared<vision::HwcToChwOperation>();
                        THROW_IF_ERROR(hwc_to_chw->ValidateParams());
                        return hwc_to_chw;
                      }));
                }));

PYBIND_REGISTER(InvertOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::InvertOperation, TensorOperation, std::shared_ptr<vision::InvertOperation>>(
                    *m, "InvertOperation")
                    .def(py::init([]() {
                      auto invert = std::make_shared<vision::InvertOperation>();
                      THROW_IF_ERROR(invert->ValidateParams());
                      return invert;
                    }));
                }));

PYBIND_REGISTER(
  MixUpBatchOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::MixUpBatchOperation, TensorOperation, std::shared_ptr<vision::MixUpBatchOperation>>(
      *m, "MixUpBatchOperation")
      .def(py::init([](float alpha) {
        auto mix_up_batch = std::make_shared<vision::MixUpBatchOperation>(alpha);
        THROW_IF_ERROR(mix_up_batch->ValidateParams());
        return mix_up_batch;
      }));
  }));

PYBIND_REGISTER(
  NormalizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NormalizeOperation, TensorOperation, std::shared_ptr<vision::NormalizeOperation>>(
      *m, "NormalizeOperation")
      .def(py::init([](std::vector<float> mean, std::vector<float> std) {
        auto normalize = std::make_shared<vision::NormalizeOperation>(mean, std);
        THROW_IF_ERROR(normalize->ValidateParams());
        return normalize;
      }));
  }));

PYBIND_REGISTER(
  NormalizePadOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NormalizePadOperation, TensorOperation, std::shared_ptr<vision::NormalizePadOperation>>(
      *m, "NormalizePadOperation")
      .def(py::init([](const std::vector<float> &mean, const std::vector<float> &std, const std::string &dtype) {
        auto normalize_pad = std::make_shared<vision::NormalizePadOperation>(mean, std, dtype);
        THROW_IF_ERROR(normalize_pad->ValidateParams());
        return normalize_pad;
      }));
  }));

PYBIND_REGISTER(PadOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::PadOperation, TensorOperation, std::shared_ptr<vision::PadOperation>>(
                    *m, "PadOperation")
                    .def(py::init(
                      [](std::vector<int32_t> padding, std::vector<uint8_t> fill_value, BorderType padding_mode) {
                        auto pad = std::make_shared<vision::PadOperation>(padding, fill_value, padding_mode);
                        THROW_IF_ERROR(pad->ValidateParams());
                        return pad;
                      }));
                }));

PYBIND_REGISTER(
  RandomAffineOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomAffineOperation, TensorOperation, std::shared_ptr<vision::RandomAffineOperation>>(
      *m, "RandomAffineOperation", "Tensor operation to apply random affine transformations on an image.")
      .def(py::init([](const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range,
                       const std::vector<float_t> &scale_range, const std::vector<float_t> &shear_ranges,
                       InterpolationMode interpolation, const std::vector<uint8_t> &fill_value) {
        auto random_affine = std::make_shared<vision::RandomAffineOperation>(degrees, translate_range, scale_range,
                                                                             shear_ranges, interpolation, fill_value);
        THROW_IF_ERROR(random_affine->ValidateParams());
        return random_affine;
      }));
  }));

PYBIND_REGISTER(RandomColorAdjustOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomColorAdjustOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomColorAdjustOperation>>(*m,
                                                                                        "RandomColorAdjustOperation")
                    .def(py::init([](std::vector<float> brightness, std::vector<float> contrast,
                                     std::vector<float> saturation, std::vector<float> hue) {
                      auto random_color_adjust =
                        std::make_shared<vision::RandomColorAdjustOperation>(brightness, contrast, saturation, hue);
                      THROW_IF_ERROR(random_color_adjust->ValidateParams());
                      return random_color_adjust;
                    }));
                }));

PYBIND_REGISTER(
  RandomColorOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomColorOperation, TensorOperation, std::shared_ptr<vision::RandomColorOperation>>(
      *m, "RandomColorOperation")
      .def(py::init([](float t_lb, float t_ub) {
        auto random_color = std::make_shared<vision::RandomColorOperation>(t_lb, t_ub);
        THROW_IF_ERROR(random_color->ValidateParams());
        return random_color;
      }));
  }));

PYBIND_REGISTER(RandomCropDecodeResizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomCropDecodeResizeOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomCropDecodeResizeOperation>>(
                    *m, "RandomCropDecodeResizeOperation")
                    .def(py::init([](std::vector<int32_t> size, std::vector<float> scale, std::vector<float> ratio,
                                     InterpolationMode interpolation, int32_t max_attempts) {
                      auto random_crop_decode_resize = std::make_shared<vision::RandomCropDecodeResizeOperation>(
                        size, scale, ratio, interpolation, max_attempts);
                      THROW_IF_ERROR(random_crop_decode_resize->ValidateParams());
                      return random_crop_decode_resize;
                    }));
                }));

PYBIND_REGISTER(
  RandomCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomCropOperation, TensorOperation, std::shared_ptr<vision::RandomCropOperation>>(
      *m, "RandomCropOperation")
      .def(py::init([](std::vector<int32_t> size, std::vector<int32_t> padding, bool pad_if_needed,
                       std::vector<uint8_t> fill_value, BorderType padding_mode) {
        auto random_crop =
          std::make_shared<vision::RandomCropOperation>(size, padding, pad_if_needed, fill_value, padding_mode);
        THROW_IF_ERROR(random_crop->ValidateParams());
        return random_crop;
      }));
  }));

PYBIND_REGISTER(RandomCropWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomCropWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomCropWithBBoxOperation>>(*m,
                                                                                         "RandomCropWithBBoxOperation")
                    .def(py::init([](std::vector<int32_t> size, std::vector<int32_t> padding, bool pad_if_needed,
                                     std::vector<uint8_t> fill_value, BorderType padding_mode) {
                      auto random_crop_with_bbox = std::make_shared<vision::RandomCropWithBBoxOperation>(
                        size, padding, pad_if_needed, fill_value, padding_mode);
                      THROW_IF_ERROR(random_crop_with_bbox->ValidateParams());
                      return random_crop_with_bbox;
                    }));
                }));

PYBIND_REGISTER(RandomHorizontalFlipOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomHorizontalFlipOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomHorizontalFlipOperation>>(
                    *m, "RandomHorizontalFlipOperation")
                    .def(py::init([](float prob) {
                      auto random_horizontal_flip = std::make_shared<vision::RandomHorizontalFlipOperation>(prob);
                      THROW_IF_ERROR(random_horizontal_flip->ValidateParams());
                      return random_horizontal_flip;
                    }));
                }));

PYBIND_REGISTER(RandomHorizontalFlipWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomHorizontalFlipWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomHorizontalFlipWithBBoxOperation>>(
                    *m, "RandomHorizontalFlipWithBBoxOperation")
                    .def(py::init([](float prob) {
                      auto random_horizontal_flip_with_bbox =
                        std::make_shared<vision::RandomHorizontalFlipWithBBoxOperation>(prob);
                      THROW_IF_ERROR(random_horizontal_flip_with_bbox->ValidateParams());
                      return random_horizontal_flip_with_bbox;
                    }));
                }));

PYBIND_REGISTER(RandomPosterizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomPosterizeOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomPosterizeOperation>>(*m, "RandomPosterizeOperation")
                    .def(py::init([](const std::vector<uint8_t> &bit_range) {
                      auto random_posterize = std::make_shared<vision::RandomPosterizeOperation>(bit_range);
                      THROW_IF_ERROR(random_posterize->ValidateParams());
                      return random_posterize;
                    }));
                }));

PYBIND_REGISTER(RandomResizedCropOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomResizedCropOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomResizedCropOperation>>(*m,
                                                                                        "RandomResizedCropOperation")
                    .def(py::init([](std::vector<int32_t> size, std::vector<float> scale, std::vector<float> ratio,
                                     InterpolationMode interpolation, int32_t max_attempts) {
                      auto random_resized_crop = std::make_shared<vision::RandomResizedCropOperation>(
                        size, scale, ratio, interpolation, max_attempts);
                      THROW_IF_ERROR(random_resized_crop->ValidateParams());
                      return random_resized_crop;
                    }));
                }));

PYBIND_REGISTER(RandomResizedCropWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomResizedCropWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomResizedCropWithBBoxOperation>>(
                    *m, "RandomResizedCropWithBBoxOperation")
                    .def(py::init([](std::vector<int32_t> size, std::vector<float> scale, std::vector<float> ratio,
                                     InterpolationMode interpolation, int32_t max_attempts) {
                      auto random_resized_crop_with_bbox = std::make_shared<vision::RandomResizedCropWithBBoxOperation>(
                        size, scale, ratio, interpolation, max_attempts);
                      THROW_IF_ERROR(random_resized_crop_with_bbox->ValidateParams());
                      return random_resized_crop_with_bbox;
                    }));
                }));

PYBIND_REGISTER(
  RandomResizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomResizeOperation, TensorOperation, std::shared_ptr<vision::RandomResizeOperation>>(
      *m, "RandomResizeOperation")
      .def(py::init([](std::vector<int32_t> size) {
        auto random_resize = std::make_shared<vision::RandomResizeOperation>(size);
        THROW_IF_ERROR(random_resize->ValidateParams());
        return random_resize;
      }));
  }));

PYBIND_REGISTER(RandomResizeWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomResizeWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomResizeWithBBoxOperation>>(
                    *m, "RandomResizeWithBBoxOperation")
                    .def(py::init([](std::vector<int32_t> size) {
                      auto random_resize_with_bbox = std::make_shared<vision::RandomResizeWithBBoxOperation>(size);
                      THROW_IF_ERROR(random_resize_with_bbox->ValidateParams());
                      return random_resize_with_bbox;
                    }));
                }));

PYBIND_REGISTER(RandomRotationOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomRotationOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomRotationOperation>>(*m, "RandomRotationOperation")
                    .def(py::init([](std::vector<float> degrees, InterpolationMode interpolation_mode, bool expand,
                                     std::vector<float> center, std::vector<uint8_t> fill_value) {
                      auto random_rotation = std::make_shared<vision::RandomRotationOperation>(
                        degrees, interpolation_mode, expand, center, fill_value);
                      THROW_IF_ERROR(random_rotation->ValidateParams());
                      return random_rotation;
                    }));
                }));

PYBIND_REGISTER(
  RandomSelectSubpolicyOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomSelectSubpolicyOperation, TensorOperation,
                     std::shared_ptr<vision::RandomSelectSubpolicyOperation>>(*m, "RandomSelectSubpolicyOperation")
      .def(py::init([](const py::list &py_policy) {
        std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> cpp_policy;
        for (auto &py_sub : py_policy) {
          cpp_policy.push_back({});
          for (auto handle : py_sub.cast<py::list>()) {
            py::tuple tp = handle.cast<py::tuple>();
            if (tp.is_none() || tp.size() != 2) {
              THROW_IF_ERROR(Status(StatusCode::kMDUnexpectedError, "Each tuple in subpolicy should be (op, prob)."));
            }
            std::shared_ptr<TensorOperation> t_op;
            if (py::isinstance<TensorOperation>(tp[0])) {
              t_op = (tp[0]).cast<std::shared_ptr<TensorOperation>>();
            } else if (py::isinstance<TensorOp>(tp[0])) {
              t_op = std::make_shared<transforms::PreBuiltOperation>((tp[0]).cast<std::shared_ptr<TensorOp>>());
            } else if (py::isinstance<py::function>(tp[0])) {
              t_op = std::make_shared<transforms::PreBuiltOperation>(
                std::make_shared<PyFuncOp>((tp[0]).cast<py::function>()));
            } else {
              THROW_IF_ERROR(
                Status(StatusCode::kMDUnexpectedError, "op is neither a tensorOp, tensorOperation nor a pyfunc."));
            }
            double prob = (tp[1]).cast<py::float_>();
            if (prob < 0 || prob > 1) {
              THROW_IF_ERROR(Status(StatusCode::kMDUnexpectedError, "prob needs to be with [0,1]."));
            }
            cpp_policy.back().emplace_back(std::make_pair(t_op, prob));
          }
        }
        auto random_select_subpolicy = std::make_shared<vision::RandomSelectSubpolicyOperation>(cpp_policy);
        THROW_IF_ERROR(random_select_subpolicy->ValidateParams());
        return random_select_subpolicy;
      }));
  }));

PYBIND_REGISTER(RandomSharpnessOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomSharpnessOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomSharpnessOperation>>(*m, "RandomSharpnessOperation")
                    .def(py::init([](std::vector<float> degrees) {
                      auto random_sharpness = std::make_shared<vision::RandomSharpnessOperation>(degrees);
                      THROW_IF_ERROR(random_sharpness->ValidateParams());
                      return random_sharpness;
                    }));
                }));

PYBIND_REGISTER(RandomSolarizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomSolarizeOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomSolarizeOperation>>(*m, "RandomSolarizeOperation")
                    .def(py::init([](std::vector<uint8_t> threshold) {
                      auto random_solarize = std::make_shared<vision::RandomSolarizeOperation>(threshold);
                      THROW_IF_ERROR(random_solarize->ValidateParams());
                      return random_solarize;
                    }));
                }));

PYBIND_REGISTER(RandomVerticalFlipOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomVerticalFlipOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomVerticalFlipOperation>>(*m,
                                                                                         "RandomVerticalFlipOperation")
                    .def(py::init([](float prob) {
                      auto random_vertical_flip = std::make_shared<vision::RandomVerticalFlipOperation>(prob);
                      THROW_IF_ERROR(random_vertical_flip->ValidateParams());
                      return random_vertical_flip;
                    }));
                }));

PYBIND_REGISTER(RandomVerticalFlipWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomVerticalFlipWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomVerticalFlipWithBBoxOperation>>(
                    *m, "RandomVerticalFlipWithBBoxOperation")
                    .def(py::init([](float prob) {
                      auto random_vertical_flip_with_bbox =
                        std::make_shared<vision::RandomVerticalFlipWithBBoxOperation>(prob);
                      THROW_IF_ERROR(random_vertical_flip_with_bbox->ValidateParams());
                      return random_vertical_flip_with_bbox;
                    }));
                }));

PYBIND_REGISTER(RescaleOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::RescaleOperation, TensorOperation, std::shared_ptr<vision::RescaleOperation>>(
                      *m, "RescaleOperation")
                      .def(py::init([](float rescale, float shift) {
                        auto rescale_op = std::make_shared<vision::RescaleOperation>(rescale, shift);
                        THROW_IF_ERROR(rescale_op->ValidateParams());
                        return rescale_op;
                      }));
                }));

PYBIND_REGISTER(ResizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::ResizeOperation, TensorOperation, std::shared_ptr<vision::ResizeOperation>>(
                    *m, "ResizeOperation")
                    .def(py::init([](std::vector<int32_t> size, InterpolationMode interpolation_mode) {
                      auto resize = std::make_shared<vision::ResizeOperation>(size, interpolation_mode);
                      THROW_IF_ERROR(resize->ValidateParams());
                      return resize;
                    }));
                }));

PYBIND_REGISTER(ResizeWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::ResizeWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::ResizeWithBBoxOperation>>(*m, "ResizeWithBBoxOperation")
                    .def(py::init([](std::vector<int32_t> size, InterpolationMode interpolation_mode) {
                      auto resize_with_bbox =
                        std::make_shared<vision::ResizeWithBBoxOperation>(size, interpolation_mode);
                      THROW_IF_ERROR(resize_with_bbox->ValidateParams());
                      return resize_with_bbox;
                    }));
                }));

PYBIND_REGISTER(SoftDvppDecodeRandomCropResizeJpegOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::SoftDvppDecodeRandomCropResizeJpegOperation, TensorOperation,
                                   std::shared_ptr<vision::SoftDvppDecodeRandomCropResizeJpegOperation>>(
                    *m, "SoftDvppDecodeRandomCropResizeJpegOperation")
                    .def(py::init([](std::vector<int32_t> size, std::vector<float> scale, std::vector<float> ratio,
                                     int32_t max_attempts) {
                      auto soft_dvpp_decode_random_crop_resize_jpeg =
                        std::make_shared<vision::SoftDvppDecodeRandomCropResizeJpegOperation>(size, scale, ratio,
                                                                                              max_attempts);
                      THROW_IF_ERROR(soft_dvpp_decode_random_crop_resize_jpeg->ValidateParams());
                      return soft_dvpp_decode_random_crop_resize_jpeg;
                    }));
                }));

PYBIND_REGISTER(SoftDvppDecodeResizeJpegOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::SoftDvppDecodeResizeJpegOperation, TensorOperation,
                                   std::shared_ptr<vision::SoftDvppDecodeResizeJpegOperation>>(
                    *m, "SoftDvppDecodeResizeJpegOperation")
                    .def(py::init([](std::vector<int32_t> size) {
                      auto soft_dvpp_decode_resize_jpeg =
                        std::make_shared<vision::SoftDvppDecodeResizeJpegOperation>(size);
                      THROW_IF_ERROR(soft_dvpp_decode_resize_jpeg->ValidateParams());
                      return soft_dvpp_decode_resize_jpeg;
                    }));
                }));

PYBIND_REGISTER(
  UniformAugOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::UniformAugOperation, TensorOperation, std::shared_ptr<vision::UniformAugOperation>>(
      *m, "UniformAugOperation")
      .def(py::init([](py::list transforms, int32_t num_ops) {
        auto uniform_aug =
          std::make_shared<vision::UniformAugOperation>(std::move(toTensorOperations(transforms)), num_ops);
        THROW_IF_ERROR(uniform_aug->ValidateParams());
        return uniform_aug;
      }));
  }));
}  // namespace dataset
}  // namespace mindspore
