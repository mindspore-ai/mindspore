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

#include "ops/crop_and_resize_grad_image.h"

#include <set>
#include <memory>

#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void CropAndResizeGradImage::Init(ResizeMethod method) { this->set_method(method); }

void CropAndResizeGradImage::set_method(ResizeMethod method) {
  auto swi = static_cast<int64_t>(method);
  (void)this->AddAttr(kMethod, api::MakeValue(swi));
}

ResizeMethod CropAndResizeGradImage::get_method() const {
  auto value_ptr = GetAttr(kMethod);
  return ResizeMethod(GetValue<int64_t>(value_ptr));
}

namespace {
constexpr size_t ImagekGrads = 0;
constexpr int64_t ImagekGradsShapeLen = 4;
constexpr size_t ImagekHeight = 1;
constexpr size_t ImagekWidth = 2;
constexpr size_t ImagekDepth = 3;
constexpr size_t ImagekImagesSize = 3;
constexpr int64_t ImagekImageSizeShapeLen = 1;
constexpr size_t ImagekBoxes = 1;
constexpr int64_t ImagekBoxesShapeLen = 2;
constexpr int64_t ImagekCoordinateLen = 4;
constexpr size_t ImagekBoxIndex = 2;
constexpr int64_t ImagekBoxIndShapeLen = 1;
constexpr size_t ImagekOutputSizeD = 1;
constexpr int64_t ImagekOutputSizeLen = 4;
constexpr int64_t ImageKMaxshapeDim0 = 16;
constexpr int64_t ImageKMaxshapeNum = 2;
abstract::ShapePtr CropAndResizeGradImageInferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[ImagekGrads]);
  auto input_shape0 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[ImagekGrads]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("grads rank", SizeToLong(input_shape0.size()), kEqual, ImagekGradsShapeLen,
                                           prim_name);
  MS_EXCEPTION_IF_NULL(input_args[ImagekBoxes]);
  auto input_shape1 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[ImagekBoxes]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("boxes rank", SizeToLong(input_shape1.size()), kEqual, ImagekBoxesShapeLen,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("shape[1] of boxes", input_shape1[1], kEqual, ImagekCoordinateLen,
                                           prim_name);
  MS_EXCEPTION_IF_NULL(input_args[ImagekBoxIndex]);
  auto input_shape2 = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[ImagekBoxIndex]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("box_index rank", SizeToLong(input_shape2.size()), kEqual,
                                           ImagekBoxIndShapeLen, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[ImagekImagesSize]);
  auto input_shape3 =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[ImagekImagesSize]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("image_size rank", SizeToLong(input_shape3.size()), kEqual,
                                           ImagekImageSizeShapeLen, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("length of image_size", input_shape3[0], kEqual, ImagekGradsShapeLen,
                                           prim_name);

  if (input_shape0[ImagekHeight] <= 0 || input_shape0[ImagekWidth] <= 0) {
    MS_EXCEPTION(ValueError) << "the height and width of grads must be over 0.";
  }
  if (input_shape0[ImagekDepth] <= 0) {
    MS_EXCEPTION(ValueError) << "the depth of grads must be over 0.";
  }
  if (input_shape0[0] != input_shape1[0] || input_shape2[0] != input_shape1[0]) {
    MS_EXCEPTION(ValueError) << "the first dimension of the tensor in {grads, boxes, box_index} must be equal.";
  }
  // Infer max shape of output
  bool gen_value_succ = false;
  std::vector<int64_t> output_size_value_vec(ImagekOutputSizeLen);
  auto output_size = input_args[ImagekImagesSize];
  MS_EXCEPTION_IF_NULL(output_size);
  auto dtype_value = primitive->GetAttr("T");
  auto output_type = dtype_value->cast<TypePtr>();
  auto type_size = GetTypeByte(output_type);
  auto max_Byte_ptr = primitive->GetAttr("max_Byte");
  MS_EXCEPTION_IF_NULL(max_Byte_ptr);
  const int64_t kMaxSize = GetValue<int64_t>(max_Byte_ptr);
  int64_t kMaxLen = 0;
  if (type_size > 0) {
    kMaxLen = kMaxSize / static_cast<int64_t>(type_size);
  } else {
    MS_EXCEPTION(ValueError) << "the value of T is incorrect.";
  }
  if (output_size->isa<abstract::AbstractTensor>()) {
    const std::set<TypePtr> output_size_valid_types = {kInt32};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("output_size dtype", output_size->BuildType(),
                                                     output_size_valid_types, prim_name);
    auto output_size_value = output_size->BuildValue();
    MS_EXCEPTION_IF_NULL(output_size_value);
    if (!output_size_value->isa<None>() && !output_size_value->isa<AnyValue>()) {
      auto output_size_tensor = output_size_value->cast<tensor::TensorPtr>();
      const std::vector<int64_t> const_output_size_shape = output_size_tensor->shape_c();
      if (const_output_size_shape.size() == ImagekOutputSizeD) {
        auto value = static_cast<int32_t *>(output_size_tensor->data_c());
        MS_EXCEPTION_IF_NULL(value);
        for (size_t i = 0; i < LongToSize(ImagekOutputSizeLen); ++i) {
          if (value[i] > 0) {
            if (value[i] > kMaxLen) {
              MS_EXCEPTION(ValueError) << "The value in output_size must be no more than max length: " << kMaxLen
                                       << ", but got " << value[i]
                                       << "! The value in output_size should be reduced or kMaxLen should be increased";
            }
            output_size_value_vec[i] = static_cast<int64_t>(value[i]);
          } else {
            MS_EXCEPTION(ValueError) << "CropAndResizeGradImage expected output_size to have "
                                        "positive data, but got "
                                     << value[i];
          }
        }
        gen_value_succ = true;
      }
    }
  }
  if (!gen_value_succ) {
    int64_t maxshape_dim0 = ImageKMaxshapeDim0;
    int64_t maxshape_dim1 = 0;
    int64_t maxshape_dim2 = 0;
    if (output_type == kFloat32) {
      maxshape_dim0 *= ImageKMaxshapeNum;
    }
    maxshape_dim2 = static_cast<int64_t>(sqrt(static_cast<double>(kMaxLen) / maxshape_dim0));
    maxshape_dim1 = maxshape_dim2 / input_shape0[ImagekDepth];
    ShapeVector output_shape = {abstract::Shape::SHP_ANY, abstract::Shape::SHP_ANY, abstract::Shape::SHP_ANY,
                                input_shape0[ImagekDepth]};
    ShapeVector shape_min = {1, 1, 1, input_shape0[ImagekDepth]};
    ShapeVector shape_max = {maxshape_dim0, maxshape_dim1, maxshape_dim2, input_shape0[ImagekDepth]};
    return std::make_shared<abstract::Shape>(output_shape, shape_min, shape_max);
  } else {
    return std::make_shared<abstract::Shape>(output_size_value_vec);
  }
}

TypePtr CropAndResizeGradImageInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t ImagekInputNums = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, ImagekInputNums, prim_name);
  const std::set<TypePtr> inputs_types = {kFloat32, kFloat64};
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grads", input_args[ImagekGrads]->BuildType(), inputs_types,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("boxes", input_args[ImagekBoxes]->BuildType(), inputs_types,
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("box_index", input_args[ImagekBoxIndex]->BuildType(), {kInt32},
                                                   prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("image_size", input_args[ImagekImagesSize]->BuildType(), {kInt32},
                                                   prim_name);
  auto out_T = prim->GetAttr("T")->cast<TypePtr>();
  (void)CheckAndConvertUtils::CheckSubClass("T", out_T, valid_types, prim_name);
  return out_T;
}
}  // namespace

MIND_API_OPERATOR_IMPL(CropAndResizeGradImage, BaseOperator);
AbstractBasePtr CropAndResizeGradImageInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto type = CropAndResizeGradImageInferType(primitive, input_args);
  auto shape = CropAndResizeGradImageInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(CropAndResizeGradImage, prim::kPrimCropAndResizeGradImage, CropAndResizeGradImageInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
