/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils//symbolic.h"
#include "ops/image_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
class MIND_API OCRRecognitionPreHandleInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    constexpr int64_t universe_max_batch = 256;
    constexpr int64_t image_h = 64;
    constexpr int64_t image_w = 512;
    constexpr int64_t images_max_batch = 256;
    constexpr int64_t images_channels = 3;
    ValuePtr format_value = primitive->GetAttr("format");
    std::string format = GetValue<std::string>(format_value);

    ShapeVector universe_max_shp = {universe_max_batch};
    (void)universe_max_shp.emplace_back(universe_max_batch);
    auto universe_shape = std::make_shared<abstract::TensorShape>(universe_max_shp);

    ShapeVector r_max_shp = {images_max_batch, image_h, image_w};
    if (format == "NHWC") {
      (void)r_max_shp.emplace(r_max_shp.end(), images_channels);
    } else {
      (void)r_max_shp.emplace(r_max_shp.begin() + 1, images_channels);
    }

    auto r_batched_shape = std::make_shared<abstract::TensorShape>(r_max_shp);

    abstract::BaseShapePtrList ret_shapes = {r_batched_shape, universe_shape, universe_shape, universe_shape};
    return std::make_shared<abstract::TupleShape>(ret_shapes);
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    CheckArgsSize(op_name, input_args, kSize5);
    return std::make_shared<Tuple>(TypePtrList{kUInt8, kInt32, kInt32, kInt32});
  }

  // This is used for frontend infer by abstract. If MakeAbstract support make env type abstract, InferShapeAndType can
  // be deleted.
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    constexpr size_t size_expected = 5;
    constexpr int64_t image_h = 64;
    constexpr int64_t image_w = 512;
    constexpr int64_t images_channels = 3;
    CheckArgsSize(op_name, input_args, size_expected);
    ValuePtr format_value = primitive->GetAttr("format");
    std::string format = GetValue<std::string>(format_value);

    ShapeVector universe_shp;
    (void)universe_shp.emplace_back(abstract::TensorShape::kShapeDimAny);
    auto universe_abstract =
      std::make_shared<abstract::AbstractTensor>(kInt32, std::make_shared<abstract::TensorShape>(universe_shp));

    ShapeVector r_shp = {abstract::TensorShape::kShapeDimAny, image_h, image_w};
    if (format == "NHWC") {
      (void)r_shp.emplace(r_shp.end(), images_channels);
    } else {
      (void)r_shp.emplace(r_shp.begin() + 1, images_channels);
    }
    auto r_batched_abstract =
      std::make_shared<abstract::AbstractTensor>(kUInt8, std::make_shared<abstract::TensorShape>(r_shp));

    AbstractBasePtrList elements = {r_batched_abstract, universe_abstract, universe_abstract, universe_abstract};
    return std::make_shared<abstract::AbstractTuple>(elements);
  }
};

class MIND_API OCRRecognitionPreHandle : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(OCRRecognitionPreHandle);
  /// \brief Constructor.
  OCRRecognitionPreHandle() : BaseOperator("OCRRecognitionPreHandle") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(OCRRecognitionPreHandle, prim::kPrimOCRRecognitionPreHandle,
                                 OCRRecognitionPreHandleInfer, false);
}  // namespace ops
}  // namespace mindspore
