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
#include "ops/depth_to_space.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void DepthToSpace::set_block_size(const int64_t block_size) {
  CheckAndConvertUtils::Check(kBlockSize, block_size, kGreaterEqual, "", 2, this->name());
  (void)this->AddAttr(kBlockSize, MakeValue(block_size));
}

int64_t DepthToSpace::get_block_size() const { return GetValue<int64_t>(GetAttr(kBlockSize)); }
void DepthToSpace::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, MakeValue(f));
}

Format DepthToSpace::get_format() const { return Format(GetValue<int64_t>(GetAttr(kFormat))); }

void DepthToSpace::Init(const int64_t block_size, const Format &format) {
  this->set_block_size(block_size);
  this->set_format(format);
}

AbstractBasePtr DepthToSpaceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", int64_t(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_x = input_args[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_x);

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto format = Format(GetValue<int64_t>(primitive->GetAttr(kFormat)));
  if (format == NHWC) {
    x_shape = {x_shape[0], x_shape[3], x_shape[1], x_shape[2]};
  }
  const int64_t x_rank = 4;
  (void)CheckAndConvertUtils::CheckInteger("x rank", SizeToLong(x_shape.size()), kEqual, x_rank, prim_name);
  int64_t block_size = GetValue<int64_t>(primitive->GetAttr(kBlockSize));
  (void)CheckAndConvertUtils::CheckInteger("x_shape[1] % (block_size*block_size)",
                                           x_shape[1] % (block_size * block_size), kEqual, 0, prim_name);
  auto out_shape = x_shape;
  out_shape[1] /= block_size * block_size;
  out_shape[2] *= block_size;
  out_shape[3] *= block_size;
  if (format == NHWC) {
    out_shape = {out_shape[0], out_shape[2], out_shape[3], out_shape[1]};
  }
  auto ret = input_x->Broaden();
  ret->set_shape(std::make_shared<abstract::Shape>(out_shape));
  return ret;
}
REGISTER_PRIMITIVE_C(kNameDepthToSpace, DepthToSpace);
}  // namespace ops
}  // namespace mindspore
