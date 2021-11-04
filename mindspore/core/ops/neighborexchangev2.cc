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

#include "ops/neighborexchangev2.h"
#include <string>
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kSendRankIds = "send_rank_ids";
constexpr auto kSendLens = "send_lens";
constexpr auto kRecvRankIds = "recv_rank_ids";
constexpr auto kRecvLens = "recv_lens";
constexpr auto kDataFormat = "format";
constexpr auto kGroup = "group";
constexpr size_t kRankIdsSize = 8;
constexpr size_t kLensSize = 4;

std::vector<int64_t> CheckAttrSize(const PrimitivePtr &primitive, const std::string &attr_name,
                                   const size_t attr_size) {
  MS_EXCEPTION_IF_NULL(primitive);
  // size of send/recv_rank_ids equal to size of send/recv_shapes
  std::vector<int64_t> attr_value;
  try {
    auto attr = primitive->GetAttr(attr_name);
    if (attr->cast<ValueListPtr>() == nullptr) {
      MS_EXCEPTION(TypeError);
    }
    attr_value = GetValue<std::vector<int64_t>>(attr);
  } catch (const std::exception &) {
    MS_EXCEPTION(TypeError) << "Attr " << attr_name << " must be a list[int, int, ...].";
  }

  if (attr_value.size() != attr_size) {
    MS_EXCEPTION(ValueError) << "Invalid " << primitive->name() << " attr " << attr_name << " size "
                             << attr_value.size() << " must be equal to size " << attr_size;
  }
  return attr_value;
}

void CheckRecvCorner(std::vector<int64_t> recv_rank_ids, int64_t idx1, int64_t idx2, int64_t idx_corner) {
  if (recv_rank_ids[idx1] != -1 && recv_rank_ids[idx2] != -1 && recv_rank_ids[idx_corner] == -1) {
    MS_EXCEPTION(ValueError) << "Invalid recv_rank_ids, as recv_rank_ids[" << idx1 << "] = " << recv_rank_ids[idx1]
                             << ", recv_rank_ids[" << idx2 << "] = " << recv_rank_ids[idx2] << ", and recv_rank_ids["
                             << idx_corner << "] = " << recv_rank_ids[idx_corner] << ".";
  }
  if ((recv_rank_ids[idx1] == -1 || recv_rank_ids[idx2] == -1) && recv_rank_ids[idx_corner] != -1) {
    MS_EXCEPTION(ValueError) << "Invalid recv_rank_ids, as recv_rank_ids[" << idx1 << "] = " << recv_rank_ids[idx1]
                             << ", recv_rank_ids[" << idx2 << "] = " << recv_rank_ids[idx2] << ", and recv_rank_ids["
                             << idx_corner << "] = " << recv_rank_ids[idx_corner] << ".";
  }
}

void Check(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);

  // check size of send_rank_ids, recv_rank_ids, send_lens, recv_lens
  (void)CheckAttrSize(primitive, kSendRankIds, kRankIdsSize);
  auto recv_rank_ids = CheckAttrSize(primitive, kRecvRankIds, kRankIdsSize);
  (void)CheckAttrSize(primitive, kSendLens, kLensSize);
  (void)CheckAttrSize(primitive, kRecvLens, kLensSize);

  // check recv rankids invalid cond
  CheckRecvCorner(recv_rank_ids, 0, 2, 1);
  CheckRecvCorner(recv_rank_ids, 2, 4, 3);
  CheckRecvCorner(recv_rank_ids, 4, 6, 5);
  CheckRecvCorner(recv_rank_ids, 6, 0, 7);

  // check data_format is NCHW
  auto format = GetValue<std::string>(primitive->GetAttr(kDataFormat));
  if (format != "NCHW") {
    MS_EXCEPTION(ValueError) << "Attr data_format only support NCHW now.";
  }

  // check group
  auto group_attr = primitive->GetAttr(kGroup);
  try {
    MS_EXCEPTION_IF_NULL(group_attr);
    (void)GetValue<std::string>(group_attr);
  } catch (const std::exception &) {
    MS_EXCEPTION(TypeError) << "Attr " << kGroup << " should be a str.";
  }
}

abstract::BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto recv_rank_ids = primitive->GetAttr(kRecvRankIds);
  MS_EXCEPTION_IF_NULL(recv_rank_ids);
  auto recv_rank_ids_value = recv_rank_ids->cast<ValueSequeuePtr>();
  MS_EXCEPTION_IF_NULL(recv_rank_ids_value);
  std::vector<int64_t> recv_rank_ids_v = GetValue<std::vector<int64_t>>(recv_rank_ids_value);
  auto recv_lens = primitive->GetAttr(kRecvLens);
  MS_EXCEPTION_IF_NULL(recv_lens);
  auto recv_lens_value = recv_lens->cast<ValueSequeuePtr>();
  MS_EXCEPTION_IF_NULL(recv_lens_value);
  std::vector<int64_t> recv_lens_v = GetValue<std::vector<int64_t>>(recv_lens_value);

  std::vector<int64_t> input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (recv_rank_ids_v[0] != -1) {
    input_shape[2] += recv_lens_v[0];
  }
  if (recv_rank_ids_v[4] != -1) {
    input_shape[2] += recv_lens_v[1];
  }
  if (recv_rank_ids_v[6] != -1) {
    input_shape[3] += recv_lens_v[2];
  }
  if (recv_rank_ids_v[2] != -1) {
    input_shape[3] += recv_lens_v[3];
  }
  BaseShapePtr output_shape = std::make_shared<abstract::Shape>(input_shape);
  if (input_shape.empty()) {
    return std::make_shared<abstract::Shape>();
  }
  return output_shape;
}

TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  // recv type
  TypePtr recv_type = input_args[0]->BuildType();
  if (recv_type == nullptr) {
    return std::make_shared<TypeNone>();
  }
  return recv_type;
}
}  // namespace
AbstractBasePtr NeighborExchangeV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  Check(primitive, input_args);
  auto type = InferType(primitive, input_args);
  auto shape = InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(NeighborExchangeV2, prim::kPrimNeighborExchangeV2, NeighborExchangeV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
