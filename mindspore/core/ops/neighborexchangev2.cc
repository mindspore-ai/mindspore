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
#include <set>
#include <string>
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kNeighborExchangeV2RecvRankIds = "recv_rank_ids";
constexpr auto kNeighborExchangeV2RecvLens = "recv_lens";
constexpr int64_t kNeighborExchangeV2InvalidIds = -1;
constexpr size_t kNeighborExchangeV2Idx0 = 0;
constexpr size_t kNeighborExchangeV2Idx1 = 1;
constexpr size_t kNeighborExchangeV2Idx2 = 2;
constexpr size_t kNeighborExchangeV2Idx3 = 3;
constexpr size_t kNeighborExchangeV2Idx4 = 4;
constexpr size_t kNeighborExchangeV2Idx5 = 5;
constexpr size_t kNeighborExchangeV2Idx6 = 6;
constexpr size_t kNeighborExchangeV2Idx7 = 7;

std::vector<int64_t> CheckAttrSize(const PrimitivePtr &primitive, const std::string &attr_name,
                                   const size_t attr_size) {
  MS_EXCEPTION_IF_NULL(primitive);
  // size of send/recv_rank_ids equal to size of send/recv_shapes
  std::vector<int64_t> attr_value;
  try {
    auto attr = primitive->GetAttr(attr_name);
    if (attr->cast<ValueListPtr>() == nullptr) {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', attr '" << attr_name
                              << "' must be a list and is necessary, but missing it.";
    }
    attr_value = GetValue<std::vector<int64_t>>(attr);
  } catch (const std::exception &) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', attr '" << attr_name
                            << "' must be a list[int, int, ...].";
  }

  if (attr_value.size() != attr_size) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', attr '" << attr_name
                             << "' size must be equal to attr_value size"
                             << ", but got attr " << attr_name << " size: " << attr_size
                             << ", attr_value size: " << attr_value.size() << ".";
  }

  return attr_value;
}

void CheckRecvCorner(const PrimitivePtr &primitive, std::vector<int64_t> recv_rank_ids, int64_t idx1, int64_t idx2,
                     int64_t idx_corner) {
  if (recv_rank_ids[idx1] != kNeighborExchangeV2InvalidIds && recv_rank_ids[idx2] != kNeighborExchangeV2InvalidIds &&
      recv_rank_ids[idx_corner] == kNeighborExchangeV2InvalidIds) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', recv_rank_ids[" << idx1 << "] or recv_rank_ids["
                             << idx2 << "] must be -1, or recv_rank_ids[" << idx_corner
                             << "] can not be -1. But got recv_rank_ids[" << idx1
                             << "]: " << recv_rank_ids[LongToSize(idx1)] << ", recv_rank_ids[" << idx2
                             << "]: " << recv_rank_ids[LongToSize(idx2)] << ", and recv_rank_ids[" << idx_corner
                             << "]: " << recv_rank_ids[LongToSize(idx_corner)] << ".";
  }
  if ((recv_rank_ids[idx1] == kNeighborExchangeV2InvalidIds || recv_rank_ids[idx2] == kNeighborExchangeV2InvalidIds) &&
      recv_rank_ids[idx_corner] != kNeighborExchangeV2InvalidIds) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', recv_rank_ids[" << idx1 << "] and recv_rank_ids["
                             << idx2 << "] can not be -1, or recv_rank_ids[" << idx_corner
                             << "] must be -1. But got recv_rank_ids[" << idx1
                             << "]: " << recv_rank_ids[LongToSize(idx1)] << ", recv_rank_ids[" << idx2
                             << "]: " << recv_rank_ids[LongToSize(idx2)] << ", and recv_rank_ids[" << idx_corner
                             << "]: " << recv_rank_ids[LongToSize(idx_corner)] << ".";
  }
}

void CheckIdsValue(const PrimitivePtr &primitive, std::vector<int64_t> rank_ids) {
  // check repeat & invalid value
  std::set<int64_t> ids_count;
  for (auto id : rank_ids) {
    if (id < 0 && id != kNeighborExchangeV2InvalidIds) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', all the rank id must be >= 0 or = -1, but got invalid id:" << id << ".";
    }
    if (ids_count.find(id) != ids_count.end() && id != -1) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', 'send_rank_ids' or 'recv_rank_ids' can not be repeated, but got id :" << id
                               << " repeated.";
    }
    (void)ids_count.insert(id);
  }
}

void CheckLensValue(const PrimitivePtr &primitive, std::vector<int64_t> lens) {
  // check len <0
  for (auto len : lens) {
    if (len < 0) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', 'send_lens' or 'recv_lens' must be >=0, but got invalid len:" << len << ".";
    }
  }
}

void NeighborExchangeV2Check(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, prim_name);

  // check size of send_rank_ids, recv_rank_ids, send_lens, recv_lens
  constexpr size_t kRankIdsSize = 8;
  constexpr size_t kLensSize = 4;
  constexpr auto kSendRankIds = "send_rank_ids";
  constexpr auto kSendLens = "send_lens";
  auto send_rank_ids = CheckAttrSize(primitive, kSendRankIds, kRankIdsSize);
  auto recv_rank_ids = CheckAttrSize(primitive, kNeighborExchangeV2RecvRankIds, kRankIdsSize);
  auto send_lens = CheckAttrSize(primitive, kSendLens, kLensSize);
  auto recv_lens = CheckAttrSize(primitive, kNeighborExchangeV2RecvLens, kLensSize);

  // check rank_ids value
  CheckIdsValue(primitive, send_rank_ids);
  CheckIdsValue(primitive, recv_rank_ids);
  // check lens value
  CheckLensValue(primitive, send_lens);
  CheckLensValue(primitive, recv_lens);

  // check recv rankids invalid cond
  CheckRecvCorner(primitive, recv_rank_ids, kNeighborExchangeV2Idx0, kNeighborExchangeV2Idx2, kNeighborExchangeV2Idx1);
  CheckRecvCorner(primitive, recv_rank_ids, kNeighborExchangeV2Idx2, kNeighborExchangeV2Idx4, kNeighborExchangeV2Idx3);
  CheckRecvCorner(primitive, recv_rank_ids, kNeighborExchangeV2Idx4, kNeighborExchangeV2Idx6, kNeighborExchangeV2Idx5);
  CheckRecvCorner(primitive, recv_rank_ids, kNeighborExchangeV2Idx6, kNeighborExchangeV2Idx0, kNeighborExchangeV2Idx7);

  // check data_format is NCHW
  constexpr auto kDataFormat = "format";
  auto format_attr = primitive->GetAttr(kDataFormat);
  string format = "";
  try {
    MS_EXCEPTION_IF_NULL(format_attr);
    format = GetValue<std::string>(format_attr);
  } catch (const std::exception &) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', attr '" << kDataFormat << "' must be a str.";
  }
  if (format != "NCHW") {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', attr 'data_format' only support NCHW now, but got "
                             << format << ".";
  }

  // check if send_lens > input_lens
  std::vector<int64_t> input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  constexpr size_t kInputSize = 4;
  if (input_shape.size() != kInputSize) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', input shape size must be 4, but got " << input_shape.size()
                             << ", and only support NCHW now.";
  }
  constexpr size_t kHDim = 2;
  constexpr size_t kWDim = 3;
  if (send_lens[kNeighborExchangeV2Idx0] > input_shape[kHDim]) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << "', attr send_lens[0] must be less than or equal to input size in H dim, but got send_lens[0]:"
      << send_lens[kNeighborExchangeV2Idx0] << ", input size in H dim: " << input_shape[kHDim] << ".";
  }
  if (send_lens[kNeighborExchangeV2Idx1] > input_shape[kHDim]) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << "', attr send_lens[1] must be less than or equal to input size in H dim, but got send_lens[1]: "
      << send_lens[kNeighborExchangeV2Idx1] << " , input size in H dim: " << input_shape[kHDim] << ".";
  }
  if (send_lens[kNeighborExchangeV2Idx2] > input_shape[kWDim]) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << "', attr send_lens[2] must be less than or equal to input size in W dim, but got send_lens[2]: "
      << send_lens[kNeighborExchangeV2Idx2] << ", input size in W dim: " << input_shape[kWDim] << ".";
  }
  if (send_lens[kNeighborExchangeV2Idx3] > input_shape[kWDim]) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << "', attr send_lens[3] must be less than or equal to input size in W dim, but got send_lens[3]: "
      << send_lens[kNeighborExchangeV2Idx3] << ", input size in W dim: " << input_shape[kWDim] << ".";
  }

  // check group
  constexpr auto kGroup = "group";
  auto group_attr = primitive->GetAttr(kGroup);
  try {
    MS_EXCEPTION_IF_NULL(group_attr);
    (void)GetValue<std::string>(group_attr);
  } catch (const std::exception &) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', attr '" << kGroup << "' must be a str.";
  }
}

abstract::BaseShapePtr NeighborExchangeV2InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto recv_rank_ids = primitive->GetAttr(kNeighborExchangeV2RecvRankIds);
  MS_EXCEPTION_IF_NULL(recv_rank_ids);
  auto recv_rank_ids_value = recv_rank_ids->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(recv_rank_ids_value);
  std::vector<int64_t> recv_rank_ids_v = GetValue<std::vector<int64_t>>(recv_rank_ids_value);
  auto recv_lens = primitive->GetAttr(kNeighborExchangeV2RecvLens);
  MS_EXCEPTION_IF_NULL(recv_lens);
  auto recv_lens_value = recv_lens->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(recv_lens_value);
  std::vector<int64_t> recv_lens_v = GetValue<std::vector<int64_t>>(recv_lens_value);

  std::vector<int64_t> input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (recv_rank_ids_v[kNeighborExchangeV2Idx0] != kNeighborExchangeV2InvalidIds) {
    input_shape[kNeighborExchangeV2Idx2] += recv_lens_v[kNeighborExchangeV2Idx0];
  }
  if (recv_rank_ids_v[kNeighborExchangeV2Idx4] != kNeighborExchangeV2InvalidIds) {
    input_shape[kNeighborExchangeV2Idx2] += recv_lens_v[kNeighborExchangeV2Idx1];
  }
  if (recv_rank_ids_v[kNeighborExchangeV2Idx6] != kNeighborExchangeV2InvalidIds) {
    input_shape[kNeighborExchangeV2Idx3] += recv_lens_v[kNeighborExchangeV2Idx2];
  }
  if (recv_rank_ids_v[kNeighborExchangeV2Idx2] != kNeighborExchangeV2InvalidIds) {
    input_shape[kNeighborExchangeV2Idx3] += recv_lens_v[kNeighborExchangeV2Idx3];
  }
  BaseShapePtr output_shape = std::make_shared<abstract::Shape>(input_shape);
  if (input_shape.empty()) {
    return std::make_shared<abstract::Shape>();
  }
  return output_shape;
}

TypePtr NeighborExchangeV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  // recv type
  TypePtr recv_type = input_args[0]->BuildType();
  if (recv_type == nullptr) {
    return std::make_shared<TypeNone>();
  }
  return recv_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(NeighborExchangeV2, BaseOperator);
AbstractBasePtr NeighborExchangeV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  NeighborExchangeV2Check(primitive, input_args);
  auto type = NeighborExchangeV2InferType(primitive, input_args);
  auto shape = NeighborExchangeV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(NeighborExchangeV2, prim::kPrimNeighborExchangeV2, NeighborExchangeV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
