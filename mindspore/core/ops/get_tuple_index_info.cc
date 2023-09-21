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

#include "ops/get_tuple_index_info.h"

#include <algorithm>
#include <memory>
#include <bitset>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/structure_ops.h"

namespace mindspore {
namespace ops {
static ShapeVector BroadCastShape(const ShapeVector &x_shape, const ShapeVector &y_shape) {
  if (x_shape == y_shape) {
    return x_shape;
  }
  const size_t x_len = x_shape.size();
  const size_t y_len = y_shape.size();
  const size_t min_length = std::min(x_len, y_len);
  ShapeVector broadcast_shape_back;
  for (size_t i = 0; i < min_length; i++) {
    size_t x_shape_index = x_len - min_length + i;
    size_t y_shape_index = y_len - min_length + i;
    if (x_shape[x_shape_index] == 1) {
      (void)broadcast_shape_back.emplace_back(y_shape[y_shape_index]);
    } else if (y_shape[y_shape_index] == 1 || x_shape[x_shape_index] == y_shape[y_shape_index]) {
      (void)broadcast_shape_back.emplace_back(x_shape[x_shape_index]);
    } else {
      MS_EXCEPTION(ValueError) << "For tensor getitem or setitem, x.shape and y.shape need to broadcast. "
                               << "The value of x.shape[" << std::to_string(x_shape_index) << "] or y.shape["
                               << std::to_string(y_shape_index) << "] must be 1 or -1 when they are not the same but "
                               << "got x.shape =" << x_shape << "and y.shape = " << y_shape;
    }
  }
  ShapeVector broadcast_shape_front;
  if (min_length == x_len) {
    (void)broadcast_shape_front.insert(
      broadcast_shape_front.end(), y_shape.begin(),
      y_shape.begin() + static_cast<int64_t>(y_len) - static_cast<int64_t>(min_length));
  } else {
    (void)broadcast_shape_front.insert(
      broadcast_shape_front.end(), x_shape.begin(),
      x_shape.begin() + static_cast<int64_t>(x_len) - static_cast<int64_t>(min_length));
  }
  (void)broadcast_shape_front.insert(broadcast_shape_front.end(), broadcast_shape_back.begin(),
                                     broadcast_shape_back.end());
  return broadcast_shape_front;
}

static ShapeVector BroadCastShape(const std::vector<ShapeVector> &tensor_indexes_shapes) {
  if (tensor_indexes_shapes.empty()) {
    return {};
  }
  return std::accumulate(tensor_indexes_shapes.begin(), tensor_indexes_shapes.end(), tensor_indexes_shapes[0],
                         [](const auto &output_shape, const auto &tensor_indexes_shape) {
                           return BroadCastShape(output_shape, tensor_indexes_shape);
                         });
}

static size_t GetFancyPosition(const std::vector<int64_t> &tuple_index_types, size_t fancy_position,
                               size_t ellipse_occupy_dims, const string &tuple_index_info_type) {
  std::vector<int64_t> final_tuple_index_types;
  for (size_t i = 0; i < tuple_index_types.size(); i++) {
    if (tuple_index_types[i] == kMetaTypeEllipsis) {
      auto ellipsis_slice = std::vector<int64_t>(ellipse_occupy_dims, kObjectTypeSlice);
      (void)final_tuple_index_types.insert(final_tuple_index_types.end(), ellipsis_slice.begin(), ellipsis_slice.end());
    } else {
      (void)final_tuple_index_types.emplace_back(tuple_index_types[i]);
    }
  }
  MS_LOG(DEBUG) << "final_tuple_index_types" << final_tuple_index_types;
  std::bitset<8> tensor_position_mask = 0;
  for (size_t i = 0; i < final_tuple_index_types.size(); i++) {
    if (final_tuple_index_types[i] == kObjectTypeTensorType) {
      tensor_position_mask[i] = 1;
    }
  }
  if (tuple_index_info_type != kSetitemByTuple && tuple_index_info_type != kSetitemByTupleWithTensor) {
    int64_t new_fancy_position = -1;
    if (tensor_position_mask == 0) {
      return 0;
    }
    const size_t max_tuple_nums = 8;
    for (size_t i = 0; i < max_tuple_nums; i++) {
      if (tensor_position_mask[i] == 0) {
        continue;
      }
      bool first_tensor_found = new_fancy_position != -1;
      if (first_tensor_found && tensor_position_mask[i - 1] == 0) {
        return 0;
      }
      if (!first_tensor_found) {
        new_fancy_position = i;
      }
    }
    return LongToSize(new_fancy_position);
  }
  return fancy_position;
}

static std::vector<ShapeVector> GetSliceShape(const std::vector<int64_t> &tuple_index_types,
                                              const std::vector<ShapeVector> &tensor_shapes, bool *has_zero_tensor,
                                              size_t *slice_nums, std::vector<ShapeVector> *tensor_indices_shapes) {
  std::vector<ShapeVector> slice_shapes;
  auto new_tuple_index_types = tuple_index_types;
  for (size_t i = 0; i < tuple_index_types.size(); i++) {
    if (new_tuple_index_types[i] == kMetaTypeEllipsis) {
      (void)new_tuple_index_types.erase(new_tuple_index_types.begin() + i);
      (void)new_tuple_index_types.emplace_back(kMetaTypeEllipsis);
      break;
    }
  }
  for (size_t i = 0; i < tensor_shapes.size(); i++) {
    if (new_tuple_index_types[i] == kObjectTypeTensorType) {
      if (!tensor_shapes[i].empty() && tensor_shapes[i][0] == 0) {
        *has_zero_tensor = true;
      }
      (void)tensor_indices_shapes->emplace_back(tensor_shapes[i]);
    } else {
      (void)slice_shapes.emplace_back(tensor_shapes[i]);
      *slice_nums = *slice_nums + 1;
    }
  }
  return slice_shapes;
}

static ShapeVector ComputeSliceShape(const ShapeVector &slice_shape, size_t broadcast_shape_len, size_t slice_cnt,
                                     int64_t fancy_position) {
  ShapeVector shape(slice_shape.size(), 1);
  if (slice_cnt < shape.size()) {
    shape[slice_cnt] = slice_shape[slice_cnt];
  }
  ShapeVector temp_shape(broadcast_shape_len, 1);
  (void)shape.insert(shape.begin() + fancy_position, temp_shape.begin(), temp_shape.end());
  return shape;
}

std::vector<ShapeVector> GetTupleIndexInfo::ConstGetTupleIndexInfo(
  const ShapeVector &data_shape, const std::vector<ShapeVector> &tensor_shapes,
  const std::vector<int64_t> &tuple_index_types, ShapeVector *broadcast_shape, ShapeVector *final_shape,
  ShapeVector *index_tensor_new_shape, size_t *fancy_position, const string &tuple_index_info_type) {
  // Get tuple index info: broadcast_shape
  size_t not_ellipse_occupy_dims = 0;
  for (size_t i = 0; i < tuple_index_types.size(); i++) {
    if (tuple_index_types[i] != kTypeUnknown && tuple_index_types[i] != kMetaTypeEllipsis) {
      not_ellipse_occupy_dims += 1;
    }
  }
  std::vector<ShapeVector> tensor_indices_shapes;
  std::vector<ShapeVector> slice_shapes;
  size_t slice_nums = 0;
  bool has_zero_tensor = false;
  slice_shapes = GetSliceShape(tuple_index_types, tensor_shapes, &has_zero_tensor, &slice_nums, &tensor_indices_shapes);
  MS_LOG(DEBUG) << "slice_shapes: " << slice_shapes;
  *broadcast_shape = BroadCastShape(tensor_indices_shapes);
  if (tuple_index_info_type == kSetitemByTupleWithTensor && broadcast_shape->size() < 2) {
    (void)broadcast_shape->insert(broadcast_shape->begin(), 1);
  }
  MS_LOG(DEBUG) << "broadcast_shape:" << *broadcast_shape;
  // Get tuple index info: fancy_position
  size_t ellipse_occupy_dims = tensor_shapes.size() - not_ellipse_occupy_dims;
  *fancy_position = GetFancyPosition(tuple_index_types, *fancy_position, ellipse_occupy_dims, tuple_index_info_type);
  MS_LOG(DEBUG) << "fancy_position:" << *fancy_position;
  if (tuple_index_info_type == kPreSetitemByTuple) {
    return {};
  }
  // Get tuple index info: final_shape
  size_t pre_size_len = 0;
  for (auto type : tuple_index_types) {
    if (type == kObjectTypeSlice) {
      pre_size_len += 1;
    } else if (type == kMetaTypeEllipsis) {
      break;
    }
  }
  ShapeVector slice_len;
  size_t not_ellipse_slice_cnt = slice_shapes.size() - ellipse_occupy_dims;
  std::transform(slice_shapes.begin(), slice_shapes.begin() + not_ellipse_slice_cnt, std::back_inserter(slice_len),
                 [](const ShapeVector &slice_shape) {
                   if (slice_shape.empty()) {
                     MS_LOG(EXCEPTION) << "Slice tensor can not be empty!";
                   }
                   return slice_shape[0];
                 });
  ShapeVector ellipse_slice;
  std::transform(slice_shapes.begin() + not_ellipse_slice_cnt, slice_shapes.end(), std::back_inserter(ellipse_slice),
                 [](const ShapeVector &slice_shape) {
                   if (slice_shape.empty()) {
                     MS_LOG(EXCEPTION) << "Slice tensor can not be empty!";
                   }
                   return slice_shape[0];
                 });
  (void)slice_len.insert(slice_len.begin() + pre_size_len, ellipse_slice.begin(), ellipse_slice.end());
  *fancy_position = std::min(*fancy_position, slice_nums);
  *final_shape = ShapeVector(slice_len.begin(), slice_len.begin() + slice_nums);
  (void)final_shape->insert(final_shape->begin() + *fancy_position, broadcast_shape->begin(), broadcast_shape->end());
  has_zero_tensor =
    has_zero_tensor || std::all_of(data_shape.begin(), data_shape.end(), [](int64_t dim) { return dim == 0; });
  if (has_zero_tensor && tensor_shapes.size() < data_shape.size()) {
    (void)final_shape->insert(final_shape->end(), data_shape.begin() + tensor_shapes.size(), data_shape.end());
  }
  MS_LOG(DEBUG) << "final_shape:" << *final_shape;
  // Get tuple index info: index_tensor_new_shape
  *index_tensor_new_shape = ShapeVector(slice_nums, 1);
  *fancy_position = std::min(*fancy_position, index_tensor_new_shape->size());
  (void)index_tensor_new_shape->insert(index_tensor_new_shape->begin() + *fancy_position, broadcast_shape->begin(),
                                       broadcast_shape->end());
  MS_LOG(DEBUG) << "index_tensor_new_shape:" << *index_tensor_new_shape;
  // Get tuple index info: new_slice_shapes
  std::vector<ShapeVector> new_slice_shapes;
  for (size_t i = 0; i < slice_nums; i++) {
    (void)new_slice_shapes.emplace_back(ComputeSliceShape(slice_len, broadcast_shape->size(), i, *fancy_position));
  }
  std::vector<ShapeVector> ellipse_slice_shape_vector(new_slice_shapes.begin() + pre_size_len,
                                                      new_slice_shapes.begin() + pre_size_len + ellipse_occupy_dims);
  (void)new_slice_shapes.erase(new_slice_shapes.begin() + pre_size_len,
                               new_slice_shapes.begin() + pre_size_len + ellipse_occupy_dims);
  (void)new_slice_shapes.insert(new_slice_shapes.end(), ellipse_slice_shape_vector.begin(),
                                ellipse_slice_shape_vector.end());
  MS_LOG(DEBUG) << "new_slice_shapes:" << new_slice_shapes;
  return new_slice_shapes;
}

static AbstractBasePtr VectorToAbstract(std::vector<int64_t> nums, bool to_tuple) {
  if (to_tuple) {
    abstract::AbstractBasePtrList elems;
    std::transform(nums.begin(), nums.end(), std::back_inserter(elems),
                   [](int64_t num) { return std::make_shared<abstract::AbstractScalar>(num); });
    return std::make_shared<abstract::AbstractTuple>(elems);
  }
  if (nums.empty()) {
    int64_t stub_num = 0;
    auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, ShapeVector{}, &stub_num, sizeof(int64_t));
    return tensor->ToAbstract();
  }
  ShapeVector tensor_shp({static_cast<int64_t>(nums.size())});
  auto shp_buf_size = sizeof(int64_t) * nums.size();
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, tensor_shp, nums.data(), shp_buf_size);
  return tensor->ToAbstract();
}

MIND_API_OPERATOR_IMPL(GetTupleIndexInfo, BaseOperator);

AbstractBasePtr GetTupleIndexInfoInferInner(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args, bool to_tuple) {
  MS_EXCEPTION_IF_NULL(primitive);
  ShapeVector data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex0]->BuildShape())[kShape];
  const AbstractBasePtr &fancy_position_abs = input_args[kIndex1];
  auto tuple_index_types = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrTupleIndexTypes));
  string tuple_index_info_type;
  if (primitive->HasAttr(kAttrTupleIndexInfoType)) {
    tuple_index_info_type = GetValue<string>(primitive->GetAttr(kAttrTupleIndexInfoType));
  }
  const size_t max_tensor_dims = 8;
  const size_t output_size = 13;
  if (fancy_position_abs->BuildType()->type_id() == kObjectTypeTensorType ||
      std::any_of(input_args.begin() + kIndex0, input_args.end(),
                  [](const AbstractBasePtr &shape_abs) { return shape_abs->BuildShape()->IsDynamic(); })) {
    auto abs = std::make_shared<abstract::AbstractTensor>(kInt64, ShapeVector({abstract::Shape::kShapeRankAny}));
    AbstractBasePtrList output_abs_list(output_size, abs);
    return std::make_shared<abstract::AbstractTuple>(output_abs_list);
  }
  if (data_shape.size() < 1 || data_shape.size() > max_tensor_dims) {
    MS_EXCEPTION(ValueError) << "The input data's dim must in the range of [1, 8], but got " << data_shape.size();
  }
  std::vector<ShapeVector> tensor_indices_shapes;
  ShapeVector slice_shapes;
  size_t valid_tensor_nums = 0;
  int64_t expand_dims = GetValue<int64_t>(primitive->GetAttr(kAttrExpandDimsCnt));
  for (size_t i = 0; i < tuple_index_types.size(); i++) {
    if (tuple_index_types[i] == kMetaTypeEllipsis) {
      valid_tensor_nums = data_shape.size() + expand_dims;
      break;
    } else if (tuple_index_types[i] != kTypeUnknown) {
      valid_tensor_nums += 1;
    }
  }
  for (size_t i = 0; i < valid_tensor_nums; i++) {
    auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[i + kIndex2]->BuildShape())[kShape];
    (void)tensor_indices_shapes.emplace_back(input_shape);
  }
  MS_LOG(DEBUG) << "valid_tensor_nums:" << valid_tensor_nums;
  ShapeVector broadcast_shape;
  ShapeVector final_shape;
  ShapeVector index_tensor_new_shape;
  int64_t fancy_position = GetValue<int64_t>(fancy_position_abs->BuildValue());
  auto new_slice_shapes = GetTupleIndexInfo::ConstGetTupleIndexInfo(
    data_shape, tensor_indices_shapes, tuple_index_types, &broadcast_shape, &final_shape, &index_tensor_new_shape,
    reinterpret_cast<size_t *>(&fancy_position), tuple_index_info_type);
  int64_t stub_num = 0;
  auto zero_dim_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, ShapeVector{1}, &stub_num, sizeof(int64_t));
  if (std::any_of(final_shape.begin(), final_shape.end(), [](auto dim) { return dim == 0; })) {
    zero_dim_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, final_shape, &stub_num, 0);
  }
  AbstractBasePtrList abs_list{
    VectorToAbstract(broadcast_shape, to_tuple), VectorToAbstract(index_tensor_new_shape, to_tuple),
    VectorToAbstract(final_shape, to_tuple), std::make_shared<abstract::AbstractScalar>(fancy_position),
    zero_dim_tensor->ToAbstract()};
  for (auto new_slice_shape : new_slice_shapes) {
    (void)abs_list.emplace_back(VectorToAbstract(new_slice_shape, to_tuple));
  }
  const size_t indices_size = final_shape.size();
  for (size_t i = 0; i < max_tensor_dims - new_slice_shapes.size(); i++) {
    ShapeVector shape(indices_size, 1);
    (void)abs_list.emplace_back(VectorToAbstract(shape, to_tuple));
  }
  auto output_abs = std::make_shared<abstract::AbstractTuple>(abs_list);
  return output_abs;
}

class MIND_API GetTupleIndexInfoInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return GetTupleIndexInfoInferInner(primitive, input_args, false)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return GetTupleIndexInfoInferInner(prim, input_args, true)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return GetTupleIndexInfoInferInner(primitive, input_args, true);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(GetTupleIndexInfo, prim::kPrimGetTupleIndexInfo, GetTupleIndexInfoInfer, false);
}  // namespace ops
}  // namespace mindspore
