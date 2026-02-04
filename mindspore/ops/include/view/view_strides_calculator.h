/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_OPS_INCLUDE_VIEW_VIEW_STRIDESCALCULATOR_H_
#define MINDSPORE_OPS_INCLUDE_VIEW_VIEW_STRIDESCALCULATOR_H_

#include <string>
#include <memory>
#include <vector>
#include <optional>
#include <utility>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <any>

#include "ir/tensor.h"
#include "utils/hash_map.h"
#include "primitive/op_name.h"
#include "ir/primitive.h"

namespace mindspore {
namespace ops {
using TensorStorageInfoPtrList = std::vector<TensorStorageInfoPtr>;
// unsupported will return {}
using StridesCalcFunc = std::function<TensorStorageInfoPtrList(const PrimitivePtr &, const std::vector<ValuePtr> &)>;
using StridesVecotr = std::vector<int64_t>;

OPS_API std::vector<int64_t> GetOriStrides(const std::vector<int64_t> &shape);
OPS_API bool IsContiguous(const ShapeVector &shape, const std::vector<int64_t> &strides);
OPS_API int64_t DynamicDimWrap(int64_t dim, int64_t dim_post_expr, bool wrap_scalar = false);
OPS_API bool IsDynamic(const std::vector<int64_t> &shape);
OPS_API bool HasZero(const std::vector<int64_t> &value);
OPS_API bool CheckInputsNull(const std::vector<ValuePtr> &inputs, const size_t &input_num);
OPS_API int64_t ComputeStorageNelements(int64_t storage_offset, const std::vector<int64_t> &shape,
                                        const std::vector<int64_t> &stride);
OPS_API TensorStorageInfoPtr CheckSetStorageInfo(const tensor::TensorPtr &origin_tensor, int64_t storage_offset,
                                                 const std::vector<int64_t> &shape, const std::vector<int64_t> &stride,
                                                 const std::string &source_device_type_name,
                                                 int64_t source_storage_size, const TypeId &source_storage_dtype);
inline OPS_API std::tuple<std::vector<int64_t>, std::vector<int64_t>, size_t> GetOriShapeStridesAndOffset(
  const std::vector<int64_t> &cur_shape, const std::vector<int64_t> &cur_strides,
  const TensorStorageInfoPtr &cur_storage_info) {
  if (cur_storage_info) {
    return std::make_tuple(cur_storage_info->ori_shape, cur_storage_info->ori_strides,
                           cur_storage_info->storage_offset);
  }
  return std::make_tuple(cur_shape, cur_strides, size_t(0));
}

struct OldTensorInfo {
  OldTensorInfo(std::vector<int64_t> old_shape, std::vector<int64_t> old_strides, std::vector<int64_t> ori_shape,
                std::vector<int64_t> ori_strides, size_t old_offset)
      : old_shape(std::move(old_shape)),
        old_strides(std::move(old_strides)),
        ori_shape(std::move(ori_shape)),
        ori_strides(std::move(ori_strides)),
        old_offset(old_offset) {}
  std::vector<int64_t> old_shape;
  std::vector<int64_t> old_strides;
  std::vector<int64_t> ori_shape;
  std::vector<int64_t> ori_strides;
  size_t old_offset;
};
using OldTensorInfoPtr = std::shared_ptr<OldTensorInfo>;

OldTensorInfoPtr GetOldTensorInfo(const tensor::TensorPtr &tensor);

class OPS_API ViewStridesCalcFactory {
 public:
  static ViewStridesCalcFactory &GetInstance();
  ViewStridesCalcFactory() = default;
  ~ViewStridesCalcFactory() = default;
  void AddStridesCalcFunc(const std::string &op_name, const StridesCalcFunc &func) {
    strides_calc_map_[op_name] = func;
  }

  void AddTupleOutStridesCalcFunc(const std::string &op_name, const StridesCalcFunc &func) {
    tuple_out_strides_calc_map_[op_name] = func;
  }

  std::pair<std::optional<StridesCalcFunc>, bool> GetStridesCalcFunc(const std::string &op_name) {
    const auto &iter = strides_calc_map_.find(op_name);
    if (iter != strides_calc_map_.end()) {
      return std::make_pair(iter->second, false);
    }

    const auto &tuple_iter = tuple_out_strides_calc_map_.find(op_name);
    if (tuple_iter != tuple_out_strides_calc_map_.end()) {
      return std::make_pair(tuple_iter->second, true);
    }

    return std::make_pair(std::nullopt, false);
  }

 private:
  mindspore::HashMap<std::string, StridesCalcFunc> strides_calc_map_;
  mindspore::HashMap<std::string, StridesCalcFunc> tuple_out_strides_calc_map_;
};

class ViewStridesCalcRegistrar {
 public:
  ViewStridesCalcRegistrar(const std::string &op_name, const StridesCalcFunc &func, bool is_tuple = false) {
    if (is_tuple) {
      ViewStridesCalcFactory::GetInstance().AddTupleOutStridesCalcFunc(op_name, func);
    } else {
      ViewStridesCalcFactory::GetInstance().AddStridesCalcFunc(op_name, func);
    }
  }

  ~ViewStridesCalcRegistrar() = default;
};

#define REG_VIEW_STRIDES_CALC_FUN(op_name, func) \
  static ViewStridesCalcRegistrar g_##op_name##StridesCalcReg(#op_name, func);

#define REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(op_name, func) \
  static ViewStridesCalcRegistrar g_##op_name##StridesCalcReg(#op_name, func, true);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_OPS_INCLUDE_VIEW_VIEW_STRIDESCALCULATOR_H_
