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
#ifndef MINDSPORE_CORE_OPS_VIEW_VIEWSTRIDESCALCULATOR_H_
#define MINDSPORE_CORE_OPS_VIEW_VIEWSTRIDESCALCULATOR_H_

#include <string>
#include <memory>
#include <vector>
#include <optional>
#include "ir/tensor.h"
#include "utils/hash_map.h"
#include "ir/value.h"
#include "ops/op_name.h"
#include "ir/primitive.h"

namespace mindspore {
namespace ops {
using TensorStorageInfoPtrList = std::vector<TensorStorageInfoPtr>;
// unsupported will return {}
using StridesCalcFunc = std::function<TensorStorageInfoPtrList(const PrimitivePtr &, const std::vector<ValuePtr> &)>;
using StridesVecotr = std::vector<int64_t>;
std::vector<int64_t> GetOriStrides(const std::vector<int64_t> &shape);
bool IsContiguous(const ShapeVector &shape, const std::vector<int64_t> &strides);
int64_t DynamicDimWrap(int64_t dim, int64_t dim_post_expr);
bool IsDynamic(const std::vector<int64_t> &shape);
bool HasZero(const std::vector<int64_t> &value);

struct OldTensorInfo {
  std::vector<int64_t> old_shape;
  std::vector<int64_t> old_strides;
  std::vector<int64_t> ori_shape;
  std::vector<int64_t> ori_strides;
  size_t old_offset;
};
using OldTensorInfoPtr = std::shared_ptr<OldTensorInfo>;

OldTensorInfoPtr GetOldTensorInfo(const tensor::TensorPtr &tensor);

class MIND_API ViewStridesCalcFactory {
 public:
  static ViewStridesCalcFactory &GetInstance();
  ViewStridesCalcFactory() = default;
  ~ViewStridesCalcFactory() = default;
  void AddStridesCalcFunc(const std::string &op_name, const StridesCalcFunc &func) {
    strides_calc_map_[op_name] = func;
  }

  std::optional<StridesCalcFunc> GetStridesCalcFunc(const std::string &op_name) {
    const auto &iter = strides_calc_map_.find(op_name);
    if (iter == strides_calc_map_.end()) {
      return std::nullopt;
    }
    return iter->second;
  }

 private:
  mindspore::HashMap<std::string, StridesCalcFunc> strides_calc_map_;
};

class ViewStridesCalcRegistrar {
 public:
  ViewStridesCalcRegistrar(const std::string &op_name, const StridesCalcFunc &func) {
    ViewStridesCalcFactory::GetInstance().AddStridesCalcFunc(op_name, func);
  }

  ~ViewStridesCalcRegistrar() = default;
};

#define REG_VIEW_STRIDES_CALC_FUN(op_name, func) \
  static ViewStridesCalcRegistrar g_##op_name##StridesCalcReg(#op_name, func);

}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_VIEWSTRIDESCALCULATOR_H_
