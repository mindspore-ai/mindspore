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

#include <unordered_map>
#include <memory>

#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
namespace {
template <typename T>
std::pair<int64_t, std::vector<int64_t>> InferShape(const T start, const T limit, const T delta) {
  if (delta == static_cast<T>(0)) {
    MS_EXCEPTION(ValueError) << "For Rank, the delta can not be 0.";
  }

  if (delta > 0 && start > limit) {
    MS_EXCEPTION(ValueError) << "For Range, delta cannot be positive when limit < start.";
  }

  if (delta < 0 && start < limit) {
    MS_EXCEPTION(ValueError) << "For Range, delta cannot be negative when limit > start.";
  }

  std::vector<int64_t> out_shape{};
  int64_t shape_size = 0;
  if (std::is_integral<T>::value) {
    shape_size = static_cast<int64_t>((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta));
  } else {
    shape_size = static_cast<int64_t>(std::ceil(std::abs((limit - start) / delta)));
  }
  out_shape.push_back(shape_size);

  return std::make_pair(shape_size, out_shape);
}

template <typename T>
ValuePtr RangeImpl(const TypeId dtype, const std::vector<AbstractBasePtr> &input_args) {
  auto start_opt = GetScalarValue<T>(input_args[kIndex0]->GetValue());
  auto limit_opt = GetScalarValue<T>(input_args[kIndex1]->GetValue());
  auto delta_opt = GetScalarValue<T>(input_args[kIndex2]->GetValue());

  ValuePtr res{nullptr};
  if (start_opt.has_value() && limit_opt.has_value() && delta_opt.has_value()) {
    auto start = start_opt.value();
    auto limit = limit_opt.value();
    auto delta = delta_opt.value();

    // infer shape
    auto [out_num, out_shape] = InferShape<T>(start, limit, delta);

    // make tensor
    auto tensor = std::make_shared<tensor::Tensor>(dtype, out_shape);
    MS_EXCEPTION_IF_NULL(tensor);

    // assign value
    auto output = static_cast<T *>(tensor->data_c());
    MS_EXCEPTION_IF_NULL(output);
    for (int64_t index = 0; index < out_num; index++) {
      output[index] = delta * static_cast<T>(index) + start;
    }
    res = tensor;
  } else {
    MS_LOG(ERROR) << "For Range, failed to get inputs' value.";
  }

  return res;
}

using RangeFunc = std::function<ValuePtr(const TypeId, const std::vector<AbstractBasePtr> &)>;
static std::unordered_map<TypeId, RangeFunc> RangeFuncMap{{kNumberTypeInt32, RangeImpl<int32_t>},
                                                          {kNumberTypeInt64, RangeImpl<int64_t>},
                                                          {kNumberTypeFloat32, RangeImpl<float>},
                                                          {kNumberTypeFloat64, RangeImpl<double>}};
}  // namespace

class RangeFrontendFuncImpl final : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    if (!IsAllValueKnown(input_args)) {
      return nullptr;
    }
    CheckTypes(input_args);
    auto tensor = CalRangeOutTensor(input_args);
    return tensor;
  }

 private:
  bool IsAllValueKnown(const std::vector<AbstractBasePtr> &input_args) const noexcept {
    auto CheckValueIsKnown = [](const AbstractBasePtr &arg) {
      const auto &value_ptr = arg->GetValue();
      if (value_ptr == nullptr || value_ptr->isa<ValueAny>()) {
        return false;
      }
      return true;
    };
    auto ret = std::all_of(input_args.begin(), input_args.end(), CheckValueIsKnown);
    return ret;
  }

  void CheckTypes(const std::vector<AbstractBasePtr> &input_args) const noexcept {
    auto start_type = input_args[kIndex0]->GetType()->type_id();
    auto limit_type = input_args[kIndex1]->GetType()->type_id();
    auto delta_type = input_args[kIndex2]->GetType()->type_id();
    if (start_type != limit_type || start_type != delta_type) {
      MS_EXCEPTION(TypeError) << "For Range, "
                              << "the dtype of input should all be same, but got: start's type "
                              << TypeIdToString(start_type) << ", limit's type " << TypeIdToString(limit_type)
                              << ", delta's type " << TypeIdToString(delta_type) << ".";
    }
  }

  ValuePtr CalRangeOutTensor(const std::vector<AbstractBasePtr> &input_args) const noexcept {
    auto type = input_args[0]->GetType()->type_id();
    auto it = RangeFuncMap.find(type);
    if (it == RangeFuncMap.end()) {
      MS_LOG(DEBUG) << "For Range, the dtype of input must be int32, int64, float32, float64, but got "
                    << TypeIdToString(type) << ".";
      return nullptr;
    }
    auto result_tensor = it->second(type, input_args);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Range", RangeFrontendFuncImpl);
}  // namespace mindspore::ops
