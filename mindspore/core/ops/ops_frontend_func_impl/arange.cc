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
std::pair<int64_t, std::vector<int64_t>> ArangeInferShape(const T start, const T end, const T step) {
  if (step == static_cast<T>(0)) {
    MS_EXCEPTION(ValueError) << "For Arange, the step can not be 0.";
  }

  if (step > 0 && start > end) {
    MS_EXCEPTION(ValueError) << "For Arange, step cannot be positive when end < start.";
  }

  if (step < 0 && start < end) {
    MS_EXCEPTION(ValueError) << "For Arange, step cannot be negative when end > start.";
  }

  std::vector<int64_t> out_shape{};
  int64_t shape_size = 0;
  if (std::is_integral<T>::value) {
    shape_size = static_cast<int64_t>((std::abs(end - start) + std::abs(step) - 1) / std::abs(step));
  } else {
    shape_size = static_cast<int64_t>(std::ceil(std::abs((end - start) / step)));
  }
  out_shape.push_back(shape_size);

  return std::make_pair(shape_size, out_shape);
}

template <typename T>
ValuePtr ArangeImpl(const TypeId dtype, const std::vector<AbstractBasePtr> &input_args) {
  auto start_opt = GetScalarValue<T>(input_args[kIndex0]->GetValue());
  auto end_opt = GetScalarValue<T>(input_args[kIndex1]->GetValue());
  auto step_opt = GetScalarValue<T>(input_args[kIndex2]->GetValue());

  ValuePtr res{nullptr};
  if (start_opt.has_value() && end_opt.has_value() && step_opt.has_value()) {
    auto start = start_opt.value();
    auto end = end_opt.value();
    auto step = step_opt.value();

    // infer shape
    auto [out_num, out_shape] = ArangeInferShape<T>(start, end, step);

    // make tensor
    auto tensor = std::make_shared<tensor::Tensor>(dtype, out_shape);
    MS_EXCEPTION_IF_NULL(tensor);

    // assign value
    auto output = static_cast<T *>(tensor->data_c());
    MS_EXCEPTION_IF_NULL(output);
    for (int64_t index = 0; index < out_num; index++) {
      output[index] = step * static_cast<T>(index) + start;
    }
    res = tensor;
  } else {
    MS_LOG(ERROR) << "For Arange, failed to get inputs' value.";
  }

  return res;
}

using ArangeFunc = std::function<ValuePtr(const TypeId, const std::vector<AbstractBasePtr> &)>;
static std::unordered_map<TypeId, ArangeFunc> ArangeFuncMap{{kNumberTypeInt32, ArangeImpl<int32_t>},
                                                            {kNumberTypeInt64, ArangeImpl<int64_t>},
                                                            {kNumberTypeFloat32, ArangeImpl<float>},
                                                            {kNumberTypeFloat64, ArangeImpl<double>}};
}  // namespace

class ArangeFrontendFuncImpl final : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    if (!IsAllValueKnown(input_args)) {
      return nullptr;
    }
    auto tensor = CalArangeOutTensor(input_args);
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

  ValuePtr CalArangeOutTensor(const std::vector<AbstractBasePtr> &input_args) const noexcept {
    auto type = input_args[0]->GetType()->type_id();
    auto it = ArangeFuncMap.find(type);
    if (it == ArangeFuncMap.end()) {
      MS_LOG(DEBUG) << "For Arange, the dtype of input must be int32, int64, float32, float64, but got "
                    << TypeIdToString(type) << ".";
      return nullptr;
    }
    auto result_tensor = it->second(type, input_args);
    return result_tensor;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Arange", ArangeFrontendFuncImpl);
}  // namespace mindspore::ops
