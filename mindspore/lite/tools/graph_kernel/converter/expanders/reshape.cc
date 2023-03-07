/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "tools/graph_kernel/converter/expanders/activation.h"
#include "mindapi/base/types.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class Reshape : public OpDesc {
 public:
  Reshape() {
    std::initializer_list<std::string> attrs{"shape"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~Reshape() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    auto shp_ptr = attrs_["shape"];
    ShapeVector shape;
    if (shp_ptr->isa<tensor::Tensor>()) {
      auto value = std::static_pointer_cast<tensor::Tensor>(shp_ptr);
      if (value->data_type_c() == TypeId::kNumberTypeInt32) {
        int32_t *data = static_cast<int32_t *>(value->data_c());
        for (size_t elem = 0; elem < value->DataSize(); elem++) {
          (void)shape.emplace_back(IntToLong(*(data + elem)));
        }
      } else if (value->data_type_c() == TypeId::kNumberTypeInt64) {
        int64_t *data = static_cast<int64_t *>(value->data_c());
        for (size_t elem = 0; elem < value->DataSize(); elem++) {
          (void)shape.emplace_back(*(data + elem));
        }
      } else {
        MS_LOG(INFO) << "Type of reshape's shape tensor is neither int64_t nor int32_t. Expand failed";
        return {};
      }
    } else if (shp_ptr->isa<ValueTuple>()) {
      shape = GetValue<ShapeVector>(shp_ptr);
    } else {
      MS_LOG(INFO) << "Reshape's attr shape is neither Tensor nor ValueTuple. Expand failed";
      return {};
    }
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] == 0) {
        if (input_x->shape.size() <= i) {
          MS_LOG(INFO) << "Reshape's attr shape[" << i << "] is 0, but input's rank is " << input_x->shape.size();
          return {};
        }
        shape[i] = input_x->shape[i];
      }
    }
    auto result = gb.Reshape(input_x, shape);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("Reshape", Reshape);
}  // namespace mindspore::graphkernel::expanders
