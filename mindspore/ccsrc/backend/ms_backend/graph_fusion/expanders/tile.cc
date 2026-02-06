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

#include "mindapi/base/shape_vector.h"
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"
#include "infer/ops_func_impl/tile.h"

namespace mindspore::graphkernel::expanders {
class Tile : public OpDesc {
 public:
  Tile() {
    std::initializer_list<std::string> attrs{"dims"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~Tile() = default;

 protected:
  bool CheckInputs() override {
    const auto &x = inputs_info_[0];
    auto x_shape = x.shape;
    auto dims = GetAxisList(attrs_["dims"]);
    if (IsDynamicRank(x_shape)) {
      MS_LOG(INFO) << "Skip dynamic rank case";
      return false;
    }
    ops::AdaptShapeAndMultipies(&x_shape, &dims);
    for (size_t i = 0; i < x_shape.size(); ++i) {
      if (x_shape[i] != 1 && dims[i] != 1) {
        MS_LOG(INFO) << "For 'Tile', input.shape = " << x.shape << ", dims = " << GetAxisList(attrs_["dims"])
                     << ", both value are not equal to 1 in matched-index " << i
                     << ", which can not be replaced by 'BroadcastTo'";
        return false;
      }
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &x = inputs[0];
    auto x_shape = x->shape;
    auto dims = GetAxisList(attrs_["dims"]);
    ops::AdaptShapeAndMultipies(&x_shape, &dims);
    ShapeVector out_shape = std::move(dims);
    for (size_t i = 0; i < x_shape.size(); ++i) {
      out_shape[i] *= x_shape[i];
    }

    auto result = gb.BroadcastTo(x, out_shape);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("Tile", Tile);
}  // namespace mindspore::graphkernel::expanders
