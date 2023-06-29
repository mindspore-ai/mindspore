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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class Tile : public OpDesc {
 public:
  Tile() {
    std::initializer_list<std::string> attrs{"multiples"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~Tile() = default;

 protected:
  bool CheckInputs() override {
    const auto &x = inputs_info_[0];
    auto x_shape = x.shape;
    auto multiples = GetAxisList(attrs_["multiples"]);
    if (IsDynamicRank(x_shape)) {
      MS_LOG(INFO) << "Skip dynamic rank case";
      return false;
    }
    if (multiples.size() < x_shape.size()) {
      MS_LOG(INFO) << "For 'Tile', the length of 'multiples' should be greater than or equal to the length of input "
                      "'x' shape, but got "
                   << multiples.size() << " and " << x_shape.size();
      return false;
    }
    auto diff_len = multiples.size() - x_shape.size();
    for (size_t i = 0; i < x_shape.size(); ++i) {
      auto m_i = i + diff_len;
      if (x_shape[i] != 1 && multiples[m_i] != 1) {
        MS_LOG(INFO) << "For 'Tile', x_shape[" << i << "] = " << x_shape[i] << ", multiples[" << m_i
                     << "] = " << multiples[m_i]
                     << ", both value are not equal to 1, which can not be replaced by 'BroadcastTo'";
        return false;
      }
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &x = inputs[0];
    auto x_shape = x->shape;
    auto multiples = GetAxisList(attrs_["multiples"]);

    // calc output shape
    ShapeVector out_shape = multiples;
    auto diff_len = multiples.size() - x_shape.size();  // already checked in CheckInputs
    for (size_t i = 0; i < x_shape.size(); ++i) {
      auto m_i = i + diff_len;
      out_shape[m_i] *= x_shape[i];
    }

    auto result = gb.BroadcastTo(x, out_shape);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("Tile", Tile);
}  // namespace mindspore::graphkernel::expanders
