/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_ANF_CONV_PARSER_H
#define MINDSPORE_ANF_CONV_PARSER_H

#include "tools/anf_importer/anf_populater/anf_node_populater.h"
#include <memory>
#include <vector>
#include "base/base_ref.h"
#include "abstract/abstract_value.h"
#include "src/param_value_lite.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
class AnfConvPopulater : public AnfNodePopulater {
 public:
  AnfConvPopulater() = default;
  ~AnfConvPopulater() override = default;
  int Populate(const PrimitivePtr &prim, PrimitiveC *primitiveCPtr,
               const std::vector<AnfNodePtr> &inputs) override;

 private:
  template <typename T>
  void ConvertConvWeight(const ParameterPtr &param_node) {
    MS_ASSERT(param_node != nullptr);
    auto param = param_node->default_param();
    auto weight = std::dynamic_pointer_cast<ParamValueLite>(param);
    MS_ASSERT(weight != nullptr);

    std::unique_ptr<T> buf(new (std::nothrow) T[weight->tensor_shape_size()]);
    if (buf == nullptr) {
      MS_LOG(ERROR) << "new buf failed";
      return;
    }

    size_t filter_k = weight->tensor_shape()[0];
    size_t filter_c = weight->tensor_shape()[1];
    size_t filter_h = weight->tensor_shape()[2];
    size_t filter_w = weight->tensor_shape()[3];
    T *p1Buff = nullptr;
    T *p2Buff = nullptr;
    for (size_t k = 0; k < filter_k; ++k) {
      for (size_t c = 0; c < filter_c; ++c) {
        for (size_t h = 0; h < filter_h; ++h) {
          for (size_t w = 0; w < filter_w; ++w) {
            p1Buff = reinterpret_cast<float *>(weight->tensor_addr()) +
                     ((k * filter_c * filter_h * filter_w) + (c * filter_h * filter_w) + (h * filter_w) + (w));
            p2Buff =
              buf.get() + ((c * filter_k * filter_h * filter_w) + (k * filter_h * filter_w) + (h * filter_w) + (w));
            *p2Buff = *p1Buff;
          }
        }
      }
    }

    auto ret = ::memcpy_s(weight->tensor_addr(), weight->tensor_shape_size() * sizeof(T), buf.get(),
                          weight->tensor_shape_size() * sizeof(T));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed: " << ret;
      return;
    }

    auto abstract_base = param_node->abstract();
    MS_ASSERT(abstract_base != nullptr);
    if (utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
      auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
      utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[0] = filter_c;
      utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[1] = filter_k;
      utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[2] = filter_h;
      utils::cast<abstract::ShapePtr>(abstract_tensor->BuildShape())->shape()[3] = filter_w;
    }
    return;
  }

  void PopulaterConv2DMultiGroup(const PrimitivePtr &prim, const std::unique_ptr<schema::PrimitiveT> &primitive,
                                 const int &group, const std::vector<AnfNodePtr> &inputs);
  void PopulaterConv2DSingleGroup(const PrimitivePtr &prim, const std::unique_ptr<schema::PrimitiveT> &primitive,
                                  const int &group);
  void PopulaterQuantParam(const PrimitivePtr &prim, std::vector<std::vector<schema::QuantParamT>> *vecInputQuantParam,
                           std::vector<std::vector<schema::QuantParamT>> *vecOutputQuantParam);
  void CalQuantParam(const double &mean, const double &stdDev, float *mMin, float *mMax);
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_ANF_CONV_PARSER_H
