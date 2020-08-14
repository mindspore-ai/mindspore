/**
 * This is the C++ adaptation and derivative work of Myia
 * (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <vector>
#include <memory>
#include "src/common/anf_importer/anf_populater/anf_node_populater.h"
namespace mindspore::lite {
class AnfConvPopulater : public AnfNodePopulater {
 public:
  AnfConvPopulater() = default;
  ~AnfConvPopulater() override = default;
  int Populate(const PrimitivePtr &prim, PrimitiveTValue *primitiveTValuePtr,
               const std::vector<AnfNodePtr> &inputs) override;

 private:
  void PopulaterConv2DMultiGroup(
      const PrimitivePtr &prim,
      const std::unique_ptr<schema::PrimitiveT> &primitive, const int &group);
  void PopulaterConv2DSingleGroup(
      const PrimitivePtr &prim,
      const std::unique_ptr<schema::PrimitiveT> &primitive, const int &group);
  void PopulaterQuantParam(const PrimitivePtr &prim,
                           std::vector<std::vector<schema::QuantParamT>> *vecQuantParam);
  void CalQuantParam(const double &mean, const double &stdDev, float *mMin,
                     float *mMax);
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_ANF_CONV_PARSER_H
