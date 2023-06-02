/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CAST_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CAST_ELIMINATE_H_

#include <map>
#include "frontend/optimizer/anf_visitor.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimCast, X, T}
class CastSameTypeEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
  void Visit(const AnfNodePtr &node) override;
  void Reset() {
    src_ = nullptr;
    tgt_ = nullptr;
  }

 private:
  AnfNodePtr src_{nullptr}, tgt_{nullptr};
};

// {prim::kPrimCast, {prim::kPrimCast, X, Y}, T}
class TwoCastEliminater : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
  void Visit(const AnfNodePtr &node) override;
  void Reset() {
    x_ = nullptr;
    t_ = nullptr;
    y_ = nullptr;
  }

 private:
  bool CheckTypesIsIncreasingOrDecreasing();
  bool CheckTwoTypes(const std::map<TypeId, int> &type_map, TypeId type1, TypeId type2) const;
  bool CheckThreeTypes(const std::map<TypeId, int> &type_map, TypeId type1, TypeId type2, TypeId type3) const;
  std::map<TypeId, int> int_map_ = {
    {kNumberTypeInt, 0}, {kNumberTypeInt8, 1}, {kNumberTypeInt16, 2}, {kNumberTypeInt32, 3}, {kNumberTypeInt64, 4}};
  std::map<TypeId, int> uint_map_ = {{kNumberTypeUInt, 0},
                                     {kNumberTypeUInt8, 1},
                                     {kNumberTypeUInt16, 2},
                                     {kNumberTypeUInt32, 3},
                                     {kNumberTypeUInt64, 4}};
  std::map<TypeId, int> float_map_ = {{kNumberTypeFloat, 0},
                                      {kNumberTypeFloat16, 1},
                                      {kNumberTypeFloat32, 2},
                                      {kNumberTypeFloat64, 3},
                                      {kNumberTypeDouble, 4}};
  AnfNodePtr x_{nullptr}, t_{nullptr}, y_{nullptr};
};

class CastEliminater : public OptimizerCaller {
 public:
  CastEliminater() : cast_same_type_eliminater_(), two_cast_eliminater_() {}
  ~CastEliminater() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    auto new_node = cast_same_type_eliminater_(optimizer, node);
    if (new_node != nullptr) {
      return new_node;
    }

    new_node = two_cast_eliminater_(optimizer, node);
    if (new_node != nullptr) {
      return new_node;
    }

    return nullptr;
  }

 private:
  CastSameTypeEliminater cast_same_type_eliminater_;
  TwoCastEliminater two_cast_eliminater_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CAST_ELIMINATE_H_
