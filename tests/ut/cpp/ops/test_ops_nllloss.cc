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
#include <vector>
#include <memory>
#include "string"
#include "common/common_test.h"
#include "ops/nllloss.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "utils/ms_context.h"
#include "ops/test_ops.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace ops {
struct NLLLossParams {
  ShapeVector logits_shape;
  TypePtr logits_type;
  ShapeVector labels_shape;
  TypePtr labels_type;
  ShapeVector weight_shape;
  TypePtr weight_type;
  string reduction;
  bool is_success;
  ShapeVector loss_shape;
  TypePtr loss_type;
  ShapeVector total_weight_shape;
  TypePtr total_weight_type;
};

class TestNLLLoss : public TestOps, public testing::WithParamInterface<NLLLossParams> {};

TEST_P(TestNLLLoss, dyn_shape) {
  const auto &param = GetParam();
  auto prim = std::make_shared<Primitive>(kNameNLLLoss);
  ASSERT_NE(prim, nullptr);
  prim->set_attr("reduction", MakeValue<std::string>(param.reduction));
  auto logits = std::make_shared<abstract::AbstractTensor>(param.logits_type, param.logits_shape);
  auto labels = std::make_shared<abstract::AbstractTensor>(param.labels_type, param.labels_shape);
  auto weight = std::make_shared<abstract::AbstractTensor>(param.weight_type, param.weight_shape);
  ASSERT_NE(logits, nullptr);
  ASSERT_NE(labels, nullptr);
  ASSERT_NE(weight, nullptr);
  if (param.is_success) {
    auto loss = std::make_shared<abstract::AbstractTensor>(param.loss_type, param.loss_shape);
    auto total_weight = std::make_shared<abstract::AbstractTensor>(param.total_weight_type, param.total_weight_shape);
    ASSERT_NE(loss, nullptr);
    ASSERT_NE(total_weight, nullptr);
    AbstractBasePtrList abstract_list{loss, total_weight};
    auto expect = std::make_shared<abstract::AbstractTuple>(abstract_list);
    ASSERT_NE(expect, nullptr);

    auto out_abstract = opt::CppInferShapeAndType(prim, {logits, labels, weight});
    ASSERT_NE(out_abstract, nullptr);
    ASSERT_TRUE(*out_abstract == *expect);
  } else {
    ASSERT_ANY_THROW(opt::CppInferShapeAndType(prim, {logits, labels, weight}));
  }
}

INSTANTIATE_TEST_CASE_P(
  TestNLLLossGroup, TestNLLLoss,
  testing::Values(
    NLLLossParams{{-1, 3}, kFloat32, {-1}, kInt32, {3}, kFloat32, "none", true, {-1}, kFloat32, {}, kFloat32},
    NLLLossParams{{-1, 3}, kFloat32, {-1}, kInt32, {3}, kFloat32, "mean", true, {}, kFloat32, {}, kFloat32},
    NLLLossParams{{-1, 3}, kFloat32, {-1}, kInt32, {3}, kFloat32, "sum", true, {}, kFloat32, {}, kFloat32},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {3}, kFloat32, "none", true, {2}, kFloat32, {}, kFloat32},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {3}, kFloat32, "mean", true, {}, kFloat32, {}, kFloat32},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {3}, kFloat32, "sum", true, {}, kFloat32, {}, kFloat32},
    NLLLossParams{{-1, -1}, kFloat32, {-1}, kInt32, {-1}, kFloat32, "none", true, {-1}, kFloat32, {}, kFloat32},
    NLLLossParams{{-1, -1}, kFloat32, {-1}, kInt32, {-1}, kFloat32, "mean", true, {}, kFloat32, {}, kFloat32},
    NLLLossParams{{-1, -1}, kFloat32, {-1}, kInt32, {-1}, kFloat32, "sum", true, {}, kFloat32, {}, kFloat32},
    NLLLossParams{{-2}, kFloat32, {-1}, kInt32, {-1}, kFloat32, "none", true, {-1}, kFloat32, {}, kFloat32},
    NLLLossParams{{-2}, kFloat32, {-1}, kInt32, {-1}, kFloat32, "mean", true, {}, kFloat32, {}, kFloat32},
    NLLLossParams{{-2}, kFloat32, {-1}, kInt32, {-1}, kFloat32, "sum", true, {}, kFloat32, {}, kFloat32},
    NLLLossParams{{2, 3, 4}, kFloat32, {2}, kInt32, {3}, kFloat32, "none", false},
    NLLLossParams{{2, 3}, kFloat32, {2, 3}, kInt32, {3}, kFloat32, "none", false},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {2, 3}, kFloat32, "none", false},
    NLLLossParams{{2, 3}, kFloat32, {2}, kInt32, {2}, kFloat32, "none", false},
    NLLLossParams{{2, 3}, kFloat32, {3}, kInt32, {3}, kFloat32, "none", false}));
}  // namespace ops
}  // namespace mindspore
