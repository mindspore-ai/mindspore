/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <iterator>
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ir/tensor_new.h"
#include "include/utils/utils.h"
#include "infer/ops_func_impl/upsample_linear1d.h"
#include "infer/ops_func_impl/upsample_nearest1d.h"
#include "infer/ops_func_impl/upsample_nearest2d.h"
#include "infer/ops_func_impl/upsample_nearest3d.h"
#include "infer/ops_func_impl/upsample_trilinear3d.h"
#include "infer/ops_func_impl/upsample_bicubic2d.h"
#include "infer/ops_func_impl/upsample_bilinear2d.h"
#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/test_ops_cmp_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace ops {
struct UpsampleForwardParams {
  ShapeVector image_shape;
  ValuePtr size;    // tuple[int]
  ValuePtr scales;  // tuple[float]
  ShapeVector out_shape;
};

static std::map<std::string, OpFuncImplPtr> upsample_forward_func_impl = {
  {kNameUpsampleNearest1D, std::make_shared<UpsampleNearest1DFuncImpl>()},
  {kNameUpsampleNearest2D, std::make_shared<UpsampleNearest2DFuncImpl>()},
  {kNameUpsampleNearest3D, std::make_shared<UpsampleNearest3DFuncImpl>()},
  {kNameUpsampleTrilinear3D, std::make_shared<UpsampleTrilinear3DFuncImpl>()},
  {kNameUpsampleLinear1D, std::make_shared<UpsampleLinear1DFuncImpl>()},
  {kNameUpsampleBilinear2D, std::make_shared<UpsampleBilinear2DFuncImpl>()},
  {kNameUpsampleBicubic2D, std::make_shared<UpsampleBicubic2DFuncImpl>()},
};

class TestUpsampleForward : public TestOps,
                            public testing::WithParamInterface<std::tuple<const char *, UpsampleForwardParams>> {};

TEST_P(TestUpsampleForward, dyn_shape) {
  const auto &upsample_mode = std::get<0>(GetParam());
  const auto &param = std::get<1>(GetParam());

  auto image = std::make_shared<abstract::AbstractTensor>(kFloat32, param.image_shape);
  ASSERT_NE(image, nullptr);
  auto size = param.size->ToAbstract();
  ASSERT_NE(size, nullptr);
  auto scales = param.scales->ToAbstract();
  ASSERT_NE(scales, nullptr);
  std::vector<AbstractBasePtr> input_args{image, size, scales};

  static std::set<std::string> white_list{kNameUpsampleLinear1D, kNameUpsampleBicubic2D, kNameUpsampleBilinear2D,
                                          kNameUpsampleTrilinear3D};
  if (white_list.find(upsample_mode) != white_list.end()) {
    auto align_corners = CreateScalar<bool>(true)->ToAbstract();
    input_args.push_back(align_corners);
  }

  auto op_itr = upsample_forward_func_impl.find(upsample_mode);
  ASSERT_TRUE(op_itr != upsample_forward_func_impl.end());
  auto op_impl = op_itr->second;
  ASSERT_NE(op_impl, nullptr);

  // Abstract Infer
  auto expect_shape = std::make_shared<abstract::Shape>(param.out_shape);
  DoFuncImplInferAndCompare(op_impl, upsample_mode, input_args, {expect_shape},
                            {std::make_shared<TensorType>(kFloat32)});

  // Simple Infer
  ValuePtrList input_values{tensor::from_spec(kNumberTypeFloat32, param.image_shape, device::DeviceType::kCPU)};
  std::transform(input_args.begin() + kIndex1, input_args.end(), std::back_inserter(input_values),
                 [](const AbstractBasePtr &abstract) { return abstract->GetValue(); });
  DoFuncImplSimpleInferAndCompare(op_impl, upsample_mode, input_values, {param.out_shape}, {kFloat32});
}

namespace {
float scale = 0.5;
auto Upsample3DDynTestCase = testing::ValuesIn(
  {UpsampleForwardParams{{1, 3, 8, 8, 8}, CreatePyIntTuple({4, 4, 4}), kNone, {1, 3, 4, 4, 4}},
   UpsampleForwardParams{{1, 3, 8, 8, 8}, kNone, CreateTuple({scale, scale, scale}), {1, 3, 4, 4, 4}},

   UpsampleForwardParams{{1, 3, 8, 8, 8}, CreatePyIntTuple({4, kValueAny, kValueAny}), kNone, {1, 3, 4, -1, -1}},
   UpsampleForwardParams{{1, 3, 8, 8, 8}, kValueAny, kNone, {1, 3, -1, -1, -1}},
   UpsampleForwardParams{{1, 3, 8, 8, 8}, kNone, CreateTuple({scale, kValueAny, kValueAny}), {1, 3, 4, -1, -1}},
   UpsampleForwardParams{{1, 3, 8, 8, 8}, kNone, kValueAny, {1, 3, -1, -1, -1}},

   UpsampleForwardParams{{1, 3, 8, -1, -1}, CreatePyIntTuple({4, 4, 4}), kNone, {1, 3, 4, 4, 4}},
   UpsampleForwardParams{{1, 3, 8, -1, -1}, kNone, CreateTuple({scale, scale, scale}), {1, 3, 4, -1, -1}},

   UpsampleForwardParams{{2, 2, -1, -1, -1}, CreatePyIntTuple({8, 8, kValueAny}), kNone, {2, 2, 8, 8, -1}},
   UpsampleForwardParams{{2, -1, 8, 8, 8}, kValueAny, kNone, {2, -1, -1, -1, -1}},
   UpsampleForwardParams{{-1, 2, -1, -1, -1}, kValueAny, kNone, {-1, 2, -1, -1, -1}},
   UpsampleForwardParams{{2, -1, 8, 4, 6}, kNone, CreateTuple({kValueAny, kValueAny, float(1.7)}), {2, -1, -1, -1, 10}},
   UpsampleForwardParams{{-1, 2, -1, -1, -1}, kNone, kValueAny, {-1, 2, -1, -1, -1}},

   UpsampleForwardParams{{-2}, CreatePyIntTuple({10, kValueAny, 8}), kNone, {-1, -1, 10, -1, 8}},
   UpsampleForwardParams{{-2}, kNone, CreateTuple({scale, scale, scale}), {-1, -1, -1, -1, -1}},

   UpsampleForwardParams{{-2}, CreatePyIntTuple({kValueAny, kValueAny, kValueAny}), kNone, {-1, -1, -1, -1, -1}},
   UpsampleForwardParams{{-2}, kValueAny, kNone, {-1, -1, -1, -1, -1}},
   UpsampleForwardParams{{-2}, kNone, CreateTuple({float(1.5), float(1.6), float(1.7)}), {-1, -1, -1, -1, -1}},
   UpsampleForwardParams{{-2}, kNone, kValueAny, {-1, -1, -1, -1, -1}}});

auto Upsample2DDynTestCase =
  testing::ValuesIn({UpsampleForwardParams{{1, 3, 8, 8}, CreatePyIntTuple({4, 4}), kNone, {1, 3, 4, 4}},
                     UpsampleForwardParams{{1, 3, 8, 8}, kNone, CreateTuple({scale, scale}), {1, 3, 4, 4}},

                     UpsampleForwardParams{{1, 3, 8, 8}, CreatePyIntTuple({4, kValueAny}), kNone, {1, 3, 4, -1}},
                     UpsampleForwardParams{{1, 3, 8, 8}, kValueAny, kNone, {1, 3, -1, -1}},
                     UpsampleForwardParams{{1, 3, 8, 8}, kNone, CreateTuple({scale, kValueAny}), {1, 3, 4, -1}},
                     UpsampleForwardParams{{1, 3, 8, 8}, kNone, kValueAny, {1, 3, -1, -1}},

                     UpsampleForwardParams{{1, 3, 8, -1}, CreatePyIntTuple({4, 4}), kNone, {1, 3, 4, 4}},
                     UpsampleForwardParams{{1, 3, 8, -1}, kNone, CreateTuple({scale, scale}), {1, 3, 4, -1}},

                     UpsampleForwardParams{{2, 2, -1, -1}, CreatePyIntTuple({8, 8}), kNone, {2, 2, 8, 8}},
                     UpsampleForwardParams{{2, -1, 8, 8}, kValueAny, kNone, {2, -1, -1, -1}},
                     UpsampleForwardParams{{-1, 2, -1, -1}, kValueAny, kNone, {-1, 2, -1, -1}},
                     UpsampleForwardParams{{2, -1, 8, 4}, kNone, CreateTuple({kValueAny, float(1.6)}), {2, -1, -1, 6}},
                     UpsampleForwardParams{{-1, 2, -1, -1}, kNone, kValueAny, {-1, 2, -1, -1}},

                     UpsampleForwardParams{{-2}, CreatePyIntTuple({10, kValueAny}), kNone, {-1, -1, 10, -1}},
                     UpsampleForwardParams{{-2}, kNone, CreateTuple({float(1.5), float(1.6)}), {-1, -1, -1, -1}},

                     UpsampleForwardParams{{-2}, CreatePyIntTuple({kValueAny, kValueAny}), kNone, {-1, -1, -1, -1}},
                     UpsampleForwardParams{{-2}, kValueAny, kNone, {-1, -1, -1, -1}},
                     UpsampleForwardParams{{-2}, kNone, CreateTuple({kValueAny, kValueAny}), {-1, -1, -1, -1}},
                     UpsampleForwardParams{{-2}, kNone, kValueAny, {-1, -1, -1, -1}}});

auto Upsample1DDynTestCase =
  testing::ValuesIn({UpsampleForwardParams{{1, 3, 8}, CreatePyIntTuple({4}), kNone, {1, 3, 4}},
                     UpsampleForwardParams{{1, 3, 8}, kNone, CreateTuple({scale}), {1, 3, 4}},

                     UpsampleForwardParams{{1, 3, 8}, CreatePyIntTuple({kValueAny}), kNone, {1, 3, -1}},
                     UpsampleForwardParams{{1, 3, 8}, kValueAny, kNone, {1, 3, -1}},
                     UpsampleForwardParams{{1, 3, 8}, kNone, CreateTuple({kValueAny}), {1, 3, -1}},
                     UpsampleForwardParams{{1, 3, 8}, kNone, kValueAny, {1, 3, -1}},

                     UpsampleForwardParams{{2, 2, -1}, CreatePyIntTuple({8}), kNone, {2, 2, 8}},
                     UpsampleForwardParams{{1, 3, -1}, kNone, CreateTuple({scale}), {1, 3, -1}},

                     UpsampleForwardParams{{2, -1, 4}, CreatePyIntTuple({8}), kNone, {2, -1, 8}},
                     UpsampleForwardParams{{2, -1, 4}, kNone, CreateTuple({float(1.5)}), {2, -1, 6}},

                     UpsampleForwardParams{{-1, 2, -1}, CreatePyIntTuple({kValueAny}), kNone, {-1, 2, -1}},
                     UpsampleForwardParams{{2, -1, 8}, CreatePyIntTuple({kValueAny}), kNone, {2, -1, -1}},
                     UpsampleForwardParams{{1, 3, -1}, kValueAny, kNone, {1, 3, -1}},

                     UpsampleForwardParams{{-1, 2, -1}, kNone, CreateTuple({kValueAny}), {-1, 2, -1}},
                     UpsampleForwardParams{{2, -1, 8}, kNone, CreateTuple({kValueAny}), {2, -1, -1}},
                     UpsampleForwardParams{{1, 3, -1}, kNone, kValueAny, {1, 3, -1}},

                     UpsampleForwardParams{{-2}, CreatePyIntTuple({10}), kNone, {-1, -1, 10}},
                     UpsampleForwardParams{{-2}, kNone, CreateTuple({scale}), {-1, -1, -1}},

                     UpsampleForwardParams{{-2}, CreatePyIntTuple({kValueAny}), kNone, {-1, -1, -1}},
                     UpsampleForwardParams{{-2}, kValueAny, kNone, {-1, -1, -1}},
                     UpsampleForwardParams{{-2}, kNone, CreateTuple({kValueAny}), {-1, -1, -1}},
                     UpsampleForwardParams{{-2}, kNone, kValueAny, {-1, -1, -1}}});
}  // namespace

INSTANTIATE_TEST_CASE_P(TestUpsampleNearest1DGroup, TestUpsampleForward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleNearest1D}), Upsample1DDynTestCase));
INSTANTIATE_TEST_CASE_P(TestUpsampleLinear1DGroup, TestUpsampleForward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleLinear1D}), Upsample1DDynTestCase));

INSTANTIATE_TEST_CASE_P(TestUpsampleNearest2DGroup, TestUpsampleForward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleNearest2D}), Upsample2DDynTestCase));
INSTANTIATE_TEST_CASE_P(TestUpsampleBilinear2DGroup, TestUpsampleForward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleBilinear2D}), Upsample2DDynTestCase));
INSTANTIATE_TEST_CASE_P(TestUpsampleBicubic2DGroup, TestUpsampleForward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleBicubic2D}), Upsample2DDynTestCase));

INSTANTIATE_TEST_CASE_P(TestUpsampleNearest3DGroup, TestUpsampleForward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleNearest3D}), Upsample3DDynTestCase));
INSTANTIATE_TEST_CASE_P(TestUpsampleTrilinear3DGroup, TestUpsampleForward,
                        testing::Combine(testing::ValuesIn({kNameUpsampleTrilinear3D}), Upsample3DDynTestCase));
}  // namespace ops
}  // namespace mindspore
