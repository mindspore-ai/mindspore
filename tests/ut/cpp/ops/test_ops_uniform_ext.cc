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
#include <vector>
#include <memory>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "common/common_test.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/ops_func_impl/uniform_ext.h"
#include "ops/test_ops.h"
#include "test_value_utils.h"

namespace mindspore {
    namespace ops {
        struct UniformExtOpParams {
            ShapeVector input_shape;
            TypePtr input_type;
            float from;
            float to;
            ShapeVector output_shape;
            TypePtr output_type;
        };

        class TestUniformExt : public TestOps, public testing::WithParamInterface<UniformExtOpParams> {};

        TEST_P(TestUniformExt, uniform_ext_dyn_shape) {
        auto primitive = std::make_shared<Primitive>("UniformExt");
        ASSERT_NE(primitive, nullptr);
        const auto &param = GetParam();
        auto input = std::make_shared<abstract::AbstractTensor>(param.input_type, param.input_shape);
        ASSERT_NE(input, nullptr);
        auto seed = std::make_shared<abstract::AbstractTensor>(param.input_type, ShapeVector{});
        ASSERT_NE(input, nullptr);
        auto offset = std::make_shared<abstract::AbstractTensor>(param.input_type, ShapeVector{});
        ASSERT_NE(input, nullptr);
        std::vector<abstract::AbstractBasePtr> input_args{std::move(input)};
        input_args.push_back(CreateScalar<double>(param.from)->ToAbstract());
        input_args.push_back(CreateScalar<double>(param.to)->ToAbstract());
        input_args.push_back(std::move(seed));
        input_args.push_back(std::move(offset));

        auto infer_impl = std::make_shared<UniformExtFuncImpl>();
        ASSERT_NE(infer_impl, nullptr);
        auto infer_shape = infer_impl->InferShape(primitive, input_args);
        ASSERT_NE(infer_shape, nullptr);
        auto infer_type = infer_impl->InferType(primitive, input_args);
        ASSERT_NE(infer_type, nullptr);

        auto expect_shape = std::make_shared<abstract::Shape>(param.output_shape);
        ASSERT_NE(expect_shape, nullptr);
        auto expect_type = std::make_shared<TensorType>(param.output_type);
        ASSERT_NE(expect_type, nullptr);
        ASSERT_TRUE(*infer_shape == *expect_shape);
        ASSERT_TRUE(*infer_type == *expect_type);
    }

    INSTANTIATE_TEST_CASE_P(
            TestUniformExtGroup, TestUniformExt,
            testing::Values(
            UniformExtOpParams{{10}, kFloat32, 0.0f, 1.0f,  {10}, kFloat32},
            UniformExtOpParams{{10, -1 , 10}, kFloat16, 0.0f, 1.0f,  {10, -1 , 10}, kFloat16},
            UniformExtOpParams{{-2}, kFloat64, 0.0f, 1.0f,  {-2}, kFloat64}
            ));
}  // namespace ops
}  // namespace mindspore
