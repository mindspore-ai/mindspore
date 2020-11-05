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
#ifndef TESTS_UT_TRANSFORM_UT_TRANSFORM_BASE_H_
#define TESTS_UT_TRANSFORM_UT_TRANSFORM_BASE_H_

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include "transform/graph_ir/util.h"
#include "ir/tensor.h"

#include "common/common_test.h"
#include "pipeline/jit/parse/parse.h"

#include "graph/tensor.h"
#ifdef OPEN_SOURCE
#include "ge/client/ge_api.h"
#else
#include "external/ge/ge_api.h"
#endif

namespace mindspore {
namespace transform {
std::vector<FuncGraphPtr> getAnfGraph(std::string package, std::string function);
void PrintMeTensor(MeTensor* tensor);
FuncGraphPtr MakeFuncGraph(const PrimitivePtr prim, unsigned int nparam);
MeTensorPtr MakeTensor(const TypePtr& t, std::initializer_list<int64_t> shp);
}  // namespace transform
}  // namespace mindspore

#endif  // TESTS_UT_TRANSFORM_UT_TRANSFORM_BASE_H_
