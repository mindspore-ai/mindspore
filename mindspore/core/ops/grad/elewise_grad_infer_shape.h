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

#ifndef MINDSPORE_CORE_OPS_ELEWISE_GRAD_INFER_SHAPE_H_
#define MINDSPORE_CORE_OPS_ELEWISE_GRAD_INFER_SHAPE_H_

#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "ir/anf.h"

namespace mindspore {
abstract::ShapePtr ElewiseGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args);
}

#endif  // MINDSPORE_CORE_OPS_ELEWISE_GRAD_INFER_SHAPE_H_
