/**
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
#include "dataset/kernels/image/resize_bilinear_op.h"
#include <random>

#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t ResizeBilinearOp::kDefWidth = 0;

void ResizeBilinearOp::Print(std::ostream &out) const { out << "ResizeBilinearOp: "; }
}  // namespace dataset
}  // namespace mindspore
