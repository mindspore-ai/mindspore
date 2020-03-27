/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_CLEAN_H_
#define MINDSPORE_CCSRC_OPTIMIZER_CLEAN_H_

#include <memory>
#include "ir/anf.h"
#include "operator/ops.h"
#include "utils/any.h"
#include "ir/manager.h"
#include "pipeline/static_analysis/dshape.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {

// Remove the class type from graphs
void SimplifyDataStructures(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager);

// Remove most uses of tuples from the graph
// tuples that are returned will be kept
void EraseTuple(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager);

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPTIMIZER_CLEAN_H_
