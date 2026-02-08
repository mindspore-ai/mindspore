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

#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/graph_kernel_callback.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"

namespace mindspore::graphkernel::test {
void TestGraphKernelExpander::CompareShapeAndType(const AnfNodePtr &node, size_t output_idx, const ShapeVector &expect_shape,
                                       const TypeId &expect_type) {
  auto cb = graphkernel::Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  auto output_shape = cb->GetOutputShape(node, output_idx);
  auto output_type = cb->GetOutputType(node, output_idx);
  if (output_shape != expect_shape || output_type != expect_type) {
    MS_LOG(ERROR) << "output[" << output_idx << "] compare failed";
    MS_LOG(ERROR) << "expect shape: " << expect_shape << " data type: " << TypeIdToString(expect_type);
    MS_LOG(ERROR) << "output shape: " << output_shape << " data type: " << TypeIdToString(output_type);
    ASSERT_TRUE(false);
  }
}
}  // namespace mindspore::graphkernel::test