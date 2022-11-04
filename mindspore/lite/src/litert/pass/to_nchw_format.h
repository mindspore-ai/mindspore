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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_PASS_TO_NCHW_FORMAT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_PASS_TO_NCHW_FORMAT_H_

#include <vector>
#include <set>
#include "src/litert/kernel_exec.h"
#include "src/litert/pass/runtime_optimizer.h"
#include "schema/ops_generated.h"

namespace mindspore::lite::pass {
/* ToNCHWFormat PASS
 * The Lite default src_format_ is NHWC
 * for cpu fp32 graph, dst_format_ is NC4HW4
 * for cpu fp16 graph, dst_format_ is NC8HW8
 * for NPU graph, dst_format_ is NCHW
 * */

/* In ToNCHWFormat Run function, handle the kernels that are related to format, like: conv2d, pooling, etc.
 * The kernel with dynamic format will be ignored, like: activation, arithmetic, arithmetic_self, etc. */

class ToNCHWFormat : public RuntimePass {
 public:
  ToNCHWFormat(const Format &src_format, const Format &dst_format, std::set<schema::PrimitiveType> to_trans_kernels)
      : src_format_(src_format), dst_format_(dst_format), to_trans_kernels_(to_trans_kernels) {}
  ~ToNCHWFormat() override = default;
  int Run(kernel::SubGraphKernel *subgraph, std::vector<Tensor *> *tensors) override;

 private:
  int InsertPreTransKernel(kernel::SubGraphKernel *subgraph, kernel::KernelExec *kernel,
                           std::vector<Tensor *> *tensors);
  int InsertPostTransKernel(kernel::SubGraphKernel *subgraph, kernel::KernelExec *kernel,
                            std::vector<Tensor *> *tensors);

  /* src_format_ is the origin format of model kernel. The Model must only be one format: NHWC/NCHW. */
  Format src_format_;
  /* dst_format_ is the executed format of some specific kernels. */
  Format dst_format_;
  /* to_trans_kernels_ contains the specific kernels that will be changed it's  executed format. */
  /* In ToNCHWFormat Run function, the pre and post transpose kernels are inserted to ensure the format in the graph. */
  std::set<schema::PrimitiveType> to_trans_kernels_;
};
}  // namespace mindspore::lite::pass
#endif  // MINDSPORE_LITE_SRC_RUNTIME_PASS_TO_NCHW_FORMAT_H_
