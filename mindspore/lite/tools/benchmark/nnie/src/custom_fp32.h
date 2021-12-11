/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CUSTOM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CUSTOM_H_

#include <vector>
#include <string>
#include "include/schema/model_generated.h"
#include "include/context.h"
#include "include/api/kernel.h"
#include "src/custom_infer.h"

using mindspore::kernel::Kernel;
using mindspore::tensor::MSTensor;
namespace mindspore {
namespace nnie {
class CustomCPUKernel : public Kernel {
 public:
  CustomCPUKernel(int seg_id, bool forward_bbox, const std::vector<MSTensor> &inputs,
                  const std::vector<MSTensor> &outputs, const mindspore::schema::Primitive *primitive,
                  const mindspore::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx), seg_id_(seg_id), forward_bbox_(forward_bbox) {
    if (forward_bbox) {
      roi_used_ = true;
    }
  }

  ~CustomCPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Execute() override;

  int seg_id(void) const { return seg_id_; }

  void set_seg_id(int id) { seg_id_ = id; }

  int forward_bbox(void) const { return forward_bbox_; }

  void set_forward_bbox(bool flag) { forward_bbox_ = flag; }

 private:
  static bool load_model_;
  static int run_seg_;
  static bool roi_used_;
  int seg_id_ = 0;
  bool forward_bbox_ = false;
  std::vector<std::vector<int64_t>> outputs_shapes_;
};
}  // namespace nnie
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CUSTOM_H_
