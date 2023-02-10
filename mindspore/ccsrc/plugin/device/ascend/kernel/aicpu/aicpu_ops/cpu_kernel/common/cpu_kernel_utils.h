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
#ifndef AICPU_CONTEXT_INC_UTILS_H_
#define AICPU_CONTEXT_INC_UTILS_H_
#include <functional>
#include <memory>
#include <string>

#include "cpu_kernel/inc/cpu_attr_value.h"
#include "cpu_kernel/inc/cpu_context.h"
#include "cpu_kernel/common/cpu_node_def.h"
#include "cpu_kernel/inc/cpu_tensor.h"

namespace aicpu {
class AICPU_VISIBILITY CpuKernelUtils {
 public:
  /*
   * create Tensor.
   * @return std::shared_ptr<Tensor>: Tensor ptr
   */
  static std::shared_ptr<Tensor> CreateTensor();

  /*
   * create Tensor.
   * @param tensor: Tensor impl
   * @return std::shared_ptr<Tensor>: Tensor ptr
   */
  static std::shared_ptr<Tensor> CreateTensor(TensorImpl *tensor);

  /*
   * get tensor impl.
   */
  static std::shared_ptr<TensorImpl> GetImpl(const Tensor *tensor);

  /*
   * get tensor name.
   */
  static std::string GetTensorName(const Tensor *tensor);

  /*
   * set tensor name.
   */
  static void SetTensorName(const std::string &name, std::shared_ptr<Tensor> &tensor);

  /*
   * create Tensor shape.
   * @return std::shared_ptr<TensorShape>: TensorShape ptr
   */
  static std::shared_ptr<TensorShape> CreateTensorShape();

  /*
   * create Tensor Shape.
   * @param tensorShape: Tensor shape impl
   * @return std::shared_ptr<TensorShape>: TensorShape ptr
   */
  static std::shared_ptr<TensorShape> CreateTensorShape(TensorShapeImpl *tensor_shape);

  /*
   * get tensor shape impl.
   */
  static std::shared_ptr<TensorShapeImpl> GetImpl(const TensorShape *tensorShape);

  /*
   * create attr value.
   * @return std::shared_ptr<AttrValue>: attr value ptr
   */
  static std::shared_ptr<AttrValue> CreateAttrValue();

  /*
   * create attr value.
   * @param attr_value: attr value impl
   * @return std::shared_ptr<AttrValue>: attr value ptr
   */
  static std::shared_ptr<AttrValue> CreateAttrValue(AttrValueImpl *attr_value);

  /*
   * get attr value impl.
   */
  static std::shared_ptr<AttrValueImpl> GetImpl(const AttrValue *attr_value);

  /*
   * create node def.
   * @return std::shared_ptr<NodeDef>: node def ptr
   */
  static std::shared_ptr<NodeDef> CreateNodeDef();

  /*
   * ParallelFor shards the "total" units of work.
   * @param ctx: context info of kernel
   * @param total: size of total work
   * @param per_unit_size: expect size of per unit work
   * @param work: process of per unit work
   * @return uint32_t: 0->success other->failed
   */
  static uint32_t ParallelFor(const CpuKernelContext &ctx, int64_t total, int64_t perUnitSize,
                              const std::function<void(int64_t, int64_t)> &work);

  /*
   * Get CPU number
   * @param ctx: context info of kernel
   * @return CPU number
   */
  static uint32_t GetCPUNum(const CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_INC_UTILS_H_
