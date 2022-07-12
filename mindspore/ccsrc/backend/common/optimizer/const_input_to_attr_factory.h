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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_FACTORY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_FACTORY_H_
#include <string>
#include <vector>
#include <utility>

#include "ir/anf.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "include/backend/visible.h"

namespace mindspore::opt {
class BACKEND_EXPORT ConstInputToAttrRegister {
 public:
  static ConstInputToAttrRegister &GetInstance();
  [[nodiscard]] mindspore::HashSet<size_t> GetConstToAttr(const std::string &name, const std::string &backend,
                                                          bool is_dynamic_shape) const;
  class RegisterHelper {
   public:
    RegisterHelper(const std::string &name, const std::string &backend, bool is_dynamic_shape, int len, ...);
    ~RegisterHelper() = default;
  };

 private:
  ConstInputToAttrRegister() = default;
  ~ConstInputToAttrRegister() = default;
  DISABLE_COPY_AND_ASSIGN(ConstInputToAttrRegister)
  void RegConstToAttr(const std::string &name, const std::string &backend, bool is_dynamic_shape,
                      const mindspore::HashSet<size_t> &input_to_attr);
  static std::string GenerateKey(const std::string &name, const std::string &backend, bool is_dynamic_shape);
  // key: (node_name + bankend + is_dynamic), value: <input_index>
  HashMap<std::string, mindspore::HashSet<size_t>> input_to_attr_;
};

#define RER_CONST_TO_ATTR(op_name, backend, dynamic, ...)                                  \
  static ConstInputToAttrRegister::RegisterHelper g_reg_##backend##_##dynamic##_##op_name( \
    op_name, backend, dynamic, std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value, __VA_ARGS__)

#define RER_CPU_DYNAMIC_CONST_TO_ATTR(op_name, ...) RER_CONST_TO_ATTR(op_name, kCPUDevice, true, __VA_ARGS__)

#define RER_GPU_DYNAMIC_CONST_TO_ATTR(op_name, ...) RER_CONST_TO_ATTR(op_name, kGPUDevice, true, __VA_ARGS__)

#define RER_ASCEND_DYNAMIC_CONST_TO_ATTR(op_name, ...) RER_CONST_TO_ATTR(op_name, kAscendDevice, true, __VA_ARGS__)

#define RER_CPU_STATIC_CONST_TO_ATTR(op_name, ...) RER_CONST_TO_ATTR(op_name, kCPUDevice, false, __VA_ARGS__)

#define RER_GPU_STATIC_CONST_TO_ATTR(op_name, ...) RER_CONST_TO_ATTR(op_name, kGPUDevice, false, __VA_ARGS__)

#define RER_ASCEND_STATIC_CONST_TO_ATTR(op_name, ...) RER_CONST_TO_ATTR(op_name, kAscendDevice, false, __VA_ARGS__)
}  // namespace mindspore::opt

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_FACTORY_H_
