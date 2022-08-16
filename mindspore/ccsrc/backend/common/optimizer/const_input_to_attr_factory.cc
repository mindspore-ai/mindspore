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
#include "backend/common/optimizer/const_input_to_attr_factory.h"

#include "kernel/oplib/oplib.h"
#include "include/common/utils/utils.h"
#include "utils/log_adapter.h"

namespace mindspore::opt {
ConstInputToAttrRegister &ConstInputToAttrRegister::GetInstance() {
  static ConstInputToAttrRegister inst;
  return inst;
}

mindspore::HashSet<size_t> ConstInputToAttrRegister::GetConstToAttr(const std::string &name, const std::string &backend,
                                                                    bool is_dynamic_shape) const {
  auto key = GenerateKey(name, backend, is_dynamic_shape);
  auto iter = input_to_attr_.find(key);
  if (iter == input_to_attr_.end()) {
    if (backend != kAscendDevice) {
      return {};
    }
    auto op_info = kernel::OpLib::FindOp(name, kernel::kTBE, is_dynamic_shape);
    if (op_info == nullptr) {
      return {};
    }
    mindspore::HashSet<size_t> ret = {};
    auto input_to_attr_index_info = op_info->input_to_attr_index();
    (void)std::for_each(input_to_attr_index_info.begin(), input_to_attr_index_info.end(),
                        [&](auto &index) { ret.insert(index); });
    return ret;
  }

  return iter->second;
}

void ConstInputToAttrRegister::RegConstToAttr(const std::string &name, const std::string &backend,
                                              bool is_dynamic_shape, const mindspore::HashSet<size_t> &input_to_attr) {
  auto key = GenerateKey(name, backend, is_dynamic_shape);
  auto find = input_to_attr_.find(key);
  if (find != input_to_attr_.end()) {
    return;
  }
  input_to_attr_[key] = input_to_attr;
}

std::string ConstInputToAttrRegister::GenerateKey(const std::string &name, const std::string &backend,
                                                  bool is_dynamic_shape) {
  if (backend != kCPUDevice && backend != kGPUDevice && backend != kAscendDevice) {
    MS_LOG(EXCEPTION) << "Backend type is error, " << backend;
  }
  std::string is_dynamic_shape_str = is_dynamic_shape ? "true" : "false";
  return std::string(name + backend + is_dynamic_shape_str);
}

ConstInputToAttrRegister::RegisterHelper::RegisterHelper(const string &name, const string &backend,
                                                         bool is_dynamic_shape, int len, ...) {
  mindspore::HashSet<size_t> input_to_attr;
  input_to_attr.reserve(static_cast<size_t>(IntToUint(len)));
  va_list var_ptr;
  va_start(var_ptr, len);
  for (int i = 0; i < len; ++i) {
    (void)input_to_attr.insert(static_cast<size_t>(IntToUint(va_arg(var_ptr, int))));
  }
  va_end(var_ptr);
  ConstInputToAttrRegister::GetInstance().RegConstToAttr(name, backend, is_dynamic_shape, input_to_attr);
}
}  // namespace mindspore::opt
