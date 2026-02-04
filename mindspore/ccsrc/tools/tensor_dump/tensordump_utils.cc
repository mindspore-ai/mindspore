/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "tools/tensor_dump/tensordump_utils.h"

#include <memory>
#include <string>
#include <vector>

#include "ir/tensor.h"
#include "ir/tensor_new.h"
#include "tools/data_dump/utils.h"
#include "tools/tensor_dump/tensordump.h"
#include "utils/log_adapter.h"

namespace mindspore {

namespace datadump {
void MbufTensorDumpCallback(const std::string &tensor_name,
                            const std::vector<std::variant<std::string, mindspore::tensor::TensorPtr>> &data_items) {
  MS_VLOG(VL_PRINT_DUMP_V0) << "For 'TensorDump' ops, acltdt received Tensor name is " << tensor_name;
  if (tensor_name.empty()) {
    MS_LOG(ERROR) << "For 'TensorDump' ops, the args of 'file' is empty, skip this data.";
    return;
  }
  if (data_items.size() != 1) {
    MS_LOG(ERROR) << "For 'TensorDump' ops, the args of 'input_x' only support one input, bug got "
                  << data_items.size();
    return;
  }
  auto data_elem = data_items.front();
  if (std::holds_alternative<std::string>(data_elem)) {
    MS_LOG(WARNING) << "Ignore data of string type: " << std::get<std::string>(data_elem);
  }
  auto tensor_ptr = std::get<mindspore::tensor::TensorPtr>(data_elem);
  datadump::TensorDumpManager::GetInstance().Exec(tensor_name, tensor_ptr);
}
}  // namespace datadump

}  // namespace mindspore
