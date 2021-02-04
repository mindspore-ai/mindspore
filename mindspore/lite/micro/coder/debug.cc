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

#include "micro/coder/debug.h"
#include <memory>
#include <map>
#include <vector>
#include <utility>
#include "include/errorcode.h"
#include "micro/coder/utils/print_utils.h"
#include "micro/coder/coder_context.h"

namespace mindspore::lite::micro {
void MicroDebug::DumpTensorData(Tensor *tensor, const std::string &tensor_addr, std::string *code_block_str,
                                bool is_input) {
  *code_block_str += "\t\t\t{\n\t\t\t\tMicroTensor tensor;\n";

  std::string format_str = "\t\t\t\ttensor.format = " + std::to_string(tensor->format()) + ";\n";
  std::string type_str = "\t\t\t\ttensor.type = " + GetMicroTensorDataType(tensor->data_type()) + ";\n";
  std::string ndim_str = "\t\t\t\ttensor.ndim = " + std::to_string(static_cast<int>(tensor->shape().size())) + ";\n";

  *code_block_str += "\t\t\t\tint dim[] = {";
  for (size_t i = 0; i < tensor->shape().size(); ++i) {
    *code_block_str += std::to_string(tensor->shape().at(i)) + ", ";
  }
  *code_block_str += "};\n";
  *code_block_str += "\t\t\t\ttensor.dim = dim;\n";
  std::string data_str = "\t\t\t\ttensor.data = (void *)(" + tensor_addr + ");\n";
  std::string in_or_out = (is_input == 1 ? "input" : "output");
  std::string fprint_str = "\t\t\t\tfprintf(output_file, \"" + in_or_out + " Tensor:" + tensor_addr + "\\n\");\n";
  std::string print_str = "\t\t\t\tPrintTensor(&tensor,output_file," + std::to_string(is_input) + ");\n\t\t\t}\n";

  *code_block_str += ndim_str;
  *code_block_str += type_str;
  *code_block_str += format_str;
  *code_block_str += data_str;
  *code_block_str += fprint_str;
  *code_block_str += print_str;
}

int MicroDebug::DumpNodeData(const std::unique_ptr<OperatorCoder> &op_coder,
                             const std::map<Tensor *, std::string> &tensor_addrs, std::string *code_block_str) {
  auto config = Configurator::GetInstance();
  if (!config->debug_mode()) {
    return RET_OK;
  }
  std::string node_name = op_coder->ID();
  std::string file_str = "\n\t\t{\n\t\t\tFILE *output_file = fopen( \"./" + node_name +
                         ".ir\", \"w\");\n\t\t\tfprintf(output_file, \"Node:" + op_coder->ID() + "\\n\");\n";

  *code_block_str += file_str;
  auto runtime_tensor_iterator = [&op_coder, tensor_addrs, &code_block_str](const std::vector<Tensor *> &tensors,
                                                                            bool dump_data) {
    for (const auto &tensor : tensors) {
      if (tensor->data_c() != nullptr) {
        continue;
      }
      auto find_item =
        std::find_if(tensor_addrs.begin(), tensor_addrs.end(),
                     [tensor](const std::pair<Tensor *, std::string> &item) { return item.first == tensor; });
      if (find_item != tensor_addrs.end()) {
        DumpTensorData(tensor, find_item->second, code_block_str, dump_data);
      }
    }
    return RET_OK;
  };
  int status = runtime_tensor_iterator(op_coder->input_tensors(), true);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "dump runtime input tensor failed!";
    return status;
  }
  status = runtime_tensor_iterator(op_coder->output_tensors(), false);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "dump runtime input tensor failed!";
    return status;
  }
  std::string end_file_str = "\t\t\tfclose(output_file);\n\t\t}\n";
  *code_block_str += end_file_str;
  return RET_OK;
}

}  // namespace mindspore::lite::micro
