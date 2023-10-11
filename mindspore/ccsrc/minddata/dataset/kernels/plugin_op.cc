/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/plugin_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/plugin/plugin_loader.h"

namespace mindspore {
namespace dataset {
Status PluginOp::PluginToTensorRow(const std::vector<plugin::Tensor> &in_row, TensorRow *out_row) {
  CHECK_FAIL_RETURN_UNEXPECTED(out_row != nullptr && out_row->empty(), "null/empty out_row received!");
  out_row->reserve(in_row.size());
  for (const auto &tensor : in_row) {
    std::shared_ptr<Tensor> output;
    auto tp = DataType(tensor.type_);
    CHECK_FAIL_RETURN_UNEXPECTED(tp.IsNumeric() && tp != DataType::DE_UNKNOWN,
                                 "Input datatype should be numeric, got Unsupported type: " + tensor.type_);
    RETURN_IF_NOT_OK(Tensor::CreateFromMemory(TensorShape(tensor.shape_), tp, tensor.buffer_.data(), &output));
    out_row->emplace_back(output);
  }
  return Status::OK();
}

Status PluginOp::TensorRowToPlugin(const TensorRow &in_row, std::vector<plugin::Tensor> *out_row) {
  CHECK_FAIL_RETURN_UNEXPECTED(out_row != nullptr && out_row->empty(), "null/empty out_row received!");
  out_row->resize(in_row.size());
  for (size_t ind = 0; ind < in_row.size(); ind++) {
    plugin::Tensor &tensor = (*out_row)[ind];
    if (in_row[ind]->type().IsNumeric()) {
      dsize_t buffer_size = in_row[ind]->SizeInBytes();
      tensor.buffer_.resize(buffer_size);
      if (buffer_size < SECUREC_MEM_MAX_LEN) {
        int ret_code = memcpy_s(tensor.buffer_.data(), tensor.buffer_.size(), in_row[ind]->GetBuffer(), buffer_size);
        CHECK_FAIL_RETURN_UNEXPECTED(ret_code == 0, "Failed to copy data into plugin tensor.");
      } else {
        int ret_code = memcpy_s(tensor.buffer_.data(), buffer_size, in_row[ind]->GetBuffer(), buffer_size);
        CHECK_FAIL_RETURN_UNEXPECTED(ret_code == 0, "Failed to copy data into plugin tensor.");
      }
    } else {  // string tensor, for now, only tensor with 1 string is supported!
      CHECK_FAIL_RETURN_UNEXPECTED(in_row[ind]->shape().NumOfElements() == 1,
                                   "String tensor with more than 1 element is not yet supported.");
      // get the first and only string in this tensor
      std::string str1(*(in_row[ind]->begin<std::string_view>()));
      tensor.buffer_.resize(str1.size());
      auto ret_code = memcpy_s(tensor.buffer_.data(), tensor.buffer_.size(), str1.data(), str1.size());
      CHECK_FAIL_RETURN_UNEXPECTED(ret_code == 0, "memcpy_s failed when copying string tensor.");
    }
    tensor.shape_ = in_row[ind]->shape().AsVector();
    tensor.type_ = in_row[ind]->type().ToString();
  }
  return Status::OK();
}

Status PluginOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  // Compute should quit if init fails. Error code has already been logged, no need to repeat
  RETURN_IF_NOT_OK(init_code_);
  std::vector<plugin::Tensor> in_row;
  std::vector<plugin::Tensor> out_row;
  RETURN_IF_NOT_OK(TensorRowToPlugin(input, &in_row));
  plugin::Status rc = plugin_op_->Compute(&in_row, &out_row);
  CHECK_FAIL_RETURN_UNEXPECTED(rc.IsOk(), rc.ToString());
  RETURN_IF_NOT_OK(PluginToTensorRow(out_row, output));
  return Status::OK();
}

PluginOp::PluginOp(const std::string &lib_path, const std::string &func_name, const std::string &user_args)
    : plugin_op_(nullptr), lib_path_(lib_path), func_name_(func_name), user_args_(user_args) {
  init_code_ = Init();
}

Status PluginOp::Init() {
  plugin::PluginManagerBase *plugin = nullptr;
  RETURN_IF_NOT_OK(PluginLoader::GetInstance()->LoadPlugin(lib_path_, &plugin));
  // casting a void pointer to specific type
  plugin_op_ = dynamic_cast<plugin::TensorOp *>(plugin->GetModule(func_name_));
  RETURN_UNEXPECTED_IF_NULL(plugin_op_);
  plugin::Status rc = plugin_op_->ParseSerializedArgs(user_args_);
  CHECK_FAIL_RETURN_UNEXPECTED(rc.IsOk(), rc.ToString());
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
