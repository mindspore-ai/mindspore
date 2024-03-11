/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/tensorprint_utils.h"
#include <fstream>
#include <iostream>
#include <string>
#include "pybind11/pybind11.h"
#include "ir/tensor.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "proto/print.pb.h"

namespace py = pybind11;

namespace mindspore::device::ascend {
namespace {
void OutputReceiveData2StdOut(const ScopeAclTdtDataset &dataset) {
  // Acquire Python GIL
  py::gil_scoped_acquire gil_acquire;

  for (auto data_elem : dataset.GetDataItems()) {
    if (std::holds_alternative<std::string>(data_elem)) {
      std::cout << std::get<std::string>(data_elem) << std::endl;
    } else {
      auto tensor_ptr = std::get<mindspore::tensor::TensorPtr>(data_elem);
      std::cout << tensor_ptr->ToStringNoLimit() << std::endl;
    }
  }
}

void OutputReceiveData2PbFile(const ScopeAclTdtDataset &dataset, const std::string &print_file_path) {
  prntpb::Print print;
  ChangeFileMode(print_file_path, S_IWUSR);
  std::fstream output(print_file_path, std::ios::out | std::ios::trunc | std::ios::binary);

  for (auto data_elem : dataset.GetDataItems()) {
    prntpb::Print_Value *value = print.add_value();

    if (std::holds_alternative<std::string>(data_elem)) {
      value->set_desc(std::get<std::string>(data_elem));
      continue;
    }

    auto tensor_ptr = std::get<mindspore::tensor::TensorPtr>(data_elem);
    prntpb::TensorProto *tensor = value->mutable_tensor();
    for (const auto &dim : tensor_ptr->shape()) {
      tensor->add_dims(static_cast< ::google::protobuf::int64>(dim));
    }

    tensor->set_tensor_type(tensor_ptr->type_name());
    tensor->set_tensor_content(tensor_ptr->data_c(), tensor_ptr->data().nbytes());
  }

  if (!print.SerializeToOstream(&output)) {
    MS_LOG(ERROR) << "Save print file:" << print_file_path << " fail.";
  }
  print.Clear();
  output.close();
  ChangeFileMode(print_file_path, S_IRUSR);
}
}  // namespace

void PrintReceiveData(const ScopeAclTdtDataset &dataset, const string &print_file_path) {
  if (print_file_path.empty()) {
    OutputReceiveData2StdOut(dataset);
  } else {
    // output data to file in protobuf binary format
    OutputReceiveData2PbFile(dataset, print_file_path);
  }
}
}  // namespace mindspore::device::ascend
