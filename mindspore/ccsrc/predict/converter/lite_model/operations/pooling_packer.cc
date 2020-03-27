/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "predict/converter/lite_model/op_attr_packer.h"

namespace mindspore {
namespace predict {
namespace convert {
bool PoolingPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op) {
  if (c_node_ptr == nullptr || ms_op == nullptr) {
    return false;
  }
  std::unique_ptr<PoolingT> attr(new PoolingT());
  MS_EXCEPTION_IF_NULL(attr);
  std::string kernel_format_value = AnfAlgo::GetNodeAttr<std::string>(c_node_ptr, "data_format");
  attr->format = GetAttrFormat(kernel_format_value);
  auto c_name = AnfAlgo::GetCNodeName(c_node_ptr);
  if (c_name == "MaxPool") {
    ms_op->name = c_node_ptr->fullname_with_scope();
    attr->poolingMode = mindspore::predict::PoolMode::PoolMode_MAX_POOLING;
  } else if (c_name == "MeanPool") {
    ms_op->name = c_node_ptr->fullname_with_scope();
    attr->poolingMode = mindspore::predict::PoolMode::PoolMode_MEAN_POOLING;
  } else if (c_name == "GlobalPool") {
    ms_op->name = c_node_ptr->fullname_with_scope();
    attr->poolingMode = mindspore::predict::PoolMode::PoolMode_GLOBAL_POOING;
  } else {
    MS_LOG(ERROR) << "unknowed pooling type.";
    return false;
  }
  std::vector<int> kernel_ksize = AnfAlgo::GetNodeAttr<std::vector<int>>(c_node_ptr, "ksize");
  attr->windowW = kernel_ksize[kHIndex];
  attr->windowH = kernel_ksize[kWIndex];
  std::vector<int> kernel_strides = AnfAlgo::GetNodeAttr<std::vector<int>>(c_node_ptr, "strides");
  attr->strideW = kernel_strides[kHIndex];
  attr->strideH = kernel_strides[kWIndex];
  std::string kernel_pad_mode_value = AnfAlgo::GetNodeAttr<std::string>(c_node_ptr, "padding");
  attr->padMode = GetAttrPadMode(kernel_pad_mode_value);
  attr->padUp = 0;
  attr->padDown = 0;
  attr->padLeft = 0;
  attr->padRight = 0;
  attr->caffeMode = false;
  ms_op->attr.type = OpT_Pooling;
  ms_op->attr.value = attr.release();
  return true;
}
}  // namespace convert
}  // namespace predict
}  // namespace mindspore
