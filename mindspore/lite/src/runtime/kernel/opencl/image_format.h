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

#ifndef MINDSPORE_LITE_SRC_BACKEND_OPENCL_IMAGE_FORMAT_H_
#define MINDSPORE_LITE_SRC_BACKEND_OPENCL_IMAGE_FORMAT_H_

#include "src/runtime/opencl/opencl_runtime.h"

namespace mindspore {
namespace kernel {

/**
 * MindSpore to OpenCL channel order.
 * @param num_channels
 * @return opencl_channels
 */
cl_channel_order ToChannelOrder(int num_channels) {
  switch (num_channels) {
    case 1:
      return CL_R;
    case 2:
      return CL_RG;
    case 3:
      return CL_RGB;
    case 4:
      return CL_RGBA;
    default:
      return -1;
  }
}

/**
 * MindSpore image channel type to OpenCL channel data type.
 * @param data_type
 * @return opencl_data_type
 */
cl_channel_type ToImageChannelType(TypeId data_type) {
  switch (data_type) {
    case kNumberTypeFloat32:
      return CL_FLOAT;
    case kNumberTypeFloat16:
      return CL_HALF_FLOAT;
    default:
      return -1;
  }
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_BACKEND_OPENCL_IMAGE_FORMAT_H_

