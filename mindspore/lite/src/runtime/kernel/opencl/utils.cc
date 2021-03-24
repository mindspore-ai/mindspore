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

#include "src/runtime/kernel/opencl/utils.h"
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include "src/kernel_registry.h"
#include "src/common/file_utils.h"

using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::ActivationType_TANH;

namespace mindspore::lite {
kernel::LiteKernel *GetOpenCLKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                    OpParameter *parameter, const InnerContext *ctx, const kernel::KernelKey &key) {
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  if (creator != nullptr) {
    auto kernel = creator(in_tensors, out_tensors, parameter, nullptr, key);
    return kernel;
  }
  return nullptr;
}
}  // namespace mindspore::lite

namespace mindspore::kernel {

const std::set<schema::PrimitiveType> ArithmeticPrimitives = {schema::PrimitiveType_MulFusion,
                                                              schema::PrimitiveType_AddFusion,
                                                              schema::PrimitiveType_SubFusion,
                                                              schema::PrimitiveType_DivFusion,
                                                              schema::PrimitiveType_LogicalAnd,
                                                              schema::PrimitiveType_LogicalOr,
                                                              schema::PrimitiveType_Maximum,
                                                              schema::PrimitiveType_Minimum,
                                                              schema::PrimitiveType_FloorDiv,
                                                              schema::PrimitiveType_FloorMod,
                                                              schema::PrimitiveType_SquaredDifference,
                                                              schema::PrimitiveType_Equal,
                                                              schema::PrimitiveType_NotEqual,
                                                              schema::PrimitiveType_Less,
                                                              schema::PrimitiveType_LessEqual,
                                                              schema::PrimitiveType_Greater,
                                                              schema::PrimitiveType_GreaterEqual,
                                                              schema::PrimitiveType_Eltwise,
                                                              schema::PrimitiveType_BiasAdd};

const std::set<schema::PrimitiveType> ArithmeticSelfPrimitives = {
  schema::PrimitiveType_Abs,        schema::PrimitiveType_Ceil,  schema::PrimitiveType_Cos,
  schema::PrimitiveType_ExpFusion,  schema::PrimitiveType_Floor, schema::PrimitiveType_Log,
  schema::PrimitiveType_LogicalNot, schema::PrimitiveType_Round, schema::PrimitiveType_Rsqrt,
  schema::PrimitiveType_Sin,        schema::PrimitiveType_Neg,   schema::PrimitiveType_Sqrt,
  schema::PrimitiveType_Square};

std::string GetActDefines() {
  static std::string act_defines = "#define ActivationType_RELU " + std::to_string(ActivationType_RELU) +
                                   "\n#define ActivationType_RELU6 " + std::to_string(ActivationType_RELU6) +
                                   "\n#define ActivationType_LEAKY_RELU " + std::to_string(ActivationType_LEAKY_RELU) +
                                   "\n#define ActivationType_TANH " + std::to_string(ActivationType_TANH) +
                                   "\n#define ActivationType_SIGMOID " + std::to_string(ActivationType_SIGMOID) + "\n";
  return act_defines;
}

int GetUpPow2(int n) {
  int i = 0;
  int j = 0;
  while (n > 0) {
    j += n & 1;
    n = n >> 1;
    i++;
  }
  return 1 << (i - (j == 1));
}

int GetMaxDivisor(int x, int divisor) {
  int i = divisor;
  while (i > 0) {
    if (x % i == 0) {
      return i;
    }
    i--;
  }
  return 1;
}

int GetMaxDivisorStrategy0(int x, int divisor) {
  if (divisor >= 8 && x % 8 == 0) {
    return 8;
  } else if (divisor >= 4 && x % 4 == 0) {
    return 4;
  } else if (divisor >= 2 && x % 2 == 0) {
    return 2;
  } else {
    return GetMaxDivisor(x, divisor);
  }
}

int GetMaxDivisorStrategy1(int x, int divisor) {
  if (divisor >= 8 && x % 8 == 0) {
    return x / 8;
  } else if (divisor >= 4 && x % 4 == 0) {
    return x / 4;
  } else if (divisor >= 2 && x % 2 == 0) {
    return x / 2;
  } else {
    return GetMaxDivisor(x, divisor);
  }
}

std::map<cl_int, std::string> error_infos = {
  {CL_SUCCESS, "Success"},
  {CL_DEVICE_NOT_FOUND, "Device not found"},
  {CL_DEVICE_NOT_AVAILABLE, "Device not available"},
  {CL_COMPILER_NOT_AVAILABLE, "Compiler not available"},
  {CL_MEM_OBJECT_ALLOCATION_FAILURE, "Memory object allocation failure"},
  {CL_OUT_OF_RESOURCES, "Out of resources"},
  {CL_OUT_OF_HOST_MEMORY, "Out of host memory"},
  {CL_PROFILING_INFO_NOT_AVAILABLE, "Profiling information not available"},
  {CL_MEM_COPY_OVERLAP, "Memory copy overlap"},
  {CL_IMAGE_FORMAT_MISMATCH, "Image format mismatch"},
  {CL_IMAGE_FORMAT_NOT_SUPPORTED, "Image format not supported"},
  {CL_BUILD_PROGRAM_FAILURE, "Build program failure"},
  {CL_MAP_FAILURE, "Mapping failure"},
  {CL_MISALIGNED_SUB_BUFFER_OFFSET, "Misaligned sub-buffer offset"},
  {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, "Execution status error for events in wait list"},
  {CL_COMPILE_PROGRAM_FAILURE, "Compile program failure"},
  {CL_LINKER_NOT_AVAILABLE, "Linker not available"},
  {CL_LINK_PROGRAM_FAILURE, "Link program failure"},
  {CL_DEVICE_PARTITION_FAILED, "Device partition failed"},
  {CL_KERNEL_ARG_INFO_NOT_AVAILABLE, "Kernel argument information not available"},
  {CL_INVALID_VALUE, "Invalid value"},
  {CL_INVALID_DEVICE_TYPE, "Invalid device type"},
  {CL_INVALID_PLATFORM, "Invalid platform"},
  {CL_INVALID_DEVICE, "Invalid device"},
  {CL_INVALID_CONTEXT, "Invalid context"},
  {CL_INVALID_QUEUE_PROPERTIES, "Invalid queue properties"},
  {CL_INVALID_COMMAND_QUEUE, "Invalid command queue"},
  {CL_INVALID_HOST_PTR, "Invalid host pointer"},
  {CL_INVALID_MEM_OBJECT, "Invalid memory object"},
  {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "Invalid image format descriptor"},
  {CL_INVALID_IMAGE_SIZE, "Invalid image size"},
  {CL_INVALID_SAMPLER, "Invalid sampler"},
  {CL_INVALID_BINARY, "Invalid binary"},
  {CL_INVALID_BUILD_OPTIONS, "Invalid build options"},
  {CL_INVALID_PROGRAM, "Invalid program"},
  {CL_INVALID_PROGRAM_EXECUTABLE, "Invalid program executable"},
  {CL_INVALID_KERNEL_NAME, "Invalid kernel name"},
  {CL_INVALID_KERNEL_DEFINITION, "Invalid kernel definition"},
  {CL_INVALID_KERNEL, "Invalid kernel"},
  {CL_INVALID_ARG_INDEX, "Invalid argument index"},
  {CL_INVALID_ARG_VALUE, "Invalid argument value"},
  {CL_INVALID_ARG_SIZE, "Invalid argument size"},
  {CL_INVALID_KERNEL_ARGS, "Invalid kernel arguments"},
  {CL_INVALID_WORK_DIMENSION, "Invalid work dimension"},
  {CL_INVALID_WORK_GROUP_SIZE, "Invalid work group size"},
  {CL_INVALID_WORK_ITEM_SIZE, "Invalid work item size"},
  {CL_INVALID_GLOBAL_OFFSET, "Invalid global offset"},
  {CL_INVALID_EVENT_WAIT_LIST, "Invalid event wait list"},
  {CL_INVALID_EVENT, "Invalid event"},
  {CL_INVALID_OPERATION, "Invalid operation"},
  {CL_INVALID_GL_OBJECT, "Invalid GL object"},
  {CL_INVALID_BUFFER_SIZE, "Invalid buffer size"},
  {CL_INVALID_MIP_LEVEL, "Invalid mip-level"},
  {CL_INVALID_GLOBAL_WORK_SIZE, "Invalid global work size"},
  {CL_INVALID_PROPERTY, "Invalid property"},
  {CL_INVALID_IMAGE_DESCRIPTOR, "Invalid image descriptor"},
  {CL_INVALID_COMPILER_OPTIONS, "Invalid compiler options"},
  {CL_INVALID_LINKER_OPTIONS, "Invalid linker options"},
  {CL_INVALID_DEVICE_PARTITION_COUNT, "Invalid device partition count"},
  {CL_INVALID_PIPE_SIZE, "Invalid pipe size"},
  {CL_INVALID_DEVICE_QUEUE, "Invalid device queue"},
  {CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR, "Invalid GL share group reference KHR"}};

std::string CLErrorCode(cl_int error_code) {
  auto it = error_infos.find(error_code);
  if (it == error_infos.end()) {
    return "Unknown OpenCL error code";
  } else {
    return it->second;
  }
}

int WriteToBin(const std::string &file_path, void *data, size_t size) {
  MS_ASSERT(data);
  std::ofstream out_file;

  out_file.open(file_path.c_str(), std::ios::binary);
  if (!out_file.good()) {
    MS_LOG(ERROR) << "file is bad";
    return -1;
  }

  if (!out_file.is_open()) {
    MS_LOG(ERROR) << "file open failed";
    return -1;
  }
  out_file.write(reinterpret_cast<char *>(data), size);
  return 0;
}

int GetBroadcastGpuAxis(int ndim, int ori_axis) {
  if (ori_axis >= ndim) {
    return ndim - 1;
  }
  int axis = 0;
  if (ndim == 1) {
    axis = 3;
  } else if (ndim == 2) {
    axis = ori_axis == 0 ? 0 : 3;
  } else if (ndim == 3) {
    axis = ori_axis == 0 ? 0 : ori_axis == 1 ? 2 : 3;
  } else if (ndim == 4) {
    axis = ori_axis;
  } else if (ndim > 4) {
    MS_LOG(ERROR) << "GPU doesn't support ndim>=" << ndim;
  }
  return axis;
}

void PackNHWCToNHWC4(void *src, void *dst, bool src_is_fp16, bool dst_is_fp16, const GpuTensorInfo &tensor) {
  MS_ASSERT(src);
  MS_ASSERT(dst);
  auto src_fp16 = reinterpret_cast<float16_t *>(src);
  auto src_fp32 = reinterpret_cast<float32_t *>(src);
  auto dst_fp16 = reinterpret_cast<float16_t *>(dst);
  auto dst_fp32 = reinterpret_cast<float32_t *>(dst);
  for (int n = 0, src_idx = 0; n < tensor.N; n++) {
    for (int h = 0; h < tensor.H; ++h) {
      for (int w = 0; w < tensor.W; ++w) {
        for (int c = 0; c < tensor.C; ++c, ++src_idx) {
          int dst_idx = ((n * tensor.H + h) * tensor.W + w) * tensor.Slice * C4NUM + c;
          if (dst_is_fp16) {
            dst_fp16[dst_idx] = src_is_fp16 ? src_fp16[src_idx] : static_cast<float16_t>(src_fp32[src_idx]);
          } else {
            dst_fp32[dst_idx] = src_is_fp16 ? static_cast<float32_t>(src_fp16[src_idx]) : src_fp32[src_idx];
          }
        }
      }
    }
  }
  // scalar
  if (tensor.ElementsNum == 1) {
    if (dst_is_fp16) {
      dst_fp16[3] = dst_fp16[2] = dst_fp16[1] = dst_fp16[0];
    } else {
      dst_fp32[3] = dst_fp32[2] = dst_fp32[1] = dst_fp32[0];
    }
  }
}

int CheckParamLikeTensor(const std::string &kernel_name, const std::string &tensor_name, lite::Tensor *tensor,
                         TypeId expect_data_type, const std::vector<int> &expect_shape) {
  if (!tensor->IsConst()) {
    MS_LOG(ERROR) << "in " << kernel_name << ": tensor " << tensor_name << " must be Const.";
    return RET_ERROR;
  }
  if (tensor->data_type() != expect_data_type) {
    MS_LOG(ERROR) << "in " << kernel_name << ": tensor's data_type must be " << expect_data_type;
    return RET_ERROR;
  }
  if (tensor->shape() != expect_shape) {
    std::string expect_shape_str = "(";
    for (auto i : expect_shape) {
      expect_shape_str += std::to_string(i) + ",";
    }
    expect_shape_str += ")";

    std::string tensor_shape_str = "(";
    for (auto i : tensor->shape()) {
      tensor_shape_str += std::to_string(i) + ",";
    }
    tensor_shape_str += ")";

    MS_LOG(ERROR) << "in " << kernel_name
                  << ": tensor's shape is error. expect_shape: " + expect_shape_str +
                       " tensor->shape(): " + tensor_shape_str;
    return RET_ERROR;
  }
  return RET_OK;
}

static std::set<void *> tmp_weights;

void StoreTmpWeight(lite::Tensor *tensor) {
  MS_LOG(WARNING) << "store weight when kernel don't infer shape!";
  if (tensor && tensor->data_c() && tensor->Size()) {
    void *new_data = malloc(tensor->Size());
    MS_ASSERT(new_data);
    if (new_data == nullptr) {
      return;
    }
    memcpy(new_data, tensor->data_c(), tensor->Size());
    tensor->set_data(new_data);
    tmp_weights.insert(new_data);
  }
}

void FreeTmpWeight(void *data) {
  if (tmp_weights.count(data)) {
    free(data);
    tmp_weights.erase(data);
  }
}

}  // namespace mindspore::kernel
