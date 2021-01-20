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
    auto kernel = creator(in_tensors, out_tensors, parameter, nullptr, key, nullptr);
    return kernel;
  }
  return nullptr;
}
}  // namespace mindspore::lite

namespace mindspore::kernel {

const std::set<schema::PrimitiveType> ArithmeticPrimitives = {schema::PrimitiveType_Mul,
                                                              schema::PrimitiveType_Add,
                                                              schema::PrimitiveType_Sub,
                                                              schema::PrimitiveType_Div,
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
  schema::PrimitiveType_Exp,        schema::PrimitiveType_Floor, schema::PrimitiveType_Log,
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

std::string CLErrorCode(cl_int error_code) {
  switch (error_code) {
    case CL_SUCCESS:
      return "Success";
    case CL_DEVICE_NOT_FOUND:
      return "Device not found";
    case CL_DEVICE_NOT_AVAILABLE:
      return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:
      return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:
      return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:
      return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:
      return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:
      return "Build program failure";
    case CL_MAP_FAILURE:
      return "Mapping failure";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "Misaligned sub-buffer offset";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "Execution status error for events in wait list";
    case CL_COMPILE_PROGRAM_FAILURE:
      return "Compile program failure";
    case CL_LINKER_NOT_AVAILABLE:
      return "Linker not available";
    case CL_LINK_PROGRAM_FAILURE:
      return "Link program failure";
    case CL_DEVICE_PARTITION_FAILED:
      return "Device partition failed";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "Kernel argument information not available";
    case CL_INVALID_VALUE:
      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:
      return "Invalid device type";
    case CL_INVALID_PLATFORM:
      return "Invalid platform";
    case CL_INVALID_DEVICE:
      return "Invalid device";
    case CL_INVALID_CONTEXT:
      return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:
      return "Invalid command queue";
    case CL_INVALID_HOST_PTR:
      return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:
      return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:
      return "Invalid image size";
    case CL_INVALID_SAMPLER:
      return "Invalid sampler";
    case CL_INVALID_BINARY:
      return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:
      return "Invalid build options";
    case CL_INVALID_PROGRAM:
      return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:
      return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:
      return "Invalid kernel definition";
    case CL_INVALID_KERNEL:
      return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:
      return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:
      return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:
      return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:
      return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:
      return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:
      return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "Invalid event wait list";
    case CL_INVALID_EVENT:
      return "Invalid event";
    case CL_INVALID_OPERATION:
      return "Invalid operation";
    case CL_INVALID_GL_OBJECT:
      return "Invalid GL object";
    case CL_INVALID_BUFFER_SIZE:
      return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:
      return "Invalid mip-level";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "Invalid global work size";
    case CL_INVALID_PROPERTY:
      return "Invalid property";
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "Invalid image descriptor";
    case CL_INVALID_COMPILER_OPTIONS:
      return "Invalid compiler options";
    case CL_INVALID_LINKER_OPTIONS:
      return "Invalid linker options";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "Invalid device partition count";
    case CL_INVALID_PIPE_SIZE:
      return "Invalid pipe size";
    case CL_INVALID_DEVICE_QUEUE:
      return "Invalid device queue";
    case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
      return "Invalid GL share group reference KHR";
    default:
      return "Unknown OpenCL error code";
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

}  // namespace mindspore::kernel
