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
#include <cmath>
#include <cstring>
#include <memory>
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/litert/lite_session.h"
#include "include/registry/register_kernel_interface.h"
#include "include/registry/register_kernel.h"
#include "include/registry/opencl_runtime_wrapper.h"
#include "include/api/data_type.h"

using mindspore::kernel::Kernel;
using mindspore::kernel::KernelInterface;
using mindspore::schema::PrimitiveType_AddFusion;
#define UP_ROUND(x, y) (((x) + (y) - (1)) / (y) * (y))
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

namespace mindspore {
namespace {
constexpr int kDimIndex2 = 2;
constexpr int kDimIndex3 = 3;
constexpr int kDimIndex4 = 4;
constexpr size_t kDimSize2 = 2;
constexpr auto kFloat32 = DataType::kNumberTypeFloat32;
static const char *arithmetic_source =
  "\n"
  "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
  "__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
  "\n"
  "__kernel void ElementAdd(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t "
  "output,\n"
  "                         const int2 output_shape) {\n"
  "  int X = get_global_id(0);\n"
  "  int Y = get_global_id(1);\n"
  "  if (X >= output_shape.x || Y >= output_shape.y) {\n"
  "    return;\n"
  "  }\n"
  "\n"
  "  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n"
  "  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n"
  "  FLT4 result = a + b;\n"
  "\n"
  "  WRITE_IMAGE(output, (int2)(X, Y), result);\n"
  "}\n";

template <typename SrcT, typename DstT>
void Broadcast2GpuShape(DstT *dst, const SrcT *src, int src_num) {
  if (src == nullptr || src_num <= 0) {
    return;
  }
  auto *N = dst;
  auto *H = dst + 1;
  auto *W = dst + kDimIndex2;
  auto *C = dst + kDimIndex3;
  if (src_num == 1) {  // 1 1 1 C
    *C = src[0];
  } else if (src_num == kDimIndex2) {  // N 1 1 C
    *N = src[0];
    *C = src[1];
  } else if (src_num == kDimIndex3) {  // N 1 W C
    *N = src[0];
    *W = src[1];
    *C = src[2];
  } else if (src_num == kDimIndex4) {  // N H W C
    *N = src[0];
    *H = src[1];
    *W = src[2];
    *C = src[3];
  } else if (src_num > 4) {
    std::cerr << "GPU doesn't support ndim>=" << src_num;
  }
}

template <typename SrcT, typename DstT>
void Broadcast2GpuShape(DstT *dst, const SrcT *src, int src_num, DstT default_value) {
  for (int i = 0; i < 4; ++i) {
    dst[i] = default_value;
  }
  if (src == nullptr || src_num <= 0) {
    return;
  }
  Broadcast2GpuShape(dst, src, src_num);
}

struct GpuTensorInfo {
  GpuTensorInfo() = default;
  explicit GpuTensorInfo(const MSTensor *tensor, registry::opencl::OpenCLRuntimeWrapper *opencl_run) {
    if (tensor == nullptr) {
      return;
    }
    auto shape_ori = tensor->Shape();
    int64_t shape[kDimIndex4];
    Broadcast2GpuShape(shape, shape_ori.data(), shape_ori.size(), 1l);
    N = shape[0];
    H = shape[1];
    W = shape[kDimIndex2];
    C = shape[kDimIndex3];
    Slice = UP_DIV(C, C4NUM);
    if (tensor->DataType() == mindspore::DataType::kNumberTypeFloat16) {
      FLT_size = sizeof(cl_half);
    } else {
      FLT_size = sizeof(cl_float);
    }
    FLT4_size = FLT_size * 4;
    if (W * Slice <= opencl_run->GetMaxImage2DWidth()) {
      height = N * H;
      width = W * Slice;
    } else {
      height = N * H * W;
      width = Slice;
      if (height > opencl_run->GetMaxImage2DHeight()) {
        height = -1;
        width = -1;
      }
    }

    ElementsNum = N * H * W * C;
    Image2DSize = height * width * FLT4_size;
  }
  size_t N{1};
  size_t H{1};
  size_t W{1};
  size_t C{1};
  size_t Slice{};
  size_t width{};
  size_t height{};
  size_t FLT_size{4};
  size_t FLT4_size{16};
  size_t ElementsNum{};
  size_t Image2DSize{};
};
}  // namespace

class CustomAddKernel : public kernel::Kernel {
 public:
  CustomAddKernel(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                  const schema::Primitive *primitive, const mindspore::Context *ctx, bool fp16_enable)
      : Kernel(inputs, outputs, primitive, ctx), fp16_enable_(fp16_enable) {}
  ~CustomAddKernel() override { FreeWeight(); }

  int CheckInputsDataTypes() { return kSuccess; }

  // Prepare will be called during graph compilation
  int Prepare() override {
    auto ret = CheckSpecs();
    if (ret != kSuccess) {
      std::cerr << "Prepare failed for check kernel specs!";
      return ret;
    }
    const std::string kernel_name_ = "ElementAdd";
    const std::string program_name = "Arithmetic";
    std::string source = arithmetic_source;
    if (opencl_runtime_.LoadSource(program_name, source) != kSuccess) {
      std::cerr << "Load source failed.";
      return kLiteError;
    }
    std::vector<std::string> build_options_ext = {"-cl-mad-enable -cl-fast-relaxed-math -Werror"};
    if (fp16_enable_) {
      build_options_ext.push_back(" -DFLT4=half4 -DWRITE_IMAGE=write_imageh -DREAD_IMAGE=read_imageh");
    } else {
      build_options_ext.push_back(" -DFLT4=float4 -DWRITE_IMAGE=write_imagef -DREAD_IMAGE=read_imagef");
    }

    if (opencl_runtime_.BuildKernel(&kernel_, program_name, kernel_name_, build_options_ext) != kSuccess) {
      std::cerr << "Build kernel failed.";
      return kLiteError;
    }

    auto out_shape = GpuTensorInfo(&outputs_[0], &opencl_runtime_);
    local_range_ = cl::NullRange;
    global_range_ = cl::NDRange(out_shape.width, out_shape.height);
    for (int i = 0; i < inputs_.size(); ++i) {
      auto &in_tensor = inputs_.at(i);
      if (in_tensor.IsConst()) {
        GpuTensorInfo in_shape = GpuTensorInfo(&in_tensor, &opencl_runtime_);
        std::vector<char> weight(in_shape.Image2DSize, 0);
        bool src_is_fp16 = in_tensor.DataType() == mindspore::DataType::kNumberTypeFloat16;
        PackNHWCToNHWC4(in_tensor.MutableData(), weight.data(), src_is_fp16, fp16_enable_, in_shape,
                        in_tensor.DataType());
        DataType dtype =
          fp16_enable_ ? mindspore::DataType::kNumberTypeFloat16 : mindspore::DataType::kNumberTypeFloat32;
        auto allocator = opencl_runtime_.GetAllocator();
        if (allocator == nullptr) {
          std::cerr << "GetAllocator fail.";
          FreeWeight();
          return kLiteError;
        }
        auto weight_ptr = allocator->Malloc(in_shape.width, in_shape.height, dtype);
        if (weight_ptr == nullptr) {
          std::cerr << "Malloc fail.";
          FreeWeight();
          return kLiteError;
        }
        weight_ptrs_.push_back(weight_ptr);
        if (opencl_runtime_.WriteImage(weight_ptr, weight.data()) != kSuccess) {
          std::cerr << "WriteImage fail.";
          FreeWeight();
          return kLiteError;
        }
      } else {
        weight_ptrs_.push_back(nullptr);
      }
    }

    int arg_idx = 3;
    cl_int2 output_shape{static_cast<int>(global_range_[0]), static_cast<int>(global_range_[1])};
    if (opencl_runtime_.SetKernelArg(kernel_, arg_idx, output_shape) != kSuccess) {
      std::cerr << "Set kernel arg" << arg_idx << "failed.";
      FreeWeight();
      return kLiteError;
    }

    std::cout << kernel_name_ << " Prepare Done!" << std::endl;
    return kSuccess;
  }

  // Execute is called to compute.
  int Execute() override {
    if (inputs_.size() != kDimSize2) {
      return kLiteParamInvalid;
    }
    PreProcess();
    std::cout << this->name() << " Running!" << std::endl;
    auto input_0_ptr = weight_ptrs_[0] == nullptr ? inputs_[0].MutableData() : weight_ptrs_[0];
    auto input_1_ptr = weight_ptrs_[1] == nullptr ? inputs_[1].MutableData() : weight_ptrs_[1];
    int arg_idx = 0;
    if (opencl_runtime_.SetKernelArg(kernel_, arg_idx++, input_0_ptr) != kSuccess) {
      std::cerr << "Set kernel arg" << arg_idx - 1 << "failed.";
      return kLiteError;
    }
    if (opencl_runtime_.SetKernelArg(kernel_, arg_idx++, input_1_ptr) != kSuccess) {
      std::cerr << "Set kernel arg" << arg_idx - 1 << "failed.";
      return kLiteError;
    }
    if (opencl_runtime_.SetKernelArg(kernel_, arg_idx++, outputs_[0].MutableData()) != kSuccess) {
      std::cerr << "Set kernel arg" << arg_idx - 1 << "failed.";
      return kLiteError;
    }
    if (opencl_runtime_.RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != kSuccess) {
      std::cerr << "Run kernel failed.";
      return kLiteError;
    }

    return kSuccess;
  }

  int CheckSpecs() {
    for (auto &tensor : inputs_) {
      if (tensor.DataType() != DataType::kNumberTypeFloat32 && tensor.DataType() != DataType::kNumberTypeFloat16) {
        std::cerr << "ArithmeticOpenCLKernel only support fp32/fp16 input";
        return kLiteError;
      }
    }
    for (auto &tensor : outputs_) {
      if (tensor.DataType() != DataType::kNumberTypeFloat32 && tensor.DataType() != DataType::kNumberTypeFloat16) {
        std::cerr << "ArithmeticOpenCLKernel only support fp32/fp16 output";
        return kLiteError;
      }
    }

    if (inputs_.size() != kDimSize2 || outputs_.size() != 1) {
      std::cerr << "in size: " << inputs_.size() << ", out size: " << outputs_.size();
      return kLiteError;
    }

    for (int i = 0; i < inputs_.size(); ++i) {
      auto &in_tensor = inputs_.at(i);
      if (!in_tensor.IsConst()) {
        if (fp16_enable_ && in_tensor.DataType() == mindspore::DataType::kNumberTypeFloat32) {
          std::cerr << "Inputs data type error, expectation kNumberTypeFloat16 but kNumberTypeFloat32.";
          return kLiteError;
        } else if (!fp16_enable_ && in_tensor.DataType() == mindspore::DataType::kNumberTypeFloat16) {
          std::cerr << "Inputs data type error, expectation kNumberTypeFloat32 but kNumberTypeFloat16.";
          return kLiteError;
        }
      }
    }

    return kSuccess;
  }

  // Resize is used to update some parameters if current node can change along with inputs.
  int ReSize() override {
    if (CheckOutputs(outputs_) == kSuccess) {
      return kSuccess;
    }
    auto status =
      registry::RegisterKernelInterface::GetKernelInterface("", primitive_)->Infer(&inputs_, &outputs_, primitive_);
    if (status != kSuccess) {
      std::cerr << "infer failed." << std::endl;
      return kLiteError;
    }
    auto ret = Prepare();
    if (ret != kSuccess) {
      std::cerr << "ReSize failed for kernel prepare!";
      return ret;
    }
    return kSuccess;
  }

 private:
  const bool fp16_enable_;
  cl::Kernel kernel_;
  cl::Event event_;
  cl::NDRange global_range_{cl::NullRange};
  cl::NDRange local_range_{cl::NullRange};
  std::vector<void *> weight_ptrs_;
  registry::opencl::OpenCLRuntimeWrapper opencl_runtime_;

  int PreProcess() {
    int ret;
    ret = ReSize();
    if (ret != kSuccess) {
      return ret;
    }
    for (auto i = 0; i < outputs_.size(); ++i) {
      auto *output = &outputs_.at(i);
      auto img_info = GpuTensorInfo(output, &opencl_runtime_);
      auto allocator = output->allocator();
      if (allocator == nullptr) {
        std::cerr << "The output tensor of OpenCL kernel must have an allocator.";
        return kLiteError;
      }
      auto data_ptr = allocator->Malloc(img_info.width, img_info.height, output->DataType());
      if (data_ptr == nullptr) {
        std::cerr << "Malloc data failed";
        return kLiteError;
      }
      output->SetData(data_ptr);
    }
    return kSuccess;
  }

  int CheckOutputs(const std::vector<mindspore::MSTensor> &outputs) {
    for (auto &output : outputs) {
      auto output_shape = output.Shape();
      if (std::find(output_shape.begin(), output_shape.end(), -1) != output_shape.end()) {
        return kLiteInferInvalid;
      }
    }
    return kSuccess;
  }

  void PackNHWCToNHWC4(void *src, void *dst, bool src_is_fp16, bool dst_is_fp16, const GpuTensorInfo &tensor,
                       mindspore::DataType data_type) {
    auto src_fp16 = static_cast<float16_t *>(src);
    auto src_fp32 = static_cast<float32_t *>(src);
    auto src_int32 = static_cast<int32_t *>(src);
    auto dst_fp16 = static_cast<float16_t *>(dst);
    auto dst_fp32 = static_cast<float32_t *>(dst);
    auto dst_int32 = static_cast<int32_t *>(dst);
    for (int n = 0, src_idx = 0; n < tensor.N; n++) {
      for (int h = 0; h < tensor.H; ++h) {
        for (int w = 0; w < tensor.W; ++w) {
          for (int c = 0; c < tensor.C; ++c, ++src_idx) {
            int dst_idx = ((n * tensor.H + h) * tensor.W + w) * tensor.Slice * C4NUM + c;
            if (data_type == mindspore::DataType::kNumberTypeInt32) {
              dst_int32[dst_idx] = src_int32[src_idx];
            } else if (dst_is_fp16) {
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
        dst_fp16[kDimIndex3] = dst_fp16[kDimIndex2] = dst_fp16[1] = dst_fp16[0];
      } else {
        dst_fp32[kDimIndex3] = dst_fp32[kDimIndex2] = dst_fp32[1] = dst_fp32[0];
      }
    }
  }

  void FreeWeight() {
    auto allocator = opencl_runtime_.GetAllocator();
    if (allocator == nullptr) {
      std::cerr << "GetAllocator fail.";
      return;
    }
    for (auto &weight_ptr : weight_ptrs_) {
      if (weight_ptr != nullptr) {
        allocator->Free(weight_ptr);
        weight_ptr = nullptr;
      }
    }
  }
};

class CustomAddInfer : public kernel::KernelInterface {
 public:
  CustomAddInfer() = default;
  ~CustomAddInfer() = default;

  Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
               const schema::Primitive *primitive) override {
    (*outputs)[0].SetFormat((*inputs)[0].format());
    (*outputs)[0].SetDataType((*inputs)[0].DataType());
    (*outputs)[0].SetShape((*inputs)[0].Shape());
    return kSuccess;
  }
};

namespace {
std::shared_ptr<kernel::Kernel> CustomAddCreator(const std::vector<MSTensor> &inputs,
                                                 const std::vector<MSTensor> &outputs,
                                                 const schema::Primitive *primitive, const mindspore::Context *ctx) {
  bool fp16_enable = false;

  std::cout << "using fp32 add.\n" << std::endl;
  return std::make_shared<CustomAddKernel>(inputs, outputs, primitive, ctx, fp16_enable);
}

std::shared_ptr<kernel::KernelInterface> CustomAddInferCreator() { return std::make_shared<CustomAddInfer>(); }
}  // namespace

REGISTER_CUSTOM_KERNEL_INTERFACE(BuiltInTest, Custom_Add, CustomAddInferCreator)
// Register custom “Custom_Add” operator
REGISTER_CUSTOM_KERNEL(GPU, BuiltInTest, kFloat32, Custom_Add, CustomAddCreator)

class TestGPURegistryCustomOp : public mindspore::CommonTest {
 public:
  TestGPURegistryCustomOp() = default;
};

TEST_F(TestGPURegistryCustomOp, TestGPUCustomAdd) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_Custom;
  auto primitive = new schema::CustomT;
  primitive->type = "Custom_Add";
  node->primitive->value.value = primitive;
  node->name = "Add";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0, 1};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 28, 28, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = lite::NodeType_ValueNode;
  weight->format = schema::Format_NHWC;
  weight->dataType = TypeId::kNumberTypeFloat32;
  weight->dims = {1, 28, 28, 3};

  weight->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(weight));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(output));

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  // create a context
  auto context = std::make_shared<mindspore::Context>();
  context->SetThreadNum(1);
  context->SetEnableParallel(false);
  context->SetThreadAffinity(lite::HIGHER_CPU);
  auto &device_list = context->MutableDeviceInfo();

  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_info->SetEnableFP16(false);
  device_list.push_back(device_info);

  std::shared_ptr<GPUDeviceInfo> provider_gpu_device_info = std::make_shared<GPUDeviceInfo>();
  provider_gpu_device_info->SetEnableFP16(false);
  provider_gpu_device_info->SetProviderDevice("GPU");
  provider_gpu_device_info->SetProvider("BuiltInTest");
  device_list.push_back(provider_gpu_device_info);

  // build a model
  auto model = std::make_shared<mindspore::Model>();
  auto ret = model->Build(content, size, kMindIR_Lite, context);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  auto inputs = model->GetInputs();
  ASSERT_EQ(inputs.size(), 2);
  auto inTensor = inputs.front();
  auto impl = inTensor.impl();
  ASSERT_NE(nullptr, impl);
  float *in0_data = static_cast<float *>(inTensor.MutableData());
  in0_data[0] = 10.0f;
  auto inTensor1 = inputs.back();
  impl = inTensor1.impl();
  ASSERT_NE(nullptr, impl);
  float *in1_data = static_cast<float *>(inTensor1.MutableData());
  in1_data[0] = 20.0f;
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  impl = outputs.front().impl();
  ASSERT_NE(nullptr, impl);
  ASSERT_EQ(28 * 28 * 3, outputs.front().ElementNum());
  ASSERT_EQ(DataType::kNumberTypeFloat32, outputs.front().DataType());
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_NE(nullptr, outData);
  ASSERT_EQ(30.0f, outData[0]);
  MS_LOG(INFO) << "Register add op test pass.";
}
}  // namespace mindspore
