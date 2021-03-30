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

#include <iostream>
#include <cstring>
#include <random>
#include <fstream>
#include <thread>
#include <algorithm>
#include "include/errorcode.h"
#include "include/model.h"
#include "include/context.h"
#include "include/lite_session.h"
#include "include/version.h"

std::string RealPath(const char *path) {
  const size_t max = 4096;
  if (path == nullptr) {
    std::cerr << "path is nullptr" << std::endl;
    return "";
  }
  if ((strlen(path)) >= max) {
    std::cerr << "path is too long" << std::endl;
    return "";
  }
  auto resolved_path = std::make_unique<char[]>(max);
  if (resolved_path == nullptr) {
    std::cerr << "new resolved_path failed" << std::endl;
    return "";
  }
#ifdef _WIN32
  char *real_path = _fullpath(resolved_path.get(), path, 1024);
#else
  char *real_path = realpath(path, resolved_path.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    std::cerr << "file path is not valid : " << path << std::endl;
    return "";
  }
  std::string res = resolved_path.get();
  return res;
}

char *ReadFile(const char *file, size_t *size) {
  if (file == nullptr) {
    std::cerr << "file is nullptr." << std::endl;
    return nullptr;
  }

  std::ifstream ifs(file);
  if (!ifs.good()) {
    std::cerr << "file: " << file << " is not exist." << std::endl;
    return nullptr;
  }

  if (!ifs.is_open()) {
    std::cerr << "file: " << file << " open failed." << std::endl;
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<char[]> buf(new (std::nothrow) char[*size]);
  if (buf == nullptr) {
    std::cerr << "malloc buf failed, file: " << file << std::endl;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), *size);
  ifs.close();

  return buf.release();
}

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  if (data == nullptr) {
    std::cerr << "data is nullptr." << std::endl;
    return;
  }
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&]() { return static_cast<T>(distribution(random_engine)); });
}

std::shared_ptr<mindspore::lite::Context> CreateCPUContext() {
  auto context = std::make_shared<mindspore::lite::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed while running." << std::endl;
    return nullptr;
  }
  // Configure the number of worker threads in the thread pool to 2, including the main thread.
  context->thread_num_ = 2;
  // CPU device context has default values.
  auto &cpu_device_info = context->device_list_[0].device_info_.cpu_device_info_;
  // The large core takes priority in thread and core binding methods. This parameter will work in the BindThread
  // interface. For specific binding effect, see the "Run Graph" section.
  cpu_device_info.cpu_bind_mode_ = mindspore::lite::HIGHER_CPU;
  // Use float16 operator as priority.
  cpu_device_info.enable_float16_ = true;
  return context;
}

std::shared_ptr<mindspore::lite::Context> CreateGPUContext() {
  auto context = std::make_shared<mindspore::lite::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed while running. " << std::endl;
    return nullptr;
  }

  // If GPU device context is set. The preferred backend is GPU, which means, if there is a GPU operator, it will run on
  // the GPU first, otherwise it will run on the CPU.
  mindspore::lite::DeviceContext gpu_device_ctx{mindspore::lite::DT_GPU, {false}};
  // GPU use float16 operator as priority.
  gpu_device_ctx.device_info_.gpu_device_info_.enable_float16_ = true;
  // The GPU device context needs to be push_back into device_list to work.
  context->device_list_.push_back(gpu_device_ctx);
  return context;
}

std::shared_ptr<mindspore::lite::Context> CreateNPUContext() {
  auto context = std::make_shared<mindspore::lite::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed while running. " << std::endl;
    return nullptr;
  }
  mindspore::lite::DeviceContext npu_device_ctx{mindspore::lite::DT_NPU};
  npu_device_ctx.device_info_.npu_device_info_.frequency_ = 3;
  // The NPU device context needs to be push_back into device_list to work.
  context->device_list_.push_back(npu_device_ctx);
  return context;
}

int GetInputsAndSetData(mindspore::session::LiteSession *session) {
  auto inputs = session->GetInputs();

  // The model has only one input tensor.
  auto in_tensor = inputs.front();
  if (in_tensor == nullptr) {
    std::cerr << "Input tensor is nullptr" << std::endl;
    return -1;
  }
  auto input_data = in_tensor->MutableData();
  if (input_data == nullptr) {
    std::cerr << "MallocData for inTensor failed." << std::endl;
    return -1;
  }
  GenerateRandomData<float>(in_tensor->Size(), input_data, std::uniform_real_distribution<float>(0.1f, 1.0f));

  return 0;
}

int GetInputsByTensorNameAndSetData(mindspore::session::LiteSession *session) {
  auto in_tensor = session->GetInputsByTensorName("2031_2030_1_construct_wrapper:x");
  if (in_tensor == nullptr) {
    std::cerr << "Input tensor is nullptr" << std::endl;
    return -1;
  }
  auto input_data = in_tensor->MutableData();
  if (input_data == nullptr) {
    std::cerr << "MallocData for inTensor failed." << std::endl;
    return -1;
  }
  GenerateRandomData<float>(in_tensor->Size(), input_data, std::uniform_real_distribution<float>(0.1f, 1.0f));
  return 0;
}

void GetOutputsByNodeName(mindspore::session::LiteSession *session) {
  // model has a output node named output_node_name_0.
  auto output_vec = session->GetOutputsByNodeName("Default/head-MobileNetV2Head/Softmax-op204");
  // output node named output_node_name_0 has only one output tensor.
  auto out_tensor = output_vec.front();
  if (out_tensor == nullptr) {
    std::cerr << "Output tensor is nullptr" << std::endl;
    return;
  }
  std::cout << "tensor size is:" << out_tensor->Size() << " tensor elements num is:" << out_tensor->ElementsNum()
            << std::endl;
  // The model output data is float 32.
  if (out_tensor->data_type() != mindspore::TypeId::kNumberTypeFloat32) {
    std::cerr << "Output should in float32" << std::endl;
    return;
  }
  auto out_data = reinterpret_cast<float *>(out_tensor->MutableData());
  if (out_data == nullptr) {
    std::cerr << "Data of out_tensor is nullptr" << std::endl;
    return;
  }
  std::cout << "output data is:";
  for (int i = 0; i < out_tensor->ElementsNum() && i < 10; i++) {
    std::cout << out_data[i] << " ";
  }
  std::cout << std::endl;
}

void GetOutputByTensorName(mindspore::session::LiteSession *session) {
  // We can use GetOutputTensorNames method to get all name of output tensor of model which is in order.
  auto tensor_names = session->GetOutputTensorNames();
  // Use output tensor name returned by GetOutputTensorNames as key
  for (const auto &tensor_name : tensor_names) {
    auto out_tensor = session->GetOutputByTensorName(tensor_name);
    if (out_tensor == nullptr) {
      std::cerr << "Output tensor is nullptr" << std::endl;
      return;
    }
    std::cout << "tensor size is:" << out_tensor->Size() << " tensor elements num is:" << out_tensor->ElementsNum()
              << std::endl;
    // The model output data is float 32.
    if (out_tensor->data_type() != mindspore::TypeId::kNumberTypeFloat32) {
      std::cerr << "Output should in float32" << std::endl;
      return;
    }
    auto out_data = reinterpret_cast<float *>(out_tensor->MutableData());
    if (out_data == nullptr) {
      std::cerr << "Data of out_tensor is nullptr" << std::endl;
      return;
    }
    std::cout << "output data is:";
    for (int i = 0; i < out_tensor->ElementsNum() && i < 10; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
}

void GetOutputs(mindspore::session::LiteSession *session) {
  auto out_tensors = session->GetOutputs();
  for (auto out_tensor : out_tensors) {
    std::cout << "tensor name is:" << out_tensor.first << " tensor size is:" << out_tensor.second->Size()
              << " tensor elements num is:" << out_tensor.second->ElementsNum() << std::endl;
    // The model output data is float 32.
    if (out_tensor.second->data_type() != mindspore::TypeId::kNumberTypeFloat32) {
      std::cerr << "Output should in float32" << std::endl;
      return;
    }
    auto out_data = reinterpret_cast<float *>(out_tensor.second->MutableData());
    if (out_data == nullptr) {
      std::cerr << "Data of out_tensor is nullptr" << std::endl;
      return;
    }
    std::cout << "output data is:";
    for (int i = 0; i < out_tensor.second->ElementsNum() && i < 10; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
}

mindspore::session::LiteSession *CreateSessionAndCompileByModel(mindspore::lite::Model *model) {
  // Create and init CPU context.
  // If you need to use GPU or NPU, you can refer to CreateGPUContext() or CreateNPUContext().
  auto context = CreateCPUContext();
  if (context == nullptr) {
    std::cerr << "New context failed while." << std::endl;
    return nullptr;
  }

  // Create the session.
  mindspore::session::LiteSession *session = mindspore::session::LiteSession::CreateSession(context.get());
  if (session == nullptr) {
    std::cerr << "CreateSession failed while running." << std::endl;
    return nullptr;
  }

  // Compile graph.
  auto ret = session->CompileGraph(model);
  if (ret != mindspore::lite::RET_OK) {
    delete session;
    std::cerr << "Compile failed while running." << std::endl;
    return nullptr;
  }

  return session;
}

mindspore::session::LiteSession *CreateSessionAndCompileByModelBuffer(char *model_buf, size_t size) {
  auto context = std::make_shared<mindspore::lite::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed while running" << std::endl;
    return nullptr;
  }
  // Use model buffer and context to create Session.
  auto session = mindspore::session::LiteSession::CreateSession(model_buf, size, context.get());
  if (session == nullptr) {
    std::cerr << "CreateSession failed while running" << std::endl;
    return nullptr;
  }
  return session;
}

int ResizeInputsTensorShape(mindspore::session::LiteSession *session) {
  auto inputs = session->GetInputs();
  std::vector<int> resize_shape = {1, 128, 128, 3};
  // Assume the model has only one input,resize input shape to [1, 128, 128, 3]
  std::vector<std::vector<int>> new_shapes;
  new_shapes.push_back(resize_shape);
  return session->Resize(inputs, new_shapes);
}

int Run(const char *model_path) {
  // Read model file.
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }
  // Load the .ms model.
  auto model = mindspore::lite::Model::Import(model_buf, size);
  delete[](model_buf);
  if (model == nullptr) {
    std::cerr << "Import model file failed." << std::endl;
    return -1;
  }
  // Compile MindSpore Lite model.
  auto session = CreateSessionAndCompileByModel(model);
  if (session == nullptr) {
    delete model;
    std::cerr << "Create session failed." << std::endl;
    return -1;
  }

  // Note: when use model->Free(), the model can not be compiled again.
  model->Free();

  // Set inputs data.
  // You can also get input through other methods, and you can refer to GetInputsAndSetData()
  GetInputsByTensorNameAndSetData(session);

  session->BindThread(true);
  auto ret = session->RunGraph();
  if (ret != mindspore::lite::RET_OK) {
    delete model;
    delete session;
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }
  session->BindThread(false);

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(session);

  // Delete model buffer.
  delete model;
  // Delete session buffer.
  delete session;
  return 0;
}

int RunResize(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }
  // Load the .ms model.
  auto model = mindspore::lite::Model::Import(model_buf, size);
  delete[](model_buf);
  if (model == nullptr) {
    std::cerr << "Import model file failed." << std::endl;
    return -1;
  }
  // Compile MindSpore Lite model.
  auto session = CreateSessionAndCompileByModel(model);
  if (session == nullptr) {
    delete model;
    std::cerr << "Create session failed." << std::endl;
    return -1;
  }

  // Resize inputs tensor shape.
  auto ret = ResizeInputsTensorShape(session);
  if (ret != mindspore::lite::RET_OK) {
    delete model;
    delete session;
    std::cerr << "Resize input tensor shape error." << ret << std::endl;
    return ret;
  }

  // Set inputs data.
  // You can also get input through other methods, and you can refer to GetInputsAndSetData()
  GetInputsByTensorNameAndSetData(session);

  session->BindThread(true);
  ret = session->RunGraph();
  if (ret != mindspore::lite::RET_OK) {
    delete model;
    delete session;
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }
  session->BindThread(false);

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(session);

  // Delete model buffer.
  delete model;
  // Delete session buffer.
  delete session;
  return 0;
}

int RunCreateSessionSimplified(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }

  // Compile MindSpore Lite model.
  auto session = CreateSessionAndCompileByModelBuffer(model_buf, size);
  if (session == nullptr) {
    std::cerr << "Create session failed." << std::endl;
    return -1;
  }

  // Set inputs data.
  // You can also get input through other methods, and you can refer to GetInputsAndSetData()
  GetInputsByTensorNameAndSetData(session);

  session->BindThread(true);
  auto ret = session->RunGraph();
  if (ret != mindspore::lite::RET_OK) {
    delete session;
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }
  session->BindThread(false);

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(session);

  // Delete session buffer.
  delete session;
  return 0;
}

int RunSessionParallel(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }
  // Load the .ms model.
  auto model = mindspore::lite::Model::Import(model_buf, size);
  delete[](model_buf);
  if (model == nullptr) {
    std::cerr << "Import model file failed." << std::endl;
    return -1;
  }
  // Compile MindSpore Lite model.
  auto session1 = CreateSessionAndCompileByModel(model);
  if (session1 == nullptr) {
    delete model;
    std::cerr << "Create session failed." << std::endl;
    return -1;
  }

  // Compile MindSpore Lite model.
  auto session2 = CreateSessionAndCompileByModel(model);
  if (session2 == nullptr) {
    delete model;
    std::cerr << "Create session failed." << std::endl;
    return -1;
  }
  // Note: when use model->Free(), the model can not be compiled again.
  model->Free();

  std::thread thread1([&]() {
    GetInputsByTensorNameAndSetData(session1);
    auto status = session1->RunGraph();
    if (status != 0) {
      std::cerr << "Inference error " << status << std::endl;
      return;
    }
    std::cout << "Session1 inference success" << std::endl;
  });

  std::thread thread2([&]() {
    GetInputsByTensorNameAndSetData(session2);
    auto status = session2->RunGraph();
    if (status != 0) {
      std::cerr << "Inference error " << status << std::endl;
      return;
    }
    std::cout << "Session2 inference success" << std::endl;
  });

  thread1.join();
  thread2.join();

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(session1);
  GetOutputsByNodeName(session2);

  // Delete model buffer.
  if (model != nullptr) {
    delete model;
    model = nullptr;
  }
  // Delete session buffer.
  delete session1;
  delete session2;
  return 0;
}

int RunWithSharedMemoryPool(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }
  auto model = mindspore::lite::Model::Import(model_buf, size);
  delete[](model_buf);
  if (model == nullptr) {
    std::cerr << "Import model file failed." << std::endl;
    return -1;
  }

  auto context1 = std::make_shared<mindspore::lite::Context>();
  if (context1 == nullptr) {
    delete model;
    std::cerr << "New context failed while running." << std::endl;
    return -1;
  }
  auto session1 = mindspore::session::LiteSession::CreateSession(context1.get());
  if (session1 == nullptr) {
    delete model;
    std::cerr << "CreateSession failed while running." << std::endl;
    return -1;
  }
  auto ret = session1->CompileGraph(model);
  if (ret != mindspore::lite::RET_OK) {
    delete model;
    delete session1;
    std::cerr << "Compile failed while running." << std::endl;
    return -1;
  }

  auto context2 = std::make_shared<mindspore::lite::Context>();
  if (context2 == nullptr) {
    delete model;
    std::cerr << "New  context failed while running." << std::endl;
    return -1;
  }
  // Use the same allocator to share the memory pool.
  context2->allocator = context1->allocator;

  auto session2 = mindspore::session::LiteSession::CreateSession(context2.get());
  if (session2 == nullptr) {
    delete model;
    delete session1;
    std::cerr << "CreateSession failed while running " << std::endl;
    return -1;
  }

  ret = session2->CompileGraph(model);
  if (ret != mindspore::lite::RET_OK) {
    delete model;
    delete session1;
    delete session2;
    std::cerr << "Compile failed while running " << std::endl;
    return -1;
  }

  // Note: when use model->Free(), the model can not be compiled again.
  model->Free();

  // Set inputs data.
  // You can also get input through other methods, and you can refer to GetInputsAndSetData()
  GetInputsByTensorNameAndSetData(session1);
  GetInputsByTensorNameAndSetData(session2);

  ret = session1->RunGraph();
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }

  ret = session2->RunGraph();
  if (ret != mindspore::lite::RET_OK) {
    delete model;
    delete session1;
    delete session2;
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(session1);
  GetOutputsByNodeName(session2);

  // Delete model buffer.
  delete model;
  // Delete session buffer.
  delete session1;
  delete session2;
  return 0;
}

int RunCallback(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }
  // Load the .ms model.
  auto model = mindspore::lite::Model::Import(model_buf, size);
  delete[](model_buf);
  if (model == nullptr) {
    std::cerr << "Import model file failed." << std::endl;
    return -1;
  }
  // Compile MindSpore Lite model.
  auto session = CreateSessionAndCompileByModel(model);
  if (session == nullptr) {
    delete model;
    std::cerr << "Create session failed." << std::endl;
    return -1;
  }

  // Note: when use model->Free(), the model can not be compiled again.
  model->Free();

  // Set inputs data.
  // You can also get input through other methods, and you can refer to GetInputsAndSetData()
  GetInputsByTensorNameAndSetData(session);

  // Definition of callback function before forwarding operator.
  auto before_call_back = [&](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                              const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                              const mindspore::CallBackParam &call_param) {
    std::cout << "Before forwarding " << call_param.node_name << " " << call_param.node_type << std::endl;
    return true;
  };
  // Definition of callback function after forwarding operator.
  auto after_call_back = [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                             const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                             const mindspore::CallBackParam &call_param) {
    std::cout << "After forwarding " << call_param.node_name << " " << call_param.node_type << std::endl;
    return true;
  };

  session->BindThread(true);
  auto ret = session->RunGraph(before_call_back, after_call_back);
  if (ret != mindspore::lite::RET_OK) {
    delete model;
    delete session;
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }
  session->BindThread(false);

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(session);

  // Delete model buffer.
  delete model;
  // Delete session buffer.
  delete session;
  return 0;
}

int main(int argc, const char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./runtime_cpp model_path Option" << std::endl;
    std::cerr << "Example: ./runtime_cpp ../model/mobilenetv2.ms 0" << std::endl;
    std::cerr << "When your Option is 0, you will run MindSpore Lite inference." << std::endl;
    std::cerr << "When your Option is 1, you will run MindSpore Lite inference with resize." << std::endl;
    std::cerr << "When your Option is 2, you will run MindSpore Lite inference with CreateSession simplified API."
              << std::endl;
    std::cerr << "When your Option is 3, you will run MindSpore Lite inference with session parallel." << std::endl;
    std::cerr << "When your Option is 4, you will run MindSpore Lite inference with shared memory pool." << std::endl;
    std::cerr << "When your Option is 5, you will run MindSpore Lite inference with callback." << std::endl;
    return -1;
  }
  std::string version = mindspore::lite::Version();
  std::cout << "MindSpore Lite Version is " << version << std::endl;
  auto model_path = RealPath(argv[1]);
  if (model_path.empty()) {
    std::cerr << "model path " << argv[1] << " is invalid.";
    return -1;
  }
  auto flag = argv[2];
  if (strcmp(flag, "0") == 0) {
    return Run(model_path.c_str());
  } else if (strcmp(flag, "1") == 0) {
    return RunResize(model_path.c_str());
  } else if (strcmp(flag, "2") == 0) {
    return RunCreateSessionSimplified(model_path.c_str());
  } else if (strcmp(flag, "3") == 0) {
    return RunSessionParallel(model_path.c_str());
  } else if (strcmp(flag, "4") == 0) {
    return RunWithSharedMemoryPool(model_path.c_str());
  } else if (strcmp(flag, "5") == 0) {
    return RunCallback(model_path.c_str());
  } else {
    std::cerr << "Unsupported Flag " << flag << std::endl;
    return -1;
  }
}
