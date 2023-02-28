/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#if defined(__linux__) && !defined(Debug)
#include <csignal>
#endif
#define USE_DEPRECATED_API
#include <iostream>
#include "include/converter.h"
#include "include/api/status.h"
#include "tools/converter/converter_lite/converter_flags.h"

#if defined(__linux__) && !defined(Debug)
void SignalHandler(int sig) {
  printf("encounter an unknown error, please verify the input model file or build the debug version\n");
  exit(1);
}
#endif

int main(int argc, const char **argv) {
#if defined(__linux__) && !defined(Debug)
  signal(SIGSEGV, SignalHandler);
  signal(SIGABRT, SignalHandler);
  signal(SIGFPE, SignalHandler);
  signal(SIGBUS, SignalHandler);
#endif

#ifndef Debug
  try {
#endif
    mindspore::converter::Flags flags;
    auto ret = flags.PreInit(argc, argv);
    if (static_cast<uint32_t>(ret) == mindspore::kLiteSuccessExit) {
      return mindspore::kSuccess;
    } else if (ret != mindspore::kSuccess) {
      MS_LOG(ERROR) << "Flags PreInit failed. Ret: " << ret;
      std::cout << "Flags PreInit failed. Ret: " << ret << std::endl;
      return ret;
    }

    int multi_model_argc = 2;
    if (argc <= multi_model_argc) {
      mindspore::Converter converter;
      converter.SetConfigFile(flags.configFile);
      auto status = converter.Convert();
      if (status != mindspore::kSuccess) {
        MS_LOG(ERROR) << "Convert failed. Ret: " << status;
        std::cout << "Convert failed. Ret: " << status << std::endl;
      }
      return status.StatusCode();
    } else {
      ret = flags.Init(argc, argv);
      if (ret != mindspore::kSuccess) {
        MS_LOG(ERROR) << "Flags Init failed. Ret: " << ret;
        std::cout << "Flags Init failed. Ret: " << ret << std::endl;
        return ret;
      }
      mindspore::Converter converter(flags.fmk, flags.modelFile, flags.outputFile, flags.weightFile);
      converter.SetConfigFile(flags.configFile);

      converter.SetWeightFp16(flags.saveFP16);
      converter.SetInputShape(flags.graph_input_shape_map);
      converter.SetInputFormat(flags.graphInputFormat);
      converter.SetInputDataType(flags.inputDataType);
      converter.SetOutputDataType(flags.outputDataType);
      converter.SetSaveType(flags.save_type);
      converter.SetDecryptKey(flags.dec_key);
      flags.dec_key.clear();
      converter.SetDecryptMode(flags.dec_mode);
      converter.SetEnableEncryption(flags.encryption);
      converter.SetEncryptKey(flags.encKeyStr);
      flags.encKeyStr.clear();
      converter.SetInfer(flags.infer);
      converter.SetTrainModel(flags.trainModel);
      converter.SetNoFusion(flags.disableFusion);
      converter.SetDevice(flags.device);
      converter.SetOptimizeTransformer(flags.optimizeTransformer);

      auto status = converter.Convert();
      if (status != mindspore::kSuccess) {
        MS_LOG(ERROR) << "Convert failed. Ret: " << status;
        std::cout << "Convert failed. Ret: " << status << std::endl;
      }
      return status.StatusCode();
    }

#ifndef Debug
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
    std::cout << "encounter an unknown error, please verify the input model file or build the debug version\n";
    return mindspore::kLiteError;
  }
#endif
}
