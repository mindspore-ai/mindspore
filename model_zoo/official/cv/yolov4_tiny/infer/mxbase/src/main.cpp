/*
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
#include <vector>
#include "Yolov4TinyDetection.h"
#include "MxBase/Log/Log.h"

std::vector<double> g_inferCost;

void SplitString(const std::string &s, std::vector<std::string> *v, const std::string &c) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        v->push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }

    if (pos1 != s.length()) {
        v->push_back(s.substr(pos1));
    }
}

void InitYolov4TinyParam(InitParam *initParam) {
    initParam->deviceId = 0;
    initParam->labelPath = "../data/models/coco2017.names";
    initParam->checkTensor = true;
    initParam->modelPath = "../data/models/yolov4_tiny.om";
    initParam->classNum = 80;
    initParam->biasesNum = 12;
    initParam->biases = "10,14,23,27,37,58,81,82,135,169,344,319";
    initParam->objectnessThresh = "0.001";
    initParam->iouThresh = "0.45";
    initParam->scoreThresh = "0.001";
    initParam->yoloType = 2;
    initParam->modelType = 0;
    initParam->inputType = 0;
    initParam->anchorDim = 3;
}

APP_ERROR ReadImagesPath(const std::string &path, std::vector<std::string> *imagesPath) {
    std::ifstream inFile;
    inFile.open(path, std::ios_base::in);
    std::string line;
    // Check images path file validity
    if (inFile.fail()) {
        LogError << "Failed to open label file: " << path;
        return APP_ERR_COMM_OPEN_FAIL;
    }
    std::vector<std::string> vectorStr;
    std::string splitStr = " ";
    // construct label map
    while (std::getline(inFile, line)) {
        if (line[0] == '#') {
            continue;
        }
        vectorStr.clear();
        SplitString(line, &vectorStr, splitStr);
        imagesPath->push_back(vectorStr[1]);
    }

    inFile.close();
    return APP_ERR_OK;
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './yolov4tiny infer.txt'.";
        return APP_ERR_OK;
    }
    InitParam initParam;
    InitYolov4TinyParam(&initParam);
    auto yolov4tiny = std::make_shared<Yolov4TinyDetectionOpencv>();
    APP_ERROR ret = yolov4tiny->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "Yolov4TinyDetectionOpencv init failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "End to Init yolov4tiny.";
    std::string inferText = argv[1];
    std::vector<std::string> imagesPath;
    ret = ReadImagesPath(inferText, &imagesPath);
    if (ret != APP_ERR_OK) {
        LogError << "ReadImagesPath failed, ret=" << ret << ".";
        return ret;
    }
    for (uint32_t i = 0; i < imagesPath.size(); i++) {
        LogInfo << "read image path " << imagesPath[i];
        ret = yolov4tiny->Process(imagesPath[i]);
        if (ret != APP_ERR_OK) {
            LogError << "Yolov4TinyDetectionOpencv process failed, ret=" << ret << ".";
            yolov4tiny->DeInit();
            return ret;
        }
    }
    yolov4tiny->DeInit();
    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " images/sec.";
    return APP_ERR_OK;
}
