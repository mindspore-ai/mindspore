#include "MSNetWork.h"
#include <iostream>
#include <android/log.h>
#include "errorcode.h"

#define MS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MSJNI", format, ##__VA_ARGS__)

MSNetWork::MSNetWork(void) : session(nullptr) {}
MSNetWork::~MSNetWork(void) {}


void MSNetWork::CreateSessionMS(char* modelBuffer, size_t bufferLen, mindspore::lite::Context* ctx)
{
    session = mindspore::session::LiteSession::CreateSession(ctx);
    if (session == nullptr){
        MS_PRINT("Create Session failed.");
        return;
    }

    // Compile model.
    auto model = mindspore::lite::Model::Import(modelBuffer, bufferLen);
    if (model == nullptr){
        MS_PRINT("Import model failed.");
        return;
    }

    int ret = session->CompileGraph(model);
    if (ret != mindspore::lite::RET_OK){
        MS_PRINT("CompileGraph failed.");
        return;
    }

}

int MSNetWork::ReleaseNets(void)
{
    delete session;
//    delete model;
    return 0;
}

