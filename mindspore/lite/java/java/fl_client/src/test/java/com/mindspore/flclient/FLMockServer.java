package com.mindspore.flclient;

import com.mindspore.flclient.common.FLLoggerGenerater;
import mindspore.schema.RequestFLJob;
import mindspore.schema.ResponseFLJob;
import mindspore.schema.ResponseGetModel;
import mindspore.schema.ResponseUpdateModel;
import okhttp3.mockwebserver.Dispatcher;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import okio.Buffer;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.logging.Logger;

/**
 * Using this class to Mock the FL server node
 * @author       : zhangzhaoju
 * @since  : 2022/4/14
 */
class FLMockServer {
    private static final Logger LOGGER = FLLoggerGenerater.getModelLogger(FLMockServer.class.toString());
    private ArrayList<FLHttpRes> httpRes;
    private int httpResCnt = 0;
    private int curIter = 0;
    private int maxIter = 1;
    private MockWebServer server = new MockWebServer();
    private final Dispatcher dispatcher = new Dispatcher() {
        @Override
        public MockResponse dispatch(RecordedRequest request) throws InterruptedException {
            if (httpRes == null) {
                LOGGER.severe("httpRes size is:" + Integer.toString(httpRes.size()) + " httpResCnt is:" + Integer.toString(httpResCnt));
                return new MockResponse().setResponseCode(404);
            }

            FLHttpRes curRes = httpRes.get(httpResCnt % httpRes.size());
            httpResCnt++;

            if(!reqMsgCheck(request, curRes)){
                return new MockResponse().setResponseCode(404);
            }

            if(curRes.getContendMode() != 0){
                return new MockResponse().setResponseCode(curRes.getResCode()).setBody(curRes.getContentData());
            }

            Buffer res = genResMsgBody(curRes);
            return new MockResponse().setResponseCode(curRes.getResCode()).setBody(res);
        }
    };

    public void setCaseRes(ArrayList<FLHttpRes> httpRes) {
        this.httpRes = httpRes;
        httpResCnt = 0;
    }

    private boolean reqMsgCheck(RecordedRequest request, FLHttpRes curRes){
        // check msg type
        if (!request.getPath().equals("/" + curRes.getResName())) {
            LOGGER.severe("The " + Integer.toString(httpResCnt) + "th expect msg is :" + curRes.getResName() + " but got " + request.getPath());
            return false;
        }
        // check msg content
        String msgName = curRes.getResName();
        if (msgName.equals("startFLJob")) {
            byte[] reqBody = request.getBody().readByteArray();
            ByteBuffer reqBuffer = ByteBuffer.wrap(reqBody);
            RequestFLJob sDataBuf = RequestFLJob.getRootAsRequestFLJob(reqBuffer);
            return true;
        }

        if (msgName.equals("updateModel")) {
            // do check for updateModel
            return true;
        }

        if (msgName.equals("getModel")) {
            // do check for getModel
            return true;
        }

        LOGGER.severe("Got unsupported msg " + request.getPath());
        return false;
    }

    private Buffer genResMsgBody(FLHttpRes curRes) {
         String msgName = curRes.getResName();
        byte[] msgBody = getMsgBodyFromFile(curRes.getContentData());
        if (msgName.equals("startFLJob")){
            curIter++;
            ByteBuffer resBuffer = ByteBuffer.wrap(msgBody);
            ResponseFLJob responseDataBuf = ResponseFLJob.getRootAsResponseFLJob(resBuffer);
            responseDataBuf.flPlanConfig().mutateEpochs(1);  // change the flbuffer
            responseDataBuf.flPlanConfig().mutateIterations(maxIter); // only hase one iteration
            responseDataBuf.mutateIteration(curIter); // cur iteration
            Buffer buffer = new Buffer();
            buffer.write(responseDataBuf.getByteBuffer().array());
            return buffer;
        }

        if(msgName.equals( "updateModel")){
            Buffer buffer = new Buffer();
            ByteBuffer resBuffer = ByteBuffer.wrap(msgBody);
            ResponseUpdateModel responseDataBuf = ResponseUpdateModel.getRootAsResponseUpdateModel(resBuffer);
            buffer.write(responseDataBuf.getByteBuffer().array());
            return buffer;
        }

        if(msgName.equals( "getModel")){
            Buffer buffer = new Buffer();
            ByteBuffer resBuffer = ByteBuffer.wrap(msgBody);
            ResponseGetModel responseDataBuf = ResponseGetModel.getRootAsResponseGetModel(resBuffer);
            responseDataBuf.mutateIteration(curIter);
            buffer.write(responseDataBuf.getByteBuffer().array());
            return buffer;
        }
        return new Buffer();
    }

    private byte[] getMsgBodyFromFile(String resFileName) {
        byte[] res = null;
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(resFileName));
            res = (byte[]) ois.readObject();
            ois.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        return res;
    }

    public void run(int port) {
        server.setDispatcher(dispatcher);
        try {
            server.start(port);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public void stop() {
        try {
            server.close();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(0);
        }
    }
}
