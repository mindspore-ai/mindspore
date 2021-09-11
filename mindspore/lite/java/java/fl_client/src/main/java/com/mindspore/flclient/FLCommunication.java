/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

package com.mindspore.flclient;

import static com.mindspore.flclient.FLParameter.TIME_OUT;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.logging.Logger;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSession;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

/**
 * Define the communication interface.
 *
 * @since 2021-06-30
 */
public class FLCommunication implements IFLCommunication {
    private static int timeOut;
    private static boolean ifCertificateVerify = false;
    private static final MediaType MEDIA_TYPE_JSON = MediaType.parse("applicatiom/json;charset=utf-8");
    private static final Logger LOGGER = Logger.getLogger(FLCommunication.class.toString());
    private static volatile FLCommunication communication;

    private FLParameter flParameter = FLParameter.getInstance();
    private OkHttpClient client;

    private FLCommunication() {
        if (flParameter.getTimeOut() != 0) {
            timeOut = flParameter.getTimeOut();
        } else {
            timeOut = TIME_OUT;
        }
        ifCertificateVerify = flParameter.isUseSSL();
        client = getOkHttpClient();
    }

    private static OkHttpClient getOkHttpClient() {
        X509TrustManager trustManager = new X509TrustManager() {
            @Override
            public X509Certificate[] getAcceptedIssuers() {
                return new X509Certificate[]{};
            }

            @Override
            public void checkServerTrusted(X509Certificate[] arg0, String arg1) throws CertificateException {

            }

            @Override
            public void checkClientTrusted(X509Certificate[] arg0, String arg1) throws CertificateException {

            }
        };
        final TrustManager[] trustAllCerts = new TrustManager[]{trustManager};
        try {
            LOGGER.info(Common.addTag("the set timeOut in OkHttpClient: " + timeOut));
            OkHttpClient.Builder builder = new OkHttpClient.Builder();
            builder.connectTimeout(timeOut, TimeUnit.SECONDS);
            builder.writeTimeout(timeOut, TimeUnit.SECONDS);
            builder.readTimeout(3 * timeOut, TimeUnit.SECONDS);
            if (ifCertificateVerify) {
                builder.sslSocketFactory(SSLSocketFactoryTools.getInstance().getmSslSocketFactory(),
                        SSLSocketFactoryTools.getInstance().getmTrustManager());
                builder.hostnameVerifier(SSLSocketFactoryTools.getInstance().getHostnameVerifier());
            } else {
                final SSLContext sslContext = SSLContext.getInstance("TLS");
                sslContext.init(null, trustAllCerts, Common.getSecureRandom());
                final SSLSocketFactory sslFactory = sslContext.getSocketFactory();
                builder.sslSocketFactory(sslFactory, trustManager);
                builder.hostnameVerifier(new HostnameVerifier() {
                    @Override
                    public boolean verify(String arg0, SSLSession arg1) {
                        return true;
                    }
                });
            }
            return builder.build();
        } catch (NoSuchAlgorithmException | KeyManagementException ex) {
            LOGGER.severe(Common.addTag("[OkHttpClient] catch NoSuchAlgorithmException or KeyManagementException: " + ex.getMessage()));
            throw new IllegalArgumentException(ex);
        }
    }

    /**
     * Get the singleton object of the class FLCommunication.
     *
     * @return the singleton object of the class FLCommunication.
     */
    public static FLCommunication getInstance() {
        FLCommunication localRef = communication;
        if (localRef == null) {
            synchronized (FLCommunication.class) {
                localRef = communication;
                if (localRef == null) {
                    communication = localRef = new FLCommunication();
                }
            }
        }
        return localRef;
    }

    @Override
    public void setTimeOut(int timeout) throws TimeoutException {
    }

    @Override
    public byte[] syncRequest(String url, byte[] msg) throws IOException {
        Request request = new Request.Builder()
                .url(url)
                .post(RequestBody.create(MEDIA_TYPE_JSON, msg)).build();
        Response response = this.client.newCall(request).execute();
        if (!response.isSuccessful()) {
            throw new IOException("Unexpected code " + response);
        }
        if (response.body() == null) {
            throw new IOException("the returned response is null");
        }
        return response.body().bytes();
    }

    @Override
    public void asyncRequest(String url, byte[] msg, IAsyncCallBack callBack) throws Exception {
        Request request = new Request.Builder()
                .url(url)
                .header("Accept", "application/proto")
                .header("Content-Type", "application/proto; charset=utf-8")
                .post(RequestBody.create(MEDIA_TYPE_JSON, msg)).build();

        client.newCall(request).enqueue(new Callback() {
            IAsyncCallBack asyncCallBack = callBack;

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                asyncCallBack.onResponse(response.body().bytes());
                call.cancel();
            }

            @Override
            public void onFailure(Call call, IOException ioException) {
                asyncCallBack.onFailure(ioException);
                call.cancel();
            }
        });
    }
}
