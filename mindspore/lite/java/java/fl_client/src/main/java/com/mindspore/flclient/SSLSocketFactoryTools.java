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
package com.mindspore.flclient;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSession;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.SignatureException;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.util.logging.Logger;

public class SSLSocketFactoryTools {
    private static final Logger LOGGER = Logger.getLogger(SSLSocketFactory.class.toString());
    private FLParameter flParameter = FLParameter.getInstance();
    private X509Certificate x509Certificate;
    private SSLSocketFactory sslSocketFactory;
    private SSLContext sslContext;
    private MyTrustManager myTrustManager;
    private static volatile SSLSocketFactoryTools sslSocketFactoryTools;

    private SSLSocketFactoryTools() {
        initSslSocketFactory();
    }

    private void initSslSocketFactory() {
        try {
            sslContext = SSLContext.getInstance("TLS");
            x509Certificate = readCert(flParameter.getCertPath());
            myTrustManager = new MyTrustManager(x509Certificate);
            sslContext.init(null, new TrustManager[]{
                    myTrustManager
            }, new java.security.SecureRandom());
            sslSocketFactory = sslContext.getSocketFactory();

        } catch (Exception e) {
            LOGGER.severe(Common.addTag("[SSLSocketFactoryTools]catch Exception in initSslSocketFactory: " + e.getMessage()));
        }
    }

    public static SSLSocketFactoryTools getInstance() {
        SSLSocketFactoryTools localRef = sslSocketFactoryTools;
        if (localRef == null) {
            synchronized (SSLSocketFactoryTools.class) {
                localRef = sslSocketFactoryTools;
                if (localRef == null) {
                    sslSocketFactoryTools = localRef = new SSLSocketFactoryTools();
                }
            }
        }
        return localRef;
    }

    public X509Certificate readCert(String assetName) {
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream(assetName);
        } catch (Exception e) {
            LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] catch Exception of read inputStream in readCert: " + e.getMessage()));
            return null;
        }
        X509Certificate cert = null;
        try {
            CertificateFactory cf = CertificateFactory.getInstance("X.509");
            cert = (X509Certificate) cf.generateCertificate(inputStream);
        } catch (Exception e) {
            LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] catch Exception of creating CertificateFactory in readCert: " + e.getMessage()));
        } finally {
            try {
                if (inputStream != null) {
                    inputStream.close();
                }
            } catch (Throwable ex) {
            }
        }

        return cert;
    }

    public HostnameVerifier getHostnameVerifier() {
        return hostnameVerifier;
    }

    public SSLSocketFactory getmSslSocketFactory() {
        return sslSocketFactory;
    }

    public MyTrustManager getmTrustManager() {
        return myTrustManager;
    }


    private static final class MyTrustManager implements X509TrustManager {
        X509Certificate cert;

        MyTrustManager(X509Certificate cert) {
            this.cert = cert;
        }

        @Override
        public void checkClientTrusted(X509Certificate[] chain, String authType) throws CertificateException {
        }

        @Override
        public void checkServerTrusted(X509Certificate[] chain, String authType) throws CertificateException {
            for (X509Certificate cert : chain) {

                // Make sure that it hasn't expired.
                cert.checkValidity();

                // Verify the certificate's public key chain.
                try {
                    cert.verify(((X509Certificate) this.cert).getPublicKey());
                } catch (NoSuchAlgorithmException e) {
                    LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] catch NoSuchAlgorithmException in checkServerTrusted: " + e.getMessage()));
                    throw new RuntimeException();
                } catch (InvalidKeyException e) {
                    LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] catch InvalidKeyException in checkServerTrusted: " + e.getMessage()));
                    throw new RuntimeException();
                } catch (NoSuchProviderException e) {
                    LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] catch NoSuchProviderException in checkServerTrusted: " + e.getMessage()));
                    throw new RuntimeException();
                } catch (SignatureException e) {
                    LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] catch SignatureException in checkServerTrusted: " + e.getMessage()));
                    throw new RuntimeException();
                }
                LOGGER.info(Common.addTag("checkServerTrusted success!"));
            }
        }

        @Override
        public X509Certificate[] getAcceptedIssuers() {
            return new java.security.cert.X509Certificate[0];
        }
    }

    private final HostnameVerifier hostnameVerifier = new HostnameVerifier() {
        @Override
        public boolean verify(String hostname, SSLSession session) {
            LOGGER.info(Common.addTag("[SSLSocketFactoryTools] server hostname: " + flParameter.getHostName()));
            LOGGER.info(Common.addTag("[SSLSocketFactoryTools] client request hostname: " + hostname));
            return hostname.equals(flParameter.getHostName());
        }
    };

}
