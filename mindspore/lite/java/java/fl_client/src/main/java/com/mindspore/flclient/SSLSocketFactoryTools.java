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

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.security.InvalidKeyException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.SignatureException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.util.logging.Logger;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSession;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

/**
 * Define SSL socket factory tools for https communication.
 *
 * @since 2021-06-30
 */
public class SSLSocketFactoryTools {
    private static final Logger LOGGER = Logger.getLogger(SSLSocketFactory.class.toString());
    private static volatile SSLSocketFactoryTools sslSocketFactoryTools;

    private FLParameter flParameter = FLParameter.getInstance();
    private X509Certificate x509Certificate;
    private SSLSocketFactory sslSocketFactory;
    private SSLContext sslContext;
    private MyTrustManager myTrustManager;
    private final HostnameVerifier hostnameVerifier = new HostnameVerifier() {
        @Override
        public boolean verify(String hostname, SSLSession session) {
            if (hostname == null || hostname.isEmpty()) {
                LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] the parameter of <hostname> is null or empty, " +
                        "please check!"));
                throw new IllegalArgumentException();
            }
            if (session == null) {
                LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] the parameter of <session> is null, please " +
                        "check!"));
                throw new IllegalArgumentException();
            }
            String domainName = flParameter.getDomainName();
            if ((domainName == null || domainName.isEmpty() || domainName.split("//").length < 2)) {
                LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] the <domainName> is null or not valid, it should" +
                        " be like as https://...... , please check!"));
                throw new IllegalArgumentException();
            }
            if (domainName.split("//")[1].split(":").length < 2) {
                LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] the format of <domainName> is not valid, it " +
                        "should be like as https://127.0.0.1:6666 when setting <useSSL> to true, please check!"));
                throw new IllegalArgumentException();
            }
            String ip = domainName.split("//")[1].split(":")[0];
            return hostname.equals(ip);
        }
    };

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
            }, Common.getSecureRandom());
            sslSocketFactory = sslContext.getSocketFactory();
        } catch (NoSuchAlgorithmException | KeyManagementException ex) {
            LOGGER.severe(Common.addTag("[SSLSocketFactoryTools]catch Exception in initSslSocketFactory: " +
                    ex.getMessage()));
        }
    }

    /**
     * Get the singleton object of the class SSLSocketFactoryTools.
     *
     * @return the singleton object of the class SSLSocketFactoryTools.
     */
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

    private X509Certificate readCert(String assetName) {
        if (assetName == null || assetName.isEmpty()) {
            LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] the parameter of <assetName> is null or empty, " +
                    "please check!"));
            return null;
        }
        InputStream inputStream = null;
        X509Certificate cert = null;
        try {
            inputStream = new FileInputStream(assetName);
            CertificateFactory cf = CertificateFactory.getInstance("X.509");
            Certificate certificate = cf.generateCertificate(inputStream);
            if (certificate instanceof X509Certificate) {
                cert = (X509Certificate) certificate;
            } else {
                LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] cf.generateCertificate(inputStream) can not " +
                        "convert to X509Certificate"));
            }
        } catch (FileNotFoundException | CertificateException ex) {
            LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] catch FileNotFoundException or CertificateException " +
                    "when creating " +
                    "CertificateFactory in readCert: " + ex.getMessage()));
        } finally {
            try {
                if (inputStream != null) {
                    inputStream.close();
                }
            } catch (IOException ex) {
                LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] catch IOException: " + ex.getMessage()));
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
                    cert.verify(this.cert.getPublicKey());
                } catch (NoSuchAlgorithmException e) {
                    LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] checkServerTrusted failed, catch " +
                            "NoSuchAlgorithmException in checkServerTrusted: " + e.getMessage()));
                } catch (InvalidKeyException e) {
                    LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] checkServerTrusted failed, catch " +
                            "InvalidKeyException in checkServerTrusted: " + e.getMessage()));
                } catch (NoSuchProviderException e) {
                    LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] checkServerTrusted failed, catch " +
                            "NoSuchProviderException in checkServerTrusted: " + e.getMessage()));
                } catch (SignatureException e) {
                    LOGGER.severe(Common.addTag("[SSLSocketFactoryTools] checkServerTrusted failed, catch " +
                            "SignatureException in checkServerTrusted: " + e.getMessage()));
                }
                LOGGER.info(Common.addTag("**********************checkServerTrusted success!**********************"));
            }
        }

        @Override
        public X509Certificate[] getAcceptedIssuers() {
            return new java.security.cert.X509Certificate[0];
        }
    }
}
