/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.flclient.cipher;

import com.mindspore.flclient.Common;
import com.mindspore.flclient.FLParameter;

import org.bouncycastle.asn1.ASN1OctetString;
import org.bouncycastle.asn1.x509.AuthorityKeyIdentifier;
import org.bouncycastle.asn1.x509.SubjectKeyIdentifier;
import org.bouncycastle.util.encoders.Hex;

import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.InvalidKeyException;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.PublicKey;
import java.security.SignatureException;
import java.security.UnrecoverableEntryException;
import java.security.cert.CRLException;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509CRL;
import java.security.cert.X509CRLEntry;
import java.security.cert.X509Certificate;
import java.util.Base64;
import java.util.Date;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Certificate verification class
 *
 * @since 2021-8-27
 */
public class CertVerify {
    private static final Logger LOGGER = Logger.getLogger(CertVerify.class.toString());

    /**
     * Verify the legitimacy of certificate chain
     *
     * @param clientID clientID of this client
     * @param x509Certificates certificate chain
     * @return verification result
     */
    public static boolean verifyCertificateChain(String clientID, X509Certificate[] x509Certificates) {
        if (clientID == null || clientID.isEmpty()) {
            LOGGER.severe(Common.addTag("[CertVerify] the parameter clientID is null or empty, please check!"));
            return false;
        }
        if (x509Certificates == null || x509Certificates.length < 2) {
            LOGGER.severe(Common.addTag("[CertVerify] the parameter x509Certificates is null or the length is not " +
                    "valid: < 2, please check!"));
            return false;
        }
        if (verifyChain(clientID, x509Certificates) && verifyCommonName(clientID, x509Certificates)
                && verifyCrl(clientID, x509Certificates) && verifyValidDate(x509Certificates) &&
                verifyKeyIdentifier(clientID, x509Certificates)) {
            LOGGER.info(Common.addTag("[CertVerify] verifyCertificateChain success!"));
            return true;
        }
        LOGGER.severe(Common.addTag("[CertVerify] verifyCertificateChain failed!"));
        return false;
    }

    private static boolean verifyCommonName(String clientID, X509Certificate[] x509Certificate) {
        if (clientID == null || clientID.isEmpty()) {
            LOGGER.severe(Common.addTag("[CertVerify] the parameter clientID is null or empty, please check!"));
            return false;
        }
        if (x509Certificate == null || x509Certificate.length < 2) {
            LOGGER.severe(Common.addTag("[CertVerify] x509Certificate chains is null or the length is not valid: < 2," +
                    " please check!"));
            return false;
        }
        X509Certificate[] certificateChains = getX509CertificateChain(clientID);
        if (certificateChains == null || certificateChains.length < 4) {
            LOGGER.severe(Common.addTag("[CertVerify] certificateChains is null or the length is not valid: < 4, " +
                    "please check!"));
            return false;
        }
        X509Certificate localEquipCACert = certificateChains[2];
        // get subjectDN of local root equipment CA certificate
        String localEquipCAName = localEquipCACert.getSubjectDN().getName();
        // get issueDN of client's equipment certificate
        X509Certificate remoteEquipCert = x509Certificate[1];
        String equipIssueName = remoteEquipCert.getIssuerDN().getName();
        return localEquipCAName.equals(equipIssueName);
    }

    // check whether the former certificate owner is the publisher of next one.
    private static boolean verifyChain(String clientID, X509Certificate[] x509Certificates) {
        if (x509Certificates == null || x509Certificates.length < 2) {
            LOGGER.severe(Common.addTag("[CertVerify] certificateChains is null or the length is not valid: < 2, " +
                    "please check!"));
            return false;
        }

        // check remote equipment certificate
        try {
            X509Certificate[] certificateChains = getX509CertificateChain(clientID);
            if (certificateChains == null || certificateChains.length < 3) {
                LOGGER.severe(Common.addTag("[CertVerify] certificateChains is null or the length is not valid: < 3, " +
                        "please check!"));
                return false;
            }
            X509Certificate localEquipCA = certificateChains[2];
            PublicKey publicKey = localEquipCA.getPublicKey();
            x509Certificates[1].verify(publicKey);
        } catch (NoSuchProviderException | CertificateException | NoSuchAlgorithmException |
                InvalidKeyException | SignatureException e) {
            LOGGER.severe(Common.addTag("[CertVerify] catch Exception: " + e.getMessage()));
            return false;
        }

        // check remote service certificate
        X509Certificate remoteEquipCert = x509Certificates[1];
        X509Certificate remoteServiceCert = x509Certificates[0];

        try {
            remoteEquipCert.checkValidity();
            remoteServiceCert.checkValidity();
        } catch (java.security.cert.CertificateExpiredException |
                java.security.cert.CertificateNotYetValidException e) {
            e.printStackTrace();
            return false;
        }

        try {
            PublicKey publicKey = remoteEquipCert.getPublicKey();
            remoteServiceCert.verify(publicKey);
        } catch (CertificateException | NoSuchAlgorithmException | InvalidKeyException |
                NoSuchProviderException | SignatureException e) {
            LOGGER.severe(Common.addTag("verifyChain failed!"));
            LOGGER.severe(Common.addTag("[verifyChain] catch Exception: " + e.getMessage()));
            return false;
        }
        LOGGER.severe(Common.addTag("verifyChain success!"));
        return true;
    }

    /**
     * get certificate chain according to clientID
     *
     * @param clientID clientID of this client
     * @return certificate chain
     */
    public static X509Certificate[] getX509CertificateChain(String clientID) {
        if (clientID == null || clientID.isEmpty()) {
            LOGGER.severe(Common.addTag("[CertVerify] the parameter clientID is null or empty, please check!"));
            return null;
        }
        X509Certificate[] x509Certificates = null;
        try {
            Certificate[] certificates = null;
            KeyStore keyStore = KeyStore.getInstance(CipherConsts.KEYSTORE_TYPE);
            keyStore.load(null);
            KeyStore.Entry entry = keyStore.getEntry(clientID, null);
            if (entry == null || !(entry instanceof KeyStore.PrivateKeyEntry)) {
                return null;
            }
            certificates = ((KeyStore.PrivateKeyEntry) entry).getCertificateChain();
            if (certificates == null) {
                return null;
            }
            x509Certificates = (X509Certificate[]) certificates;
        } catch (IOException | NoSuchAlgorithmException | UnrecoverableEntryException | KeyStoreException |
                CertificateException e) {
            LOGGER.severe(Common.addTag("[CertVerify] catch Exception: " + e.getMessage()));
        }
        return x509Certificates;
    }

    /**
     * transform pem format to X509 format
     *
     * @param pemCerts pem format certificate
     * @return X509 format certificates
     */
    public static X509Certificate[] transformPemArrayToX509Array(String[] pemCerts) {
        if (pemCerts == null || pemCerts.length == 0) {
            LOGGER.severe(Common.addTag("[CertVerify] pemCerts is null or empty, please check!"));
            throw new IllegalArgumentException();
        }
        int nSize = pemCerts.length;
        X509Certificate[] x509Certificates = new X509Certificate[nSize];
        for (int i = 0; i < nSize; ++i) {
            x509Certificates[i] = transformPemToX509(pemCerts[i]);
        }
        return x509Certificates;
    }

    private static X509Certificate transformPemToX509(String pemCert) {
        X509Certificate x509Certificate = null;
        CertificateFactory cf;
        try {
            if (pemCert != null && !pemCert.trim().isEmpty()) {
                byte[] certificateData = Base64.getDecoder().decode(pemCert);
                cf = CertificateFactory.getInstance("X509");
                x509Certificate = (X509Certificate) cf.generateCertificate(new ByteArrayInputStream(certificateData));
            }
        } catch (CertificateException e) {
            LOGGER.severe(Common.addTag("[CertVerify] catch Exception: " + e.getMessage()));
            return null;
        }
        return x509Certificate;
    }

    private static boolean verifyCrl(String clientID, X509Certificate[] x509Certificates) {
        if (x509Certificates == null || x509Certificates.length < 2) {
            LOGGER.severe(Common.addTag("[verifyCrl] the number of certificate in x509Certificates is less than 2, " +
                    "please check!"));
            throw new IllegalArgumentException();
        }
        FLParameter flParameter = FLParameter.getInstance();
        X509Certificate equipCert = x509Certificates[1];
        if (equipCert == null) {
            LOGGER.severe(Common.addTag("[verifyCrl] equipCert is null, please check it!"));
            return false;
        }
        String equipCertSerialNum = equipCert.getSerialNumber().toString();
        if (verifySingleCrl(clientID, equipCertSerialNum, flParameter.getEquipCrlPath())) {
            LOGGER.info(Common.addTag("[verifyCrl] verify crl certificate success!"));
            return true;
        }
        LOGGER.info(Common.addTag("[verifyCrl] verify crl certificate failed!"));
        return false;
    }

    private static boolean verifySingleCrl(String clientID, String caSerialNumber, String crlPath) {
        if (caSerialNumber == null || caSerialNumber.isEmpty()) {
            LOGGER.severe(Common.addTag("[CertVerify] caSerialNumber is null or empty, please check!"));
            throw new IllegalArgumentException();
        }
        // crlPath does not exist
        if (crlPath.equals("null")) {
            LOGGER.severe(Common.addTag("[CertVerify] crlPath is null, please set crlPath with setEquipCrlPath " +
                    "method!"));
            return false;
        }
        boolean notInFlag = true;
        try {
            X509CRL crl = (X509CRL) readCrl(crlPath);
            if (crl != null) {
                // check CRL cert with local equipment CA publicKey
                X509Certificate[] certificateChains = getX509CertificateChain(clientID);
                if (certificateChains == null || certificateChains.length < 3) {
                    LOGGER.severe(Common.addTag("[CertVerify] certificateChains is null or the length is not" +
                            " valid: < 3, please check!"));
                    return false;
                }
                X509Certificate localEquipCA = certificateChains[2];
                PublicKey publicKey = localEquipCA.getPublicKey();
                crl.verify(publicKey);

                // check whether remote equipmentCert in CRL
                Set<?> set = crl.getRevokedCertificates();
                if (set == null) {
                    LOGGER.info(Common.addTag("[verifySingleCrl] verifyCrl Revoked Cert list is null"));
                    return true;
                }
                for (Object obj : set) {
                    X509CRLEntry crlEntity = (X509CRLEntry) obj;
                    if (crlEntity.getSerialNumber().toString().equals(caSerialNumber)) {
                        LOGGER.info(Common.addTag("[verifySingleCrl] Find same SerialNumber during the crl!"));
                        notInFlag = false;
                        break;
                    }
                }
            }
        } catch (java.security.cert.CRLException | java.security.NoSuchAlgorithmException |
                java.security.InvalidKeyException | java.security.NoSuchProviderException |
                java.security.SignatureException e) {
            LOGGER.severe(Common.addTag("[verifySingleCrl] judgeCAInCRL error: " + e.getMessage()));
            notInFlag = false;
        }
        return notInFlag;
    }

    private static Object readCrl(String assetName) {
        if (assetName == null || assetName.isEmpty()) {
            LOGGER.severe(Common.addTag("[readCrl] the parameter of <assetName> is null or empty, please check!"));
            return null;
        }
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream(assetName);
        } catch (IOException e) {
            LOGGER.severe(Common.addTag("[readCrl] catch Exception of read inputStream in readCert: " +
                    e.getMessage()));
            return null;
        }
        Object crlCert = null;
        try {
            CertificateFactory cf = CertificateFactory.getInstance("X.509");
            crlCert = cf.generateCRL(inputStream);
        } catch (CertificateException | CRLException e) {
            LOGGER.severe(Common.addTag("[readCrl] catch Exception of creating CertificateFactory in readCert: "
                    + e.getMessage()));
        } finally {
            try {
                inputStream.close();
            } catch (IOException e) {
                LOGGER.severe(Common.addTag("[readCrl] catch Exception of close inputStream: "
                        + e.getMessage()));
            }
        }

        return crlCert;
    }

    private static boolean verifyValidDate(X509Certificate[] x509Certificates) {
        if (x509Certificates == null) {
            LOGGER.severe(Common.addTag("[CertVerify] x509Certificates is null, please check!"));
            throw new IllegalArgumentException();
        }
        Date date = new Date();
        try {
            int nSize = x509Certificates.length;
            for (int i = 0; i < nSize; ++i) {
                x509Certificates[i].checkValidity(date);
            }
        } catch (java.security.cert.CertificateExpiredException |
                java.security.cert.CertificateNotYetValidException e) {
            LOGGER.severe(Common.addTag("[verifyValidDate] catch Exception: " + e.getMessage()));
            return false;
        }
        return true;
    }

    private static boolean verifyKeyIdentifier(String clientID, X509Certificate[] x509Certificates) {
        if (clientID == null || clientID.isEmpty()) {
            LOGGER.severe(Common.addTag("[CertVerify] the parameter clientID is null or empty, please check!"));
            return false;
        }
        if (x509Certificates == null || x509Certificates.length < 2) {
            LOGGER.severe(Common.addTag("[CertVerify] x509Certificate chains is null or the length is not valid: < 2," +
                    " please check!"));
            return false;
        }

        X509Certificate[] certificateChains = getX509CertificateChain(clientID);
        if (certificateChains == null || certificateChains.length < 3) {
            LOGGER.severe(Common.addTag("[CertVerify] certificateChains is null or the length is not valid: < 3, " +
                    "please check!"));
            return false;
        }

        X509Certificate localEquipCACert = certificateChains[2];
        String subjectIdentifier = "null";

        // get subjectKeyIdentifier of local root equipment CA certificate
        try {
            String subjectIdentifierOid = "2.5.29.14";
            byte[] subjectExtendData = localEquipCACert.getExtensionValue(subjectIdentifierOid);
            ASN1OctetString asn1OctetString = ASN1OctetString.getInstance(subjectExtendData);
            byte[] tmpData = asn1OctetString.getOctets();
            SubjectKeyIdentifier subjectKeyIdentifier = SubjectKeyIdentifier.getInstance(tmpData);
            byte[] octKeyIdentifier = subjectKeyIdentifier.getKeyIdentifier();
            subjectIdentifier = new String(Hex.encode(octKeyIdentifier));
        } catch (ExceptionInInitializerError e) {
            e.printStackTrace();
        }

        // get authorityKeyIdentifier of client's equipment certificate
        X509Certificate remoteEquipCert = x509Certificates[1];
        String authorityIdentifier = "null";
        try {
            if (remoteEquipCert == null) {
                LOGGER.severe(Common.addTag("[CertVerify] remoteEquipCert is null, please check it!"));
                return false;
            }
            String authorityIdentifierOid = "2.5.29.35";
            byte[] authExtendData = remoteEquipCert.getExtensionValue(authorityIdentifierOid);
            ASN1OctetString asn1OctetString = ASN1OctetString.getInstance(authExtendData);
            byte[] tmpData = asn1OctetString.getOctets();
            AuthorityKeyIdentifier authorityKeyIdentifier = AuthorityKeyIdentifier.getInstance(tmpData);
            byte[] octKeyIdentifier = authorityKeyIdentifier.getKeyIdentifier();
            authorityIdentifier = new String(Hex.encode(octKeyIdentifier));
        } catch (ExceptionInInitializerError e) {
            e.printStackTrace();
        }

        if (authorityIdentifier.equals("null") || subjectIdentifier.equals("null")) {
            LOGGER.severe(Common.addTag("[CertVerify] authorityKeyIdentifier or subjectKeyIdentifier is null, check " +
                    "failed!"));
            return false;
        } else {
            return authorityIdentifier.equals(subjectIdentifier);
        }
    }
}
