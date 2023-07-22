#include <iostream>
#include <string.h>
#include <fstream>
#include <cassert>  
#include <string>    
#include <vector>  
#include <sstream>
#include <openssl/md5.h> 
#include <openssl/bn.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/sha.h>

#pragma comment(lib,"libssl.lib")
#pragma comment(lib,"libcrypto.lib")
using namespace std;


// global variable
char sourceA[] = "./localUser";
char sourceB[] = "./localUser";
string source1 = ".txt";
RSA* r;
BIGNUM* bne; //public key e
unsigned long e = RSA_F4; //  RSA Public key index
int bits = 2048;  // RSA  key length
// Alice message sha256 hash->hm
unsigned char hm[SHA256_DIGEST_LENGTH + 1] = { 0 };
//Alice hm hash->hh
unsigned char hh[33] = { 0 };
//unsigned char hh[SHA256_DIGEST_LENGTH + 1];
//Bob message sha256 hash->hp
unsigned char hp[SHA256_DIGEST_LENGTH + 1] = { 0 };
//Bob hp hash->hq
//unsigned char hq[SHA256_DIGEST_LENGTH + 1];
unsigned char hq[33] = { 0 };
int inter_num = 0;


class Alice {
public:
    const BIGNUM* privateKey;
    const BIGNUM* mod;
    char* m1;
};
class Bob {
public:
    BIGNUM* blind_k;
    const BIGNUM* mod;
    char* m2;
};

Alice a;
Bob b;
int KeyGen() {
    int ret;
    bne = BN_new();
    ret = BN_set_word(bne, e);
    r = RSA_new();
    ret = RSA_generate_key_ex(r, bits, bne, NULL);  // generate RSA key
    if (ret != 1) {
        printf("RSA_generate_key_ex err!\n");
        return -1;
    }
    else {
        //              printf("generate RSA key\n");
    }
    return 0;
}

BIGNUM* AlicePro(char* databuf, long size) {
    SHA256((unsigned char*)databuf, strlen(databuf), (unsigned char*)hm);
    //      cout << "................................Alice:one........................................." << endl;

    const BIGNUM* nn;  
    const BIGNUM* dd;  
    RSA_get0_key(r, &nn, NULL, &dd);
    a.privateKey = dd;
    a.mod = nn;

    // hm->BN bn_hm
    BIGNUM* bn_hm;
    bn_hm = BN_new();
    BN_bin2bn(hm, sizeof(hm), bn_hm);
    BN_CTX* ctx;
    ctx = BN_CTX_new();
    //sign the hashed message
    BIGNUM* signed_an;
    signed_an = BN_new();
    BN_mod_exp(signed_an, bn_hm, dd, nn, ctx);

    a.m1 = BN_bn2hex(signed_an);


    SHA256((const unsigned char*)a.m1, strlen(a.m1), hh);
    //hh->BN tn_hh
    BIGNUM* tn_hh;
    tn_hh = BN_new();
    BN_bin2bn(hh, sizeof(hh), tn_hh);

    return tn_hh;
}

//Alice get message
BIGNUM* AliceGet(BIGNUM* c) {
    BN_CTX* ctx;
    ctx = BN_CTX_new();
    //      cout << "................................Alice:three........................................." << endl;
         
    BIGNUM* c_get;
    c_get = BN_new();
    BN_mod_exp(c_get, c, a.privateKey, a.mod, ctx);

    return c_get;
}


BIGNUM* BobPro(char* databuf, long size) {
    // message sha256 hash->hm
    SHA256((unsigned char*)databuf, strlen(databuf), (unsigned char*)hp);
    //      cout << "................................Bob:two........................................." << endl;
    const BIGNUM* nn;  
    RSA_get0_key(r, &nn, NULL, NULL);
    BIGNUM* k;  
    k = BN_new();
    BN_rand_range(k, nn);  
    b.blind_k = k;
    b.mod = nn;
    BN_CTX* ctx;
    ctx = BN_CTX_new();
   
    BIGNUM* blind;
    blind = BN_new();
    BN_mod_exp(blind, k, bne, nn, ctx);

    // hm->BN bn_hp
    BIGNUM* bn_hp;
    bn_hp = BN_new();
    BN_bin2bn(hp, sizeof(hp), bn_hp);

    
    BIGNUM* m_blind;
    m_blind = BN_new();
    BN_mod_mul(m_blind, bn_hp, blind, nn, ctx);

    return m_blind;
}

//Bob get message
int BobGet(BIGNUM* tn_hh, BIGNUM* c_get, long size) {
 
    BN_CTX* ctx;
    ctx = BN_CTX_new();
    //      cout << "................................Bob:four........................................." << endl;
    const BIGNUM* nn; 
    RSA_get0_key(r, &nn, NULL, NULL);
    BIGNUM* k_inverse;
    k_inverse = BN_new();
    BN_mod_inverse(k_inverse, b.blind_k, b.mod, ctx);


    BIGNUM* cc;
    cc = BN_new();
    BN_mod_mul(cc, c_get, k_inverse, nn, ctx);
    b.m2 = BN_bn2hex(cc);

   
    SHA256((unsigned char*)b.m2, strlen(b.m2), hq);
    //hh->BN tt_hq
    BIGNUM* tt_hq;
    tt_hq = BN_new();
    BN_bin2bn(hq, sizeof(hq), tt_hq);

    //equal?
    if (!BN_cmp(tn_hh, tt_hq)) {
        //printf("Element is equal!\n");
        inter_num++;
    }
    else {
        //printf("Not equal!\n");
    }
    return 0;
}


void imp_P(string m1, string m2, long size) {
    BIGNUM* tn_hh;
    BIGNUM* c_get;
    BIGNUM* result;

    tn_hh = AlicePro(const_cast<char*>(m1.c_str()), size);
    c_get = BobPro(const_cast<char*>(m2.c_str()), size);
    result = AliceGet(c_get);
    BobGet(tn_hh, result, size);

}

vector<vector<string>> get_string(string path) {
    
    ifstream inFile(path, ios::in);
  
    string lineStr;
    vector<vector<string>> strArray;
    while (getline(inFile, lineStr)) {
     
        
        //cout << lineStr << endl;
       
       
        stringstream ss(lineStr);
        string str;
        vector<string> lineArray;
        //Tab
        while (getline(ss, str, '	'))
           

            lineArray.push_back(str);

               
        strArray.push_back(lineArray);
    }


    //getchar();
   
    return strArray;
}
//read a file
vector<vector<string>>  readfile(string m,string n) {
    int number = 0;
    string fullPath_Alice = sourceA + m + source1;
    string fullPath_Bob = sourceB + n + source1;
    
    vector<vector<string>> set_Alice;
    set_Alice = get_string(fullPath_Alice);
    
    vector<vector<string>> inter;
    vector<string> inter_strB;

    vector<vector<string>> set_Bob;
    set_Bob = get_string(fullPath_Bob);

    cout << n << endl;

    clock_t start = clock();
    /*
    for (int i = 0; i < set_Alice.size(); i++) {
        cout << set_Alice[i][1]<< endl;
    }
 
    for (int i = 0; i < set_Bob.size(); i++) {
        cout << set_Bob[i][1]<< endl;
    }*/
    for (int i = 0; i < set_Alice.size(); i++) {
    	for(int j = 0; j < set_Bob.size(); j++){
    	    number = inter_num;
            imp_P(set_Alice[i][1], set_Bob[j][1], 1024);
            if(number!=inter_num){
                inter_strB.push_back(set_Bob[j][0]);
                inter_strB.push_back(set_Bob[j][1]);
                inter_strB.push_back(set_Bob[j][2]);
                inter.push_back(inter_strB);
                inter_strB.clear();
              
	        }
           
          
	    }
    }
   
    
    clock_t end = clock();

    double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("number");
    cout << n<<endl;
    char* nn = const_cast<char*>(n.c_str()); 
    printf("number %s user implementing agreements takes time %f  seconds\n", nn, time);
    printf("number %s user  implementing agreements takes time %f minutes\n", nn, time / 60);

    cout << "Intersection number" << inter_num << endl;
    inter_num = 0;
    return inter;
}

int pseudonymize(int m,int n){

    return m,n;
}

int main() {

    clock_t start = clock();
    KeyGen();
    int  m,n;
    printf("choose file m,file count n ：");
    scanf("%d %d", &m,&n);

    for(m =1;m<=n;m++){
        ofstream dataFile;
        string filePath = "./Process-Data/data/douban/douban_psi_all/localUser" + to_string(m)+".txt";
        dataFile.open(filePath, ofstream::app);
        fstream file(filePath, ios::out);
        
        string fullPath_Alice = sourceA + to_string(m) + source1;
        vector<vector<string>> set;
        vector<vector<string>> inter;
        set = get_string(fullPath_Alice);

        for (int i = 0; i < set.size(); i++) {
            dataFile <<set[i][0] << '\t'<< set[i][1] << '\t'<< set[i][2]<< endl;    
        }
    
        for(int i =1;i<=n;i++){
            pseudonymize(m,i);
            if(i!=m){
        
                inter = readfile(to_string(m),to_string(i));
                for (int i = 0; i < inter.size(); i++) {
                    dataFile <<inter[i][0] << '\t'<< inter[i][1] << '\t'<< inter[i][2]<< endl; 
                }
        
            }
        }
        dataFile.close();
    }
    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "Protocol takes total time" << time << "seconds" << endl;
    cout << "Protocol takes total time" << time / 60 << "minutes" << endl;
    cout << "Protocol takes total time" << (time / 3600) << "hours" << endl;
    //cout << "Intersection number" << inter_num << endl;
    return 0;
}