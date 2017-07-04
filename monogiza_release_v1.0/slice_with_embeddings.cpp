#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/unordered_map.hpp>
#include "lm.h"
#include <boost/random.hpp>
#include <math.h>
#include <vector>
#include "slice_with_embeddings.h"

using namespace std;


int main(int argc, char** argv) {
    /*boost::mt19937 int_gen;
    boost::uniform_real<double> real_distribution(0,1);
    int i = 0;
    while(true){
        double random = real_distribution(int_gen);
        if(random == 1) {
            cout << "pass" << endl;
            break;
        }
    }*/
    LM test_lm;
    test_lm.load_lm(argv[2]);
    int embedding_dimension = atoi(argv[10]);
    int opt_itr = atoi(argv[11]);
    int interval = atoi(argv[12]);
    float alpha = atof(argv[13]);
    int base_scale = atoi(argv[14]);
    Decipherment decipherer(test_lm, atoi(argv[5]), argv[8], argv[9], 
    embedding_dimension, opt_itr, interval, alpha, base_scale);
    decipherer.loadSeedTable(argv[6]);

    decipherer.loadSlice(argv[3]);
    decipherer.loadCipherBigrams(argv[4]);
    decipherer.initSamples();
    //decipherer.buildCountsMatrix();
    //decipherer.doMappingOptimization();
    //decipherer.loadBaseFromFile("/home/nlg-05/qingdou/sampling_base/lex.id.f2e");
    decipherer.runSampling(atoi(argv[1])); 
    string tmp_dir = getenv("TMPDIR");
    decipherer.printTTable((tmp_dir + "/cipher.id.ptable.final").c_str());
    //decipherer.printBase(tmp_dir + "/base", tmp_dir + "/alphas");
    //decipherer.printAccCounts((string("") + getenv("TMPDIR") + "/cipher.id.counts.final").c_str());
    return 0;
}

