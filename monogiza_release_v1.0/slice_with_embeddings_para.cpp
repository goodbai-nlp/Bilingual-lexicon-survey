#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/unordered_map.hpp>
#include "lm.h"
#include <boost/random.hpp>
#include <math.h>
#include <vector>
#include "slice_with_embeddings_x.h"
#include <tclap/CmdLine.h>

using namespace std;
using namespace TCLAP;

int main(int argc, char** argv) {
    
    LM lm;
    int max_threads = setup_threads(0);
    CmdLine cmd("Slice sampler with multinomial dirichlet regression for finding word-to-word translations from monolingual data", ' ', "1.0");
    ValueArg<int> arg_iteration("", "total_iteration", "number of total iterations (default: 1)", false, 1, "int", cmd);
    ValueArg<int> arg_m_iteration("", "m_iteration", "number of M steps in stochastic EM (default: 5)", false, 5, "int", cmd);
    ValueArg<int> arg_interval("", "interval_iteration", "number of sampling iterations between each M step in stochastic EM (default: 100000)", false, 100000, "int", cmd);
    ValueArg<int> arg_vocab_size("", "vocab_size", "max number of word types in plain and cipher text (default: 10000)", false, 10000, "int", cmd);    
    ValueArg<int> arg_num_threads("", "num_threads", "number of threads for slice sampler (default: 1)", false, 1, "int", cmd); 
    ValueArg<int> arg_dimension("", "dimension", "size of word embeddings vectors (default: 50)", false, 50, "int", cmd);
    ValueArg<float> arg_alpha("", "alpha", "weight of prior/base distribution (default: 1.0)", false, 1.0, "float", cmd);
    ValueArg<float> arg_base_threshold("", "base_threshold", "controls the number of candidates to consider at each sampling operations. candidates whose prior are above this threshold will always be considered. (default: 1|/V_observe|)", false, 0.002, "float", cmd);
    ValueArg<bool> arg_fast_mode("", "fast_mode", "whether perform approximate slice sampling. 1 yes, 0 no. (default: 0)", false, false, "bool", cmd);
    ValueArg<double> arg_learning_rate("", "learning_rate", "learning rate for gradient ascent in stochastic EM (default: 10)", false, 10, "double", cmd);
    ValueArg<double> arg_reg("", "reg", "regularization parameter for the M step in stochastic EM (default: 0.001)", false, 0.001, "double", cmd);
    ValueArg<int> arg_random_seed("", "random_seed", "random seed for sampler", false, 1, "int", cmd);
    ValueArg<bool> arg_use_uniform_base("", "use_uniform_base", "whether use uniform prior for decipherment. 1 yes, 0 no (default: 1)", false, true, "bool", cmd);
    ValueArg<int> arg_sorted_list_size("", "sorted_list_size", "number of top k candidates for each context in pre-sorted list (default: 2000)", true, 2000, "int", cmd);
    ValueArg<bool> arg_use_embeddings("", "use_embeddings", "whether use embeddings during decipherment to reestimate prior. 1 yes, 0 no. (default: 0)", false, false, "bool", cmd);

    ValueArg<string> arg_lm("", "lm", "language model file", true, "", "string", cmd);
    ValueArg<string> arg_sorted_list("", "sorted_list", "pre-sorted list file", true, "", "string", cmd);
    ValueArg<string> arg_cipher_bigrams("", "cipher_bigrams", "cipher bigrams file", true, "", "string", cmd);
    ValueArg<string> arg_seed_table("", "seed_table", "seed table file used to initialize first sample. same format as output ttable", false, "ptable.final", "string", cmd);
    ValueArg<string> arg_mapping_seed("", "mapping_seed", "seed file used to initialize mapping matrix M, previous output of M can be used as seed", false, "m_seed", "string", cmd);
    ValueArg<string> arg_plain_embeddings("", "plain_embeddings", "plaintext word embeddings file", false, "", "string", cmd);
    ValueArg<string> arg_cipher_embeddings("", "cipher_embeddings", "ciphertext word embeddings file", false, "", "string", cmd);
    ValueArg<string> arg_output_ttable("", "output_ttable", "output file name for translation table. Format: f ||| e ||| p(f|e) p(e|f)", true, "ttable.final", "string", cmd);
    ValueArg<string> arg_output_mapping("", "output_mapping", "output file name for mapping matrix M", false, "mapping.final", "string", cmd);
    ValueArg<string> arg_base_file("", "base_file", "base distribution file used to initialize base distribution(prior) in Bayesian decipherment. Format: f e p(f|e)", false, "", "string", cmd);

    cmd.parse(argc, argv);
   
    int iteration = arg_iteration.getValue();
    lm.load_lm(arg_lm.getValue().c_str()); 
    int embedding_dimension = arg_dimension.getValue(); 
    int m_iteration = arg_m_iteration.getValue();
    int interval = arg_interval.getValue(); 
    float alpha = arg_alpha.getValue(); 
    int vocab_size = arg_vocab_size.getValue(); 
    int random_seed = arg_random_seed.getValue(); 
    int num_threads = arg_num_threads.getValue(); 
    bool fast_mode = arg_fast_mode.getValue();
    float base_threshold = arg_base_threshold.getValue();
    bool use_uniform_base = arg_use_uniform_base.getValue();
    bool use_embeddings = arg_use_embeddings.getValue();
    Decipherment decipherer(lm, random_seed, arg_mapping_seed.getValue(), arg_plain_embeddings.getValue(), arg_cipher_embeddings.getValue(), 
    embedding_dimension, m_iteration, interval, alpha, arg_sorted_list_size.getValue(), fast_mode, base_threshold, num_threads, max_threads, vocab_size, arg_learning_rate.getValue(), arg_reg.getValue(), use_uniform_base, use_embeddings);
    decipherer.loadSeedTable(arg_seed_table.getValue().c_str());
    decipherer.loadSlice(arg_sorted_list.getValue().c_str()); 
    decipherer.loadCipherBigrams(arg_cipher_bigrams.getValue().c_str());
    decipherer.shuffle();
    decipherer.initSamples();
    decipherer.initBaseDistribution(arg_base_file.getValue().c_str());
    for(int i = 0; i < iteration; i += interval) {
        vector<ThreadData> targs(num_threads);
        vector<pthread_t> t_handler(num_threads);
        for(int tid = 0; tid < num_threads; tid++) {
            targs[tid] = ThreadData(tid, interval, num_threads, &decipherer);
            int rc = pthread_create(&t_handler[tid], 0, &ThreadWrapper::runSampling, (void*)(&targs[tid]));
            if(rc != 0) {
                cout << "error creating thread" << endl;
                exit(0);
            }
        } 
        for(int tid = 0; tid < num_threads; tid++) {
            pthread_join(t_handler[tid], 0);
        }
        // update mapping matrix m
        if(i + interval < iteration && use_embeddings){
            cout << "building counts matrix" << endl;
            decipherer.buildCountsMatrix();
            decipherer.doMappingOptimization(max_threads);
            decipherer.updateCache();   
        }
    }
    decipherer.printTTable(arg_output_ttable.getValue().c_str());
    //decipherer.printBase(tmp_dir + "/base");
    decipherer.printMapMatrix(arg_output_mapping.getValue().c_str());
    //decipherer.printAccCounts((string("") + getenv("TMPDIR") + "/cipher.id.counts.final").c_str());
    return 0;
}

