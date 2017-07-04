/* 
 * File:   slice_with_embeddings.h
 * Author: 
 *
 * Created on January 15, 2014, 3:02 PM
 */
#pragma once
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include "lm.h"
#include <math.h>
#include <float.h>
#include <limits.h>
#include <queue> 
#include <boost/random.hpp>
#include <pthread.h>
#include "util.h"
#include "tbb/concurrent_hash_map.h"
#include <Eigen/Dense>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include "maybe_omp.h"

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

#define MAX_ID 50000

using namespace std;
using namespace tbb;
using namespace Eigen;

class Decipherment;
void reestimateMapping(Decipherment* myData,
double reg_lambda,
int epochs,
double learning_rate,
int num_threads);

class Bigram {
public:
    unsigned int tokens[2];
    unsigned int sample_tokens[2];
    long count;
    Bigram(unsigned int tok1, unsigned int tok2, long c) {
        tokens[0] = tok1;
        tokens[1] = tok2;
        count = c;
    }
};

class CountList{
  public:
    boost::unordered_map<unsigned int,unsigned int> members; // to quickly check if a given word is in the list
    vector<unsigned int> member_list;
};

class Candidate{
  public:
  unsigned int token;
  float prob;
  Candidate(){
  }
  Candidate(int t, float p) {
    token = t;
    prob = p;
  }
  bool operator<(const Candidate& c) const {
    return (prob > c.prob);
  }
};

class Decipherment {
   
  public:
  LM &lm;
  float** slice_list;
  int slice_list_size;
  int base_scale;
  vector<Bigram> cipher_bigrams;
  pthread_mutex_t* plain_vocab_locks;
  pthread_mutex_t* cipher_vocab_locks;
  pthread_mutex_t hash_key_lock;
  boost::unordered_map<long,long> acc_counts; // accumulated cooc count for last 1000 iterations
  concurrent_hash_map<long,long> counts; // the cache
  boost::unordered_map<long,long> o_counts; // counts of observed cipher tokens
  boost::unordered_map<unsigned int,CountList> channel_list;
  boost::unordered_map<unsigned int, vector<Candidate> > seed_table;
  boost::unordered_map<unsigned int, unsigned int> observed_v;
  //boost::unordered_map<long, float> base; // base distribution
  string map_file_name; 
   
  // embedding parts
  Matrix<double,Dynamic,Dynamic> counts_matrix, base_distribution,plain_embeddings,cipher_embeddings, embeddings_product_sum;
  unsigned int plain_dis2con_map[MAX_ID]; // map discontinuous word id to continuous word ids
  unsigned int plain_con2dis_map[MAX_ID]; // map continous word id to discontinuous
  unsigned int cipher_dis2con_map[MAX_ID];
  unsigned int cipher_con2dis_map[MAX_ID];
  Matrix<double, Dynamic, 1> alphas, source_counts;
  Matrix<double, Dynamic, Dynamic> M; // mapping weights matrix 
  int embedding_dimension; 
  int opt_itr;   // number of optimization itr
  int interval; // number of intervals between mapping matrix optimization
  bool valid_embeddings;
  // end of embedding parts   
  int max_iter;
  float uniform_base;
  float alpha;
  double corpus_prob;
  bool is_viterbi;
  unsigned int seed;
  int last_iterations;
  int max_threads;  
  int vocab_size;
  bool fast_mode;
  float base_threshold;
  double learning_rate;
  double reg;
  bool use_uniform_base;
  bool use_embeddings;

  boost::mt19937* int_gen;
  boost::mt19937* flt_gen;
  boost::uniform_int<int>* int_vocab_distribution;
  boost::uniform_real<double>* real_distribution; 
 

  Decipherment(LM &dep_lm, unsigned int s, string map_file, string plain_embeddings_file, string cipher_embeddings_file, int dimension
    , int opitr, int intv, float al, int arg_sorted_list_size, bool arg_fast_mode, float arg_base_threshold, int num_threads, int max_threads_arg, int vocab_size_arg, double learning_rate_arg, double reg_arg, bool arg_use_uniform_base, bool arg_use_embeddings):
    lm(dep_lm),
    opt_itr(opitr),
    interval(intv),
    alpha(al),
    map_file_name(map_file),
    seed(s){
    use_uniform_base = arg_use_uniform_base;
    use_embeddings = arg_use_embeddings;
    vocab_size = vocab_size_arg;
    base_threshold = arg_base_threshold;
    fast_mode = arg_fast_mode;
    cout << "fast_mode: " << fast_mode << endl;
    embedding_dimension = dimension;
    corpus_prob = 0;
    reg = reg_arg;
    learning_rate = learning_rate_arg;
    max_threads = max_threads_arg;
    valid_embeddings = true;
    slice_list_size = arg_sorted_list_size;
    int_gen = new boost::mt19937[num_threads];
    flt_gen = new boost::mt19937[num_threads];
    int_vocab_distribution = new boost::uniform_int<int>[num_threads];
    real_distribution = new boost::uniform_real<double>[num_threads];
    for(int i = 0; i < num_threads; i++) {
        int_gen[i].seed(seed);
        flt_gen[i].seed(seed + 100);
        int_vocab_distribution[i] = boost::uniform_int<int>(0, lm.vocab_size - 1);
        real_distribution[i] = boost::uniform_real<double>(0,1);
    }
    plain_embeddings.setZero(vocab_size, embedding_dimension);
    cipher_embeddings.setZero(vocab_size, embedding_dimension);
    for(int i = 0; i < MAX_ID; i++) {
        plain_dis2con_map[i] = 0;
        plain_con2dis_map[i] = 0;
        cipher_dis2con_map[i] = 0;
        cipher_con2dis_map[i] = 0;
    }
    if(use_embeddings) {
        readEmbeddings(plain_embeddings_file, plain_embeddings, plain_dis2con_map, plain_con2dis_map);
        readEmbeddings(cipher_embeddings_file, cipher_embeddings, cipher_dis2con_map, cipher_con2dis_map);
    } else {
        for(int i = 1; i < MAX_ID; i++) {
            plain_dis2con_map[i] = i - 1;
            plain_con2dis_map[i - 1] = i;
            cipher_dis2con_map[i] = i - 1;
            cipher_con2dis_map[i - 1] = i;
        }
    }
    base_distribution.setZero(vocab_size, vocab_size);
    plain_embeddings /= 10;
    cipher_embeddings /= 10;
    /*Matrix<int,Dynamic,Dynamic> training_data; 
    training_data.setZero(2, 54556);    
    counts_matrix.setZero(5001, 5001); 
    for(int j = 0; j < training_data.cols(); j++) {
        int pid = training_data(0, j);
        int cid = training_data(1, j);
        counts_matrix(pid, cid) += 1;
    } */
    embeddings_product_sum.setZero(vocab_size, embedding_dimension * embedding_dimension);
    M.setZero(embedding_dimension, embedding_dimension);
    boost::mt19937 rng(1234);
    initMatrix(rng, M, 1, 0.01);
    alphas.setOnes(vocab_size);
    //doMappingOptimization(); 

    //pthread_mutex_init(&hash_key_lock,0);

  }

  template<typename Derived>
  void readEmbeddings(string embeddings_filename, Eigen::MatrixBase<Derived>& const_embeddings_matrix, unsigned int* dis2con_map, unsigned int* con2dis_map) {
      cout << "reading embeddings: " << embeddings_filename << endl;
      ifstream embeddings_file(embeddings_filename.c_str());
      if (embeddings_file) {
          readMatrix(embeddings_file,const_embeddings_matrix, dis2con_map, con2dis_map);
      } else {
          valid_embeddings = false;
          cout << "could not open file " + embeddings_filename << endl;
      }
  }

  long concatNumbers(unsigned int w1, unsigned int w2){
    return (((long)w1) << 30 | ((long)w2));
  } 

  void loadBaseFromFile(string file) {
      ifstream ifs(file.c_str());
      string line;
      if(!ifs.is_open()) {
          cout << file << "base file not found" << endl;
          exit(0);
      }
      boost::unordered_map<unsigned int, boost::unordered_map<unsigned int, float> > bestTrans;
      boost::unordered_map<unsigned int, float> scoreSum;
      boost::unordered_map<unsigned int, int> counts;
      while(getline(ifs,line)) {
          vector<string> entries;
          LM::split(line, entries, " ");
          unsigned int eid = atoi(entries[1].c_str());
          unsigned int fid = atoi(entries[0].c_str());
          float score = atof(entries[2].c_str());
          bestTrans[eid][fid] = score;
          if(observed_v.count(fid) != 0) {
              scoreSum[eid] += score;
              counts[eid] += 1;
          }
      }
      // now construct the base distribution matrix;
      for(int i = 0; i < lm.vocab_size; i++) {
          unsigned int eid = lm.hidden_vocab[i];
          bool evenBase = true;
          if(scoreSum.count(eid) != 0 && scoreSum[eid] >= 0.1) {
              evenBase = false;
          }
          for(boost::unordered_map<unsigned int, unsigned int>::iterator itr = observed_v.begin();
              itr != observed_v.end(); itr++) {
              unsigned int fid = itr->first;
              if(evenBase) {
                  base_distribution(eid,fid) = uniform_base;
              } else {
                  if(bestTrans[eid].count(fid) != 0) {
                      base_distribution(eid,fid) = bestTrans[eid][fid] / 1.1;
                      //cout << "found" << " " << eid << " " << fid << " " << bestTrans[eid].second << endl;
                  } else {
                      if(scoreSum[eid] > 1.1) {
                          cout << "illegal sum " << eid << " " << scoreSum[eid] << endl;
                          exit(0);
                      }
                      base_distribution(eid,fid) = (1.1 - scoreSum[eid]) / (float)(observed_v.size() - counts[eid]) / 1.1;
                  }
              }
              if(base_distribution(eid, fid) <= 0) {
                  cout << "illegal prior: " << eid << " " << fid << " " << base_distribution(eid,fid) << endl;
                  exit(0);
              }
              if(base_distribution(eid,fid) > uniform_base) {
                  //cout << eid << " " << fid << " " << base_distribution(eid,fid) << endl;
                  if(channel_list[fid].members.count(eid) == 0) {
                      channel_list[fid].members[eid] = 1;
                      channel_list[fid].member_list.push_back(eid);
                  } 
              } 
          }
      }
      cout << "base distribution loaded from file" << endl;
  }
  
  void buildCountsMatrix() {
      /*counts_matrix.setZero(5001, 5001);
      cout << "building counts matrix" << endl;
      ifstream counts_file(counts_file_name.c_str());
      if (!counts_file) throw runtime_error("Could not open file " + counts_file_name);
          readMatrix(counts_file,counts_matrix); 
      return;*/    

     long mask = INT_MAX >> 1;
     counts_matrix.setZero(vocab_size, vocab_size);
     for(concurrent_hash_map<long,long>::iterator itr = counts.begin();
           itr != counts.end(); itr++) {
      long key = itr->first;
      long token1 = key & mask;
      long token0 = (key >> 30) & mask;
      if(token0 != 0) {
        counts_matrix(plain_dis2con_map[token0], cipher_dis2con_map[token1]) = (double)itr->second;
      }
    }       
  }
 
  void doMappingOptimization(int threads) {
      source_counts = counts_matrix.rowwise().sum();
      reestimateMapping(this,
      reg, // 0.001 default
      opt_itr,
      learning_rate,
      threads);
      //writeMatrix(base_distribution.transpose(), "base.grd");
  } 
  
  void initUniformBase() {
      cout << "initializing base distribution to uniform" << endl;
      //unsigned int pid, cid;      
      for(int i = 0; i < base_distribution.rows(); i++) {
          for(int j = 0; j < base_distribution.cols(); j++) {
              //pid = plain_con2dis_map[i];
              //cid = cipher_con2dis_map[j];
              base_distribution(i, j) = uniform_base;
              //base[(long)pid << 30 | cid] = uniform_base;
          }
      }
  }

  void loadMappingFromFile(string map_file_name) {
      cout << "reading M from file: " << map_file_name << endl;
      readMatrix(map_file_name, M);
  }

  // put the translation pair whose base distribution is greater than uniform into exist_trans
  void updateCache() {
      unsigned int pid, cid;
      for(int i = 0; i < base_distribution.rows(); i++) {
          for(int j = 0; j < base_distribution.cols(); j++) {
              pid = plain_con2dis_map[i];
              cid = cipher_con2dis_map[j];
              if(base_distribution(i, j) > base_threshold) {
                  if(channel_list[cid].members.count(pid) == 0) {
                      channel_list[cid].members[pid] = 1;
                      channel_list[cid].member_list.push_back(pid); 
                  }      
              }
              //base[(long)pid << 30 | cid] = base_distribution(i, j);
          }
      }
  }

  void shuffle() {
    unsigned int n = cipher_bigrams.size();
    boost::mt19937 rd_gen(/*time(0)*/ + seed);
    for (int i = 0; i < n; i++) {
      boost::uniform_int<int> int_distribution(0, n - i - 1);
      int change = i + int_distribution(rd_gen);
      Bigram buffer = cipher_bigrams[i];
      cipher_bigrams[i] = cipher_bigrams[change];
      cipher_bigrams[change] = buffer;
    }
  }

  void loadSeedTable(const char* file) {
       ifstream ifs(file);
       string line;
       if(!ifs.is_open()) {
         cout << "seed table file not found, will initialize with random samples." << endl;
         return;
       }
       while(getline(ifs,line)){
          vector<string> entries;
          LM::split(line, entries, " ||| ");
          vector<string> scores;
          LM::split(entries[2], scores, " ");
          //cerr << entries.size() << " " << entries[0] << " " << entries[1] << " " << entries[2] << endl;
          unsigned int plain = atoi(entries[1].c_str());
          unsigned int cipher = atoi(entries[0].c_str());
          float score = atof(scores[1].c_str());
          if(seed_table.count(cipher) == 0) {
            seed_table[cipher] = vector<Candidate>();
          }
          seed_table[cipher].push_back(Candidate(plain, score));          
       }
       for(boost::unordered_map<unsigned int, vector<Candidate> >::iterator itr = seed_table.begin();
           itr != seed_table.end(); itr++) {
           stable_sort(itr->second.begin(), itr->second.end());
           //for(int i = 0; i < itr->second.size(); i++) {
           //  cout << itr->first << " " << (itr->second)[i].token << " " << (itr->second)[i].prob << endl;
           //}
       }
       cout << "seed table loaded" << endl;
  }

  void loadSlice(const char* file) {
    ifstream ifs(file);
    string line;
    if(!ifs.is_open()) {
      cout << "file not found" << endl;
      exit(0);
    }
    unsigned int rank = 0;
    unsigned int context = 0;
    slice_list = new float*[2 * lm.pseu_end];
    for(int i = 0; i < 2 * lm.pseu_end; i++) {
      slice_list[i] = 0;
    }
    while(getline(ifs,line)) {
      vector<string> entries;
      LM::split(line, entries, " ");
      if(entries.size() == 1) {
        cout << "slice list format err" << endl;
        exit(0);
        //slice_list = atoi(entries[0].c_str());
        //continue;
      } 
      vector<string> tokens;
      LM::split(entries[0], tokens, "|");
      context = lm.get_token_id(tokens[0]) + lm.get_token_id(tokens[1]);
      float prob = pow(10.0, 
                       lm.get_ngram_prob(lm.get_token_id(tokens[0]), lm.get_token_id(entries[1])) 
                       + lm.get_ngram_prob(lm.get_token_id(entries[1]), lm.get_token_id(tokens[1])));
      if(slice_list[context] == 0) {
        slice_list[context] = new float[2 * slice_list_size];
      }

      slice_list[context][rank] = lm.get_token_id(entries[1]);
      slice_list[context][rank + 1] = prob;
      
      //cout << context <<  " " << prob << endl;
      rank += 2;
      if(rank == 2 * slice_list_size) {
        rank = 0;
      }
    }
    cout << "slice table loaded" << endl;
  }

  void getBestSequence(const Bigram& f, unsigned int* e){
    if(seed_table.count(f.tokens[0]) == 0 
       || seed_table.count(f.tokens[1]) == 0){
      for(int i = 0; i < 2; i++) {
        unsigned int hidden = 0;
        if(seed_table.count(f.tokens[i]) > 0){
          hidden = seed_table[f.tokens[i]][0].token;
          if(hidden >= lm.pseu_end + 1 || lm.uni_gram_prob[hidden][0] == -10.0){
            hidden = lm.hidden_vocab[int_vocab_distribution[0](int_gen[0])];
          } 
        }else{
          hidden = lm.hidden_vocab[int_vocab_distribution[0](int_gen[0])];
        }
        e[i] = hidden;
      } 
      is_viterbi = false;
    }else{
      Candidate lattice[20][2];
      // populate lattice with candidates for first token
      vector<Candidate> cands = seed_table[f.tokens[0]];
      int i;
      for(i = 0; i < cands.size() && i < 20; i++){
        lattice[i][0].token = cands[i].token;
        lattice[i][0].prob = log10(cands[i].prob) + 
                             lm.get_ngram_prob(lm.pseu_start, cands[i].token);
      }
      cands = seed_table[f.tokens[1]];
      float best_prob = -FLT_MAX;
      for(int j = 0; j < cands.size() && j < 20; j++){
        lattice[j][1].token = cands[j].token;
        lattice[j][1].prob = -FLT_MAX;
        // compute viterbi sequence
        for(int k = 0; k < i; k++) {
          float prob = lattice[k][0].prob +
                       log10(cands[j].prob) +
                       lm.get_ngram_prob(lattice[k][0].token, cands[j].token) +
                       lm.get_ngram_prob(cands[j].token, lm.pseu_end);
          if(prob > best_prob) {
            best_prob = prob;
            e[0] = lattice[k][0].token;
            e[1] = lattice[j][1].token; 
          }
                       
        }
      }
      is_viterbi = true;
      //cout << e[0] << " " << e[1] << endl; 
    }    
  }
  
  float getChannelProb(unsigned int hidden, unsigned int observe){
    int joint_count = 0;
    int condition_count = 0;
    long joint = concatNumbers(hidden, observe);
    long condition = hidden;
    unsigned int pid = plain_dis2con_map[hidden];
    unsigned int cid = cipher_dis2con_map[observe];
    
    concurrent_hash_map<long, long>::const_accessor accessor;
    if(counts.find(accessor, joint)) {
      joint_count = accessor->second;
      accessor.release();
    }
    //alpha = alphas(pid);

    if(counts.find(accessor, condition)) {
      condition_count = accessor->second;
      accessor.release();
    }
    float prob = (alpha * base_distribution(pid, cid) + joint_count)/(alpha + condition_count);
    if(prob <= 0) {
        cout << "wrong prob: " << pid << " " << cid << " " << hidden << " " << observe << endl;
        cout << "alpha: " << alpha << endl;
        cout << base_distribution(pid, cid) << " " << joint_count << " " << condition_count << endl;
        exit(0);
    }
    return (alpha * base_distribution(pid, cid) + joint_count)/(alpha + condition_count);
  }

  int getCandidates(float* cand_score, boost::unordered_map<unsigned int,unsigned int>& members, 
                    float threshold, float raw_channel, vector<unsigned int>& candidates) {
    int location = 2;
    while(location <= 2 * slice_list_size && cand_score[location - 1] * raw_channel >= threshold) {
      location += 4;
    }
    if(location > (2 * slice_list_size)) {
      return slice_list_size;
    }else {
      return (location - 2) >> 1;
    }  
  }

  void loadCipherBigrams(const char* file) {
       ifstream ifs(file);
       string line;
       if(!ifs.is_open()) {
           cout << "file not found" << endl;
           exit(0);
       }
       int maxid = 0;
       while(getline(ifs,line)){
          vector<string> entries;
          LM::split(line, entries, "\t");
          vector<string> s_tokens;
          LM::split(entries[1], s_tokens, " ");
          int id1 = atoi(s_tokens[0].c_str());
          int id2 = atoi(s_tokens[1].c_str());
          if(id1 > maxid) {
              maxid = id1;
          }
          if(id2 > maxid) {
              maxid = id2;
          }
          Bigram tmp(id1, id2, atoi(entries[0].c_str()));
          cipher_bigrams.push_back(tmp);
       }
       plain_vocab_locks = new pthread_mutex_t[lm.max_id + 1];
       for(int i = 0; i < lm.max_id + 1; i++) {
           pthread_mutex_init(&plain_vocab_locks[i], 0);
       }
       cipher_vocab_locks = new pthread_mutex_t[maxid + 1];
       for(int i = 0; i < maxid + 1; i++) {
           pthread_mutex_init(&cipher_vocab_locks[i], 0);
       }
       cout << cipher_bigrams.size() << " bigrams loaded" << endl;
  }

  void initSamples() {
    unsigned int sample[2];
    for(int i = 0; i < cipher_bigrams.size(); i++){
      Bigram& cipher = cipher_bigrams[i];
      getBestSequence(cipher, sample);
      for(int j = 0; j < 2; j++) {
        observed_v[cipher.tokens[j]] = 1;
        if(o_counts.count(cipher.tokens[j]) == 0) {
          o_counts[cipher.tokens[j]] = cipher.count;
        }else {
          o_counts[cipher.tokens[j]] += cipher.count; 
        }
        // update sample and  counts
        if(sample[j] >= lm.pseu_end + 1 || lm.uni_gram_prob[sample[j]][0] == -10.0){
          sample[j] = lm.hidden_vocab[int_vocab_distribution[0](int_gen[0])];
        }
        cipher.sample_tokens[j] = sample[j];
        long plain_token = sample[j];
        
        concurrent_hash_map<long, long>::accessor accessor;
        counts.insert(accessor, plain_token);
        accessor->second += cipher.count;
       
        long plain_cipher_pair = concatNumbers(sample[j], cipher.tokens[j]);
        counts.insert(accessor, plain_cipher_pair);
        accessor->second += cipher.count;
        accessor.release();
 
        // update countlist
        if(channel_list.count(cipher.tokens[j]) == 0) {
          channel_list[cipher.tokens[j]] = CountList();
        }
        CountList& count_list = channel_list[cipher.tokens[j]];
        if(count_list.members.count(sample[j]) == 0) {
          count_list.members[sample[j]] = 1;
          count_list.member_list.push_back(sample[j]);
        }
        
      }
    }
    uniform_base = 1.0 / (float)observed_v.size();
    if(use_uniform_base || fast_mode || base_threshold == 0.0) 
    {
      cout << "using uniform threshold" << endl;
      base_threshold = uniform_base;
    }
    //uniform_base = 1.0 / 500; 
    cout << "Observe Vocab: " << observed_v.size() << endl;
  }
  
  void initBaseDistribution(string base_file = "") {
    if(seed_table.size() == 0 || !valid_embeddings || !use_embeddings) {
        if(use_uniform_base || base_file == "") {
            initUniformBase();
        } else {
            loadBaseFromFile(base_file);
        }
    } else {
        loadMappingFromFile(map_file_name);
        buildCountsMatrix();
        doMappingOptimization(max_threads);
        updateCache();        
    }
  }

  unsigned int drawSample(int thread_id, int pos, Bigram& cipher, unsigned int hidden, unsigned int observed, double& optr_count, double& token_count) {
    unsigned int pre_hidden, post_hidden;
    if(pos == 0) {
      pre_hidden = lm.pseu_start;
      post_hidden = cipher.sample_tokens[1];
    }else {
      pre_hidden = cipher.sample_tokens[0];
      post_hidden = lm.pseu_end;
    }
    float ngram_prob = pow(10, lm.get_ngram_prob(pre_hidden, hidden) +
                                       lm.get_ngram_prob(hidden, post_hidden));
    float channel_prob = getChannelProb(hidden, observed);
    double random_num = real_distribution[thread_id](flt_gen[thread_id]);
    float threshold = ngram_prob * channel_prob * random_num;
    unsigned int context = pre_hidden + post_hidden;
    token_count++;
    CountList& exist_trans = channel_list[observed];
    float* slice_cand_list = slice_list[context]; 
    int slice_cand_size = slice_list_size * 2;
    float score = 1.0;
    float raw_channel = (alpha * base_threshold + 0) / (alpha + 0);
    if(slice_cand_list[slice_cand_size - 1] * raw_channel < threshold) {
      vector<unsigned int> candidates;
      int range1 = getCandidates(slice_cand_list, exist_trans.members, threshold, raw_channel, candidates);
      int range2 = exist_trans.member_list.size();
      int range = range1 + range2;
      int cand_index = 0;
      //boost::unordered_set<unsigned int> cand_to_remove;
      while(true) {
        optr_count++;
        boost::uniform_int<int> int_cand_distribution(0, range - 1);
        cand_index = int_cand_distribution(int_gen[thread_id]);
        if(cand_index < range1) { // drop samples from set A: P(trigram)*prior > T
          int location = cand_index << 1;
          unsigned int new_hidden = slice_cand_list[location];
          if(exist_trans.members.count(new_hidden) != 0) {
            continue;
          }
          float ngram_prob = slice_cand_list[location + 1];
          float channel_prob = getChannelProb(new_hidden, observed);
          if(ngram_prob * channel_prob >= threshold) {
            exist_trans.members[new_hidden] = 1;
            exist_trans.member_list.push_back(new_hidden);
            return new_hidden;
          }
        }else { // draw samples from set B: count(e,f) > 0
          cand_index -= range1;
          unsigned int new_hidden = exist_trans.member_list[cand_index];
          if(new_hidden == hidden) {
            /*for(boost::unordered_set<unsigned int>::iterator itr = cand_to_remove.begin();
                itr != cand_to_remove.end(); itr++) {
              exist_trans.members.erase(*itr);
            }
            exist_trans.member_list.clear();
            for(boost::unordered_map<unsigned int,unsigned int>::iterator itr = exist_trans.members.begin();
                itr != exist_trans.members.end(); itr++) {
              exist_trans.member_list.push_back(itr->first);
            }*/
            return new_hidden;
          } 
          float channel_prob = getChannelProb(new_hidden, observed);
          score = pow(10, lm.get_ngram_prob(pre_hidden, new_hidden) +
                        lm.get_ngram_prob(new_hidden, post_hidden)) * channel_prob;
          if(score >= threshold) {
            /*for(boost::unordered_set<unsigned int>::iterator itr = cand_to_remove.begin();
                itr != cand_to_remove.end(); itr++) {
              exist_trans.members.erase(*itr);
            }
            exist_trans.member_list.clear();
            for(boost::unordered_map<unsigned int,unsigned int>::iterator itr = exist_trans.members.begin();
                itr != exist_trans.members.end(); itr++) {
              exist_trans.member_list.push_back(itr->first);
            }*/
            return new_hidden;
          } 
          // remove candidates that won't pass threshold
          long cand_pair = (long)new_hidden << 30 | observed;
          if(counts.count(cand_pair) == 0 && (fast_mode || (float)(base_distribution(plain_dis2con_map[new_hidden], cipher_dis2con_map[observed])) <= base_threshold)) {
            exist_trans.members.erase(new_hidden);
            exist_trans.member_list.erase(exist_trans.member_list.begin() + cand_index);
            --range;
          }                 
        }
      }
    }else { // back off to slow mode when P(k)*prior >= threshold
      while(true) { // while loop
        optr_count++;
        unsigned int new_hidden = lm.hidden_vocab[int_vocab_distribution[thread_id](int_gen[thread_id])];
        if(new_hidden == hidden) {
          return new_hidden;
        }
        float channel_prob = getChannelProb(new_hidden, observed);
        score = pow(10, lm.get_ngram_prob(pre_hidden, new_hidden) +
                        lm.get_ngram_prob(new_hidden, post_hidden)) * channel_prob;
        if(score >= threshold) {
          if(exist_trans.members.count(new_hidden) == 0) {
            exist_trans.members[new_hidden] = 1;
            exist_trans.member_list.push_back(new_hidden);
          }
          /*if(counts.count(concatNumbers(hidden, observed)) == 0) {
            exist_trans.members.erase(hidden);
          }*/
          return new_hidden;
        }
      } // while loop
    }
    return 0;
  }

  void printTTable(const char* file) {
    ofstream ofs(file);
    if(!ofs.is_open()) {
      cout << "can't open file for writing" << endl;
      exit(0);
    }
    std::cerr << "Writing ID ttable to " << file << std::endl; 
    long mask = INT_MAX >> 1; 
    for(concurrent_hash_map<long,long>::iterator itr = counts.begin();
           itr != counts.end(); itr++) {
      long key = itr->first;
      long token1 = key & mask;
      long token0 = (key >> 30) & mask;
      if(token0 != 0) {
        concurrent_hash_map<long, long>::const_accessor accessor;
        counts.find(accessor, token0);
        float prob = (float)itr->second / (float)accessor->second;
        accessor.release();
        float reverse_prob = (float)itr->second / (float)o_counts[token1];
        ofs << token1 << " ||| " << token0 << " ||| " << prob << " " << reverse_prob << endl;
      }  
    }      
    ofs.close();
  }

  
  void printMapMatrix(string file_name) {
     writeMatrix(M, file_name);
  } 
  
  void printBase(string file_name) {
     writeMatrix(base_distribution, file_name);
  } 

  /*void printBase(string base_file_name, string alphas_file_name) {
     ofstream bofs(base_file_name.c_str());
     ofstream aofs(alphas_file_name.c_str());
     for(int row = 0; row < base_distribution.rows(); row++) {
         bofs << plain_con2dis_map[row] << "\t";
         for(int col = 0; col < base_distribution.cols(); col++) {
             if(base_distribution(row, col) >= 0.001 ) {
                 bofs << cipher_con2dis_map[col] << ":" << base_distribution(row, col) << "\t";
             } 
         }
         bofs << endl;
     }
     // output alphas
     for(int row = 0; row < alphas.rows(); row++) {
         aofs << plain_con2dis_map[row] << "\t";
         for(int col = 0; col < alphas.cols(); col++) {
             aofs << alphas(row, col) << "\t";
         }        
         aofs << endl;
     }
  }*/

  void printAccCounts(const char* file) {
    ofstream ofs(file);
    if(!ofs.is_open()) {
      cout << "can't open file for writing" << endl;
      exit(0);
    }
    long mask = INT_MAX >> 1; 
    for(boost::unordered_map<long,long>::iterator itr = acc_counts.begin();
           itr != acc_counts.end(); itr++) {
      long key = itr->first;
      long token1 = key & mask;
      long token0 = (key >> 30) & mask;
      if(token0 != 0) {
        ofs << token1 << " ||| " << token0 << " ||| " << acc_counts[key] << endl;
      }  
    }      
    ofs.close();
  }
  
}; // end of Decipherment class


struct ThreadData{
    int thread_id;
    int iteration;
    int thread_count;
    Decipherment* decipherer;
    ThreadData(){}
    ThreadData(int tid, int i, int tc, Decipherment* d){
        thread_id = tid;
        iteration = i;
        thread_count = tc;
        decipherer = d;
    }
};

class ThreadWrapper{
    public:
    static void* runSampling(void* args) {
        Decipherment* decipherer = ((ThreadData *)args)->decipherer;
        int thread_id = ((ThreadData *)args)->thread_id;
        int iteration = ((ThreadData *)args)->iteration;
        int thread_count = ((ThreadData *)args)->thread_count;
        int start_pos = thread_id * (decipherer->cipher_bigrams.size()/thread_count);
        int end_pos = start_pos + (decipherer->cipher_bigrams.size()/thread_count);
        if(thread_id == thread_count - 1) {
          end_pos = decipherer->cipher_bigrams.size();
        }
        //cout << "thread " << thread_id << " from " << start_pos << "to " << end_pos << endl;
        double optr_count = 0;
        double token_count = 0; 
        double corpus_prob = 0.0;
        for(int i = 0; i < iteration; i++) {
            corpus_prob = 0;
            optr_count = 0;
            token_count = 0;
            for(int k = start_pos; k < end_pos; k++) {
                Bigram& cipher = decipherer->cipher_bigrams[k];
                long count = cipher.count;
                unsigned int pre_hidden = decipherer->lm.pseu_start;
                for(int j = 0; j < 2; j++) { // for each token in a bigram
                    unsigned int old_hidden = cipher.sample_tokens[j];
                    unsigned int observed = cipher.tokens[j];
                    long plain_cipher_pair = decipherer->concatNumbers(old_hidden, observed);
                    unsigned int new_hidden = 0;
                    pthread_mutex_lock(&(decipherer->cipher_vocab_locks[observed]));
                    // reduce old counts
                    concurrent_hash_map<long, long>::accessor accessor;
                    decipherer->counts.find(accessor, plain_cipher_pair);
                    accessor->second -= count;
                    
                    /*if(accessor->second  < 0) { // check that counts never become smaller than zero
                    cout << "fatal error" << endl;
                    exit(0);
                    }*/

                    if(accessor->second == 0) {
                        if(!decipherer->counts.erase(accessor)) {
                            cout << "fail to remove obsolte items" << endl; exit(0);
                        }
                    }
                    accessor.release();

                    pthread_mutex_lock(&(decipherer->plain_vocab_locks[old_hidden]));
                    
                    decipherer->counts.find(accessor, (long)old_hidden);
                    accessor->second -= count;
                    /*if(counts[(long)old_hidden] < 0) {
                    cout << "fatal error" << endl;
                    exit(0);
                    }*/
                    if(accessor->second == 0) {
                        //pthread_mutex_lock(&(decipherer->hash_key_lock));
                        decipherer->counts.erase(accessor);
                        //pthread_mutex_unlock(&(decipherer->hash_key_lock));
                    }
                    accessor.release();
          
                    new_hidden = decipherer->drawSample(thread_id, j, cipher, old_hidden, observed, optr_count, token_count);
                    pthread_mutex_unlock(&(decipherer->plain_vocab_locks[old_hidden]));
                    cipher.sample_tokens[j] = new_hidden; // update sample
                    if(i % 500 == 0) {
                        corpus_prob += decipherer->lm.get_ngram_prob(pre_hidden, old_hidden);
                        corpus_prob += log10(decipherer->getChannelProb(old_hidden, observed));
                    }
                    pre_hidden = new_hidden;
                    // accumulate count
                    plain_cipher_pair = decipherer->concatNumbers(new_hidden, observed);
                    /*if(iteration - i <= 1000) {
                           if(i % 100 == 0) {
                              acc_counts[plain_cipher_pair] += count;  
                          }
                    }*/
                    // update count
                    decipherer->counts.insert(accessor, plain_cipher_pair);
                    accessor->second += count;
                    accessor.release();

                    pthread_mutex_lock(&(decipherer->plain_vocab_locks[new_hidden]));
                    decipherer->counts.insert(accessor,(long)new_hidden);
                    accessor->second += count;
                    accessor.release();
                    pthread_mutex_unlock(&(decipherer->plain_vocab_locks[new_hidden]));
                    pthread_mutex_unlock(&(decipherer->cipher_vocab_locks[observed]));
               }
               if(i % 500 == 0) {
                   corpus_prob += decipherer->lm.get_ngram_prob(pre_hidden, decipherer->lm.pseu_end);
               }
           } 
           // output corpus probability
           if(i % 500 == 0) {
               cout << thread_id << " " << i << " " << corpus_prob << " avg optr:" << optr_count/token_count << endl;
           }
        }
    }
};

struct MappingDataWrapper{
    int thread_id;
    int begin_source_id;
    int end_source_id;
    bool computeBase;
    Decipherment* data;
    Matrix<double, Dynamic, Dynamic> M_gradient;
    double objective_function_value;
};


static void* computeGradient(void* args){
                int thread_id = ((MappingDataWrapper*)args)->thread_id;
                
                Matrix<double, Dynamic, Dynamic>& base_distribution = ((MappingDataWrapper*)args)->data->base_distribution;
                Matrix<double, Dynamic, Dynamic>& source_target_counts = ((MappingDataWrapper*)args)->data->counts_matrix;
                Matrix<double, Dynamic, 1>& source_counts = ((MappingDataWrapper*)args)->data->source_counts;
                Matrix<double, Dynamic, 1>& alphas = ((MappingDataWrapper*)args)->data->alphas;
                Matrix<double, Dynamic, Dynamic>& source_embeddings = ((MappingDataWrapper*)args)->data->plain_embeddings;
                Matrix<double, Dynamic, Dynamic>& target_embeddings = ((MappingDataWrapper*)args)->data->cipher_embeddings; 
                Matrix<double, Dynamic, Dynamic> &M = ((MappingDataWrapper*)args)->data->M;
                Matrix<double, Dynamic, Dynamic> &M_gradient = ((MappingDataWrapper*)args)->M_gradient;
                double& objective_function_value = ((MappingDataWrapper*)args)->objective_function_value;
                int begin_source_id = ((MappingDataWrapper*)args)->begin_source_id;
                int end_source_id = ((MappingDataWrapper*)args)->end_source_id;
                int num_target_words = target_embeddings.rows();
                bool computeBase = ((MappingDataWrapper*)args)->computeBase;
                //cout << "from: " << begin_source_id << "to: " << end_source_id << endl; 
		for (int source_index = begin_source_id; source_index < end_source_id; source_index++){
			//GET THE EXP SUMS
			if(source_counts(source_index) == 0 && !computeBase) {
                            continue;
                        }
                        //cout << source_index << endl;
			double exp_sum = 0.;
			Matrix<double,Dynamic,1> affinity_scores;
			affinity_scores.setZero(num_target_words);
			//cerr<<"here1"<<endl;
			for (int target_index=0; target_index<num_target_words; target_index++) {
				affinity_scores(target_index) = exp((source_embeddings.row(source_index)*M).dot(target_embeddings.row(target_index)));
				base_distribution(source_index,target_index) = affinity_scores(target_index);
				exp_sum += affinity_scores(target_index);
				//cerr<<"affinity sore"<<affinity_scores(target_index)<<endl;
				//cerr<<"exp sum is "<<exp_sum<<endl;
			}
			alphas(source_index) = exp_sum;
			//cerr<<"exp sum is "<<exp_sum<<endl;
			double sum_digamma_diff = boost::math::digamma(exp_sum) - boost::math::digamma(exp_sum+source_counts(source_index));
			objective_function_value += boost::math::lgamma(exp_sum) - boost::math::lgamma(exp_sum+source_counts(source_index));
			//cerr<<"here 2"<<endl;
			for (int target_index=0; target_index<num_target_words; target_index++){
				Matrix<double,Dynamic,Dynamic> outer_product = source_embeddings.row(source_index).transpose()*target_embeddings.row(target_index);
				//double weight = exp((source_embeddings(source_index)*M).dot(target_embeddings(target_index)));
				//current_gradient_term = sum_digamma_diff;
				double weight = sum_digamma_diff;
				if (source_target_counts(source_index,target_index) != 0.) {
					weight += boost::math::digamma(affinity_scores(target_index)+source_target_counts(source_index,target_index))-
						boost::math::digamma(affinity_scores(target_index));
					objective_function_value += boost::math::lgamma(affinity_scores(target_index)+source_target_counts(source_index,target_index))-boost::math::lgamma(affinity_scores(target_index));
                                }
				weight *= affinity_scores(target_index);
				M_gradient += outer_product*weight;
			}
		}
}


void reestimateMapping(Decipherment* myData,
double reg_lambda,
int epochs,
double learning_rate,
int num_threads) {

	//UNCONST(DerivedD,const_M,M);
	//UNCONST(DerivedA,const_base_distribution, base_distribution);
	//UNCONST(DerivedC, const_alphas, alphas);
	
	int num_source_words = myData->plain_embeddings.rows();
	int num_target_words = myData->cipher_embeddings.rows();
	long int total_counts = myData->source_counts.sum();
        cerr<<"Total counuts is "<<total_counts<<endl;
        cerr<<"max num of threads: " << num_threads << endl;
        vector<MappingDataWrapper> targs(num_threads);
        vector<pthread_t> t_handler(num_threads);
	for (int epoch=0; epoch<epochs; epoch++){
	    cerr<<"Epoch "<<epoch<<endl;
		// start threads here
                int begin_source_id = 0;
                int end_source_id = 0;
                int totalRows = myData->source_counts.rows();
                int block_size = totalRows / num_threads;
                double objective_function_value = 0.0;
                Matrix<double, Dynamic, Dynamic> M_gradient;
                M_gradient.setZero(myData->M.rows(), myData->M.cols());
                for(int tid = 0; tid < num_threads; tid++) {
                    end_source_id = begin_source_id + block_size;
                    if(num_threads - 1 == tid) {
                        end_source_id = totalRows;
                    }
                    MappingDataWrapper& curThread = targs[tid];
                    curThread.begin_source_id = begin_source_id;
                    curThread.end_source_id = end_source_id;
                    curThread.data = myData;
                    curThread.thread_id = tid;
                    curThread.M_gradient.setZero(myData->M.rows(), myData->M.cols());
                    curThread.objective_function_value = 0;
                    curThread.computeBase = epoch == epochs - 1;
                    int rc = pthread_create(&t_handler[tid], 0, &computeGradient, (void*)(&targs[tid]));
                    if(rc != 0) {
                        cout << "error creating thread" << endl;
                        exit(0);
                    }
                    begin_source_id = end_source_id;
                }
                for(int tid = 0; tid < num_threads; tid++) {
                    pthread_join(t_handler[tid], 0);
                    objective_function_value += targs[tid].objective_function_value;
                    M_gradient += targs[tid].M_gradient;
                }                
                //Scaling the objective function value
		objective_function_value /= total_counts;
		//Now to update the weights. 
		//Matrix<double,Dynamic,Dynamic> reg_gradient = reg_lambda*(myData->M.array().square()).matrix();
		double reg_value = reg_lambda*(myData->M.array().square()).sum()/2;
                myData->M += learning_rate*(M_gradient/total_counts - reg_lambda*myData->M);
		//cerr<<"mapping matrix"<<endl;
		//cerr<<M<<endl;
		cout<<"Objective function value before reg gradient is "<<objective_function_value<<endl;
		objective_function_value -= reg_value;
		
		cout<<"Objective function value in epoch "<<epoch<<" was "<<objective_function_value<<endl;
		//learning_rate = learning_rate*(epoch+1)/(epoch+2);
		cout<<"Learning rate is "<<learning_rate<<endl;
                //exit(0);
	}
	//Now update the base distribution 

	//base_distribution = base_distribution.rowwise() *(1/alphas.transpose().array()); 
	for (int i=0; i<myData->base_distribution.rows(); i++){
	    myData->base_distribution.row(i) = myData->base_distribution.row(i)/myData->alphas(i);
	}
	cout<<"sum of base distribution is"<<myData->base_distribution.sum()<<endl;
}
							
