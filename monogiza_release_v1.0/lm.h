/* 
 * File:   lm.h
 * Author: Qing
 *
 * Created on January 15, 2014, 3:29 PM
 */

#ifndef LM_H
#define	LM_H

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/unordered_map.hpp>

using namespace std;

class LM {
public:   
    //public static HashMap<Integer,HashMap<Integer,Float>> bi_gram_prob=new HashMap<Integer,HashMap<Integer,Float>>(); 
    boost::unordered_map<unsigned int,float>** bi_gram_prob;  //new HashMap<Integer,HashMap<Integer,Float>>();
    float** uni_gram_prob;
    unsigned int* hidden_vocab; 
    unsigned int vocab_size;


    unsigned int max_id;
    float lm_weight;
   // static long mask=Integer.MAX_VALUE>>11;
    unsigned int pseu_start,pseu_end;
    static void split(const string& str,
                      vector<string>& tokens,
                      const string& delimiters = " ")
    {
      // Skip delimiters at beginning.
      string::size_type lastPos = str.find_first_not_of(delimiters, 0);
      // Find first "non-delimiter".
      string::size_type pos     = str.find(delimiters, lastPos);

      while (string::npos != pos || string::npos != lastPos)
      {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find(delimiters, lastPos);
      }
    }    
    /*void init_special_ids(int s){
        
        pseu_start=0;
        pseu_end=s;
    }
    
    public static int get_context_id(int w1,int w2){
        return w1+w2;
    }*/
    
    unsigned int get_token_id(string token){
        if(token == "<s>")
            return pseu_start;
        else if(token == "</s>")
            return pseu_end;
        else{
            return atoi(token.c_str());
        }
    }
    
    float get_ngram_prob(unsigned int id1,unsigned int id2){
       if(id1 > pseu_end || id2 > pseu_end) {
           return -20;
       } 
       if(bi_gram_prob[id1] !=0 && bi_gram_prob[id1]->find(id2) != bi_gram_prob[id1]->end()) {
           return (*(bi_gram_prob[id1]))[id2];
       }else {
           return uni_gram_prob[id2][0] + uni_gram_prob[id1][1];
       }  
    }
    
    /*public float get_ngram_prob_unk(int id1,int id2){

       try{
           return bi_gram_prob[id1].get(id2)*lm_weight;
       }catch(Exception e){
           try{
               return (uni_gram_prob[id2][0]+uni_gram_prob[id1][1])*lm_weight;
           }catch(Exception e2){
               return -5;
           }
       }
    }*/   
   
    void get_vocab_size(const char* file){
          max_id = 0;
          vocab_size = 0;              
          ifstream ifs(file);
          string line;
          if(!ifs.is_open()) {
              cout << "file not found" << endl;
              exit(0);
          }
          while(getline(ifs,line)){
             if(line.find("\t") != string::npos){
               vector<string> entries;
               split(line, entries, "\t");
               vector<string> tokens;
               split(entries[1], tokens);
               if(tokens.size() == 1 && tokens[0] != "<s>" && tokens[0] != "</s>") {
                   int id = atoi(tokens[0].c_str());
                   vocab_size++;
                   if (id > max_id)
                       max_id=id;
               }else if (tokens.size() == 2)
                   break;
             }                   
          } 
          pseu_start = 0;
          pseu_end = max_id + 1;
          cout << "max_id" << max_id << endl;
          //cout << "max_id " + max_id << endl;  
        
    }
    
    void load_lm(const char* file){
          // get vocab size first         
          get_vocab_size(file);

                
         
          
          // allocate memory              
          uni_gram_prob=new float*[pseu_end+1];  
          for(int i=0;i < pseu_end + 1;i++){
              uni_gram_prob[i] = new float[2];
              uni_gram_prob[i][0] = -10.0f;
              uni_gram_prob[i][1] = -10.0f;
          }
          bi_gram_prob = new boost::unordered_map<unsigned int,float>*[pseu_end + 1];
          for(int i = 0; i < pseu_end; i++) {
              bi_gram_prob[i] = 0;
          }
          hidden_vocab=new unsigned int[vocab_size];
          
          
          ifstream ifs(file);
          string line;
          if(!ifs.is_open()) {
              cout << "file not found" << endl;
              exit(0);
          }
          int counter=0;
          while(getline(ifs,line)){
             if(line.find("\t") != string::npos){
                 vector<string> entries;
                 split(line, entries, "\t");
                 vector<string> tokens;
                 split(entries[1], tokens);
                 if(tokens.size() == 1 && tokens[0] != "<s>" && tokens[0] != "</s>"){                   
                     hidden_vocab[counter++] = get_token_id(tokens[0]);
                 }
                 if(entries.size() == 2){
                    if(tokens.size() == 2){
                    unsigned int id1=get_token_id(tokens[0]);
                    unsigned int id2=get_token_id(tokens[1]);
                    if(bi_gram_prob[id1] == 0)
                      bi_gram_prob[id1]=new boost::unordered_map<unsigned int,float>();
                    (*(bi_gram_prob[id1]))[id2] = atof(entries[0].c_str());
                    
                    }else{
                     uni_gram_prob[get_token_id(tokens[0])][0] = atof(entries[0].c_str());
                    }
                                                                  
                 }               
                 else if(entries.size() == 3){
                    unsigned int id = get_token_id(entries[1]);
                    uni_gram_prob[id][0] = atof(entries[0].c_str());
                    uni_gram_prob[id][1] = atof(entries[2].c_str());
                        
                 }
                 else
                  cout << "LM file format error" << endl;
             }   
             
          }
          //long allocatedMemory = Runtime.getRuntime().totalMemory();
          //System.out.println("allocated :"+allocatedMemory/1024/1024);
          //System.out.println(hidden_vocab.length+" "+vocab_size);
          
     }
      
};


#endif	/* LM_H */

