#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesCountMin : public BaseClf<NaiveBayesCountMin> {
    /** Two different count-min sketch matrices for ham and spams.*/
    std::vector<std::vector<int>> ham_counter_;
    std::vector<std::vector<int>> spam_counter_;
    std::vector<int> seeds_;
    /** Total email and ngrams counters. */
    int total_ham_;
    int total_spam_;
    int total_ngrams_ham_;
    int total_ngrams_spam_;
    int ngram_;
    int log_num_buckets_;
    int num_hashes_;

public:
    NaiveBayesCountMin(int ngram, int num_hashes, int log_num_buckets)
        : BaseClf(-1) /** Change to -17 for n-grams > 1*/
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , num_hashes_(num_hashes)
        , total_ham_(1)
        , total_spam_(1)
        , total_ngrams_ham_(1)
        , total_ngrams_spam_(1)
        , ham_counter_(num_hashes,std::vector<int>(1 << log_num_buckets,1))
        , spam_counter_(num_hashes, std::vector<int>(1 << log_num_buckets,1))
    {
        /** Initialize random seeds. */
        seeds_.resize(num_hashes_,0);
        for (int i = 0; i<num_hashes_; ++i) {
            size_t seed = rand();
            seeds_[i] = seed;
        }
    }

    void update_(const Email &email) {

        /** Obtain label and update count of spam/ham emails */
        int is_spam(email.is_spam());
        /** Create a vector pointer and assign it to the corresponding counts vector (ham or spam).
         * Create pointer to the total ngrams counter.*/
        std::vector<std::vector<int>>* counts;
        int* total_ngrams;

        if(is_spam){
            counts = &spam_counter_;
            /** Increment the spam email counter. */
            total_spam_ += 1;
            total_ngrams = &total_ngrams_spam_;
        } else {
            counts = &ham_counter_;
            /** Increment the ham email counter. */
            total_ham_ += 1;
            total_ngrams = &total_ngrams_ham_;
        }

        /** Iterate over ngrams, increment bucket counts in every hash function. */
        EmailIter iter(email,ngram_);
        while(iter){
            std::string_view ngram = iter.next();
            for(int j = 0; j < num_hashes_; ++j){
                size_t bucket(get_bucket(ngram,j));
                (*counts)[j][bucket]+=1;
                /** Increment the total number of n-grams observed count. */
                (*total_ngrams) += 1;
            }
        }

    }

    double predict_(const Email& email) const {
        /** Compute ham/spam prior probabilities. */
        float ham_prior = static_cast<float>(total_ham_)/(total_ham_+total_spam_);
        float spam_prior = 1 - ham_prior;

        /**  Iterate over the ngrams and compute the sum of the log ratios of likelihoods. */
        double sum_logs = 0;

        EmailIter iter(email,ngram_);
        while(iter){
            std::string_view ngram = iter.next();
            /** Find n-gram counts for ham and spam. */
            int observation_ham = find_count_min(ngram, ham_counter_);
            int observation_spam = find_count_min(ngram, spam_counter_);
            /** Compute the corresponding probabilities. */
            float likelihood_ham = static_cast<float>(observation_ham) / total_ngrams_ham_;
            float likelihood_spam = static_cast<float>(observation_spam) / total_ngrams_spam_;
            /** Add the log ratio of these probabilities. */
            sum_logs += log2(likelihood_spam/likelihood_ham);
        }
        /** Add the log ratio of prior probabilities. */
        return log2(spam_prior/ham_prior) + sum_logs;
    }



private:
    size_t get_bucket(std::string_view ngram, int hash_num) const {
        /** Hash n-grams with the corresponding seed.*/
        return get_bucket(hash(ngram, seeds_[hash_num]));
    }

    size_t get_bucket(size_t hash) const {
        /** Compute the number of buckets. */
        size_t num_buckets = 1 << log_num_buckets_;
        return (hash % num_buckets); /** Modulo operator to limit to max number of buckets. */
    }
    int find_count_min(std::string_view ngram, std::vector<std::vector<int>> counts_matrix) const {
        /** Check the bucket values for a given n-gram in every hashing function and return the lowest one. */
        int min_count = std::numeric_limits<int>::max(); /** Initialize to largest integer. */
        /** Iterate over bucket values. */
        for(int i = 0; i < num_hashes_; ++i){
            size_t bucket = get_bucket(ngram,i);
            int current_count = counts_matrix[i][bucket];
            /** If a lowe value is found, update min_count variable. */
            if(current_count<min_count){
                min_count = current_count;
            }
        }
        return min_count;
    }
};

}; // namespace bdap
