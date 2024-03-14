#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronCountMin : public BaseClf<PerceptronCountMin> {
    int ngram_;
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    int num_hashes_;
    /** Save weights in count-min sketch matrix. */
    std::vector<std::vector<double>> weights_;
    std::vector<int> seeds_;

public:
    PerceptronCountMin(int ngram, int num_hashes, int log_num_buckets,
                       double learning_rate)
        : BaseClf(0.17)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , num_hashes_(num_hashes)
        /** Initialize all weights to 0. */
        , weights_(num_hashes,std::vector<double>(1 << log_num_buckets_,0.0))
    {
        /** Initialize seeds randomly. */
        seeds_.resize(num_hashes_,0);
        for (int i = 0; i<num_hashes_; ++i) {
            size_t seed = rand();
            seeds_[i] = seed;
        }
    }

    void update_(const Email& email) {
        /** Obtain label and transform ham emails to -1. */
        int is_spam = email.is_spam();
        int true_label = (is_spam) ? 1 : -1;

        /** Obtain email prediction. */
        double pred_label = predict_(email);

        /** Iterate over the n-grams updating the corresponding weight in each hash function. */
        EmailIter iter(email,ngram_);

        /** Perform the updates. */
        double difference = pred_label-true_label;
        while(iter){
            std::string_view ngram = iter.next();
            /** Find the corresponding bucket in each hash function. */
            std::vector<size_t> buckets = find_buckets(ngram);
            /** For each hash function, apply the delta rule to the corresponding bucket. */
            for(int j = 0; j < num_hashes_; ++j){
                size_t bucket = buckets[j];
                weights_[j][bucket] = weights_[j][bucket] - learning_rate_ * 2 * difference;
            }
        }
        /** Update the bias term. */
        bias_ = bias_ - learning_rate_ * difference;
    }

    double predict_(const Email& email) const {
        /** Iterate over the n-grams and obtain the median weight of the buckets that they are assigned in the
         * count-min sketch matrix. */
        EmailIter iter(email,ngram_);
        double sum = 0;
        sum += bias_;
        /** Since it will iterate over the entire email, each weight is multiplied by an observation count of 1. Thus
         * if a given n-gram appears k times in the email, the operation will happen k times, which is equivalent to
         * multiplying the median weight by k. */
        while(iter){
            std::string_view ngram = iter.next();
            double weight = find_median_weight(ngram);
            sum += weight;
        }
        /** Apply hyperbolic tangent activation function. */
        double result = tanh(sum);
        return result;
    }

private:
    size_t get_bucket(std::string_view ngram, int hash_num) const
    {   return get_bucket(hash(ngram, seeds_[hash_num])); }

    size_t get_bucket(size_t hash) const {
        size_t num_buckets = 1 << log_num_buckets_;
        return (hash % num_buckets); /** Modulo operator to limit to max number of buckets. */
    }

    std::vector<size_t> find_buckets(std::string_view ngram) const {
        /** This function returns the index of the buckets in every hash function to which a given n-gram is assigned */
        std::vector<size_t> buckets(num_hashes_,0);
        for(int i = 0; i < num_hashes_; ++i){
            size_t bucket = get_bucket(ngram, i);
            buckets[i] = bucket;
        }
        return buckets;
    }

    double find_median_weight(std::string_view ngram) const {
        /** This function finds the corresponding weights to a given n-gram, and returns their median value. */
        std::vector<double> hashed_weights(num_hashes_,0.0);
        for(int i = 0; i < num_hashes_; ++i){
            size_t bucket = get_bucket(ngram, i);
            hashed_weights[i] = weights_[i][bucket];
        }
        /** Sort the weights. */
        std::sort(hashed_weights.begin(), hashed_weights.end());
        if(num_hashes_ % 2 == 0){
            /** In the case of an even number of hash functions, find the average between the two middle values. */
            return (hashed_weights[num_hashes_ / 2] + hashed_weights[(num_hashes_ / 2) - 1]) / 2.0;
        } else {
            return hashed_weights[num_hashes_ / 2];
        }
    }
};

} // namespace bdap
