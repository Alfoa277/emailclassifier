#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronFeatureHashing : public BaseClf<PerceptronFeatureHashing> {
    int ngram_;
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    /** Save the weights as a vector. */
    std::vector<double> weights_;
    int seed_;

public:
    PerceptronFeatureHashing(int ngram, int log_num_buckets, double learning_rate)
        : BaseClf(0.15)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , seed_(0xa738cc)
    {
        /** Initialize weights to 0. */
        weights_.resize(1 << log_num_buckets_, 0.0);
    }

    void update_(const Email& email) {
        /** Obtain label and transform ham emails to -1. */
        int is_spam = email.is_spam();
        int true_label = (is_spam) ? 1 : -1;

        /** Obtain email prediction. */
        double pred_label = predict_(email);

        /** Initialize observations vector. */
        std::vector<double> observations(1 << log_num_buckets_,0.0);

        /** Iterate through n-grams updating the observations vector. */
        EmailIter iter(email,ngram_);
        while(iter) {
            std::string_view ngram = iter.next();
            size_t bucket= get_bucket(ngram);
            observations[bucket]+=1;
        }

        /** Apply delta rule to update the weights. */
        double difference = pred_label - true_label;
        for(int i = 0; i < (1 << log_num_buckets_); ++i){
            weights_[i] = weights_[i] - learning_rate_ * 2 * difference * observations[i];
        }
        bias_ = bias_ - learning_rate_ * difference;

    }

    double predict_(const Email& email) const {
        /** Initialize observations vector. */
        std::vector<double> observations(1 << log_num_buckets_,0.0);

        /** Iterate through n-grams updating the observations vector. */
        EmailIter iter(email,ngram_);
        while(iter) {
            std::string_view ngram = iter.next();
            size_t bucket(get_bucket(ngram));
            observations[bucket]+=1;
        }
        /** Add bias and multiply the weights and observations vectors. */
        double sum = 0;
        sum += bias_;
        for(int i = 0; i < (1 << log_num_buckets_); ++i){
            sum += weights_[i] * observations[i];
        }

        /** Apply hyperbolic tangent activation function, which maps the weighted sum to a value between
         * 1 and -1. */
        double result = tanh(sum);
        return result;
    }


private:
    size_t get_bucket(std::string_view ngram) const
    {   return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const {
        size_t num_buckets = 1 << log_num_buckets_;
        return (hash % num_buckets); /** Modulo operator to limit to max number of buckets. */
    }
};

} // namespace bdap


