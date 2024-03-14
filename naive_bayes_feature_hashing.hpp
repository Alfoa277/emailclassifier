#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesFeatureHashing : public BaseClf<NaiveBayesFeatureHashing> {
    int seed_;
    int ngram_;
    int log_num_buckets_;
    std::vector<int> ham_counts_; /** Counts vector for ham emails */
    std::vector<int> spam_counts_; /** Counts vector for spam emails */

public:
    NaiveBayesFeatureHashing(int ngram, int log_num_buckets)
        : BaseClf(-1.5)
        , seed_(0xfa4f8cc)
        , ngram_(ngram)
        , log_num_buckets_(log_num_buckets)
        , ham_counts_((1 << log_num_buckets_) + 2, 1) /** Initialize all values to 1 to avoid zero counts. */
        /** It has 2^log_num_buckets buckets for individual n-gram hashing counts, plus two buckets to store the number
        of observed emails and total number of observed n-grams (useful for probability computations). */
        , spam_counts_((1 << log_num_buckets_) + 2,1)
    {}

    void update_(const Email &email) {
        /** Obtain label and update count of spam/ham emails */
        int is_spam(email.is_spam());
        /** Create a vector pointer and assign it to the corresponding counts vector (ham or spam). */
        std::vector<int>* counts_;
        if (is_spam) {
            counts_ = &spam_counts_;
        } else {
            counts_ = &ham_counts_;
        }

        /** Increment the ham/spam email counter. */
        (*counts_)[0]+=1;

        /** Iterate over ngrams, increment bucket counts and total ngram counts.*/
        EmailIter iter(email, ngram_);
        while(iter){
            std::string_view ngram = iter.next();
            size_t bucket(get_bucket(ngram));
            (*counts_)[bucket]+=1;
            (*counts_)[1]+=1;
        }
    }

    double predict_(const Email& email) const {
        /** Compute ham/spam prior probabilities. */
        float ham_prior = static_cast<float>(ham_counts_[0])/(ham_counts_[0]+spam_counts_[0]);
        float spam_prior = 1 - ham_prior;

        /**  Iterate over the ngrams and compute the sum of likelihoods ratios term. */
        double sum_logs = 0;

        int total_n_grams_ham = ham_counts_[1]; /** Total number of n-grams observed in ham emails */
        int total_n_grams_spam = spam_counts_[1]; /** Total number of n-grams observed in spam emails */
        EmailIter iter(email, ngram_);
        while(iter){
            std::string_view ngram = iter.next();
            size_t bucket = get_bucket(ngram);
            /** Find n-gram counts for ham and spam. */
            int observation_ham = ham_counts_[bucket];
            int observation_spam = spam_counts_[bucket];
            /** Compute the corresponding probabilities. */
            float likelihood_ham = static_cast<float>(observation_ham) / total_n_grams_ham;
            float likelihood_spam = static_cast<float>(observation_spam) / total_n_grams_spam;
            /** Add the log ratio of these probabilities. */
            sum_logs += log2(likelihood_spam/likelihood_ham);
        }
        /** Add the log ratio of prior probabilities. */
        return log2(spam_prior/ham_prior) + sum_logs;
    }

    std::vector<int> getHamCounts() {
        return ham_counts_;
    }

    std::vector<int> getSpamCounts() {
        return spam_counts_;
    }

private:
    size_t get_bucket(std::string_view ngram) const {
        return get_bucket(hash(ngram, seed_));
    }

    size_t get_bucket(size_t hash) const {
        size_t num_buckets = 1 << log_num_buckets_; /** Compute number of buckets */
        /** Modulo operator to limit to max number of buckets. */
        return (hash % num_buckets) + 2; /** Save the first two buckets for total email and ngram counters */
    }
};

} // namespace bdap
