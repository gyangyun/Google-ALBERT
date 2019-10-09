Albert_zh
An Implementation of A Lite Bert For Self-Supervised Learning Language Representations with TensorFlow

ALBert is based on Bert, but with some improvements. It achieves state of the art performance on main benchmarks with 30% parameters less.

For albert_base_zh it only has ten percentage parameters compare of original bert model, and main accuracy is retained.

Different version of ALBERT pre-trained model for Chinese, including TensorFlow, PyTorch and Keras, is available now.

Pre-training the ALBERT model on a large amount of Chinese corpus: fewer parameters and better results. The pre-training small model can also win 13 NLP missions, and the ALBERT three major transformations topped the GLUE benchmark.

Update
***** 2019-10-06: albert_xlarge_zh *****

Released albert_xlarge_zh, 59M parameters, half parameters of bert_base, 200M.

Rank top 1 for LCQMC dataset up to now, up 0.5 percentage

***** 2019-10-04: PyTorch and Keras versions of albert were supported *****

Convert to PyTorch version and do your tasks through albert_pytorch

Load pre-trained model with keras using one line of codes through bert4keras

Releasing albert_xlarge on 6th Oct

***** 2019-10-02: albert_large_zh, albert_base_zh *****

Relesed albert_base_zh with only 10% parameters of bert_base, a small model(40M) & training can be very fast.

Relased albert_large_zh with only 16% parameters of bert_base(64M)

***** 2019-09-28: codes and test functions *****

Add codes and test functions for three main changes of albert from bert

Model Download Download Pre-trained Models of Chinese
1, albert_xlarge_zh, this week will update a better version, so stay tuned, parameter quantity, layer number 24, file size is 230M

The parameter size and model size are one-half of bert_base; on the test set of the colloquial description similarity data set LCQMC, the bert_base rises by 0.8 points; a large graphics card is required.
2, albert_large_zh, parameter quantity, layer number 24, file size is 64M

The parameter size and model size are one-sixth of bert_base; on the test set of the colloquial description similarity data set LCQMC, the bert_base rises by 0.2 points.
3, albert_base_zh (small model experience version), parameter quantity 12M, layer number 12, size 40M

The parameter quantity is one tenth of the bert_base, and the model size is also one tenth; the test set of the colloquial description similarity data set LCQMC is reduced by about 1 point compared to the bert_base;
Increases 14 points for albert_base compared to unpre-trained
4, albert_xxlarge may coming recently.

If you want use a albert model with best performance among all pre-trained models, just wait a few days.
Introduction to the ALBERT Model Introduction of ALBERT
The ALBERT model is an improved version of the BERT. Unlike other recent State of the art models, this is a pre-trained small model with better results and fewer parameters.

It made three modifications to the BERT. Three main changes of ALBert from Bert:

1) Factorization of word embedding vector parameters Factorized embedding parameterization

 O(V * H) to O(V * E + E * H)
 
 For example, take ALBert_xxlarge as an example, V=30000, H=4096, E=128
   
 Then the original parameter is V * H = 30000 * 4096 = 123 million parameters, now V * E + E * H = 30000 * 128 + 128 * 4096 = 3.84 million + 520,000 = 4.36 million,
   
 The parameter embedded in the word embedding is 28 times before the change.
2) Cross-Layer Parameter Sharing

 Parameter sharing can significantly reduce parameters. Sharing can be divided into the full connection layer and the parameter sharing of the attention layer; the parameters of the attention layer have less effect on the weakening effect.
3) Paragraph continuity task Inter-sentence coherence loss.

 Use paragraph continuity tasks. For example, use two consecutive paragraphs of text from one document; in the negative case, use two consecutive paragraphs of text from one document, but the position is changed.
 
 Avoid using the original NSP task, the original task contains an overly simple task that implies a predictive theme.

  We maintain that inter-sentence modeling is an important aspect of language understanding, but we propose a loss
  Based on on coherence. That is, for ALBERT, we use a sentence-order prediction (SOP) loss, which avoids topic
  Prediction and instead focuses on modeling inter-sentence coherence. The SOP loss uses as positive examples the
  Same technique as BERT (two consecutive segments from the same document), and as negative examples the same two
  Consistent segments but with their order swapped. This forces the model to learn finer-grained distinctions about
  Discourse-level coherence properties.
Other changes, as well as Other changes:

1) Removed dropout Remove dropout to enlarge capacity of model.
    The largest model, after training 1 million steps, still did not overfit the training data. Explain that the capacity of the model can be larger, and the dropout is removed.
    (dropout can be thought of as randomly removing a portion of the network while making the network smaller)
    We also note that, even after training for 1M steps, our largest models still do not overfit to their training data.
    As a result, we decide to remove dropout to further increase our model capacity.
    For other models, we still retain the original dropout ratio in our implementation to prevent the model from over-fitting the training data.
    
2) In order to speed up the training, use LAMB as an optimizer. Use LAMB as optimizer, to train with big batch size
  A large batch_size is used to train (4096). The LAMB optimizer allows us to train, especially large batch batch_sizes, such as up to 60,000.

3) Use n-gram (uni-gram, bi-gram, tri-gram) as the masking language model Use n-gram as make language model
   That is, using n-grams with different probabilities, uni-gram has the highest probability, bi-gram is second, and tri-gram probability is the smallest.
   The current use of this project is to do a whole word mask in Chinese, and later update the effect of the n-gram mask. N-gram comes from spanBERT.
Release Plan Release Plan
1, albert_base, parameter quantity 12M, layer number 12, October 7

2, Albert_large, parameter quantity 18M, layer number 24, October 13

3, albert_xlarge, parameter quantity 59M, layer number 24, October 6

4, albert_xxlarge, parameter quantity 233M, layer number 12, October 7 (the best effect model)

Training corpus / training configuration Training Data & Configuration
30g Chinese corpus, more than 10 billion Chinese characters, including multiple encyclopedias, news, interactive communities.

The pre-training sequence length sequence_length is set to 512, the batch batch_size is 4096, and the training generates 350 million training data; each model will train 125k steps by default, and albert_xxlarge will train longer.

For comparison, roberta_zh pre-training produced 250 million training data with a sequence length of 256. Since the albert_zh pre-training generates more training data and uses a longer sequence length,

We expect that albert_zh will have better performance than roberta_zh and will be able to handle longer text better.
Training uses the TPU v3 Pod, we are using v3-256, which contains 32 v3-8. Each v3-8 machine contains 128G of video memory.

Model Performance and Comparison (English) Performance and Comparision






Chinese task set effect comparison test Performance on Chinese datasets
Natural language inference: XNLI of Chinese Version
Model development set test set
BERT 77.8 (77.4) 77.8 (77.5)
ERNIE 79.7 (79.4) 78.6 (78.2)
BERT-wwm 79.0 (78.4) 78.2 (78.0)
BERT-wwm-ext 79.4 (78.6) 78.7 (78.3)
XLNet 79.2 78.7
RoBERTa-zh-base 79.8 78.8
RoBERTa-zh-Large 80.2 (80.0) 79.9 (79.5)
ALBERT-base 77.0 77.1
ALBERT-large 78.0 77.5
ALBERT-xlarge ? ?
ALBERT-xxlarge ? ?
Note: BERT-wwm-ext comes
