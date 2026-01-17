---
title: "Overcoming Input Length Constraints of Transformers"
date: 2021-12-14
draft: false
tags: ["transformers", "summarization", "long-documents", "nlp"]
canonicalURL: "https://towardsdatascience.com/overcoming-input-length-constraints-of-transformers-b0dd5c557f7e"
summary: "Using extractive summarization to train Transformers on long documents efficiently."
---

<!--more-->

*Originally published on [Towards Data Science](https://towardsdatascience.com/overcoming-input-length-constraints-of-transformers-b0dd5c557f7e)*

## An extractive summarization approach to long-document training

True to their name, Transformers [1] have truly transformed the field of NLP over the last few years, mostly due to their parallelization abilities allowing for large pre-trained models such as BERT [2]. While BERT and its derivatives have shown state-of-the-art results in most areas of NLP, there is one major drawback to the Transformer model: it is hard to apply to very long documents. This difficulty is due to the self-attention operation, which has an exponential complexity of O(n²) with respect to the input length. With many companies integrating AI models into their workflow this can be a problem, as not all companies have the resources to effectively handle the size and complexity of the data they have access to. Some examples are law firms (legal documents) and hospitals (medical records).

Several solutions have been proposed to mitigate the problem posed by long documents. The simplest solution is truncating, where only the first N tokens of the documents are taken as input, but this throws away a lot of potentially valuable information from the original document. Another solution is chunking, where a document is split into smaller chunks and every chunk is used as a separate input for the model. While this does allow for longer inputs, every chunk must be encoded individually and must be used in a downstream model, which still makes it computationally expensive. Another downside of this approach is that long-range attention is lost as there is no attention between the chunks. A promising new solution is to potentially reduce the computational complexity by replacing global self-attention with an approximation called locality-sensitive hashing self-attention [9], used in the Reformer model [10], which effectively reduces the complexity to O(L log L).

However, I propose an alternative solution for long document inputs: summarization. I will show how the use of extractive summarization can first extract salient information from long documents, and then be leveraged to train a Transformer-based model on these summaries. In other words, I will demonstrate how applying summarization can reduce input size, leading to less computational requirements.

To evaluate this method, we compared several inputs for the task of citation prediction. Predicting the number of citations for a scientific document is a task where demonstrable improvements have been made by utilizing the full-text of a document vs. only using the abstract [3]. The task of citation prediction has two unique properties for our experiment: firstly, we know that long-document inputs are needed for better results on this task. Secondly, we have a human-made summary in the form of abstracts to which we can compare our own summaries.

![Figure 1: Proposed model architecture](/images/blog/overcoming-model-architecture.webp)
*Figure 1: The proposed model for citation prediction. A summary is extracted from an input document using LexRank, which is then used as input for a Transformer model for citation prediction.*

## A refresher on extractive summarization

To understand how our method works, it is important to understand how our summarization algorithm works. Summarization algorithms come in two flavors: extractive and abstractive.

Extractive summarization algorithms select a set of sentences from a source document which are combined into a summary, while abstractive summarization algorithms generate a summary which contains sentences that do not necessarily appear in the source document.

While much progress has been made over the past years in abstractive summarization, mostly due to the rise of pre-trained encoder-decoder based models such as the Transformer, abstractive summarization models are considerably more complex and computationally expensive; defeating the purpose of our experiment. For this reason, we used LexRank [4], a fast extractive summarization algorithm from 2004 that is still among the state-of-the-art algorithms in extractive summarization.

LexRank uses an unsupervised graph-based approach to select the most relevant sentences from a document. LexRank works as follows: every sentence in the input document is first turned into an embedding (using the mean TF-IDF of the words in the sentence) and then considered a node in a graph. The connections, or edges, are created by computing the cosine-similarity between the sentence embeddings. In practice, this information is placed in a connectivity matrix which contains all the similarities across all sentences. LexRank then applies a similarity threshold to every connection to ensure that only "strong" connections are used in the connectivity matrix. This results in a matrix of 0's and 1's. After this, every node is divided by its degree: the number of connections it has. Finally, the power iteration method is used to calculate the scores for every sentence, and the top N highest scoring sentences are returned as the summary. Our summary is thus a strict subset of our original document. For a more extensive explanation of the algorithm, I encourage you to read the paper by Erkan et al. [4].

## Experimental setup

To evaluate our method, we generated summaries from all our full-text inputs using LexRank. We compared these summaries to the abstracts and to the full-text of every document in our training data. In addition, we also generated a set of random sentences for every document, which served as our random baseline. For a fair comparison, both the generated summaries and the random sentences had the same number of sentences as the human written abstracts.

Note that this dataset does not have section information available, which is why we took random sentences rather than the first N of every section (which is a common baseline in literature). While it is possible to take the average citation count as a baseline, this would have always resulted in an R2 value of 0 since our predictions would not explain any of the variance in our data.

As our training data, we used the ACL-BiblioMetry dataset proposed in [3]. This dataset contains full-text and abstract information for 30,950 documents from the ACL anthology database. Every paper has a label for the citations over a period of the first 10 years after it was published. Since the number of citations increases following Zips-Mandelbrot's law (or the power law) [8], and we were interested in "separating the wheat from the chaff" rather than predicting extreme impact papers, we used a log-normalization function where we take the logarithm and add one to the number of citations *n*.

A plot of the non-normalized citations in the training set is shown in Figure 2. Notice that the citation counts in our data are heavily skewed and follow the power law.

![Figure 2: Histogram of citation counts](/images/blog/overcoming-citation-histogram.webp)
*Figure 2: Histogram of the citation counts in the training data.*

We thus had four inputs for a document on which we each trained a citation prediction model: full-text, abstract, random baseline, and our LexRank summary. Our hypothesis was that we could encode salient information from the full-text document through our summary, and match, or even improve, performance over abstracts.

The average number of sentences in the abstracts and summaries is 6. For comparison, the average number of sentences in a full-text document is 150, thus our summaries are, on average, a 25× reduction in input size.

We used the SchuBERT model as proposed in [3], except we replaced BERT-base for SciBERT [5] since we worked with scientific documents. In this model, full-text documents are encoded into chunks of 512 tokens using our SciBERT layer, after which a GRU layer is trained with a single dropout layer, followed by a single linear layer for predictions. For a more comprehensive explanation of the architecture, I encourage you to read [3]. To generate our LexRank summaries, we used the Sumy package (https://pypi.org/project/sumy/).

To compare our results, we report three common regression metrics: R2, Mean Squared Error (MSE) and Mean Absolute Error (MAE). For R2, a higher score is better; for MSE and MAE, a lower score is better.

In addition to evaluating how well our summaries can be used to predict the number of citations, we also calculated the (F1) Rouge-1, Rouge-2, and Rouge-L scores [6] between our summaries and abstracts. We did this to evaluate how much overlap there is between the two since we expect them to include similar information from the full-text.

## Results of the proposed experiments

![Table 1: Results of different input types](/images/blog/overcoming-results-table.webp)
*Table 1: Results of the different input types for citation prediction.*

![Table 2: Rouge scores](/images/blog/overcoming-rouge-scores.webp)
*Table 2: F1 Rouge scores between the abstract and LexRank summaries.*

The R2, MSE, and MAE scores on the test set, averaged over three runs, show that the baseline model where we select N random sentences performs noticeably worse than the other inputs. Human-written abstracts slightly outperform LexRank summaries, while both are significantly outperformed by full-text inputs. Note that when we convert these scores back to their unnormalized values, we get an MAE of 13.09 for full-text documents and 13.62 for summarized documents. These high MAE scores are heavily influenced by papers with a large number of citations, which is why we trained on the log normalized citation counts. For reference: if we remove the top-50 highest citation papers from our test set of 1,474 papers, these scores become 6.92 and 7.19, respectively.

The Rouge-1, Rouge-2, and Rouge-L between the abstracts and summaries are on par with recent research where LexRank is used as a baseline [7].

## Discussion

Our results show that summarizations obtained with LexRank provide similar results when used as input for a downstream task as human-written abstracts. The closeness of the results of LexRank summaries and abstracts indicates that there might be a ceiling in the downstream results that we can obtain through using condensed versions of full-text inputs. Even though our extracted summaries do not improve performance over abstracts, the fact that they have a similar performance indicates that extractive summaries can serve as a good input when full-text document inputs are too expensive to use. This can be especially useful in domains where long documents are often found, but a condensed version is not generally available, such as legal or medical documents. However, full-text documents are still preferable when it is possible to use them, as shown by the significant increase in performance.

The rouge scores indicate that there is a noticeable difference between the abstract and summary in terms of content, which is interesting since the results are very close. We tried using a concatenation of the abstract and summary as input to leverage these differences, but this gave only a very minor improvement.

An interesting follow-up study would be to compare different summary lengths and see whether we can match the performance of full-text inputs with a shorter summary, e.g., by generating a summary that is half the length of our input. Another interesting study would be to compare different summarization algorithms, since only LexRank was used in this experiment. Lastly, since LexRank outputs a score for every sentence, we could also use it to reduce noise by throwing away the least salient sentences.

## Conclusion

We showed that extractive summaries have similar performance to human-written summaries when used as input for a Transformer on the task of citation prediction. This is a promising solution for long-document training tasks where resources are limited. However, full-text inputs still show a large improvement over summaries.

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. *Attention is All You Need* (2017).
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018).
3. van Dongen, T., Maillette de Buy Wenniger, G., & Schomaker, L. *SchuBERT: Scholarly Document Chunks with BERT-Encoding Boost Citation Count Prediction* (2020).
4. Erkan, G., & Radev, D. R. *LexRank: Graph-based Lexical Centrality as Salience in Text Summarization* (2004).
5. Beltagy, I., Lo, K., & Cohan, A. *SciBERT: A Pretrained Language Model for Scientific Text* (2019).
6. Lin, C.-Y. *ROUGE: A Package for Automatic Evaluation of Summaries* (2004).
7. Dong, Y., Mircea, A., & Cheung, J. C. K. *Discourse-aware Unsupervised Summarization of Long Scientific Documents* (2021).
8. Silagadze, Z. *Citations and the Zipf–Mandelbrot's Law* (1999).
9. Andoni, A., Indyk, P., Laarhoven, T., Razenshteyn, I., & Schmidt, L. *Practical and Optimal LSH for Angular Distance* (2015).
10. Kitaev, N., Kaiser, Ł., & Levskaya, A. *Reformer: The Efficient Transformer* (2020).
