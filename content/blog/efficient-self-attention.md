---
title: "Demystifying Efficient Self-Attention"
date: 2022-11-07
draft: false
tags: ["transformers", "attention", "efficient-attention", "nlp"]
canonicalURL: "https://towardsdatascience.com/demystifying-efficient-self-attention-b3de61b9b0fb"
math: true
---

*Originally published on [Towards Data Science](https://towardsdatascience.com/demystifying-efficient-self-attention-b3de61b9b0fb)*

## A practical overview

## Introduction

The Transformer architecture [1] has been essential for some of the biggest breakthroughs in deep learning in recent years. Especially in the field of Natural Language Processing (NLP), pre-trained autoencoding models (like BERT [2]) and autoregressive models (like GPT-3 [3]) have continuously managed to outperform the state-of-the-art and reach human-like levels of text generation. One of the most important innovations of the Transformer is the use of attention layers as its main way of routing information.

As the name suggests, the goal of attention is to allow the model to focus on important parts of the input. This makes sense from a human perspective: when we look at an input (e.g. an image or a text), some parts are more important for our understanding than others. We can relate certain parts of the input to each other and understand long-range context. These are all essential for our understanding and attention mechanisms allow Transformer models to learn in a similar way. While this has proven to be extremely effective, there is a practical problem with attention mechanisms: they scale quadratically with respect to the input length. Fortunately, there is a lot of research dedicated to making attention more efficient.

This blog post aims to provide a comprehensive overview of the different types of efficient attention with intuitive explanations. This is not a complete overview of every paper that has been written, but instead a coverage of the underlying methods and techniques, with in-depth examples.

## A primer on attention

Before diving into the specific methods, let's first go over the basics of self-attention mechanisms and define some terms that will be reused throughout this blog post.

Self-attention is a specific type of attention. The difference between regular attention and self-attention is that instead of relating an input to an output sequence, self-attention focuses on a single sequence. It allows the model to let a sequence learn information about itself. For example, let's take the sentence "The man walked to the river bank, and he ate a sandwich". In contrast to previous embedding methods, such as TF-IDF and word2vec [4], self-attention allows the model to learn that a "river bank" is different from a "financial bank" (context-dependent). Furthermore, it allows the model to learn that "he" refers to "the man" (can learn dependencies).

Suppose we have a sequence $\mathbf{x}$ of length $n$. Every element in $\mathbf{x}$ is represented by a $d$-dimensional vector. In the case of NLP, $\mathbf{x}$ would be the word embeddings for a sentence. $\mathbf{x}$ is projected through three (trained) weight matrices $\mathbf{W}_Q$, $\mathbf{W}_K$, and $\mathbf{W}_V$, outputting three matrices: $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$, all of dimensions $n \times d$. Self-attention can then be defined as the following general formula:

$$
\text{Attention}(Q, K, V) = \text{Score}(Q, K)V
$$

The most commonly used score function is the softmax. Taking the softmax and applying a scaling factor leads to scaled-dot product attention (SDP), as proposed in [1]:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Here, the attention of an input $\mathbf{x}$ is calculated by multiplying $\mathbf{Q}$ with $\mathbf{K}^\top$ (associating every item with every other item), applying a scaling factor, taking the row-wise softmax (normalizing every row), and multiplying every value in $\mathbf{V}$ by its computed attention so that our output is again $n \times d$. $\mathbf{Q}$ and $\mathbf{K}$ are thus used to associate every element with every other element, and $\mathbf{V}$ is used to assign the output of the softmax back to every individual element.

The authors argue that for larger values of $d_k$, the dot products grow very large, which in turn pushes the softmax function into regions where the gradients are extremely small. For this reason, the dot product is scaled by dividing it by $\sqrt{d_k}$. Note that this has no effect on the complexity of the calculation, as that is dominated by the $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)$ calculation. For this reason, it will be left out of the general formulation.

As you might already see, this formula has a problem: multiplying $\mathbf{Q}$ and $\mathbf{K}$ leads to an $n \times n$ matrix. Taking the row-wise softmax of an $n \times n$ matrix has a complexity of $O(n^2)$. This is problematic for both runtime and memory usage as $n$ can be very large. For multi-page documents, it quickly becomes infeasible to compute self-attention over the full input, which means the inputs have to be truncated or chunked. Both of these methods remove one of the main benefits of self-attention: long-range context. This type of attention, where every item is multiplied with every other item, is called "global attention", which can be visualized as follows:

![Figure 1: Global attention](/images/blog/attention-global-attention.webp)
*Figure 1: Global attention. Every item in the diagonal (dark blue) looks at all other items in its row and column (highlighted in light blue).*

For simplicity, the following definitions will be used hereafter:

$$
P = QK^T
$$

$$
A = \text{softmax}(P)
$$

Here, $P$ refers to the $n \times n$ matrix that is the result of multiplying $Q$ and $K$, and $A$ (the self-attention matrix) refers to the softmax of $P$. Note that most papers use their own definition which can be slightly confusing.

For a more detailed explanation of attention, I encourage you to read the [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).

## Alternatives

The main assumption in reducing the complexity of SDP is that not all parts of the input are equally important and some tokens do not need to attend to other specific tokens.

To avoid computing global attention, there are several alternatives:

- **Sparse attention:** sparse attention methods sparsify the global attention matrix to reduce the number of tokens that have to attend to each other
- **Matrix factorization**: matrix factorization methods operate on the notion that the attention matrix is low-rank and can be decomposed and approximated with a lower-rank matrix without losing too much information.
- **Locality-sensitive hashing:** locality-sensitive hashing provides a fast way to compute nearest-neighbor search. This can be applied directly on the attention matrix to select which tokens should attend to each other.
- **Kernel attention:** kernel attention methods interpret the softmax function as a kernel and use that to more efficiently compute the self-attention matrix.

All of these options lead to a much lower computational complexity, at the cost of some performance. Note that all of these approaches attempt to reduce the complexity with respect to the sequence length $n$. For this reason, all the complexities are reduced to the part that is dependent on $n$.

## Sparse attention

Sparse attention methods reduce the complexity by only considering a subset of the computations in the $n \times n$ self-attention matrix $\mathbf{P}$. The idea is that tokens do not need to attend to every other token, but can instead focus on the more important tokens and ignore the others. The question then becomes: how do we pick which tokens to attend to?

### Local attention — O(n·W)

Local attention, also known as windowed or sliding attention, is a simple but effective method to sparsify the self-attention matrix. In local attention, tokens only attend to their local neighborhood, or window $W$. Thus, global attention is no longer computed. By only considering tokens in $W$, it reduces the complexity from $n \times n$ to $n \cdot W$.

### Random attention — O(n·R)

In random attention, tokens only attend to random other tokens. The complexity depends on the number of selected random tokens ($R$), which is a ratio of all the tokens.

![Figure 2: Local attention (left) and random attention (right)](/images/blog/attention-local-random.webp)
*Figure 2: Local attention (left) and random attention (right).*

### Sparse Transformer — O(n√n)

The sparse transformer [5] was one of the first attempts to reduce the complexity of self-attention. The authors propose two sparse attention patterns: strided attention and fixed attention, which both reduce the complexity to $O(n\sqrt{n})$. Their two attention types can be defined using specific attention patterns. Strided attention is similar to local attention with a stride, which the authors argue is important for learning from data with a periodic structure, like images or music. However, for data without a periodic structure (like text), this pattern can fail to route information to distant items. Fixed attention is a solution for this. It lets some items attend to the entire column and create a "summary" that is propagated to other items.

![Figure 3: Strided attention (left) and fixed attention (right)](/images/blog/attention-sparse-strided-fixed.webp)
*Figure 3: Strided attention (left) and fixed attention (right).*

### Longformer — O(n)

Longformer [6] uses a combination of sliding (or local), dilated sliding, and global attention. Dilated sliding attention is based on the idea of dilated CNNs. The goal of dilated sliding attention is to gradually increase the receptive field for every layer. The authors propose to use local attention in lower-level layers with a small window $W$ (which can be seen as dilated sliding window attention with a gap $d$ of 0) and increase $W$ and $d$ in higher-level layers.

Global attention is only added for specific tokens $s$. The choice of which tokens to make global is up to the user. A logical choice for classification is to make the [CLS] token global, while for QA tasks all question mark tokens can be made global. The complexity of their algorithm is $(n \cdot W + s \cdot n)$, which scales linearly w.r.t. the sequence length $n$ and is thus simplified as $O(n)$.

Note that the implementation of Longformer requires a custom CUDA kernel, since modern GPUs are optimized for dense matrix multiplication. The authors provide a custom CUDA kernel that allows for effective computation of their proposed sparse matrix multiplications on a GPU in PyTorch and Tensorflow.

![Figure 4: Local attention (left), dilated sliding attention (middle), and global attention (right)](/images/blog/attention-longformer.webp)
*Figure 4: Local attention (left), dilated sliding attention (middle), and global attention (right).*

## Matrix factorization

In matrix factorization (or decomposition) methods, the matrix $\mathbf{P}$ is assumed to be low-rank, meaning that not all items in the matrix are independent of each other. Therefore, it can be decomposed and approximated with a smaller matrix. This way, the $n \times n$ matrix can be reduced to $n \times k$ (where $k < n$), which allows us to compute $\mathbf{A}$ (the result of the softmax) much more efficiently.

### Linformer — O(n)

The authors of Linformer [7] propose the use of low-rank factorization of the attention matrix to reach a complexity of $O(n)$. The authors first empirically show that $\mathbf{A}$ can be recovered from its first few largest singular values when applying singular value decomposition (SVD), suggesting that it is low-rank. Then, they prove that $\mathbf{A}$ can be approximated as a low-rank matrix $\tilde{\mathbf{A}}$ with very low error using the Johnson-Lindenstraus lemma (JL), which states:

> A set of points in high-dimensional space can be projected into a low-dimensional space while (nearly) preserving the distances between points.

The authors note that computing the SVD for every self-attention matrix adds additional complexity. Instead, the authors add two linear projection matrices after $\mathbf{V}$ and $\mathbf{K}$, which effectively project the original ($n \times d$) matrices to lower ($k \times d$)-dimensional matrices, where $k$ is the reduced dimensionality.

![Figure 5: Standard attention (top) and Linformer attention (bottom)](/images/blog/attention-linformer.webp)
*Figure 5: Standard attention (top) and Linformer attention (bottom).*

The new attention formula they propose is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QW_i^Q(E_iKW_i^K)^T}{\sqrt{d_k}}\right) \cdot F_iVW_i^V
$$

Here, $E_i$ and $F_i$ are the two linear projection matrices. Notice that, to reduce $A$ from $n \times n$ to $\tilde{A}$ ($n \times k$), only $K$ needs to be projected to dimension $k$. Since $V$ is still $n \times d$, $V$ is also projected to dimension $k$ to ensure that the final output matrix is $n \times d$ (which is the expected dimensionality for the next layer).

What this effectively does is reduce the sequence length $n$ by a linear projection. This makes sense for NLP, as not all words in a sentence are (equally) relevant.

The last step is choosing the value of $k$. The authors show that a value of $d \log(d)$ is sufficient for $k$, which leads to a complexity of $O(nk)$. Since $d$ does not increase with respect to the input length $n$, the complexity of the self-attention mechanism becomes $O(n)$.

### Nyströmformer — O(n)

Nyströmformer [8] uses the Nyström method to approximate the self-attention matrix. The idea is to rewrite the matrix $\mathbf{P}$ as a matrix of four parts: $\mathbf{B}$ (which is $m \times m$, where $m$ is some number $< n$), $\mathbf{C}$, $\mathbf{D}$, and $\mathbf{E}$.

![Figure 6: Explanation of the Nyström method for matrix approximation](/images/blog/attention-nystrom-decomposition.webp)
*Figure 6: Explanation of the Nyström method for matrix approximation.*

According to the Nyström method, $\mathbf{P}$ can be approximated as $\tilde{\mathbf{P}}$ by replacing $\mathbf{E}$ with $\mathbf{D}\mathbf{B}^+\mathbf{C}$ (where $\mathbf{B}^+$ is the Moore-Penrose pseudoinverse of $\mathbf{B}$). The original $n \times n$ matrix is now decomposed as a multiplication of two $n \times m$ matrices and an $m \times m$ matrix. This significantly reduces the computation, since only the selected rows of $\mathbf{K}$ and columns of $\mathbf{Q}$ have to be multiplied to create this decomposition (instead of all the rows and columns).

To better understand how this works, let's approximate a single element $e_{i,j}$ in our sub-matrix $\mathbf{E}$ using the Nyström method.

![Figure 7: Example of the Nyström method for matrix approximation](/images/blog/attention-nystrom-example.webp)
*Figure 7: Example of the Nyström method for matrix approximation.*

While for this example the first row and column from $\mathbf{Q}$ and $\mathbf{K}$ were selected, it is also possible to sample multiple rows and columns, called "landmarks", which is done in the paper. These landmarks are selected using segmental means, which is similar to local average pooling (dividing the input into segments and taking the mean of each segment).

There is, however, still a problem: to compute the attention matrix $A$, $P$ needs to be computed first because the softmax operation normalizes elements of $A$ by taking the contents of the entire row in $P$. Since the goal is to avoid computing $P$, the authors propose a workaround: they do a softmax over the three submatrices in $\tilde{P}$ and multiply those.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)Z^*\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

Here, $Z^*$ is the approximation of $B^+$.

While this is technically not allowed as the softmax is a non-linear operation, the authors show that the approximation provided by this method is still sufficient.

## Locality-Sensitive Hashing

Locality-sensitive hashing (LSH) is a technique that can be used for efficient approximate nearest-neighbor search. The idea of LSH is that it is possible to select hash functions such that for any two points in high-dimensional space $\mathbf{p}$ and $\mathbf{q}$, if $\mathbf{p}$ is close to $\mathbf{q}$, then hash($\mathbf{p}$) == hash($\mathbf{q}$). Using this property, all points can be divided into hash buckets. This makes it possible to find the nearest neighbors for any point much more efficiently since only the distance to points in the same hash bucket has to be computed. In the context of self-attention, this can be used to speed up the computation of $\mathbf{P}$ by applying LSH on $\mathbf{Q}$ and $\mathbf{K}$, and only multiplying items that are close to each other after applying LSH, instead of performing the full computation $\mathbf{Q}\mathbf{K}^\top$.

### Reformer — O(n log n)

The authors of Reformer [9] were the first to propose the use of LSH for efficient self-attention. They note that, since the softmax is dominated by the largest elements, for each query $\mathbf{q}_i$ in $\mathbf{Q}$, $\mathbf{q}_i$ only needs to focus on the keys in $\mathbf{K}$ that are closest to $\mathbf{q}_i$ (or in the same hash bucket).

To better understand how this works, let's go through an example. Imagine we have a 2-d space with a number of points. In the case of self-attention, these points would be the items in $\mathbf{P}$. The colors represent points that are close together. To divide the items into hash buckets, a number of random hyperplanes are drawn through the origin. Any hyperplane has a positive side (1) and a negative side (0). Items are then placed into hash buckets based on which side they appear on w.r.t. each hyperplane. The number of hash buckets is thus defined by the number of drawn hyperplanes. After doing this, items only have to compute the distance to items within their own hash bucket (or, in the context of self-attention, attend to items within the same hash bucket).

![Figure 8: Example of LSH](/images/blog/attention-lsh-example.webp)
*Figure 8: Example of LSH. Points are divided into hash buckets based on hyperplanes.*

As can be seen in the resulting hash buckets, it is possible that items that are close together still end up in different hash buckets. To mitigate this, it's possible to perform multiple rounds of hashing and assign every value to the hash that it ends up in most often. This does, however, increase the complexity of the algorithm. The authors show that with 8 rounds of hashing the model reaches a performance that's similar to a global-attention model.

The authors use a variant called angular LSH [10], which uses the cosine distance to compute the distance between any two points. They show that, with a high probability, two points that are close together end up in the same bucket.

After dividing the points into buckets, the points are sorted by bucket. However, some buckets might be bigger than others. The largest bucket will still dominate the memory requirements, which is an issue. For this reason, the authors chunk the buckets into fixed chunks, so that the memory requirements are dependent on the chunk size. Note that it is possible that items do not end up in the same chunk as the other items within their bucket. The items can attend to all the items in the chunk that they should have ended up in, but can not be attended to themselves (which adds a small constant cost to the complexity).

![Figure 9: Explanation of LSH Attention](/images/blog/attention-lsh-buckets.webp)
*Figure 9: Explanation of LSH Attention.*

This effectively reduces the complexity to $O(n \log n)$. An important thing to note is that a large constant is introduced due to the 8 rounds of hashing that's removed from the complexity as it is not dependent on $n$, which effectively causes the Reformer to only become more efficient when the input sequence is very long (>2048).

## Kernel attention

A kernel is a function that takes as input the dot product of two vectors $\mathbf{x}$ and $\mathbf{y}$ in some lower dimensional space and returns the result of the dot product in some higher dimensional space. This can be generalized as a function $K(\mathbf{x},\mathbf{y}) = \phi(\mathbf{x})^\top\phi(\mathbf{y})$, where $K$ is the kernel function, and $\phi$ is a mapping from low to high dimensional space. Support vector machines (SVM) are a well-known example of this in the context of machine learning. For efficient self-attention specifically, kernel methods operate on the notion that the Softmax can be interpreted as a kernel and rewritten so that we can avoid explicitly computing attention matrix $\mathbf{A}$.

### Performer — O(n)

The Performer [11] is based on a mechanism called _fast attention via positive orthogonal random features_ (or FAVOR+). The idea is that we can use the kernel method to approximate the softmax function.

Usually, when applying the kernel method we want to compute the dot product in a higher dimensional space. This can be achieved using the appropriate kernel function $K$ (as we do in kernel SVM for example). However, the Performer does the reverse: we already know what our function $K$ is (the non-linear softmax) and we want to find $\phi$ so that we can compute $\phi(\mathbf{x})^\top\phi(\mathbf{y})$ (which is linear). We can visualize this method: instead of computing the $L \times L$ matrix $\mathbf{A}$ multiplied by $\mathbf{V}$ (note that this is just the formula $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$), the method uses $\phi$ to compute $\phi(\mathbf{Q}) = \mathbf{Q}'$ and $\phi(\mathbf{K}) = \mathbf{K}'$ directly which allows us to multiply $\mathbf{K}$ and $\mathbf{V}$ first, and avoids the costly computation of matrix $\mathbf{A}$.

![Figure 10: Explanation of FAVOR+ Attention](/images/blog/attention-performer.webp)
*Figure 10: Explanation of FAVOR+ Attention.*

The main research question becomes: how do we find $\phi$? The idea is based on random Fourier features [12]. The authors indicate that most kernels can be modeled using a general function. The authors prove that the Softmax kernel can be approximated by choosing specific functions.

There is still one problem, however. Unlike a softmax, sin and cos can have negative values, which causes the variance of the approximation to become high when the actual value of the softmax would be close to 0. Since many self-attention values are close to 0, this is a problem. For this reason, the authors propose the use of different functions which only output positive values (hence the positive part in FAVOR+):

$$
\phi(x) = \frac{h(x)}{\sqrt{m}}\left(f_1(\omega_1^Tx), \ldots, f_1(\omega_m^Tx), \ldots, f_l(\omega_1^Tx), \ldots, f_l(\omega_m^Tx)\right)
$$

Lastly, the authors explain that making sure the $\omega$s are orthogonal leads to even less variance (hence the orthogonal part in FAVOR+).

## Alternatives to self-attention

Clearly, there is a lot of research dedicated to making scaled-dot product attention more efficient. There is, however, another alternative: not using self-attention at all, but instead using a simpler approach to share information between our tokens. This idea has been proposed in multiple papers recently ([13], [14], [15]). We will discuss one, as the general idea is very similar for all of these papers.

### FNet — O(n)

FNet [15] is an alternative Transformer architecture that completely replaces self-attention blocks with the discrete Fourier transform (DFT). Consequently, there are no learnable parameters anymore except for the feedforward layers. The DFT decomposes a signal into its constituent frequencies. When $N$ is infinite, we can exactly create the original signal. In the context of NLP, our signal is a sequence of tokens. Effectively, every component $n$ contains some information about every token in the input sequence.

What's interesting about their method is not so much the fact that they use DFT, but rather that they apply a linear transformation to mix their tokens. They also tried a linear encoder, which is very similar to how synthesizer models work, and even a completely random encoder. While the linear encoder has a slightly higher performance, it has learnable parameters, making it slower than FNet. BERT-Base still has a substantially higher average score on GLUE, but they report a training time speedup of ~7x. Since there are many possible linear transformations, there is an interesting open research question on what is the most suitable one for Transformers.

## Benchmarks

While all the papers discussed in this post report their theoretical complexities with respect to the input sequence length $n$, in practice, some might still be impractical due to large constants (like Reformer) or due to inefficient implementations. To this end, [Xformers](https://github.com/facebookresearch/xformers) was used to compute the memory use and runtime for different sequence lengths for a number of methods. Note that not every method discussed is implemented in Xformers, and BlockSparse, unfortunately, did not work on my GPU (an RTX 3090).

**Long sequence lengths (512, 1024, 2048):**

![Figure 11: Memory use and runtime usage for long sequence lengths](/images/blog/attention-linear-attention.webp)
*Figure 11: Memory use and runtime usage for long sequence lengths for various efficient attention methods.*

**Short sequence lengths (128, 256):**

![Figure 12: Memory use and runtime usage for short sequence lengths](/images/blog/attention-complexity-comparison.webp)
*Figure 12: Memory use and runtime usage for short sequence lengths for various efficient attention methods.*

Clearly, all methods are significantly more efficient than SDP for longer sequences. While all of the attention mechanisms compared here (except for SDP) scale linearly w.r.t the sequence length, it is interesting to see that there are still noticeable differences between the mechanisms due to constants and other scaling factors. Notably, Linformer does not scale as well as the other methods. Another interesting result is the memory usage and runtime of Nyströmformer. While it scales very well, as can be seen in the graphs for sequence lengths (512, 1024, 2048), it is actually the most inefficient method for short sequences (128 and 256). This is likely due to the number of selected landmarks, which was kept at a value of 64 as suggested in [8].

It is interesting to see that SDP performs very similarly to the other methods for sequence lengths up to 512. The only method that is notably more efficient than any other method is FNet (Fourier mix attention). It is almost completely independent of the sequence length while having no significant constants to consider.

## Conclusion

Efficient self-attention is still an active area of research, given the ever-increasing relevance of Transformer-based models. While they may seem daunting, most techniques can actually be linked back to more general mathematical concepts that you might already be familiar with. Hopefully, this blog post serves as both an introduction as well as an explanation of most of the relevant techniques and helps you gain a deeper understanding of the field.

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. *Attention is All You Need* (2017).
2. Devlin, J., Chang, M., Lee, K., & Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018).
3. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., et al. *Language Models are Few-Shot Learners* (2020).
4. Mikolov, T., Chen, K., Corrado, G., & Dean, J. *Efficient Estimation of Word Representations in Vector Space* (2013).
5. Child, R., Gray, S., Radford, A., & Sutskever, I. *Generating Long Sequences with Sparse Transformers* (2019).
6. Beltagy, I., Peters, M. E., & Cohan, A. *Longformer: The Long-Document Transformer* (2020).
7. Wang, S., Li, B. Z., Khabsa, M., Fang, H., & Ma, H. *Linformer: Self-Attention with Linear Complexity* (2020).
8. Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G., Li, Y., & Singh, V. *Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention* (2021).
9. Kitaev, N., Kaiser, Ł., & Levskaya, A. *Reformer: The Efficient Transformer* (2020).
10. Andoni, A., Indyk, P., Laarhoven, T., Razenshteyn, I., & Schmidt, L. *Practical and Optimal LSH for Angular Distance* (2015).
11. Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., et al. *Rethinking Attention with Performers* (2020).
12. Rahimi, A., & Recht, B. *Random Features for Large-Scale Kernel Machines* (2007).
13. Tay, Y., Bahri, D., Metzler, D., Juan, D., Zhao, Z., & Zheng, C. *Synthesizer: Rethinking Self-Attention in Transformer Models* (2020).
14. Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., et al. *MLP-Mixer: An all-MLP Architecture for Vision* (2021).
15. Ainslie, J., Eckstein, I., & Ontañón, S. *FNet: Mixing Tokens with Fourier Transforms* (2021).
