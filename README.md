# supervised Latent Dirichlet Allocation (sLDA)

Implementation of supervised Latent Dirichlet Allocation (sLDA) using variational Bayesian inference is provided in this class. Classification task together with topic distribution inference can be conducted simultaneously.
For furthur information on sLDA, please refer to the original paper [supervised topic model](https://www.cs.princeton.edu/~blei/papers/BleiMcAuliffe2007.pdf) .

## Sample Code

Sample code is available for applying sLDA and extracting topics from synthetically generated data.
Figures below show the obtained results after observing 1000 synthetically generated data.

![sample](https://raw.github.com/kyoheiotsuka/supervisedLDA/master/result/0.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedLDA/master/result/1.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedLDA/master/result/2.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedLDA/master/result/3.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedLDA/master/result/4.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedLDA/master/result/5.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedLDA/master/result/6.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedLDA/master/result/7.bmp)

![sample](https://raw.github.com/kyoheiotsuka/supervisedLDA/master/result/topicWord.jpg)

LDA class provided supports not only extracting topics from training data but also inferring document-topic distribution and label of unseen data. 

## Licence
[MIT](https://github.com/kyoheiotsuka/logisticRegression/blob/master/LICENSE)
