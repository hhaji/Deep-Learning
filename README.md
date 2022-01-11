# Deep Learning   
* [**Deep Learning Using PyTorch (2020)**](https://github.com/hhaji/Deep-Learning/blob/master/README.md)
* [**Deep Learning Using TensorFlow (2019)**](https://github.com/hhaji/Deep-Learning/blob/master/Deep-Learning-TensorFlow.md)

## Deep Learning Using PyTorch

<table>
  <tr>
    <th colspan="2"><span style="font-weight:bold">Lecturer: </span><a href="http://facultymembers.sbu.ac.ir/hhaji/">Hossein Hajiabolhassan</a><br><br><a href="http://ds.sbu.ac.ir/">Data Science Center</a> <br><br><a href="http://en.sbu.ac.ir/">Shahid Beheshti University</a></th>
    <th colspan="3"><img src=".\Images\HH.jpg" alt="" border='3' height='140' width='140' /></th>
  </tr>
  <tr>
    <td colspan="5"><span style="font-weight:bold">Teaching Assistants:</span></td>
  </tr>
  <tr>
    <td><a href="https://github.com/behnazhoseyni">Behnaz H.M. Hoseyni</a></td>
    <td><a href="https://github.com/YavarYeganeh">Yavar T. Yeganeh</a></td>
    <td><a href="https://github.com/MSTF4">Mostafa Khodayari</a></td>
    <td><a href="https://github.com/E008001">Esmail Mafakheri</a></td>
  </tr>
  <tr>
    <td><img src=".\Images\BH.jpeg" alt="" border='3' height='140' width='120' /></td>
    <td><img src=".\Images\Y.jpg" alt="" border='3' height='140' width='120' /></td>
    <td><img src=".\Images\K.jpg" alt="" border='3' height='140' width='120' /></td>
    <td><img src=".\Images\Mafakheri.jpg" alt="" border='3' height='140' width='120' /></td>   
  </tr>
</table>

---

### **Index:**
- [Course Overview](#Course-Overview)
- [Main TextBooks](#Main-TextBooks)
- [Slides and Papers](#Slides-and-Papers)
  1. Lecture 1: [Introduction](#Introduction) 
  2. Lecture 2: [Toolkit Lab 1: Google Colab and Anaconda](#Part-1) 
  3. Lecture 3: [Toolkit Lab 2:  Getting Started with PyTorch](#Part-2)
  4. Lecture 4: [Deep Feedforward Networks](#DFN) 
  5. Lecture 5: [Toolkit Lab 3: Preprocessing Datasets by PyTorch](#Part-3)  
  6. Lecture 6: [Regularization for Deep Learning](#RFDL) 
  7. Lecture 7: [Toolkit Lab 4: Using a Neural Network to Fit the Data with PyTorch](#Part-4)   
  8. Lecture 8: [Optimization for Training Deep Models](#OFTDM) 
  9. Lecture 9: [Convolutional Networks](#CNN) 
  10. Lecture 10: [Toolkit Lab 5: Using Convolutions to Generalize](#Part-5) 
  11. Lecture 11: [Sequence Modeling: Recurrent and Recursive Networks](#SMRARN)
  12. Lecture 12: [Toolkit Lab 6: Transfer Learning and Other Tricks](#Part-6) 
  13. Lecture 13: [Practical Methodology](#Practical-Methodology)  
  14. Lecture 14: [Toolkit Lab 7: Optimization Software (Ray Tune or Optuna)](#Part-7) 
  15. Lecture 15: [Applications](#Applications) 
  16. Lecture 16: [Autoencoders](#Autoencoders)
  17. Lecture 17: [Generative Adversarial Networks](#GAN)  
  18. Lecture 18: [Graph Neural Networks](#GNN)
  19. Lecture 19: [Atention Mechanisms and Transformers](#TRA)
- [Additional Resources](#ANAS)
- [Class Time and Location](#Class-Time-and-Location)
  - [Recitation and Assignments](#MTA)  
- [Projects](#Projects)
  - [Google Colab](#Google-Colab)
  - [Fascinating Guides For Machine Learning](#Fascinating-Guides-For-Machine-Learning)
  - [Latex](#Latex)
- [Grading](#Grading)
  - [Three Exams](#Three-Exams)
- [Prerequisites](#Prerequisites)
  - [Linear Algebra](#Linear-Algebra)
  - [Probability and Statistics](#Probability-and-Statistics)
- [Topics](#Topics)
- [Account](#Account)
- [Academic Honor Code](#Academic-Honor-Code)
- [Questions](#Questions)
- Miscellaneous: 
     * [Data Handling](https://github.com/hhaji/Deep-Learning/tree/master/Data-Handling)  

---

## <a name="Course-Overview"></a>Course Overview:
```javascript
In this course, you will learn the foundations of Deep Learning, understand how to build 
neural networks, and learn how to lead successful machine learning projects. You will learn 
about Convolutional networks, RNNs, LSTM, Adam, Dropout, BatchNorm, and more.
```

## <a name="Main-TextBooks"></a>Main TextBooks:
![Book 1](/Images/DL.jpg)  ![Book 2](/Images/Deep-Learning-PyTorch.jpg) ![Book 3](/Images/PPDL.jpg) ![Book 4](/Images/GDL.jpg) ![Book 5](/Images/Dive-Into-DL.png) 

```
Main TextBooks:
```

* [Deep Learning](http://www.deeplearningbook.org) (available in online) by Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville   
* [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch) by Eli Stevens and Luca Antiga  
    - GitHub: [Codes](https://github.com/deep-learning-with-pytorch/dlwpt-code)    
* [Programming PyTorch for Deep Learning](https://www.oreilly.com/library/view/programming-pytorch-for/9781492045342/) by Ian Pointer    
    - GitHub: [Codes](https://github.com/falloutdurham/beginners-pytorch-deep-learning)      


```
Additional TextBooks:
```

* [Generative Deep Learning](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/) by David Foster  
    - GitHub: [Codes](https://github.com/davidADSP/GDL_code)
* [Dive into Deep Learning](https://d2l.ai) by  Mag Gardner, Max Drummy, Joanne Quinn, Joanne McEachen, and Michael Fullan   
    - GitHub: [Codes](https://github.com/dsgiitr/d2l-pytorch)    

## <a name="Slides-and-Papers"></a>Slides and Papers:  
  Recommended Slides & Papers:
  
1. ### <a name="Introduction"></a>Introduction (1 Session)  

```
Required Reading:
```

* [Chapter 1](http://www.deeplearningbook.org/contents/intro.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
* Slide: [Introduction](https://www.deeplearningbook.org/slides/01_intro.pdf)  by Ian Goodfellow
 
```
Suggested Reading:
```
 
* Demo: [3D Fully-Connected Network Visualization](http://scs.ryerson.ca/~aharley/vis/fc/) by Adam W. Harley  

```
Additional Resources:
```

* [Video](https://www.youtube.com/embed//vi7lACKOUao) of lecture by Ian Goodfellow and discussion of Chapter 1 at a reading group in San Francisco organized by Alena Kruchkova <br>
* Paper: [On the Origin of Deep Learning](https://arxiv.org/pdf/1702.07800.pdf) by Haohan Wang and Bhiksha Raj <br>

```
Applied Mathematics and Machine Learning Basics:
```

* Slide: [Mathematics for Machine Learning](http://www.deeplearningindaba.com/uploads/1/0/2/6/102657286/2018_maths4ml_vfinal.pdf) by Avishkar Bhoopchand, Cynthia Mulenga, Daniela Massiceti, Kathleen Siminyu, and Kendi Muchungi 
* Blog: [A Gentle Introduction to Maximum Likelihood Estimation and Maximum A Posteriori Estimation (Getting Intuition of MLE and MAP with a Football Example)](https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-and-maximum-a-posteriori-estimation-d7c318f9d22d) by Shota Horii  
    
2. ### <a name="Part-1"></a>Toolkit Lab 1: Google Colab and Anaconda (1 Session)  

```
Required Reading:
```

* Blog: [Google Colab Free GPU Tutorial](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d) by Fuat <br>
* Blog: [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments) <br>
* Blog: [Kernels for Different Environments](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) <br>
 
```
Suggested Reading:
```
 
* Blog: [Using Pip in a Conda Environment](https://www.anaconda.com/using-pip-in-a-conda-environment/) by Jonathan Helmus <br> 
* Blog: [How to Import Dataset to Google Colab Notebook?](https://mc.ai/how-to-import-dataset-to-google-colab-notebook/) 
* Blog: [How to Upload Large Files to Google Colab and Remote Jupyter Notebooks ](https://www.freecodecamp.org/news/how-to-transfer-large-files-to-google-colab-and-remote-jupyter-notebooks-26ca252892fa/)(For Linux Operating System) by Bharath Raj  <br>


```
Additional Resources:
```
* PDF: [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/_downloads/1f5ecf5a87b1c1a8aaf5a7ab8a7a0ff7/conda-cheatsheet.pdf) 
* Blog: [Conda Commands (Create Virtual Environments for Python with Conda)](http://deeplearning.lipingyang.org/2018/12/25/conda-commands-create-virtual-environments-for-python-with-conda/) by LipingY <br>  
* Blog: [Colab Tricks](https://rohitmidha23.github.io/Colab-Tricks/) by  Rohit Midha    
  
  
3. ### <a name="Part-2"></a>Toolkit Lab 2:  Getting Started with PyTorch (2 Sessions)         
```
Required Reading:
```

* NoteBook: [Chapter 3: It Starts with a Tensor](https://github.com/deep-learning-with-pytorch/dlwpt-code/tree/master/p1ch3) from [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch) by Eli Stevens and Luca Antiga         

```
Suggested Reading:
```

* Blog: [Why PyTorch is the Deep Learning Framework of the Future](https://blog.paperspace.com/why-use-pytorch-deep-learning-framework/) by Dhiraj Kumar   
* Blog: [Torch Tensors & Types:](https://pytorch.org/docs/stable/tensors.html) A torch.Tensor is a multi-dimensional 
matrix containing elements of a single data type. Torch defines nine CPU tensor types and nine GPU tensor types. 
    
```
Additional Resources:
``` 

* Blog: [Learning PyTorch with Exampls](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) by Justin Johnson. 
This tutorial introduces the fundamental concepts of PyTorch through self-contained examples.   
 
```
Building Dynamic Models Using the Subclassing API:
``` 
    
* Object-Oriented Programming:
    
   * Blog: [Object-Oriented Programming (OOP) in Python 3](https://realpython.com/python3-object-oriented-programming/) by the Real Python Team   
   * Blog: [How to Explain Object-Oriented Programming Concepts to a 6-Year-Old](https://www.freecodecamp.org/news/object-oriented-programming-concepts-21bb035f7260/)  
   * Blog: [Understanding Object-Oriented Programming Through Machine Learning](https://dziganto.github.io/classes/data%20science/linear%20regression/machine%20learning/object-oriented%20programming/python/Understanding-Object-Oriented-Programming-Through-Machine-Learning/) by David Ziganto  
   * Blog: [Object-Oriented Programming for Data Scientists: Build your ML Estimator](https://towardsdatascience.com/object-oriented-programming-for-data-scientists-build-your-ml-estimator-7da416751f64) by Tirthajyoti Sarkar  
   * Blog: [Python Callable Class Method](https://medium.com/@nunenuh/python-callable-class-1df8e122b30c) by Lalu Erfandi Maula Yusnu  
    

4. ### <a name="DFN"></a>Deep Feedforward Networks (6 Sessions)  

```
Required Reading:
```

* [Chapter 6](https://www.deeplearningbook.org/contents/mlp.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook.   
* Slides: [Deep Feed-forward Networks](https://cedar.buffalo.edu/~srihari/CSE676/) by Sargur Srihari  
* Chapter 20 of [Understanding Machine Learning: From Theory to Algorithms](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning) <br>
* Slide: [Neural Networks](https://www.cs.huji.ac.il/~shais/Lectures2014/lecture10.pdf) by Shai Shalev-Shwartz <br>
* Slide: [Backpropagation and Neural Networks](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture4.pdf) by Fei-Fei Li, Justin Johnson, and  Serena Yeung  
* Blog: [7 Types of Neural Network Activation Functions: How to Choose?](https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/) <br>
* Blog: [Back-Propagation, an Introduction](https://www.offconvex.org/2016/12/20/backprop/) by Sanjeev Arora and Tengyu Ma <br>

```
Interesting Questions:
```

* [Why are non Zero-Centered Activation Functions a Problem in Backpropagation?](https://stats.stackexchange.com/questions/237169/why-are-non-zero-centered-activation-functions-a-problem-in-backpropagation)   
  
```
Suggested Reading:
```

  * Blog: [Epoch vs Batch Size vs Iterations](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9) by Sagar Sharma  
  * Blog: [The Gradient](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/the-gradient) by Khanacademy <br>
  * Blog: [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/) by Christopher Olah 
  * PDF: [SVM (Section 5: Lagrange Duality)](http://cs229.stanford.edu/notes/cs229-notes3.pdf) by Andrew Ng   
  * Blog: [Killer Combo: Softmax and Cross Entropy](https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba) by Paolo Perrotta  
  
```
Additional Resources:
```

  * Blog: [Activation Functions](https://sefiks.com/tag/activation-function/) by Sefik Ilkin Serengil   
  * Paper: [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681v2) by Diganta Misra  
  * Blog: [Activation Functions](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#id5)  
  * Blog: [Analytical vs Numerical Solutions in Machine Learning](https://machinelearningmastery.com/analytical-vs-numerical-solutions-in-machine-learning/) by Jason Brownlee  
  * Blog: [Validating Analytic Gradient for a Neural Network](https://medium.com/@shivajbd/how-to-validate-your-gradient-expression-for-a-neural-network-8284ede6272) by  Shiva Verma  
  * Blog: [Stochastic vs Batch Gradient Descent](https://medium.com/@divakar_239/stochastic-vs-batch-gradient-descent-8820568eada1) by Divakar Kapil  
  * [Video](https://drive.google.com/file/d/0B64011x02sIkRExCY0FDVXFCOHM/view?usp=sharing): (.flv) of a presentation by Ian  Goodfellow and a group discussion at a reading group at Google organized by Chintan Kaur. <br>
  * **Extra Slide:**
    - Slide: [Deep Feedforward Networks](https://www.deeplearningbook.org/slides/06_mlp.pdf)  by Ian Goodfellow  
    - Slide: [Feedforward Neural Networks (Lecture 2)](http://wavelab.uwaterloo.ca/wp-content/uploads/2017/04/Lecture_2.pdf) by Ali Harakeh   
    - Slides: Deep Feedforward Networks [1](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L8-deep_feedforward_networks.pdf)  and [2](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L9-deep_feedforward_networks-2.pdf) by U Kang   
    
5. ### <a name="Part-3"></a>Toolkit Lab 3: Preprocessing Datasets by PyTorch (1 Session)

```
Required Reading:
```

  * NoteBook: [Chapter 4: Real-World Data Representation Using Tensors](https://github.com/deep-learning-with-pytorch/dlwpt-code/tree/master/p1ch4) from [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch) by Eli Stevens and Luca Antiga       
  * Blog: [How to Build a Streaming DataLoader with PyTorch](https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd) by David MacLeod   
  * Blog: [Building Efficient Custom Datasets in PyTorch](https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f) by Syafiq Kamarul Azman    
  * Blog: [A Beginner’s Tutorial on Building an AI Image Classifier using PyTorch](https://towardsdatascience.com/a-beginners-tutorial-on-building-an-ai-image-classifier-using-pytorch-6f85cb69cba7) by Alexander Wu    
  
```
Suggested Reading:
```
   
  * Blog: [A Quick Guide To Python Generators and Yield Statements](https://medium.com/@jasonrigden/a-quick-guide-to-python-generators-and-yield-statements-89a4162c0ef8) by Jason Rigden 
  * NoteBook: [Iterable, Generator, and Iterator](https://github.com/hhaji/Deep-Learning/blob/master/NoteBooks/Generator.ipynb)  
  * Blog: [Vectorization in Python](https://www.geeksforgeeks.org/vectorization-in-python/) 
  * Blog: [numpy.vectorize](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.vectorize.html)
 
```
Additional Resources:
```

  * Blog: [Iterables vs. Iterators vs. Generators](https://nvie.com/posts/iterators-vs-generators/) by Vincent Driessen   
  * Blog: [Writing Custum Datasets, Dataloaders and Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) by Sasank Chilamkurthy   
  * Blog: [TORCHVISION.DATASETS](https://pytorch.org/docs/stable/torchvision/datasets.html)   

6. ### <a name="RFDL"></a>Regularization for Deep Learning (5 Sessions)     

```
Required Reading:
```

  * [Chapter 7](http://www.deeplearningbook.org/contents/regularization.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
  Regularization  
  * Slides: [Regularization](https://cedar.buffalo.edu/~srihari/CSE676/) by Sargur Srihari  
  * Slide: [Bagging and Random Forests](https://davidrosenberg.github.io/mlcourse/Archive/2017/Lectures/9a.bagging-random-forests.pdf) by David Rosenberg <br>
  * Slide: [Deep Learning Tutorial](http://speech.ee.ntu.edu.tw/~tlkagk/slide/Deep%20Learning%20Tutorial%20Complete%20(v3)) (Read the Part of Dropout) by Hung-yi Lee   
 
```
Suggested Reading:
```

 * Blog: [Train Neural Networks With Noise to Reduce Overfitting](https://machinelearningmastery.com/train-neural-networks-with-noise-to-reduce-overfitting/) by Jason Brownlee   
 * Blog: [Generative VS Discriminative Models](https://medium.com/@mlengineer/generative-and-discriminative-models-af5637a66a3) by Prathap Manohar Joshi   
 * Paper: [Ensemble Methods in Machine Learnin](http://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf) by Thomas G. Dietterich   
 * Blog: [Conventional guide to Supervised learning with scikit-learn — Orthogonal Matching Pursuit (OMP)- Generalized Linear Models (10)](https://medium.com/@venali/conventional-guide-to-supervised-learning-with-scikit-learn-orthogonal-matching-pursuit-omp-d3de1ffef841) by Venali Sonone  
 * Paper: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf) by Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov   
 * Blog: [A Gentle Introduction to Weight Constraints in Deep Learning](https://machinelearningmastery.com/introduction-to-weight-constraints-to-reduce-generalization-error-in-deep-learning/) by Jason Brownlee    

```
Additional Reading:
```
    
  * Blog: [Analysis of Dropout](https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/) by Paolo Galeone  
  * **Extra Slides:** 
    - Slide: [Regularization For Deep Models (Lecture 3)](http://wavelab.uwaterloo.ca/wp-content/uploads/2017/04/Lecture_3.pdf) by Ali Harakeh  
    - Slide: [Regularization for Deep Learning](https://www.deeplearningbook.org/slides/07_regularization.pdf)  by Ian Goodfellow
    - Slides: Regularization for Deep Learning [1](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L13-regularization.pdf)  and [2](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L14-regularization-2.pdf) by U Kang 
    - Slide: [Training Deep Neural Networks](https://web.cs.hacettepe.edu.tr/~aykut/classes/spring2018/cmp784/slides/lec4-training-deep-nets.pdf) by Aykut Erdem 
    
    
7. ### <a name="Part-4"></a>Toolkit Lab 4: Using a Neural Network to Fit the Data with PyTorch (2 Sessions)   
```
Required Reading:
```
   * NoteBook: [Chapter 5: The Mechanics of Learning](https://github.com/deep-learning-with-pytorch/dlwpt-code/tree/master/p1ch5) from [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch) by Eli Stevens and Luca Antiga         
   * NoteBook: [Chapter 6: Using a Neural Network to Fit the Data](https://github.com/deep-learning-with-pytorch/dlwpt-code/tree/master/p1ch6) from [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch) by Eli Stevens and Luca Antiga         
   * NoteBook: [Chapter 2:  Image Classification with PyTorch](https://github.com/falloutdurham/beginners-pytorch-deep-learning/blob/master/chapter2) from [Programming PyTorch for Deep Learning](https://www.oreilly.com/library/view/programming-pytorch-for/9781492045342/)  by Ian Pointer   

```
Suggested Reading:
```

  * Blog: [Properly Setting the Random Seed in ML Experiments. Not as Simple as You Might Imagine](https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752) by [Open Data Science](https://opendatascience.com)    
  * Blog & NoteBook: [How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/) by Jason Brownlee    * Blog: [Why is my Validation Loss Lower than my Training Loss?](https://www.pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/) by Adrian Rosebrock   
  * Blog: [Saving/Loading Your Model in PyTorch](https://medium.com/udacity-pytorch-challengers/saving-loading-your-model-in-pytorch-741b80daf3c) by David Ashraf  
  * Blog: [Saving and Loading Your Model to Resume Training in PyTorch](https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61) by Rachit Jain   
  * Blog: [Deep Learning with PyTorch: A 60 Minute Blitz — PyTorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) by Soumith Chintala  


```
Additional Resources:
``` 
  * PDF: [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515.pdf) by Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter  
  * Deep Learning via Pytorch by Ayoosh Kathuria  
    - [PyTorch 101, Part 1: Understanding Graphs, Automatic Differentiation and Autograd](https://blog.paperspace.com/pytorch-101-understanding-graphs-and-automatic-differentiation/)  
    - [PyTorch 101, Part 2: Building Your First Neural Network](https://blog.paperspace.com/pytorch-101-building-neural-networks/)  
    - [PyTorch 101, Part 3: Going Deep with PyTorch](https://blog.paperspace.com/pytorch-101-advanced/)   
 
8. ### <a name="OFTDM"></a>Optimization for Training Deep Models (5 Sessions)  

```
Required Reading:
```  

   * [Chapter 8](http://www.deeplearningbook.org/contents/optimization.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook.    
   * Slides: [Optimization for Training Deep Models](https://cedar.buffalo.edu/~srihari/CSE676/) by Sargur Srihari  
   * Blog: [Batch Normalization in Deep Networks](https://www.learnopencv.com/batch-normalization-in-deep-networks/) by Sunita Nayak  
   

```
Suggested Reading:
```

   * Lecture Note: [Matrix Norms and Condition Numbers](http://faculty.nps.edu/rgera/MA3042/2009/ch7.4.pdf) by Ralucca Gera  
   * Blog: [Initializing Neural Networks](https://www.deeplearning.ai/ai-notes/initialization/) by Katanforoosh & Kunin, [deeplearning.ai](https://www.deeplearning.ai), 2019   
   * Blog: [How to Initialize Deep Neural Networks? Xavier and Kaiming Initialization](https://pouannes.github.io/blog/initialization/) by Pierre Ouannes  
   * Blog: [What Is Covariate Shift?](https://medium.com/@izadi/what-is-covariate-shift-d7a7af541e6) by Saeed Izadi  
   * Book: [Dataset Shift in Machine Learning](https://www.amazon.com/Dataset-Machine-Learning-Information-Processing/dp/0262170051) by by Joaquin Quinonero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D. Lawrence  
   * Blog: [Stay Hungry, Stay Foolish:](https://www.adityaagrawal.net/blog/) This interesting blog contains the computation of back propagation of different layers of deep learning prepared by Aditya Agrawal    
   * Paper: [FitNets: Hints for Thin Deep Nets](https://arxiv.org/pdf/1412.6550.pdf) by Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio   
  * Paper: [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/pdf/1603.06560.pdf) by Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar  

   
```
Additional Reading:
```
   
   * Blog: [Why Momentum Really Works](https://distill.pub/2017/momentum/) by Gabriel Goh  
   * Blog: [Understanding the Backward Pass Through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html) by Frederik Kratzert   
   * [Video](https://www.youtube.com/watch?v=Xogn6veSyxA) of lecture / discussion: This video covers a presentation by Ian Goodfellow and group discussion on the end of Chapter 8 and entirety of Chapter 9 at a reading group in San Francisco organized by Taro-Shigenori Chiba. <br>         
   * Blog: [Preconditioning the Network](https://cnl.salk.edu/~schraudo/teach/NNcourse/precond.html) by Nic Schraudolph and Fred Cummins  
   * Paper: [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun  
   * Blog: [Neural Network Optimization](https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0) by Matthew Stewart  
   * Paper: [Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift](https://arxiv.org/pdf/1801.05134.pdf) by Xiang Li, Shuo Chen, Xiaolin Hu, and Jian Yang   
   * Blog: [Restricted Boltzmann Machine, a Complete Analysis. Part 3: Contrastive Divergence Algorithm](https://medium.com/datatype/restricted-boltzmann-machine-a-complete-analysis-part-3-contrastive-divergence-algorithm-3d06bbebb10c) by Nguyễn Văn Lĩnh  
   * Blog: [A (Slightly) Better Budget Allocation for Hyperband](https://medium.com/data-from-the-trenches/a-slightly-better-budget-allocation-for-hyperband-bbd45af14481) by Alexandre Abraham   
   * Blog: [Massively Parallel Hyperparameter Optimization](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/) by Liam Li  

   * **Extra Slides:**  
       - Slide: [Optimization for Training Deep Models (Lecture 4)](http://wavelab.uwaterloo.ca/wp-content/uploads/2017/04/Lecture-4-1.pdf) by Ali Harakeh  
       - Slide: [Optimization for Training Deep Models - Algorithms (Lecture 4)](http://wavelab.uwaterloo.ca/wp-content/uploads/2017/04/Lecture_4_2-1.pdf) by Ali Harakeh  
       - Slide: [Conjugate Gradient Descent](http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/conjugate_direction_methods.pdf) by Aarti Singh  
       - Slide: [Training Deep Neural Networks](https://web.cs.hacettepe.edu.tr/~aykut/classes/spring2018/cmp784/slides/lec4-training-deep-nets.pdf) by Aykut Erdem   
       - Slides: Optimization for Training Deep Models [1](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L15-opt.pdf)  and [2](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L16-opt-2.pdf) by U Kang    

9. ### <a name="CNN"></a>Convolutional Networks  (3 Sessions)

```
Required Reading:
```

   * [Chapter 9](http://www.deeplearningbook.org/contents/convnets.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook.    
   * Slides: [ Convolutional Networks](https://cedar.buffalo.edu/~srihari/CSE676/) by Sargur Srihari  
  
```
Suggested Reading:
```

   * Blog: [Color Vision](https://phet.colorado.edu/sims/html/color-vision/latest/color-vision_en.html)
   * Blog: [Convolutional Neural Networks CheatSheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks) by Afshine Amidi and Shervine Amidi  
   * Blog: [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/) by Christopher Olah    
   * Blog: [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 Way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) by Sumit Saha  
   * Blog: [A Gentle Introduction to 1×1 Convolutions to Manage Model Complexity](https://machinelearningmastery.com/introduction-to-1x1-convolutions-to-reduce-the-complexity-of-convolutional-neural-networks/) by Jason Brownlee   
   * Blog: [Global Average Pooling Layers for Object Localization](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/) by Alexis Cook  
   * Blog: [Review: Spatial Pyramid Pooling](https://medium.com/analytics-vidhya/review-spatial-pyramid-pooling-1406-4729-bfc142988dd2) by Sanchit Tanwar  
   * Blog: [Maxpooling vs Minpooling vs Average Pooling](https://medium.com/@bdhuma/which-pooling-method-is-better-maxpooling-vs-minpooling-vs-average-pooling-95fb03f45a9) by Madhushree Basavarajaiah    
   * Blog: [A Basic Introduction to Separable Convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) by Chi-Feng Wang   
   * Blog: [Depth wise Separable Convolutional Neural Networks](https://www.geeksforgeeks.org/depth-wise-separable-convolutional-neural-networks/) by Mayank Chaurasia  
   * Blog: [Type of convolutions: Deformable and Transformable Convolution](https://towardsdatascience.com/type-of-convolutions-deformable-and-transformable-convolution-1f660571eb91) by Ali Raza  
   * Blog: [Review: DilatedNet — Dilated Convolution (Semantic Segmentation)](https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5) by Sik-Ho Tsang  
   * Blog: [Region of Interest Pooling Explained](https://deepsense.ai/region-of-interest-pooling-explained/) by Tomasz Grel     
   * Blog: [Understanding the Receptive Field of Deep Convolutional Networks](https://theaisummer.com/receptive-field/) by Nikolas Adaloglou   
  

```
Additional Reading:  
```  
   
   * Blog: [Image Convolution Examples](http://aishack.in/tutorials/image-convolution-examples/) by Utkarsh Sinha  
   * Blog: [Convolutions and Backpropagations](https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c) by Pavithra Solai  
   * **Extra Slides:**  
       - Slide: [Convolutional Neural Networks (Lecture 6)](http://wavelab.uwaterloo.ca/wp-content/uploads/2017/04/Lecture_6.pdf) by Ali Harakeh   
       - Slide: [Convolutional Networks](http://www.deeplearningbook.org/slides/09_conv.pdf)  by Ian Goodfellow    
   
```
Fourier Transformation:
```   

   * Blog: [Fourier Transformation and Its Mathematics](https://towardsdatascience.com/fourier-transformation-and-its-mathematics-fff54a6f6659) by Akash Dubey    
   * Blog: [Fourier Transformation for a Data Scientist](https://towardsdatascience.com/fourier-transformation-for-a-data-scientist-1f3731115097) by Nagesh Singh Chauhan        
   * Blog: [Purrier Series (Meow) and Making Images Speak](http://bilimneguzellan.net/en/purrier-series-meow-and-making-images-speak/) by Bilim Ne Güzel Lan   
   * Blog: [Follow up to Fourier Series](http://bilimneguzellan.net/en/follow-up-to-fourier-series-2/) by Bilim Ne Güzel Lan  
   
10. ### <a name="Part-5"></a>Toolkit Lab 5: Using Convolutions to Generalize (2 Sessions)

```
Required Reading:    
```
  
   * NoteBook: [Chapter 8: Using Convolutions to Generalize](https://github.com/deep-learning-with-pytorch/dlwpt-code/tree/master/p1ch8) from [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch) by Eli Stevens and Luca Antiga       
   * NoteBook: [Chapter 3:  Convolutional Neural Networks](https://github.com/falloutdurham/beginners-pytorch-deep-learning/tree/master/chapter3)  by Ian Pointer          

```
Suggested Reading:
```

   * Blog: [Pytorch (Basics) — Intro to CNN](https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-cnn-26a14c2ea29) by Akshaj Verma    

```
Additional Resources:
```
   
   * Blog: [PyTorch Image Recognition with Convolutional Networks](https://nestedsoftware.com/2019/09/09/pytorch-image-recognition-with-convolutional-networks-4k17.159805.html)   

11. ### <a name="SMRARN"></a>Sequence Modeling: Recurrent and Recursive Networks  (4 Sessions)

```
Required Reading:
```
  
   * [Chapter 10](http://www.deeplearningbook.org/contents/rnn.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook.    
   * Slides: [Sequence Modeling: Recurrent and Recursive Nets](https://cedar.buffalo.edu/~srihari/CSE676/) by Sargur Srihari  

```
Suggested Reading:
```
 
   * Blog: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah  <br>
   * Blog: [Illustrated Guide to LSTM’s and GRU’s: A Step by Step Explanation](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21) by Michael Nguyen  
  
```
Additional Reading:
```
  
   * [Video](https://www.youtube.com/watch?v=ZVN14xYm7JA&feature=youtu.be) of lecture / discussion. This video covers a presentation by Ian Goodfellow and a group discussion of Chapter 10 at a reading group in San Francisco organized by Alena Kruchkova. <br>
   * Blog: [Gentle introduction to Echo State Networks](https://towardsdatascience.com/gentle-introduction-to-echo-state-networks-af99e5373c68) by Madalina Ciortan   
   * Blog: [Understanding GRU Networks](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be) by Simeon Kostadinov    
   * Blog: [Animated RNN, LSTM and GRU](https://towardsdatascience.com/animated-rnn-lstm-and-gru-ef124d06cf45) by Raimi Karim   
   * **Extra Slides:**  
       * Slide: [Sequence Modeling: Recurrent and Recursive Networks](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L12-rnn.pdf) by U Kang    
       * Slide: [Training Recurrent Nets](http://web.eecs.utk.edu/~hqi/deeplearning/lecture14-rnn-training.pdf) by Arvind Ramanathan  
       * Slide: [An Introduction to: Reservoir Computing and Echo State Networks](http://didawiki.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/aa2/rnn4-esn.pdf) by Claudio Gallicchio   
       * Slide: [Graphical Models vs. Deep Networks](https://perso.liris.cnrs.fr/christian.wolf/papers/pres-workshop-df-170104.pdf) by Christian Wolf   
   
12. ### <a name="Part-6"></a>Toolkit Lab 6: Transfer Learning and Other Tricks (1 Session)

```
Required Reading:    
```
  
  * NoteBook: [Chapter 4:  Transfer Learning and Other Tricks](https://github.com/falloutdurham/beginners-pytorch-deep-learning/blob/master/chapter4) from [Programming PyTorch for Deep Learning](https://www.oreilly.com/library/view/programming-pytorch-for/9781492045342/)  by Ian Pointer         

```
Suggested Reading:
```

  * Blog: [Ideas on How to Fine-Tune a Pre-Trained Model in PyTorch](https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20) by Florin-Daniel Cioloboc   
  * Blog: [Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
  
```
Additional Resources:
```

  * Deep Learning via Pytorch by Ayoosh Kathuria   
      - [PyTorch 101, Part 4: Memory Management and Using Multiple GPUs](https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/)  
      - [PyTorch 101, Part 5: Understanding Hooks](https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/)

13. ### <a name="Practical-Methodology"></a>Practical Methodology  (2 Sessions)

```
Required Reading:
```

  * [Chapter 11](http://www.deeplearningbook.org/contents/guidelines.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook.  
  * Slides: [Practical Methodology](https://cedar.buffalo.edu/~srihari/CSE676/) by Sargur Srihari

```
Suggested Reading:
```

  * Metrics:
      - Blog: [Demystifying KL Divergence](https://medium.com/activating-robotic-minds/demystifying-kl-divergence-7ebe4317ee68) by Naoki Shibuya  
      - Blog: [Demystifying Cross-Entropy](https://medium.com/activating-robotic-minds/demystifying-cross-entropy-e80e3ad54a8) by Naoki Shibuya  
      - Blog: [Deep Quantile Regression](https://towardsdatascience.com/deep-quantile-regression-c85481548b5a) by Sachin Abeywardana  
      - Blog: [An Illustrated Guide to the Poisson Regression Model](https://towardsdatascience.com/an-illustrated-guide-to-the-poisson-regression-model-50cccba15958) by Sachin Date   
      - Blog: [Generalized Linear Models](https://towardsdatascience.com/generalized-linear-models-8738ae0fb97d) by Semih Akbayrak  
      - Blog: [ROC curves and Area Under the Curve Explained (Video)](https://www.dataschool.io/roc-curves-and-auc-explained/) by Data School
      - Blog: [Introduction to the ROC (Receiver Operating Characteristics) Plot](https://classeval.wordpress.com/introduction/introduction-to-the-roc-receiver-operating-characteristics-plot/) 
      - Slide: [ROC Curves](https://web.uvic.ca/~maryam/DMSpring94/Slides/9_roc.pdf) by Maryam Shoaran  
      - Blog: [Precision-Recall Curves](https://www.andybeger.com/2015/03/16/precision-recall-curves/) by Andreas Beger  

```
Additional Reading:
```
  * Paper: [The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf) by Jesse Davis and Mark Goadrich     
  * **Extra Slides:**  
      * Slide: [Practical Methodology](http://www.deeplearningbook.org/slides/11_practical.pdf)  by Ian Goodfellow  <br>
      * Slide: [Practical Methodology](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L17-practical-method.pdf) by U Kang <br> 

14. ### <a name="Part-7"></a> Toolkit Lab 7: Optimization Software ([Ray Tune](https://docs.ray.io/en/latest/index.html) or [Optuna](https://optuna.org/)) (1 Session)   

```
Required Reading:
```

  **Ray** provides a simple, universal API for building distributed applications:  
  * Blog: [Key Concepts](https://docs.ray.io/en/latest/tune/key-concepts.html)
  * Code: [A Basic Tune Tutorial](https://docs.ray.io/en/latest/tune/tutorials/tune-tutorial.html)

```
Suggested Reading:
```

  * Blog: [A Novice's Guide to Hyperparameter Optimization at Scale](https://wood-b.github.io/post/a-novices-guide-to-hyperparameter-optimization-at-scale/) by Brandon M. Wood   
  * Blog: [Tips for First-Time Users](https://docs.ray.io/en/latest/auto_examples/tips-for-first-time.html)  
  
  * **Algorithms:**
  * Blog: [Tutorial #8: Bayesian Optimization](https://www.borealisai.com/en/blog/tutorial-8-bayesian-optimization/) by M.O. Ahmed and S. Prince   
  * Blog: [Hyper-parameter Optimization Algorithms: a Short Review](https://medium.com/criteo-engineering/hyper-parameter-optimization-algorithms-2fe447525903) by Aloïs Bissuel   
  * Blog: [Band and BOHB: Understanding State of the Art Hyperparameter Optimization Algorithms](https://neptune.ai/blog/hyperband-and-bohb-understanding-state-of-the-art-hyperparameter-optimization-algorithms) by MJ Bahmani  
  * Blog: [A (Slightly) Better Budget Allocation for Hyperband](https://medium.com/data-from-the-trenches/a-slightly-better-budget-allocation-for-hyperband-bbd45af14481) by Alexandre Abraham   
  * Paper: [Hyperband: Bandit-Based Configuration Evaluation for Hyperparameter Optimization](https://jmlr.org/papers/volume18/16-558/16-558.pdf)
by Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, Ameet Talwalkar  
  * Blog: [Population Based Bandits: Provably Efficient Online Hyperparameter Optimization](https://www.anyscale.com/blog/population-based-bandits) by Jack Parker-Holder and Amog Kamsetty    
  * Blog: [Population Based Training of Neural Networks](https://deepmind.com/blog/article/population-based-training-neural-networks) by  by Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando and  Koray Kavukcuoglu.  
  * Code: [Guide to Population Based Training (PBT)](https://docs.ray.io/en/latest/tune/tutorials/tune-advanced-tutorial.html)
  * **Extra Slides:**  
      - [Multifidelity Modelling](https://mlatcl.github.io/mlphysical/slides/05-02-multifidelity.slides.html#/title-slide) by Neil Lawrence   


```
Additional Resources:
```

  * Blog: [Spark, Dask, and Ray: Choosing the Right Framework](http://blog.dominodatalab.com/spark-dask-ray-choosing-the-right-framework) by Nikolay Manchev   
  
  **Optuna** is an automatic hyperparameter optimization software framework, particularly designed for machine learning:  
  * Blog: [Using Optuna to Optimize PyTorch Hyperparameters](https://medium.com/pytorch/using-optuna-to-optimize-pytorch-hyperparameters-990607385e36) by Crissman Loomis  
  * Colab: [Optuna](https://colab.research.google.com/github/pfnet-research/optuna-hands-on/blob/master/en/01_Optuna_Quick_Start.ipynb#scrollTo=DjQFH-2x-WJa)  
  * Blog: [Tutorial](https://optuna.readthedocs.io/en/latest/tutorial/index.html)
  * Blog: [Efficient Hyperparameter Optimization with Optuna Framework](https://broutonlab.com/blog/efficient-hyperparameter-optimization-with-optuna-framework) by Brotoun Lab   
  * Blog: [Announcing Optuna 2.0](https://medium.com/optuna/optuna-v2-3165e3f1fc2) by Hiroyuki Vincent Yamazaki   
    
15. ### <a name="Applications"></a>Applications (1 Session)

```
Required Reading:
```
   * [Chapter 12](http://www.deeplearningbook.org/contents/applications.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
   * Slide: [Applications](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L18-applications.pdf) by U Kang  
   
```
Suggested Reading:
```

   * Blog: [How Neural Networks Learn Distributed Representations](https://www.oreilly.com/ideas/how-neural-networks-learn-distributed-representations) By Garrett Hoffman <br>

```
Additional Reading:
```
   * Blog: [30 Amazing Applications of Deep Learning](http://www.yaronhadad.com/deep-learning-most-amazing-applications/) by Yaron Hadad  
   * **Extra Slides:**  
       - Slides: [Applications](https://cedar.buffalo.edu/~srihari/CSE676/) by Sargur Srihari  

    
16. ### <a name="Autoencoders"></a>Autoencoders (2 Sessions)

```
Required Reading:
```
   * [Chapter 14](http://www.deeplearningbook.org/contents/autoencoders.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
   * Slide: [Autoencoders](http://www.cedar.buffalo.edu/%7Esrihari/CSE676/14%20Autoencoders.pdf) by Sargur Srihari  
   * Blog: [Variational Autoencoders](https://www.jeremyjordan.me/variational-autoencoders/) by Jermey Jordan
  
```
Suggested Reading:
```
   
   * Blog: [Introduction to Autoencoders](https://www.jeremyjordan.me/autoencoders/) by Jermey Jordan
   * Blog: [Understanding Variational Autoencoders (VAEs)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) by Joseph Rocca  
   * Blog: [Tutorial - What is a Variational Autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) by Jaan Altosaar 
   * Blog: [Autoencoders vs PCA: When to Use?](https://towardsdatascience.com/autoencoders-vs-pca-when-to-use-which-73de063f5d7) by Urwa Muaz  
   * Blog: [Generative Modeling: What is a Variational Autoencoder (VAE)?](https://www.mlq.ai/what-is-a-variational-autoencoder/) by Peter Foy  
   * Blog: [Variational Autoencoder: Intuition and Implementation](https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/) by Agustinus Kristiadi     
   * Blog: [Conditional Variational Autoencoder: Intuition and Implementation](https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/) by Agustinus Kristiadi    
    
```
Additional Reading:
```
    
   * Blog: [Tutorial - What is a Variational Autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) by Jaan Altosaar   
   * Blog: [Intuitively Understanding Variational Autoencoder: And Why They’re so Useful in Creating Your Own Generative Text, Art and Even Music](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf) by Irhum Shafkat  
   * Blog: [A High-Level Guide to Autoencoders](https://towardsdatascience.com/a-high-level-guide-to-autoencoders-b103ccd45924) by Shreya Chaudhary  

   * **Extra Slides:**  
       - Slide: [Variational Autoencoders](http://slazebni.cs.illinois.edu/spring17/lec12_vae.pdf) by Raymond Yeh, Junting Lou, and Teck-Yian Lim  
       - Slide: [Autoencoders](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L19-autoencoder.pdf) by U Kang 
       - Slide: [Generative Models](https://hpi.de/fileadmin/user_upload/fachgebiete/meinel/team/Haojin/competitive_problem_solving_with_deep_learning/Class_Generative_Models.pptx.pdf) by Mina Rezaei  


    
17. ### <a name="GAN"></a>Generative Adversarial Networks (1 Session)

```
Required Reading:
```

Slide: [Generative Adversarial Networks (GANs)](http://slazebni.cs.illinois.edu/spring17/lec11_gan.pdf) by Binglin, Shashank, and Bhargav  
Paper: [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf) by Ian Goodfellow  
  
```
Suggested Reading:
```

* Blog: [Generative Adversarial Networks (GANs), Some Open Questions](https://www.offconvex.org/2017/03/15/GANs/) by Sanjeev Arora   
* Paper: [Generative Adversarial Networks: An Overview](https://arxiv.org/pdf/1710.07035.pdf) by Antonia Creswell, Tom White, Vincent Dumoulin, Kai Arulkumaran, Biswa Sengupta, and Anil A Bharath  

```
Additional Reading:
```

* Blog: [GANs Comparison Without Cherry-Picking](https://github.com/khanrc/tf.gans-comparison) by Junbum Cha  
* Blog: [New Progress on GAN Theory and Practice](https://casmls.github.io/general/2017/04/13/gan.html) by Liping Liu  
* Blog: [Play with Generative Adversarial Networks (GANs) in your browser!](https://poloclub.github.io/ganlab/)  
* Blog: [The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo) by Avinash Hindupur  
* [Generative Adversarial Networks (GANs), Some Open Questions](https://www.offconvex.org/2017/03/15/GANs/) by Sanjeev Arora  

18. ### <a name="GNN"></a>Graph Neural Networks (1 Session)

```
Required Reading:
```

* Slide: [Graph Neural Networks](http://ir.hit.edu.cn/~xiachongfeng/slides/Graph%20Neural%20Networks.pdf) by Xiachong Feng  
* Paper: [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596.pdf) by Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu  

```
Suggested Reading:
```

* Book: [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) by William L. Hamilton   
* Tutorial: [Representation Learning on Networks](http://snap.stanford.edu/proj/embeddings-www/index.html) by Jure Leskovec      
* Paper: [A Practical Guide to Graph Neural Networks](https://arxiv.org/pdf/2010.05234.pdf) by Isaac Ronald Ward, Jack Joyner, Casey Lickfold, Stash Rowe, Yulan Guo, and Mohammed Bennamoun      
* Blog: [Deep Graph Library (DGL)](https://www.dgl.ai): A Python package that interfaces between existing tensor libraries and data being expressed as graphs.   

```
Additional Reading:
```  

* GitHub: [Graph Neural Networks](https://github.com/hhaji/Deep-Learning/tree/master/Graph-Neural-Networks)   
* **Extra Slides:**  
    - Slide: [Graph Neural Networks: Models and Applications](https://cse.msu.edu/~mayao4/tutorials/aaai2020/) by Yao Ma, Wei Jin, Jiliang Tang, Lingfei Wu and Tengfei Ma. This is a perfect slide about graph neural networks. 


19. ### <a name="TRA"></a> Attention Mechanisms and Transformers(1 Session)

```
Required Reading:
```

* Slide: [Attention Mechanisms](https://andre-martins.github.io/docs/dsl2020/attention-mechanisms.pdf) by Marcos V. Treviso   
* Book: [Dive into Deep Learning (Chapter 10)](http://d2l.ai/chapter_attention-mechanisms/) by Mag Gardner, Max Drummy, Joanne Quinn, Joanne McEachen, and Michael Fullan   
    - [10.1. Attention Cues](http://d2l.ai/chapter_attention-mechanisms/attention-cues.html#attention-cues-in-biology) 
    - [10.2. Attention Pooling: Nadaraya-Watson Kernel Regression](http://d2l.ai/chapter_attention-mechanisms/nadaraya-watson.html)  
    - [10.3. Attention Scoring Functions](http://d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html) 
    - [10.4. Bahdanau Attention](http://d2l.ai/chapter_attention-mechanisms/bahdanau-attention.html) 
    - [10.5. Multi-Head Attention](http://d2l.ai/chapter_attention-mechanisms/multihead-attention.html) 
    - [10.6. Self-Attention and Positional Encoding](http://d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html) 
    - [10.7. Transformer](http://d2l.ai/chapter_attention-mechanisms/transformer.html)  

```
Suggested Reading:
```

* Blog: [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) by Amirhossein Kazemnejad   
* Blog: [Transformers From Scratch](http://peterbloem.nl/blog/transformers) by Peter Bloem   
    - [Video](https://www.youtube.com/playlist?list=PLIXJ-Sacf8u60G1TwcznBmK6rEL3gmZmV)
* Blog: [Transformers are Graph Neural Networks](https://towardsdatascience.com/transformers-are-graph-neural-networks-bca9f75412aa) by Chaitanya K. Joshi    
* Blog: [How Transformers work in deep learning and NLP: an Intuitive Introduction](https://theaisummer.com/transformer/) by Nikolas Adaloglou   
* Blog: [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar     


```
Additional Reading:
```  

* Blog: [Illustrated: Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a) by Raimi Karim  

    
### <a name="ANAS"></a>Additional Resources:    
- Papers:  
  * [Papers with Code:](https://paperswithcode.com) The mission of Papers With Code is to create a free and open resource with Machine Learning papers, code and evaluation tables.
  * [Deep Learning Papers Reading Roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap) by Flood Sung <br>
  * [Awesome - Most Cited Deep Learning Papers](https://github.com/terryum/awesome-deep-learning-papers) by  Terry Taewoong Um <br>
 - Deep Learning Courses:  
    * [Deep Learning Do It Yourself!](https://dataflowr.github.io/website/) by Marc Lelarge, Jill-Jênn Vie, and Andrei Bursuc 
    * [The Deep Learning Lecture Series 2020](https://deepmind.com/learning-resources/deep-learning-lecture-series-2020) by  DeepMind Research Scientists and Research Engineers    
    * [Deep Learning](https://fleuret.org/ee559/) by François Fleuret   
    * [Deep Learning](https://web.cs.hacettepe.edu.tr/~aykut/classes/spring2018/cmp784/index.html) by Aykut Erdem <br>
    * [Mini Course in Deep Learning with PyTorch for AIMS](https://github.com/Atcold/pytorch-Deep-Learning-Minicourse) 
by Alfredo Canziani   
    * [Introduction to Pytorch Code Examples](https://cs230.stanford.edu/blog/pytorch/) by Andrew Ng and Kian Katanforoosh     
- The blog of [Christopher Olah:](http://colah.github.io) Fascinating tutorials about neural networks  
- The blog of [Adit Deshpande:](https://adeshpande3.github.io/adeshpande3.github.io/) The Last 5 Years In Deep Learning  
- [Fascinating Tutorials on Deep Learning](https://r2rt.com/)   
- [Deep Learning (Faster Data Science Education by Kaggle)](https://www.kaggle.com/learn/deep-learning) by Dan Becker   

## <a name="Class-Time-and-Location"></a>Class Time and Location:
Saturday and Monday 10:30-12:00 AM (Fall 2020)

### <a name="MTA"></a>Recitation and Assignments:
Tuesday 16:00-18:00 PM (Fall 2020), 
Refer to the following [link](https://github.com/hhaji/Deep-Learning/tree/master/Recitation-Assignments) to check the assignments.  

## <a name="Projects"></a>Projects:
Projects are programming assignments that cover the topic of this course. Any project is written by **[Jupyter Notebook](http://jupyter.org)**. Projects will require the use of Python 3.7, as well as additional Python libraries. 

### <a name="Google-Colab"></a>Google Colab:
[Google Colab](https://colab.research.google.com) is a free cloud service and it supports free GPU! 
  - [How to Use Google Colab](https://www.geeksforgeeks.org/how-to-use-google-colab/) by Souvik Mandal <br> 
  - [Primer for Learning Google Colab](https://medium.com/dair-ai/primer-for-learning-google-colab-bb4cabca5dd6)
  - [Deep Learning Development with Google Colab, TensorFlow, Keras & PyTorch](https://www.kdnuggets.com/2018/02/google-colab-free-gpu-tutorial-tensorflow-keras-pytorch.html)

### <a name="Fascinating-Guides-For-Machine-Learning"></a>Fascinating Guides For Machine Learning:
* [Technical Notes On Using Data Science & Artificial Intelligence: To Fight For Something That Matters](https://chrisalbon.com) by Chris Albon

### <a name="Latex"></a>Latex:
The students can include mathematical notation within markdown cells using LaTeX in their **[Jupyter Notebooks](http://jupyter.org)**.<br>
  - A Brief Introduction to LaTeX [PDF](https://www.seas.upenn.edu/~cis519/spring2018/assets/resources/latex/latex.pdf)  <br>
  - Math in LaTeX [PDF](https://www.seas.upenn.edu/~cis519/spring2018/assets/resources/latex/math.pdf) <br>
  - Sample Document [PDF](https://www.seas.upenn.edu/~cis519/spring2018/assets/resources/latex/sample.pdf) <br>
  - [TikZ:](https://github.com/PetarV-/TikZ) A collection Latex files of PGF/TikZ figures (including various neural networks) by Petar Veličković. 

## <a name="Grading"></a>Grading:
* Projects and Midterm – 50%
* Endterm – 50%

### <a name="Three-Exams"></a>ُThree Exams:

* First Midterm Examination: Saturday 1399/09/01, 10:30-12:00 
* Second Midterm Examination: Saturday 1399/10/06, 10:30-12:00 
* Final Examination: Wednesday 1399/10/24, 14:00-16:00 

## <a name="Prerequisites"></a>Prerequisites:
General mathematical sophistication; and a solid understanding of Algorithms, Linear Algebra, and 
Probability Theory, at the advanced undergraduate or beginning graduate level, or equivalent.

### <a name="Linear-Algebra"></a>Linear Algebra:
* Video: Professor Gilbert Strang's [Video Lectures](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/) on linear algebra.

### <a name="Probability-and-Statistics"></a>Probability and Statistics:
* [Learn Probability and Statistics Through Interactive Visualizations:](https://seeing-theory.brown.edu/index.html#firstPage) Seeing Theory was created by Daniel Kunin while an undergraduate at Brown University. The goal of this website is to make statistics more accessible through interactive visualizations (designed using Mike Bostock’s JavaScript library D3.js).
* [Statistics and Probability:](https://stattrek.com) This website provides training and tools to help you solve statistics problems quickly, easily, and accurately - without having to ask anyone for help.
* Jupyter NoteBooks: [Introduction to Statistics](https://github.com/rouseguy/intro2stats) by Bargava
* Video: Professor John Tsitsiklis's [Video Lectures](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/video-lectures/) on Applied Probability.
* Video: Professor Krishna Jagannathan's [Video Lectures](https://nptel.ac.in/courses/108106083/) on Probability Theory.

## <a name="Topics"></a>Topics:
Have a look at some reports of [Kaggle](https://www.kaggle.com/) or Stanford students ([CS224N](http://nlp.stanford.edu/courses/cs224n/2015/), [CS224D](http://cs224d.stanford.edu/reports_2016.html)) to get some general inspiration.

## <a name="Account"></a>Account:
It is necessary to have a [GitHub](https://github.com/) account to share your projects. It offers 
plans for both private repositories and free accounts. Github is like the hammer in your toolbox, 
therefore, you need to have it!

## <a name="Academic-Honor-Code"></a>Academic Honor Code:
Honesty and integrity are vital elements of the academic works. All your submitted assignments must be entirely your own (or your own group's).

We will follow the standard of Department of Mathematical Sciences approach: 
* You can get help, but you MUST acknowledge the help on the work you hand in
* Failure to acknowledge your sources is a violation of the Honor Code
*  You can talk to others about the algorithm(s) to be used to solve a homework problem; as long as you then mention their name(s) on the work you submit
* You should not use code of others or be looking at code of others when you write your own: You can talk to people but have to write your own solution/code

## <a name="Questions"></a>Questions?
I will be having office hours for this course on Saturday (09:00 AM--10:00 AM). If this is not convenient, email me at hhaji@sbu.ac.ir or talk to me after class.

