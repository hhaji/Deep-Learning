Lecturer: [Hossein Hajiabolhassan](http://facultymembers.sbu.ac.ir/hhaji/) <br>
The Webpage of the Course: [Deep Learning](https://hhaji.github.io/Deep-Learning/) <br>
[Data Science Center](http://ds.sbu.ac.ir), [Shahid Beheshti University](http://www.sbu.ac.ir/) <br>

---

### **Index:**
- [Course Overview](#Course-Overview)
- [Main TextBooks](#Main-TextBooks)
- [Slides and Papers](#Slides-and-Papers)
  1. Lecture 1: [Toolkit Lab (Part 1)](#Part-1) 
  2. Lecture 2: [Introduction](#Introduction) 
  3. Lecture 3: [Deep Feedforward Networks](#DFN) 
  4. Lecture 4: [Regularization for Deep Learning](#RFDL) 
  5. Lecture 5: [Optimization for Training Deep Models](#OFTDM) 
  6. Lecture 6: [Convolutional Networks](#CNN) 
  7. Lecture 7: [Sequence Modeling: Recurrent and Recursive Networks](#SMRARN) 
- [Additional NoteBooks and Slides](#ANAS)
- [Class Time and Location](#Class-Time-and-Location)
- [Projects](#Projects)
  - [Google Colab](#Google-Colab)
  - [Fascinating Guides For Machine Learning](#Fascinating-Guides-For-Machine-Learning)
  - [Latex](#Latex)
- [Grading](#Grading)
- [Prerequisites](#Prerequisites)
  - [Linear Algebra](#Linear-Algebra)
  - [Probability and Statistics](#Probability-and-Statistics)
- [Topics](#Topics)
- [Account](#Account)
- [Academic Honor Code](#Academic-Honor-Code)
- [Questions](#Questions)
- Miscellaneous

---

## <a name="Course-Overview"></a>Course Overview:
```javascript
In this course, you will learn the foundations of Deep Learning, understand how to build 
neural networks, and learn how to lead successful machine learning projects. You will learn 
about Convolutional networks, RNNs, LSTM, Adam, Dropout, BatchNorm, and more.
```

## <a name="Main-TextBooks"></a>Main TextBooks:
![Book 1](/Images/DL.jpg) ![Book 2](/Images/DDLP.png) ![Book 3](/Images/TF2-Packt.png)![Book 4](/Images/ProDeep.jpg)  ![Book 5](/Images/NNDL.jpg)

```
Main TextBooks:
```
* [Deep Learning](http://www.deeplearningbook.org) (available in online) by Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville <br>
* [Deep Learning with Python](https://machinelearningmastery.com/deep-learning-with-python/) by J. Brownlee <br>

```
Additional TextBooks:
```

* [TensorFlow 2.0 Quick Start Guide](https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-20-quick-start-guide) by Tony Holdroyd <br> 
* [Pro Deep Learning with TensorFlow: A Mathematical Approach to Advanced Artificial Intelligence in Python](https://www.amazon.com/Pro-Deep-Learning-TensorFlow-Mathematical-ebook/dp/B077Z79LVJ) by Santanu Pattanayak <br>
* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen <br>

## <a name="Slides-and-Papers"></a>Slides and Papers:  
  Recommended Slides & Papers:

1. ### <a name="Part-1"></a>Toolkit Lab (Part 1)  

    ```
    Required Reading:
    ```

    * Blog: [How to Use Google Colab](https://www.geeksforgeeks.org/how-to-use-google-colab/) by Souvik Mandal <br>  
    * [Conda Commands (Create Virtual Environments for Python with Conda)](http://deeplearning.lipingyang.org/2018/12/25/conda-commands-create-virtual-environments-for-python-with-conda/) by LipingY <br> 
    * Blog: [Install TensorFlow 2.0 with pip](https://www.tensorflow.org/install/pip) <br>
    * NoteBook: [TensorFlow 2.0 Quick Start Guide (Chapter 1)](https://github.com/PacktPublishing/Tensorflow-2.0-Quick-Start-Guide/blob/master/Chapter01/Chapter1_TF2_alpha.ipynb) by Tony Holdroyd <br>
    * NoteBook: [TensorFlow 2.0 Examples](https://github.com/aymericdamien/TensorFlow-Examples/tree/master/tensorflow_v2) by Aymeric Damien <br>
    * Blog: [What’s coming in TensorFlow 2.0](https://medium.com/tensorflow/whats-coming-in-tensorflow-2-0-d3663832e9b8) by the TensorFlow Team 

    ```
    Additional Reading:
    ```
    * [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) 
    * Blog: [Mathematics Operations in Tensorflow](https://www.tensorflow.org/api_docs/python/tf/math) <br>
    * Blog: [TensorFlow 2.0 in 5 Minutes (Tutorial)](https://gdcoder.com/tensorflow-2-0-in-5-minutes/) by Georgios Drakos<br>
    * Blog: [Introducing Ragged Tensors](https://medium.com/tensorflow/introducing-ragged-tensors-ac301c31fd38) by Laurence Moroney <br>
    * Blog: [Operations in Ragged Tensors](https://www.tensorflow.org/api_docs/python/tf/ragged)   

    ```
    TensorFlow 1.
    ```
 
    * GitHub: [Set up Tensorflow 1.](https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/setup)  by Chip Huyen <br>
    * Slide: [Overview of TensorFlow 1.](https://docs.google.com/presentation/d/1dizKPtp9hkuTwVDzoGZdYQb_61ULSsSUvaFfDFuhIc4/edit?usp=sharing)  by Chip Huyen <br>
        * [NoteBook:](https://github.com/hhaji/Deep-Learning/blob/master/NoteBooks/Lec1.ipynb) This was adopted from the slide ([Overview of TensorFlow 1.](https://docs.google.com/presentation/d/1dizKPtp9hkuTwVDzoGZdYQb_61ULSsSUvaFfDFuhIc4/edit?usp=sharing)) of Chip Huyen <br>
    * Slide: [Operations (TensorFlow 1.)](https://docs.google.com/presentation/d/1iO_bBL_5REuDQ7RJ2F35vH2BxAiGMocLC6t_N-6eXaE/edit#slide=id.g1bd10f151e_0_0) by Chip Huyen <br>
        * [NoteBook:](https://github.com/hhaji/Deep-Learning/blob/master/NoteBooks/Lec2.ipynb) This was adopted from the slide ([Operations](https://docs.google.com/presentation/d/1iO_bBL_5REuDQ7RJ2F35vH2BxAiGMocLC6t_N-6eXaE/edit#slide=id.g1bd10f151e_0_0)) of Chip Huyen <br>
    * NoteBook: [Useful Operations in TensorFlow 1](https://github.com/hhaji/Deep-Learning/blob/master/NoteBooks/Lec3.ipynb) <br>
    * Blog: [Variables Sharing in TensorFlow 1.: Variable vs get_variable](http://stefanocappellini.com/tf-variable-vs-get_variable-sharing/) by Stefano Cappellini 
        * [NoteBook](https://github.com/StefanoCappellini/tensorflow_tips/blob/master/TF-variable-sharing.ipynb) <br>
    * Blog: [Tensorflow 1.: The Confusing Parts (2)](https://jacobbuckman.com/post/tensorflow-the-confusing-parts-2/) by Jacob Buckman  
        * [NoteBook](https://github.com/hhaji/Deep-Learning/blob/master/NoteBooks/The-Confusing-Parts-2.ipynb) <br>  
    * Slide: [Eager Execution](https://docs.google.com/presentation/d/1e1gE2JJXipWm1UJgor_y8pHcM8L8oMaCVtvQvZUBlQY/edit?usp=sharing) by Chip Huyen 

2. ### <a name="Introduction"></a>Introduction  

    ```
    Required Reading:
    ```

    * [Chapter 1](http://www.deeplearningbook.org/contents/intro.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
    * Slide: [Introduction](https://www.deeplearningbook.org/slides/01_intro.pdf)  by Ian Goodfellow

    ```
    Additional Reading:
    ```

    * [Video](https://www.youtube.com/embed//vi7lACKOUao) of lecture by Ian Goodfellow and discussion of Chapter 1 at a reading group in San Francisco organized by Alena Kruchkova <br>
    * [Mathematics for Machine Learning](http://www.deeplearningindaba.com/uploads/1/0/2/6/102657286/2018_maths4ml_vfinal.pdf) by Avishkar Bhoopchand, Cynthia Mulenga, Daniela Massiceti, Kathleen Siminyu, and Kendi Muchungi 

3. ### <a name="DFN"></a>Deep Feedforward Networks  

    ```
    Required Reading:
    ```

    * [Chapter 6](https://www.deeplearningbook.org/contents/mlp.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
    * Slide: [Deep Feedforward Networks](https://www.deeplearningbook.org/slides/06_mlp.pdf)  by Ian Goodfellow 
    * Blog: [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/) by Christopher Olah 

    ```
    Additional Reading:
    ```

    * [Video](https://drive.google.com/file/d/0B64011x02sIkRExCY0FDVXFCOHM/view?usp=sharing): (.flv) of a presentation by Ian  Goodfellow and a group discussion at a reading group at Google organized by Chintan Kaur. <br>
    * Slides: Deep Feedforward Networks [1](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L8-deep_feedforward_networks.pdf)  and [2](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L9-deep_feedforward_networks-2.pdf) by U Kang 

4. ### <a name="RFDL"></a>Regularization for Deep Learning  

    ```
    Required Reading:
    ```

    * [Chapter 7](http://www.deeplearningbook.org/contents/regularization.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
    * Slide: [Regularization for Deep Learning](https://www.deeplearningbook.org/slides/07_regularization.pdf)  by Ian Goodfellow

    ```
    Additional Reading:
    ```

    * Slides: Regularization for Deep Learning [1](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L13-regularization.pdf)  and [2](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L14-regularization-2.pdf) by U Kang 

5. ### <a name="OFTDM"></a>Optimization for Training Deep Models  

    ```
    Required Reading:
    ```  

    * [Chapter 8](http://www.deeplearningbook.org/contents/optimization.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
    * Slide: [Gradient Descent and Structure of Neural Network Cost Functions](https://www.deeplearningbook.org/slides/sgd_and_cost_structure.pdf) by Ian Goodfellow   
These slides describe how gradient descent behaves on different kinds of cost function surfaces. Intuition for the structure of the cost function can be built by examining a second-order Taylor series approximation of the cost function. This quadratic function can give rise to issues such as poor conditioning and saddle points. Visualization of neural network cost functions shows how these and some other geometric features of neural network cost functions affect the performance of gradient descent.
    * Slide: [Tutorial on Optimization for Deep Networks](https://www.deeplearningbook.org/slides/dls_2016.pdf) by Ian Goodfellow    
Ian Goodfellow's presentation at the 2016 Re-Work Deep Learning Summit. Covers Google Brain research on optimization, including visualization of neural network cost functions, Net2Net, and batch normalization.
    * Slide: [Batch Normalization](https://www.deeplearningbook.org/slides/batch_norm.pdf) by Ian Goodfellow
 
    ```
    Additional Reading:
    ```

    * [Video](https://www.youtube.com/watch?v=Xogn6veSyxA) of lecture / discussion: This video covers a presentation by Ian Goodfellow and group discussion on the end of Chapter 8 and entirety of Chapter 9 at a reading group in San Francisco organized by Taro-Shigenori Chiba. <br>
    * Slides: Optimization for Training Deep Models [1](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L15-opt.pdf)  and [2](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L16-opt-2.pdf) by U Kang 

6. ### <a name="CNN"></a>Convolutional Networks  

    ```
    Required Reading:
    ```

    * [Chapter 9](http://www.deeplearningbook.org/contents/convnets.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
    * Slide: [Convolutional Networks](https://www.deeplearningbook.org/slides/09_conv.pdf)  by Ian Goodfellow  <br>
A presentation summarizing Chapter 9, based directly on the textbook itself. <br>  
    * Blog: [Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/) by Christopher Olah <br>
    * Blog: [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 Way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) by Sumit Saha

    ```
    Additional Reading:  
    ```  
    
   * [Video](https://www.youtube.com/watch?v=Xogn6veSyxA) of lecture / discussion: This video covers a presentation by Ian Goodfellow and group discussion on the end of Chapter 8 and entirety of Chapter 9 at a reading group in San Francisco organized by Taro-Shigenori Chiba. <br>
   * [A Basic Introduction to Separable Convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) by Chi-Feng Wang <br>
   * [Depth wise Separable Convolutional Neural Networks](https://www.geeksforgeeks.org/depth-wise-separable-convolutional-neural-networks/) by Mayank Chaurasia <br>  
   * Slide: [Convolutional Models](http://www.deeplearningindaba.com/uploads/1/0/2/6/102657286/dl_indaba_2018_convnets.pdf) by Naila Murray <br>
   * Blog: [A Convolutional Neural Network Tutorial in Keras and TensorFlow 2](https://www.machineislearning.com/convolutional-neural-network-keras-tensorflow-2/) by Isak Bosman <br>
   * Slide: [Convolutional Networks](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L11-cnn.pdf) by U Kang <br> 
   * Blog: [An Intuitive Guide to Convolutional Neural Networks](https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/) by  Daphne Cornelisse <br>
   * Paper: [A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/pdf/1603.07285.pdf) by Vincent Dumoulin and Francesco Visin 

7. ### <a name="SMRARN"></a>Sequence Modeling: Recurrent and Recursive Networks  

    ```
    Required Reading:
    ```

    * [Chapter 10](http://www.deeplearningbook.org/contents/rnn.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
    * Slide: [Sequence Modeling](https://www.deeplearningbook.org/slides/10_rnn.pdf)  by Ian Goodfellow  <br>
A presentation summarizing Chapter 10, based directly on the textbook itself. <br>
    * [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah

    ```
    Additional Reading:
    ```
    
   * [Video](https://www.youtube.com/watch?v=ZVN14xYm7JA&feature=youtu.be) of lecture / discussion. This video covers a presentation by Ian Goodfellow and a group discussion of Chapter 10 at a reading group in San Francisco organized by Alena Kruchkova. <br>
    * Slide: [Sequence Modeling: Recurrent and Recursive Networks](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L12-rnn.pdf) by U Kang <br> 
    
### <a name="ANAS"></a>Additional NoteBooks and Slides  
- [Deep Learning (Faster Data Science Education by Kaggle)](https://www.kaggle.com/learn/deep-learning) by Dan Becker <br>

## <a name="Class-Time-and-Location"></a>Class Time and Location:
Sunday and Tuesday 13:00-14:30 AM (Fall 2019)

## <a name="Projects"></a>Projects:
Projects are programming assignments that cover the topic of this course. Any project is written by **[Jupyter Notebook](http://jupyter.org)**. Projects will require the use of Python 3.7, as well as additional Python libraries. 

* [Get Started with TensorFlow](https://www.tensorflow.org/tutorials/)

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

## <a name="Grading"></a>Grading:
* Projects and Midterm – 50%
* Endterm – 50%

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
I will be having office hours for this course on Sunday (09:00 AM--10:00 AM). If this is not convenient, email me at hhaji@sbu.ac.ir or talk to me after class.
