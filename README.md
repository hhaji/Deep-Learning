Lecturer: [Hossein Hajiabolhassan](http://facultymembers.sbu.ac.ir/hhaji/) <br>
The Webpage of the Course: [Deep Learning](https://hhaji.github.io/Deep-Learning/) <br>
[Data Science Center](http://ds.sbu.ac.ir), [Shahid Beheshti University](http://www.sbu.ac.ir/) <br>

---

### **Index:**
- [Course Overview](#Course-Overview)
- [Main TextBooks](#Main-TextBooks)
- [Slides and Papers](#Slides-and-Papers)
  1. Lecture 1: [Introduction](#Introduction) 
  2. Lecture 2: [Toolkit Lab 1: Google Colab and Anaconda](#Part-1) 
  3. Lecture 3: [Toolkit Lab 2: Image Preprocessing by Keras](#Part-2) 
  4. Lecture 4: [Deep Feedforward Networks](#DFN) 
  5. Lecture 5: [Toolkit Lab 3: Deep Learning by Keras](#Part-4) 
  6. Lecture 6: [Toolkit Lab 4: TensorFlow 2.0](#Part-3) 
  7. Lecture 7: [Toolkit Lab 5: TensorBoard](#Part-5) 
  8. Lecture 8: [Regularization for Deep Learning](#RFDL) 
  9. Lecture 9: [Optimization for Training Deep Models](#OFTDM) 
  10. Lecture 10: [Convolutional Networks](#CNN) 
  11. Lecture 11: [Sequence Modeling: Recurrent and Recursive Networks](#SMRARN) 
  12. Lecture 12: [Practical Methodology](#Practical-Methodology)  
  13. Lecture 13: [Applications](#Applications) 
  14. Lecture 14: [Autoencoders](#Autoencoders)
  
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
![Book 1](/Images/DL.jpg)  ![Book 2](/Images/Hands-On-ML.jpg) ![Book 3](/Images/ProDeep.jpg) ![Book 4](/Images/NNLM.jpg) ![Book 5](/Images/DDLP.png)

```
Main TextBooks:
```

* [Deep Learning](http://www.deeplearningbook.org) (available in online) by Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville <br>
* [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurelien Geron <br>

```
Additional TextBooks:
```

* [Pro Deep Learning with TensorFlow: A Mathematical Approach to Advanced Artificial Intelligence in Python](https://www.amazon.com/Pro-Deep-Learning-TensorFlow-Mathematical-ebook/dp/B077Z79LVJ) by Santanu Pattanayak <br>
* [Neural Networks and Learning Machines (3rd Edition)](https://www.amazon.com/Neural-Networks-Learning-Machines-Comprehensive-ebook/dp/B008VIX57I) by Simon Haykin <br> 
* [Deep Learning with Python](https://machinelearningmastery.com/deep-learning-with-python/) by J. Brownlee <br>


## <a name="Slides-and-Papers"></a>Slides and Papers:  
  Recommended Slides & Papers:
  
1. ### <a name="Introduction"></a>Introduction  

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
Applied Math and Machine Learning Basics:
```

 * Slide: [Mathematics for Machine Learning](http://www.deeplearningindaba.com/uploads/1/0/2/6/102657286/2018_maths4ml_vfinal.pdf) by Avishkar Bhoopchand, Cynthia Mulenga, Daniela Massiceti, Kathleen Siminyu, and Kendi Muchungi 
* Blog: [A Gentle Introduction to Maximum Likelihood Estimation and Maximum A Posteriori Estimation (Getting Intuition of MLE and MAP with a Football Example)](https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-and-maximum-a-posteriori-estimation-d7c318f9d22d) by Shota Horii  
    
2. ### <a name="Part-1"></a>Toolkit Lab 1: Google Colab and Anaconda  

```
Required Reading:
```

  * Blog: [Google Colab Free GPU Tutorial](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d) by Fuat <br>
  * Blog: [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments) <br>
  * Blog: [Kernels for Different Environments](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) <br>
  * Install: [TensorFlow 2.0 RC is Available](https://www.tensorflow.org/install) <br>
 
```
Suggested Reading:
```
 
 * Blog: [Stop Installing Tensorflow Using pip for Performance Sake!](https://towardsdatascience.com/stop-installing-tensorflow-using-pip-for-performance-sake-5854f9d9eb0c) by Michael Nguyen <br> 
 * Blog: [Using Pip in a Conda Environment](https://www.anaconda.com/using-pip-in-a-conda-environment/) by Jonathan Helmus <br> 
 * Blog: [How to Import Dataset to Google Colab Notebook?](https://mc.ai/how-to-import-dataset-to-google-colab-notebook/) 
 * Blog: [How to Upload Large Files to Google Colab and Remote Jupyter Notebooks ](https://www.freecodecamp.org/news/how-to-transfer-large-files-to-google-colab-and-remote-jupyter-notebooks-26ca252892fa/)(For Linux Operating System) by Bharath Raj  <br>

```
Additional Resources:
```
  * PDF: [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/_downloads/1f5ecf5a87b1c1a8aaf5a7ab8a7a0ff7/conda-cheatsheet.pdf) 
  * Blog: [Conda Commands (Create Virtual Environments for Python with Conda)](http://deeplearning.lipingyang.org/2018/12/25/conda-commands-create-virtual-environments-for-python-with-conda/) by LipingY <br>  
  * Blog: [Colab Tricks](https://rohitmidha23.github.io/Colab-Tricks/) by  Rohit Midha <br>
  
3. ### <a name="Part-2"></a>Toolkit Lab 2: Image Preprocessing by Keras 

```
Required Reading:
```
    
  * Blog: [How to Load, Convert, and Save Images With the Keras API](https://machinelearningmastery.com/how-to-load-convert-and-save-images-with-the-keras-api/) by Jason Brownlee 
  * Blog: [Classify Butterfly Images with Deep Learning in Keras](https://towardsdatascience.com/classify-butterfly-images-with-deep-learning-in-keras-b3101fe0f98) by Bert Carremans  
    Read the part of Data augmentation of images  
  * Blog: [Keras ImageDataGenerator Methods: An Easy Guide](https://medium.com/datadriveninvestor/keras-imagedatagenerator-methods-an-easy-guide-550ecd3c0a92) by Ashish Verma  
  
```
Suggested Reading:
```
   
  * Blog: [Keras ImageDataGenerator and Data Augmentation](https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/) by  Adrian Rosebrock  
  * Blog: [How to Configure Image Data Augmentation in Keras](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/) by Jason Brownlee 
  * Blog: [A Quick Guide To Python Generators and Yield Statements](https://medium.com/@jasonrigden/a-quick-guide-to-python-generators-and-yield-statements-89a4162c0ef8) by Jason Rigden 
  * NoteBook: [Iterable, Generator, and Iterator](https://github.com/hhaji/Deep-Learning/blob/master/NoteBooks/Generator.ipynb)  
  * Blog: [Vectorization in Python](https://www.geeksforgeeks.org/vectorization-in-python/) 
  * Blog: [numpy.vectorize](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.vectorize.html)
 
```
Additional Resources:
```

 * Blog: [Learn about ImageDataGenerator](https://fairyonice.github.io/Learn-about-ImageDataGenerator.html) by Yumi 
 * Blog: [Images Augmentation for Deep Learning with Keras](https://rock-it.pl/images-augmentation-for-deep-learning-with-keras/) by Jakub Skałecki  
 * Blog: [A Detailed Example of How to Use Data Generators with Keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly#disqus-thread) by Afshine Amidi and Shervine Amidi  
  * Blog: [Iterables vs. Iterators vs. Generators](https://nvie.com/posts/iterators-vs-generators/) by Vincent Driessen   
  
4. ### <a name="DFN"></a>Deep Feedforward Networks  

```
Required Reading:
```

  * [Chapter 6](https://www.deeplearningbook.org/contents/mlp.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br> 
  * Slide: [Feedforward Neural Networks (Lecture 2)](http://wavelab.uwaterloo.ca/wp-content/uploads/2017/04/Lecture_2.pdf) by Ali Harakeh  
  * Slides: Deep Feedforward Networks [1](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L8-deep_feedforward_networks.pdf)  and [2](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L9-deep_feedforward_networks-2.pdf) by U Kang   
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

  * Blog: [The Gradient](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/the-gradient) by Khanacademy <br>
  * Blog: [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/) by Christopher Olah 
  
```
Additional Resources:
```

  * Blog: [Activation Functions](https://sefiks.com/tag/activation-function/) by Sefik Ilkin Serengil   
  * Blog: [Activation Functions](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#id5)  
  * Blog: [Stochastic vs Batch Gradient Descent](https://medium.com/@divakar_239/stochastic-vs-batch-gradient-descent-8820568eada1) by Divakar Kapil  
  * [Video](https://drive.google.com/file/d/0B64011x02sIkRExCY0FDVXFCOHM/view?usp=sharing): (.flv) of a presentation by Ian  Goodfellow and a group discussion at a reading group at Google organized by Chintan Kaur. <br>
  * Slide: [Deep Feedforward Networks](https://www.deeplearningbook.org/slides/06_mlp.pdf)  by Ian Goodfellow  
   
5. ### <a name="Part-4"></a>Toolkit Lab 4: Deep Learning by Keras  
```
Required Reading:
```
    
  * NoteBook: [Chapter 10 – Introduction to Artificial Neural Networks with Keras](https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb) from [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurelien Geron      

```
Suggested Reading:
```

  * Blog: [How to Load Large Datasets From Directories for Deep Learning in Keras](https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/) by Jason Brownlee  
  * Blog: [A Thing You Should Know About Keras if You Plan to Train a Deep Learning Model on a Large Dataset](https://medium.com/difference-engine-ai/keras-a-thing-you-should-know-about-keras-if-you-plan-to-train-a-deep-learning-model-on-a-large-fdd63ce66bd2) by Soumendra P   
  * Blog: [Standardizing on Keras: Guidance on High-level APIs in TensorFlow 2.0](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a) by TensorFlow Team  
    
```
Additional Resources:
``` 
 
  * PDF: [Keras Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Keras_Cheat_Sheet_Python.pdf)
  * Blog: [Technical Notes On Using Data Science & Artificial Intelligence: To Fight For Something That Matters](https://chrisalbon.com/) by Chris Albon (read the Keras section)  
  * Blog: [Keras Tutorial: Develop Your First Neural Network in Python Step-By-Step](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/) by Jason Brownlee  
  * Blog: [How to Use the Keras Functional API for Deep Learning](https://machinelearningmastery.com/keras-functional-api-deep-learning/) by Jason Brownlee  
  * Blog: [Keras Tutorial for Beginners with Python: Deep Learning Example](https://www.guru99.com/keras-tutorial.html)
  * Blog: [Learn Tensorflow 1: The Hello World of Machine Learning](https://codelabs.developers.google.com/codelabs/tensorflow-lab1-helloworld/) by [Google Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)    
  * Blog: [Learn Tensorflow 2: Introduction to Computer Vision (Fashion MNIST)](https://codelabs.developers.google.com/codelabs/tensorflow-lab2-computervision/)  by [Google Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)    
  * Blog: [Properly Setting the Random Seed in ML Experiments. Not as Simple as You Might Imagine](https://medium.com/@ODSC/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752) by [Open Data Science](https://opendatascience.com)    
  * Blog: [Your first Keras Model, with Transfer Learning](https://codelabs.developers.google.com/codelabs/keras-flowers-transfer-learning/) by [Google Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)  
  * Blog & NoteBook: [How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/) by Jason Brownlee  
  * Blog: [TensorFlow 2.0 Tutorial 02: Transfer Learning](https://lambdalabs.com/blog/tensorflow-2-0-tutorial-02-transfer-learning/) by Chuan Li   
 
    
```
Object-Oriented Programming:
``` 
    
   * Blog: [Object-Oriented Programming (OOP) in Python 3](https://realpython.com/python3-object-oriented-programming/) by the Real Python Team   
   * Blog: [Understanding Object-Oriented Programming Through Machine Learning](https://dziganto.github.io/classes/data%20science/linear%20regression/machine%20learning/object-oriented%20programming/python/Understanding-Object-Oriented-Programming-Through-Machine-Learning/) by David Ziganto  
   * Blog: [Object-Oriented Programming for Data Scientists: Build your ML Estimator](https://towardsdatascience.com/object-oriented-programming-for-data-scientists-build-your-ml-estimator-7da416751f64) by Tirthajyoti Sarkar  
   * Blog: [How Objects are Called in Keras](https://adaickalavan.github.io/tensorflow/how-objects-are-called-in-keras/) by Adaickalavan  
      * Blog: [Python Callable Class Method](https://medium.com/@nunenuh/python-callable-class-1df8e122b30c) by Lalu Erfandi Maula Yusnu  

6. ### <a name="Part-3"></a>Toolkit Lab 3: TensorFlow 2.0  
```
Required Reading:
```

  * NoteBook: [Chapter 12 – Custom Models and Training with TensorFlow](https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb) from [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurelien Geron    
  * Slide: [Introducing tf.data:](https://docs.google.com/presentation/d/16kHNtQslt-yuJ3w8GIx-eEH6t_AvFeQOchqGRFpAD7U/edit#slide=id.g254d08e080_0_38) The [tf.data](https://www.tensorflow.org/beta/guide/data) module contains a collection of classes that allows you to easily load data, manipulate it, and pipe it into your model. The slides were prepared by Derek Murray, the creator of tf.data explaining the API (don’t forget to read the speaker notes below the slides). 
  * NoteBook: [Chapter 13 – Loading and Preprocessing Data with TensorFlow](https://github.com/ageron/handson-ml2/blob/master/13_loading_and_preprocessing_data.ipynb) from [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurelien Geron    

```
Suggested Reading:
```

  * Blog: [What’s Coming in TensorFlow 2.0](https://medium.com/tensorflow/whats-coming-in-tensorflow-2-0-d3663832e9b8) by the TensorFlow Team  <br>
  * Blog: [TF.data Reborn from the Ashes](https://medium.com/@prince.canuma/tf-data-reborn-from-the-ashes-5600512c27d6) by Prince Canuma   
  * Blog: [Introducing Ragged Tensors](https://medium.com/tensorflow/introducing-ragged-tensors-ac301c31fd38) by Laurence Moroney <br>
  * Blog: [Load Images with tf.data (A File from a URL, If It is not Already in the Cache)](https://www.tensorflow.org/beta/tutorials/load_data/images)
  * Blog: Analyzing tf.function to Discover AutoGraph Strengths and Subtleties: [Part 1](https://pgaleone.eu/tensorflow/tf.function/2019/03/21/dissecting-tf-function-part-1/), [Part 2](https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/), and [Part 3](https://pgaleone.eu/tensorflow/tf.function/2019/05/10/dissecting-tf-function-part-3/)  by Paolo Galeone   
  * Blog: [TPU-Speed Data Pipelines: tf.data.Dataset and TFRecords](https://codelabs.developers.google.com/codelabs/keras-flowers-data/) by [Google Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)  
  
```
Additional Resources:
```
  * Course: [Introduction to TensorFlow in Python](https://www.datacamp.com/courses/introduction-to-tensorflow-in-python) by DataCamp  
  * Blog: [Tensorflow 2.0: Keras is not (yet) a Simplified Interface to Tensorflow](https://pgaleone.eu/tensorflow/keras/2019/01/19/keras-not-yet-interface-to-tensorflow/) by Paolo Galeone  
  * NoteBook: [TensorFlow 2.0 Quick Start Guide (Chapter 1)](https://github.com/PacktPublishing/Tensorflow-2.0-Quick-Start-Guide/blob/master/Chapter01/Chapter1_TF2_alpha.ipynb) by Tony Holdroyd <br>
  * NoteBook: [TensorFlow 2.0 Examples](https://github.com/aymericdamien/TensorFlow-Examples/tree/master/tensorflow_v2) by Aymeric Damien <br>
  * Blog: [Building a Data Pipeline (Using Tensorflow 1 and tf.data for Text and Images)](http://cs230.stanford.edu/blog/datapipeline/)
  * Blog: [How to Serve Deep Learning Models using TensorFlow 2.0 with Cloud Functions](https://cloud.google.com/blog/products/ai-machine-learning/how-to-serve-deep-learning-models-using-tensorflow-2-0-with-cloud-functions) by Rustem Feyzkhanov  
  * Blog: [Swift](https://swift.org) was announced in 2014. The Swift programming language has quickly become one of the fastest growing languages in history. Swift makes it easy to write software that is incredibly fast and safe by design. 
  * GitHub: [Swift for TensorFlow](https://github.com/tensorflow/swift)  

```
TensorFlow 1.0
```
  * To Learn TensorFlow 1.0, Check the Section of [TensorFlow-Tutorials](https://github.com/hhaji/Deep-Learning/blob/master/TensorFlow-Tutorials/README.md#tensorflow-1). 
     
7. ### <a name="Part-5"></a>Toolkit Lab 5: TensorBoard

```
Required Reading:
```
    
  * Video: [Inside TensorFlow: Summaries and TensorBoard](https://www.youtube.com/watch?v=OI4cskHUslQ)
  * Blog: [TensorBoard Overview](https://www.tensorflow.org/tensorboard/r1/overview)
  * NoteBook: [Get started with TensorBoard](https://github.com/tensorflow/tensorboard/blob/master/docs/r2/get_started.ipynb)
  * NoteBook: [Examining the TensorFlow Graph](https://github.com/tensorflow/tensorboard/blob/master/docs/r2/graphs.ipynb)
  * NoteBook: [Displaying Image Data in TensorBoard](https://github.com/tensorflow/tensorboard/blob/master/docs/r2/image_summaries.ipynb) 
  * NoteBook: [Using TensorBoard in Notebooks](https://github.com/tensorflow/tensorboard/blob/master/docs/r2/tensorboard_in_notebooks.ipynb)
    
```
Additional Reading:
```
  * Blog: [TensorBoard: Graph Visualization](https://www.tensorflow.org/tensorboard/r1/graphs)
  * Blog: [TensorBoard Histogram Dashboard](https://www.tensorflow.org/tensorboard/r1/histograms)
  * Blog: [TensorBoard: Visualizing Learning](https://www.tensorflow.org/tensorboard/r1/summaries)
  * NoteBook: [TensorBoard Scalars: Logging Training Metrics in Keras](https://github.com/tensorflow/tensorboard/blob/master/docs/r2/scalars_and_keras.ipynb)
  * NoteBook: [Hyperparameter Tuning with the HParams Dashboard](https://github.com/tensorflow/tensorboard/blob/master/docs/r2/hyperparameter_tuning_with_hparams.ipynb)
  * NoteBook: [TensorBoard Profile: Profiling basic training metrics in Keras](https://github.com/tensorflow/tensorboard/blob/master/docs/r2/tensorboard_profiling_keras.ipynb)
  * Blog: [TensorFlow 2.0 Tutorial 03: Saving Checkpoints](https://lambdalabs.com/blog/tensorflow-2-0-tutorial-03-saving-checkpoints/) by Chuan Li   


8. ### <a name="RFDL"></a>Regularization for Deep Learning  

```
Required Reading:
```

  * [Chapter 7](http://www.deeplearningbook.org/contents/regularization.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
  * Slide: [Regularization For Deep Models (Lecture 3)](http://wavelab.uwaterloo.ca/wp-content/uploads/2017/04/Lecture_3.pdf) by Ali Harakeh  

```
Additional Reading:
```
    
  * [How to Improve Deep Learning Model Robustness by Adding Noise](https://machinelearningmastery.com/how-to-improve-deep-learning-model-robustness-by-adding-noise/) by Jason Brownlee 
  * Slide: [Regularization for Deep Learning](https://www.deeplearningbook.org/slides/07_regularization.pdf)  by Ian Goodfellow
  * Slides: Regularization for Deep Learning [1](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L13-regularization.pdf)  and [2](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L14-regularization-2.pdf) by U Kang 
  * Blog: [TensorFlow 2.0 Tutorial 04: Early Stopping](https://lambdalabs.com/blog/tensorflow-2-0-tutorial-04-early-stopping/) by Chuan Li   
  * Blog: [Analysis of Dropout](https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/) by Paolo Galeone  
  * Slide: [Training Deep Neural Networks](https://web.cs.hacettepe.edu.tr/~aykut/classes/spring2018/cmp784/slides/lec4-training-deep-nets.pdf) by Aykut Erdem   

9. ### <a name="OFTDM"></a>Optimization for Training Deep Models  

```
Required Reading:
```  

   * [Chapter 8](http://www.deeplearningbook.org/contents/optimization.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br> 
   * Slide: [Optimization for Training Deep Models (Lecture 4)](http://wavelab.uwaterloo.ca/wp-content/uploads/2017/04/Lecture-4-1.pdf) by Ali Harakeh  
   * Slide: [Optimization for Training Deep Models - Algorithms (Lecture 4)](http://wavelab.uwaterloo.ca/wp-content/uploads/2017/04/Lecture_4_2-1.pdf) by Ali Harakeh  
   * Blog: [Neural Network Optimization](https://towardsdatascience.com/neural-network-optimization-7ca72d4db3e0) by Matthew Stewart  
   * Slide: [Batch Normalization in Deep Networks](https://www.learnopencv.com/batch-normalization-in-deep-networks/) by Sunita Nayak  
 
```
Additional Reading:
```

   * Slide: [Training Deep Neural Networks](https://web.cs.hacettepe.edu.tr/~aykut/classes/spring2018/cmp784/slides/lec4-training-deep-nets.pdf) by Aykut Erdem
   * Slide: [Gradient Descent and Structure of Neural Network Cost Functions](https://www.deeplearningbook.org/slides/sgd_and_cost_structure.pdf) by Ian Goodfellow   
These slides describe how gradient descent behaves on different kinds of cost function surfaces. Intuition for the structure of the cost function can be built by examining a second-order Taylor series approximation of the cost function. This quadratic function can give rise to issues such as poor conditioning and saddle points. Visualization of neural network cost functions shows how these and some other geometric features of neural network cost functions affect the performance of gradient descent.
   * Slide: [Tutorial on Optimization for Deep Networks](https://www.deeplearningbook.org/slides/dls_2016.pdf) by Ian Goodfellow    
Ian Goodfellow's presentation at the 2016 Re-Work Deep Learning Summit. Covers Google Brain research on optimization, including visualization of neural network cost functions, Net2Net, and batch normalization.  
   * [Video](https://www.youtube.com/watch?v=Xogn6veSyxA) of lecture / discussion: This video covers a presentation by Ian Goodfellow and group discussion on the end of Chapter 8 and entirety of Chapter 9 at a reading group in San Francisco organized by Taro-Shigenori Chiba. <br>         
   * Slides: Optimization for Training Deep Models [1](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L15-opt.pdf)  and [2](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L16-opt-2.pdf) by U Kang 
   * Blog: [Why Momentum Really Works](https://distill.pub/2017/momentum/) by Gabriel Goh  
   * Blog: [Preconditioning the Network](https://cnl.salk.edu/~schraudo/teach/NNcourse/precond.html) by Nic Schraudolph and Fred Cummins  
   * Blog: [How to Accelerate Learning of Deep Neural Networks With Batch Normalization](https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/) by Jason Brownlee  
   * Slide: [Conjugate Gradient Descent](http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/conjugate_direction_methods.pdf) by Aarti Singh


10. ### <a name="CNN"></a>Convolutional Networks  

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
  
   * NoteBook: [Chapter 14 – Deep Computer Vision Using Convolutional Neural Networks](https://github.com/ageron/handson-ml2/blob/master/14_deep_computer_vision_with_cnns.ipynb) from [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurelien Geron                   
   * [Video](https://www.youtube.com/watch?v=Xogn6veSyxA) of lecture / discussion: This video covers a presentation by Ian Goodfellow and group discussion on the end of Chapter 8 and entirety of Chapter 9 at a reading group in San Francisco organized by Taro-Shigenori Chiba. <br>
   * [A Basic Introduction to Separable Convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) by Chi-Feng Wang <br>
   * [Depth wise Separable Convolutional Neural Networks](https://www.geeksforgeeks.org/depth-wise-separable-convolutional-neural-networks/) by Mayank Chaurasia <br>  
   * Slide: [Convolutional Models](http://www.deeplearningindaba.com/uploads/1/0/2/6/102657286/dl_indaba_2018_convnets.pdf) by Naila Murray <br>
   * Blog: [A Convolutional Neural Network Tutorial in Keras and TensorFlow 2](https://www.machineislearning.com/convolutional-neural-network-keras-tensorflow-2/) by Isak Bosman <br>
   * Slide: [Convolutional Networks](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L11-cnn.pdf) by U Kang <br> 
   * Blog: [An Intuitive Guide to Convolutional Neural Networks](https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/) by  Daphne Cornelisse <br>
   * Paper: [A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/pdf/1603.07285.pdf) by Vincent Dumoulin and Francesco Visin 
   * Blog: [Cats and Dogs Image Classification Using Keras](https://pythonistaplanet.com/image-classification-using-deep-learning/) by Ashwin Joy   
   * Blog: [Learn Tensorflow 3: Introduction to Convolutions](https://codelabs.developers.google.com/codelabs/tensorflow-lab3-convolutions/) by [Google Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)
   * Blog: [Learn Tensorflow 4: Convolutional Neural Networks (CNNs)](https://codelabs.developers.google.com/codelabs/tensorflow-lab4-cnns/) by [Google Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)    
   * Blog: [Learn Tensorflow 5: Complex Images](https://codelabs.developers.google.com/codelabs/tensorflow-lab5-compleximages/) by [Google Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)  
   * Blog: [Learn Tensorflow 6: Use CNNS with Larger Datasets](https://codelabs.developers.google.com/codelabs/tensorflow-lab6-largecnns/) by [Google Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)  
   * Blog: [Convolutional Neural Networks, with Keras and TPUs](https://codelabs.developers.google.com/codelabs/keras-flowers-convnets/) by [Google Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)  
   * Blog: [Modern Convnets, Squeezenet, with Keras and TPUs](https://codelabs.developers.google.com/codelabs/keras-flowers-squeezenet/) by [Google Codelabs](https://codelabs.developers.google.com/?cat=TensorFlow)  
   * Blog: [TensorFlow 2.0 Tutorial 01: Basic Image Classification](https://lambdalabs.com/blog/tensorflow-2-0-tutorial-01-image-classification-basics/) by Chuan Li  


11. ### <a name="SMRARN"></a>Sequence Modeling: Recurrent and Recursive Networks  

```
Required Reading:
```
  
   * [Chapter 10](http://www.deeplearningbook.org/contents/rnn.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
   * Slide: [Sequence Modeling](https://www.deeplearningbook.org/slides/10_rnn.pdf)  by Ian Goodfellow  <br>
A presentation summarizing Chapter 10, based directly on the textbook itself. <br>
   * Slide: [An Introduction to: Reservoir Computing and Echo State Networks](http://didawiki.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/aa2/rnn4-esn.pdf) by Claudio Gallicchio <br>
   * Blog: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah  <br>

```
Additional Reading:
```
  
   * [Video](https://www.youtube.com/watch?v=ZVN14xYm7JA&feature=youtu.be) of lecture / discussion. This video covers a presentation by Ian Goodfellow and a group discussion of Chapter 10 at a reading group in San Francisco organized by Alena Kruchkova. <br>
   * Slide: [Sequence Modeling: Recurrent and Recursive Networks](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L12-rnn.pdf) by U Kang <br> 
   * Blog: [Gentle introduction to Echo State Networks](https://towardsdatascience.com/gentle-introduction-to-echo-state-networks-af99e5373c68) by Madalina Ciortan  <br>
   * Blog: [Understanding GRU Networks](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be) by Simeon Kostadinov  <br>
   * Blog: [Animated RNN, LSTM and GRU](https://towardsdatascience.com/animated-rnn-lstm-and-gru-ef124d06cf45) by Raimi Karim <br>
    
12. ### <a name="Practical-Methodology"></a>Practical Methodology  

```
Required Reading:
```
    
   * [Chapter 11](http://www.deeplearningbook.org/contents/guidelines.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
   * Slide: [Practical Methodology](http://www.deeplearningbook.org/slides/11_practical.pdf)  by Ian Goodfellow  <br>
    
```
Additional Reading:
```

   * Slide: [Practical Methodology](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L17-practical-method.pdf) by U Kang <br> 

    
13. ### <a name="Applications"></a>Applications 

```
Required Reading:
```
   * [Chapter 12](http://www.deeplearningbook.org/contents/applications.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
   * Slide: [Applications](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L18-applications.pdf) by U Kang <br> 
    
```
Additional Reading:
```
   * Blog: [How Neural Networks Learn Distributed Representations](https://www.oreilly.com/ideas/how-neural-networks-learn-distributed-representations) By Garrett Hoffman <br>
   * Blog: [Top 15 Deep Learning Applications that will Rule the World in 2018 and Beyond](https://medium.com/breathe-publication/top-15-deep-learning-applications-that-will-rule-the-world-in-2018-and-beyond-7c6130c43b01) by Vartul Mittal <br> 

    
14. ### <a name="Autoencoders"></a>Autoencoders

```
Required Reading:
```
   * [Chapter 14](http://www.deeplearningbook.org/contents/autoencoders.html) of the [Deep Learning](http://www.deeplearningbook.org) textbook. <br>
   * Slide: [Autoencoders](http://www.deeplearningbook.org/slides/14_autoencoders.pdf)  by Ian Goodfellow  <br>

    
```
Additional Reading:
```
    
   * Slide: [Autoencoders](https://datalab.snu.ac.kr/~ukang/courses/17S-DL/L19-autoencoder.pdf) by U Kang <br> 
   * Blog: [Tutorial - What is a Variational Autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) by Jaan Altosaar <br>
    
    
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
