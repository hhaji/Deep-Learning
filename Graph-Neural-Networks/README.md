# Graph Neural Networks  
Graph Neural Networks have received increasing attentions due to their superior performance in many node and graph classification tasks. 

### **Index:**
- [Graph Neural Networks](#Blogs)
  - [Applications and Limitations of Graph Neural Networks](#ALGNN)
  - [Video](#Video)
  - [Survey](#Survey)
  - [Graph Convolutionl Networks](#GCN)
  - [Graph Auto-Encoders](#GAE)
  - [Deep Belief Nets](#DBN)
- [Graph Represetation Learning](#GRL)
- [Courses](#Courses) 
- [Books](#Books)
- [Graph Neural Networks Libraries](#GNNL) 
  - [Deep Graph Library (DGL)](#DGL) 
  - [Node Classification](#NC) 
  - [Graph Classification](#GC) 
  - [Graph Nets Library](#GNL) 
  - [More Libraries](#ML) 
  - [Save & Load Graphs](#SLG) 
- [Datasets of Graphs](#DG) 
- [Tools for Creating Graphs](#TCG) 
- [Molecular Structure Analysis](#MSA) 
- [Graph Machine Learning and its Application on Molecular Science](#ML-MS)
- [Chemical Notations](#Notations) 
- [An Introduction to Basic Chemistry and Drugs](#Chemistry) 
   - Chemistry
   - Drug
- [Datasets of Molecules and Their Properties](#DMTP)  
   - Chemical Datasets
   - Biological Datasets 
- [Libraries & Packages](#Libraries) 
- [Online Softwares](#Online-Softwares) 
- [Softwares (Draw a Molecule)](#Draw-Molecule) 

## <a name="Blogs"></a>Graph Neural Networks    
- Blog: [Awesome Resources on Graph Neural Networks](https://github.com/nnzhan/Awesome-Graph-Neural-Networks) by Zonghan Wu. This is a collection of resources related with graph neural networks.  
- Blog: [Deep Learning on Graphs: Successes, Challenges, and Next Steps](https://towardsdatascience.com/deep-learning-on-graphs-successes-challenges-and-next-steps-7d9ec220ba8) by Michael Bronstein   
- Blog: [Graph Convolutionl Networks](http://tkipf.github.io/graph-convolutional-networks/) by Thomas Kipf  
- Blog: [Graph Convolutional Networks I](https://atcold.github.io/pytorch-Deep-Learning/en/week13/13-1/) by  Xavier Bresson  
- Blog: [Graph Convolutional Networks II](https://atcold.github.io/pytorch-Deep-Learning/en/week13/13-2/) by Xavier Bresson   
- Blog: [Graph Convolutional Networks III](https://atcold.github.io/pytorch-Deep-Learning/en/week13/13-3/) by  Alfredo Canziani  
- Blog: [Emotion Recognition Using Graph Convolutional Networks](https://towardsdatascience.com/emotion-recognition-using-graph-convolutional-networks-9f22f04b244e) by Kevin Shen    
- Blog & NoteBook: [Graph Convolutional Network](https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/1_gcn.html) by Qi Huang, Minjie Wang, Yu Gai, Quan Gan, and Zheng Zhang  
- Blog: [Deep Learning on Graphs (a Tutorial)](https://cloud4scieng.org/2020/08/28/deep-learning-on-graphs-a-tutorial/) by Gannon  
- Blog: [Graph Neural Networks and its Variants](https://docs.dgl.ai/en/0.4.x/tutorials/models/)   
- Blog: [Graph Neural Networks and Recommendations](https://github.com/yazdotai/graph-networks) by Yazdotai
- Blog: [Must-Read Papers on Graph Neural Networks (GNN)](https://github.com/thunlp/GNNPapers) contributed by Jie Zhou, Ganqu Cui, Zhengyan Zhang and Yushi Bai. 
- Blog: [A Gentle Introduction to Graph Neural Networks (Basics, DeepWalk, and GraphSage)](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3) by Steeve Huang  
- Blog: [Deep Learning with Knowledge Graphs](https://medium.com/octavian-ai/deep-learning-with-knowledge-graphs-3df0b469a61a) 
- Slide: [Graph Neural Networks: Models and Applications](http://cse.msu.edu/~mayao4/tutorials/aaai2020/) by Yao Ma, Wei Jin, Jiliang Tang, Lingfei Wu, and Tengfei Ma   

### <a name="ALGNN"></a>Applications and Limitations of Graph Neural Networks
- Blog: [Applications of Graph Neural Networks](https://towardsdatascience.com/https-medium-com-aishwaryajadhav-applications-of-graph-neural-networks-1420576be574) by Aishwarya Jadhav  
- Blog: [Exciting Applications of Graph Neural Networks](https://blog.fastforwardlabs.com/2019/10/30/exciting-applications-of-graph-neural-networks.html) by Keita  
- Blog: [Can Graph Neural Networks Solve Real-World Problems?](https://hackernoon.com/can-graph-neural-networks-solve-real-world-problems-7hd636dn) by Prince Canuma  
- Blog: [Limitations of Graph Neural Networks](https://towardsdatascience.com/limitations-of-graph-neural-networks-2412fffe677) by Sergei Ivanov   

### <a name="Video"></a>Video  
- Video: [Graph Neural Networks: Variations and Applications](https://www.youtube.com/watch?v=cWIeTMklzNg) 

### <a name="Survey"></a>Survey   
- Paper: [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596.pdf) by Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu  
- Paper: [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/pdf/1806.01261.pdf)
- Paper: [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf) by 
Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, Maosong Sun  
- Paper: [Attention Models in Graphs: A Survey](https://dl.acm.org/doi/10.1145/3363574) by John Boaz Lee, Ryan A Rossi, Sungchul Kim, Nesreen K Ahmed, and Eunyee Koh   

### <a name="GAE"></a>Graph Auto-Encoders        
- Blog: [Tutorial on Variational Graph Auto-Encoders](https://towardsdatascience.com/tutorial-on-variational-graph-auto-encoders-da9333281129) by Fanghao Han  

### <a name="DBN"></a>Deep Belief Nets       
- Blog: [Deep Learning meets Physics: Restricted Boltzmann Machines Part I](https://towardsdatascience.com/deep-learning-meets-physics-restricted-boltzmann-machines-part-i-6df5c4918c15) by Artem Oppermann    
- Slide: [Deep Belief Nets](https://www.cs.toronto.edu/~hinton/nipstutorial/nipstut3.pdf) by Geoffrey Hinton 

## <a name="GRL"></a>Graph Represetation Learning         
- Tutorial: [Representation Learning on Networks](http://snap.stanford.edu/proj/embeddings-www/index.html) by Jure Leskovec      
- Papers: [Papers with Code](https://paperswithcode.com/task/graph-representation-learning)
- Blog: [Graph Represetation Learning](https://towardsdatascience.com/graph-representation-learning-dd64106c9763) by Marco Brambilla   
- Slide: [Graph Represetation Learning](https://jian-tang.com/files/AAAI19/aaai-grltutorial-part0-intro.pdf) by 
William L. Hamilton and Jian Tang   
- Survey: [Representation Learning on Graphs: Methods and Applications](https://www-cs.stanford.edu/people/jure/pubs/graphrepresentation-ieee17.pdf) by William L. Hamilton, Rex Ying, and Jure Leskovec   
- Video: [Graph Representation Learning (Stanford university)](https://www.youtube.com/watch?v=YrhBZUtgG4E) by Jure Leskovec  
- Thesis: [Graph Representation Learning and Graph Classification](https://www.cs.uoregon.edu/Reports/AREA-201706-Riazi.pdf) by Sara Riazi   
- NeurIPS 2019 Workshop ([Graph Represetation Learning](https://grlearning.github.io)): [Open Problems and Challenges](https://grlearning.github.io/papers/)   

## <a name="Courses"></a>Courses       
- Blog: [Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/index.html#schedule) by Jure Leskovec  
- Blog: [Graph Represetation Learning](https://cs.mcgill.ca/~wlh/comp766/index.html) by William L. Hamilton   
- Blog: [Graph Neural Networks](https://gnn.seas.upenn.edu) by Alejandro Ribeiro   

## <a name="Books"></a>Books       
- Blog: [Introduction to Graph Neural Networks](https://www.amazon.com/Introduction-Networks-Synthesis-Artificial-Intelligence-ebook/dp/B087LJJNJK) by Zhiyuan Liu and Jie Zhou   
- Blog: [Graph Representation Learning](https://www.morganclaypoolpublishers.com/catalog_Orig/product_info.php?products_id=1576) by William L. Hamilton   
    - [The Pre-Publication](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)

## <a name="GNNL"></a>Graph Neural Networks Libraries       
### <a name="DGL"></a>Deep Graph Library (DGL)   
A Python package that interfaces between existing tensor libraries and data being expressed as graphs.  
- Library: [Deep Graph Library (DGL)](https://www.dgl.ai) 
    * Install: [DGL](https://docs.dgl.ai/install/index.html)
- Paper: [Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs](https://rlgm.github.io/papers/49.pdf) by Minjie Wang, Lingfan Yu, Da Zheng, Quan Gan, Yu Gai, Zihao Ye, Mufei Li, Jinjing Zhou, Qi Huang, Chao Ma, Ziyue Huang, Qipeng Guo, Hao Zhang, Haibin Lin, Junbo Zhao, Jinyang Li, Alexander Smola, and Zheng Zhang  
- Blog: [Hands-on Graph Neural Networks with PyTorch & PyTorch Geometric](https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8) by Huang Kung-Hsiang   
- Blog: [DGL Walkthrough 01: Data](https://xinhaoli74.github.io/posts/2019/12/DGL-Basic01-Data/) by Xinhao Li  
- Blog: [When Kernel Fusion Meets Graph Neural Networks](https://www.dgl.ai/blog/2019/05/04/kernel.html) By Minjie Wang, Lingfan Yu, Jake Zhao, Jinyang Li, Zheng Zhang  
- Blog: [Built-in Message Passing Functions](https://docs.dgl.ai/features/builtin.html)  

### <a name="NC"></a>Node Classification     
- Blog: [DGL at a Glance](https://docs.dgl.ai/tutorials/basics/1_first.html) by  Minjie Wang, Quan Gan, Jake Zhao, Zheng Zhang  

### <a name="GC"></a>Graph Classification    
- [Tutorial: Batched Graph Classification with DGL](https://docs.dgl.ai/tutorials/basics/4_batch.html)

### <a name="GNL"></a>Graph Nets Library   
A DeepMind's library for building graph networks in Tensorflow and Sonnet.  
- Blog: [Graph Nets Library](https://github.com/deepmind/graph_nets) 
- Jupyter NoteBook: [Tutorial of the Graph Nets Library](https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/graph_nets_basics.ipynb)   

### <a name="ML"></a>More Libraries   
- Blog: [StellarGraph Machine Learning Library:](https://github.com/stellargraph/stellargraph) a Python library for machine learning on graph-structured (or equivalently, network-structured) data. 
- Blog: [PyTorch Geometric (PyG)](https://github.com/rusty1s/pytorch_geometric) by Matthias Fey. PyTorch Geometric is a geometric deep learning extension library for PyTorch. 

### <a name="SLG"></a>Save & Load Graphs       
- Blog: [Reading and Writing Graphs](https://networkx.github.io/documentation/stable/reference/readwrite/index.html)

## <a name="DG"></a>Datasets of Graphs       

- Blog: [Open Graph Benchmark](https://ogb.stanford.edu/) is a collection of realistic, large-scale, and diverse benchmark datasets for machine learning on graphs.  
- Blog: [Network Repository. An Interactive Scientific Network Data Repository:](http://networkrepository.com) The first interactive data and network data repository with real-time visual analytics. Network repository is not only the first interactive repository, but also the largest network repository with thousands of donations in 30+ domains (from biological to social network data). This repository was made by Ryan A. Rossi and Nesreen K. Ahmed.  
- Blog: [Graph Classification:](https://paperswithcode.com/task/graph-classification/latest) The mission of Papers With Code is to create a free and open resource with Machine Learning papers, code and evaluation tables.  
- Blog: [Graph Challenge Data Sets:](https://graphchallenge.mit.edu/data-sets) Amazon is making the Graph Challenge data sets available to the community free of charge as part of the AWS Public Data Sets program. The data is being presented in several file formats, and there are a variety of ways to access it.    
- Blog: [The House of Graphs:](https://hog.grinvin.org) a database of interesting graphs by G. Brinkmann, K. Coolsaet, J. Goedgebeur, and H. Mélot (also see Discrete Applied Mathematics, 161(1-2): 311-314, 2013 ([DOI](http://dx.doi.org/10.1016/j.dam.2012.07.018))).   
    * [Search for Graphs](https://hog.grinvin.org/StartSearch.action)  
- Blog: [A Repository of Benchmark Graph Datasets for Graph Classification](https://github.com/shiruipan/graph_datasets) by 
Shiruipan    
- Blog: [Collection and Streaming of Graph Datasets](https://www.eecs.wsu.edu/~yyao/StreamingGraphs.html) by Yibo Yao   
- Blog: [Big Graph Data Sets](https://lgylym.github.io/big-graph/dataset.html) by Yongming Luo   
- Blog: [MIVIA LDGraphs Dataset:](https://mivia.unisa.it/datasets/graph-database/mivia2-graph-database/) The MIVIA LDGraphs (MIVIA Large Dense Graphs) dataset is a new dataset for benchmarking exact graph matching algorithms. It aims to extend the MIVIA graphs dataset, widely used in the last ten years, with bigger and more dense graphs, so as to face with the problems nowadays encountered in real applications devoted for instance to bioinformatics and social network analysis. 
- Blog: [Datasets](https://sites.wustl.edu/neumann/research/datasets/) by Marion Neumann  
- Blog: [Graph Dataset](https://sites.google.com/site/xiaomengsite/research/resources/graph-dataset) by Xiao Meng   
- Blog: [Constructors and Databases of Graphs in Sage](http://doc.sagemath.org/html/en/reference/graphs/index.html)
- Datasets in GitHub:   
  - [Benchmark Dataset for Graph Classification:](https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification) This repository contains datasets to quickly test graph classification algorithms, such as Graph Kernels and Graph Neural Networks by Filippo Bianchi.     
  - [GAM:](https://github.com/benedekrozemberczki/GAM) A PyTorch implementation of "Graph Classification Using Structural Attention" (KDD 2018) by Benedek Rozemberczki.
  - [CapsGNN:](https://github.com/benedekrozemberczki/CapsGNN) A PyTorch implementation of "Capsule Graph Neural Network" (ICLR 2019) by Benedek Rozemberczki.   

### <a name="TCG"></a>Tools for Creating Graphs  

- Package: [Networkx:](https://networkx.github.io) a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.  
  - [Graph Generators](https://networkx.github.io/documentation/stable/reference/generators.html)  
  - [Converting to and from Other Data Formats To NetworkX Graph](https://networkx.github.io/documentation/stable/reference/convert.html)  
  - [Reading and Writing Graphs](https://networkx.github.io/documentation/stable/reference/readwrite/index.html)    
  
- Package: [Sage:](https://www.sagemath.org) a viable free open source alternative to Magma, Maple, Mathematica and Matlab.
  - [CoCalc:](https://www.sagemath.org/notebook-vs-cloud.html) an [online service](https://cocalc.com/) for running SageMath computations online to avoid your own installation of Sage. CoCalc will allow you to work with multiple persistent worksheets in Sage, IPython, LaTeX, and much, much more!
  - [Graph Theory in Sage](http://doc.sagemath.org/html/en/reference/graphs/index.html)  
  
## <a name="MSA"></a>Molecular Structure Analysis   
Molecular structures have graph structures. By using machine learning and in particular GNNs, one can predict some physical, chemical, biochemical properties of a molecule by it's chemical formula. Also, it is possible to predict new formula and 3D structure for an unknown yet molecule or substance with certain desired properties. 

### <a name="ML-MS"></a>Graph Machine Learning and its Application on Molecular Science  


- Book: [Deep Learning for the Life Sciences: Applying Deep Learning to Genomics, Microscopy, Drug Discovery, and More](https://www.amazon.com/Deep-Learning-Life-Sciences-Microscopy/dp/1492039837) by Bharath Ramsundar, Peter Eastman, Patrick Walters, and Vijay Pande  
    - [Codes](https://github.com/deepchem/DeepLearningLifeSciences)  
- Blog: [Machine Learning for Drug Development (Tutorial at the 29th International Joint Conference on Artificial Intelligence (IJCAI))](https://zitniklab.hms.harvard.edu/drugml/) 
- Blog: [Graph Neural Networks for Binding Affinity Prediction](https://levelup.gitconnected.com/graph-neural-networks-for-binding-affinity-prediction-f057c495ad15) by Alex Gurbych   
- Blog: [Generating Molecules with the Help of Recurrent Neural Networks](https://towardsdatascience.com/generating-molecules-with-the-help-of-recurrent-neural-networks-c3fe23bd0de2) by Seyone Chithrananda    
- Blog: [How to Use Machine Learning for Drug Discovery](https://towardsdatascience.com/how-to-use-machine-learning-for-drug-discovery-1ccb5fdf81ad) by Chanin Nantasenamat   
- Blog: [Tutorial ML In Chemistry Research. RDkit & mol2vec](https://www.kaggle.com/vladislavkisin/tutorial-ml-in-chemistry-research-rdkit-mol2vec/data) by Vlad Kisin  
- Blog: [Papers in Drug Discovery](https://paperswithcode.com/task/drug-discovery)  
- Blog: [Review: Deep Learning In Drug Discovery](https://towardsdatascience.com/review-deep-learning-in-drug-discovery-f4c89e3321e1) by Hosein Fooladi   
- Blog: [A Practical Introduction to the Use of Molecular Fingerprints in Drug Discovery](https://towardsdatascience.com/a-practical-introduction-to-the-use-of-molecular-fingerprints-in-drug-discovery-7f15021be2b1) by Laksh  
- Blog: [Public Coronavirus Prediction Models](https://www.aicures.mit.edu/post/public-coronavirus-prediction-models)  
- Blog: [DIY Drug Discovery - Using Molecular Fingerprints and Machine Learning for Solubility Prediction](http://www.moreisdifferent.com/2017/9/21/DIY-Drug-Discovery-using-molecular-fingerprints-and-machine-learning-for-solubility-prediction/) by Daniel C. Elton  
- Slide: [Graph Neural Network and its Application on Molecular Science](https://tykimos.github.io/warehouse/2018-6-28-ISS_1st_Deep_Learning_Together_rsy_file.pdf) by Seongok Ryu   
- GitHub:
    - [Graph Neural Network (GNN) for Molecular Property Prediction (SMILES format)](https://github.com/masashitsubaki/molecularGNN_smiles) by Masashi Tsubaki   
- Competition: [Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling/overview)    
- Competition: [Fighting Secondary Effects of Covid](https://www.aicures.mit.edu/tasks) COVID-19 presents many health challenges beyond the virus itself. One of them is finding effective antibiotics for secondary infections.   
- Blog: [Machine Learning Formulation](https://www.aicures.mit.edu/post/machine-learning-formulation) 
- Blog: [How to Fight COVID-19 with Machine Learning](https://towardsdatascience.com/fight-covid-19-with-machine-learning-1d1106192d84) by Markus Schmitt  
- Paper: [MoleculeNet: A Benchmark for Molecular Machine Learning](https://arxiv.org/pdf/1703.00564.pdf) by Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, Vijay Pande  


### <a name="Notations"></a>Chemical Notations     
- Blog: [SMILES Tutorial](https://archive.epa.gov/med/med_archive_03/web/html/smiles.html) Simplified Molecular Input Line Entry System (SMILES) is a chemical notation that allows a user to represent a chemical structure in a way that can be used by the computer. SMILES is an easily learned and flexible notation. The SMILES notation requires that you learn a handful of rules.
- Blog: [SMILES - A Simplified Chemical Language](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)  
    - [Try the JSME](https://peter-ertl.com/jsme/JSME_2017-02-26/JSME.html)    
- Blog: [Smiles](https://docs.chemaxon.com/display/docs/smiles.md)
- Blog: [OpenSMILES](http://opensmiles.org/) is a community sponsored open-standards version of the SMILES language for chemistry. OpenSMILES is part of the [Blue Obelisk](https://blueobelisk.github.io/) community. 
- Blog: [SMARTS - A Language for Describing Molecular Patterns](https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html)   
- Blog: [SMIRKS - A Reaction Transform Language](https://www.daylight.com/dayhtml/doc/theory/theory.smirks.html)  
- Slide: [Rolling Smarts: You don’t Always Find What You Want, But if You Try Sometimes, You Find What You Need](https://www.nextmovesoftware.com/talks/Sayle_RollingSMARTS_RDKitUGM_202010.pdf) by Roger Sayle and John Mayfield    

### <a name="Chemistry"></a>An Introduction to Basic Chemistry and Drugs  

#### Chemistry
- Blog: [Chemistry Library](https://www.khanacademy.org/science/chemistry)  provides informations about the following subjects: Atoms, compounds, and ions, Molecular composition, Mass spectrometry, Chemical reactions and stoichiometry, Electronic structure of atoms, Periodic table, Chemical bonds, Gases and kinetic molecular theory, States of matter and intermolecular forces, Chemical equilibrium, Acids and bases, Buffers, titrations, and solubility equilibria, Thermodynamics, Redox reactions and electrochemistry, Kinetics, Alkanes, cycloalkanes, and functional groups.  
- Blog: [Functional Group Names, Properties, and Reactions](https://courses.lumenlearning.com/boundless-chemistry/chapter/functional-group-names-properties-and-reactions/)
- Simulations  by ChemThink  
    - [The Particulate Nature of Matter](https://simbucket.com/chemthinkserver/chemthink/index.html?pn)
    - [Atomic Structure](https://simbucket.com/chemthinkserver/chemthink/index.html?as)   
    - [Covalent Bonding Tutorial](https://pbslm-contrib.s3.amazonaws.com/WGBH/arct15/SimBucket/Simulations/chemthink-covalentbonding/content/index.html)   
    - [Ions](https://simbucket.com/chemthinkserver/chemthink/index.html?io)  
    - [Ionic Bonding](https://simbucket.com/chemthinkserver/chemthink/index.html?ib)  
    - [Ionic Formulas](https://simbucket.com/chemthinkserver/chemthink/index.html?if)  
    - [Molecular Shapes](https://simbucket.com/chemthinkserver/chemthink/index.html?ms)
    - [Isotopes](https://www.simbucket.com/chemthinkserver/chemthink/index.html?is)
- Blog: [Periodic Table of Elements](https://pubchem.ncbi.nlm.nih.gov/periodic-table/) by [PubChem](https://pubchem.ncbi.nlm.nih.gov/)  
- Blog: [Periodic Table of Elements](https://ptable.com/#Properties) by Michael Dayah   
- Blog: [Ionic Bond Examples](https://examples.yourdictionary.com/ionic-bond-examples.html)   
- Blog: [A Comprehensive Treatment of Aromaticity in the SMILES Language](https://depth-first.com/articles/2020/02/10/a-comprehensive-treatment-of-aromaticity-in-the-smiles-language/)   

#### Drug
- Blog: [Chiral Drugs](https://www.khanacademy.org/test-prep/mcat/chemical-processes/stereochemistry/a/chiral-drugs)   
- Blog: [Drug Lipophilicity and Absorption: The Continuous Challenge in Drug Discovery](https://emerypharma.com/blog/drug-lipophilicity-and-absorption-a-continuous-challenge-toward-the-goal-of-drug-discovery/) by Emery Pharma     
    
## <a name="DMTP"></a>Datasets of Molecules and Their Properties  

### Chemical Datasets
- Blog: [PubChem Dataset](https://pubchemdocs.ncbi.nlm.nih.gov/downloads) is an open chemistry database at the National Institutes of Health (NIH). “Open” means that you can put your scientific data in PubChem and that others may use it.  
    - Blog: [Browse PubChem Data](https://pubchem.ncbi.nlm.nih.gov/classification/#hid=1) using a classification of interest, or search for PubChem records annotated with the desired classification/term  
- Blog: [GDB Databases](https://gdb.unibe.ch/downloads/) GDB-11 enumerates small organic molecules up to 11 atoms of C, N, O and F following simple chemical stability and synthetic feasibility rules. GDB-13 enumerates small organic molecules up to 13 atoms of C, N, O, S and Cl following simple chemical stability and synthetic feasibility rules. With 977 468 314 structures, GDB-13 is the largest publicly available small organic molecule database to date.   
- Blog: [MoleculeNet](http://moleculenet.ai/) is a benchmark specially designed for testing machine learning methods of molecular properties. As we aim to facilitate the development of molecular machine learning method, this work curates a number of dataset collections, creates a suite of software that implements many known featurizations and previously proposed algorithms. All methods and datasets are integrated as parts of the open source **DeepChem** package(MIT license).  
- Blog: [ChEMBL](https://www.ebi.ac.uk/chembl/) is a manually curated database of bioactive molecules with drug-like properties. It brings together chemical, bioactivity and genomic data to aid the translation of genomic information into effective new drugs.
- Blog: [Open Data Repositories by the Blue Obelisk](https://blueobelisk.github.io/opendata.html) Open Data allows access to large data bases. A good example in chemoinformatics is the NMRShiftDB: the spectra and structures in this database can be downloaded for free, and you have the right to redistribute them.  
- Blog: [Tox21](http://bioinf.jku.at/research/DeepTox/tox21.html): The 2014 Tox21 data challenge was designed to help scientists understand the potential of the chemicals and compounds being tested through the Toxicology in the 21st Century initiative to disrupt biological pathways in ways that may result in toxic effects. The [Tox21 Program (Toxicology in the 21st Century)](https://tripod.nih.gov/tox21/challenge/data.jsp) is an ongoing collaboration among federal agencies to characterize the potential toxicity of chemicals using cells and isolated molecular targets instead of laboratory animals.  
- Blog: [ZINC15:](http://zinc15.docking.org/) a free database of commercially-available compounds for virtual screening. ZINC contains over 230 million purchasable compounds in ready-to-dock, 3D formats. ZINC also contains over 750 million purchasable compounds you can search for analogs in under a minute.  

### Biological Datasets
- Blog: [Therapeutics Data Commons (Machine Learning Datasets and Tasks for Therapeutics)](https://tdcommons.ai/)   
- Blog: [KEGG: Kyoto Encyclopedia of Genes and Genomes](https://www.genome.jp/kegg/)
- Blog: [Drug Repositioning Database (repoDB)](http://apps.chiragjpgroup.org/repoDB/) contains a standard set of drug repositioning successes and failures that can be used to fairly and reproducibly benchmark computational repositioning methods. repoDB data was extracted from DrugCentral and ClinicalTrials.gov.
- Blog: [STRING (Protein-Protein Interaction Networks)](https://string-db.org/) In molecular biology, STRING (Search Tool for the Retrieval of Interacting Genes/Proteins) is a biological database and web resource of known and predicted protein–protein interactions.

### <a name="Libraries"></a>Libraries & Packages  
- Package: RDKit is an open source toolkit for cheminformatics    
    - PDF: [RDKit Documentation](https://buildmedia.readthedocs.org/media/pdf/rdkit/latest/rdkit.pdf)   
    - Blog: [RDKit Cookbook](https://www.rdkit.org/docs/Cookbook.html) by Greg Landrum and Vincent Scalfani   
    - Blog: [RDKit: Simple File Input and Output](https://medium.com/@camkirk/rdkit-simple-file-input-and-output-e6764fc1e35c) by Cam Kirk 
    - Blog: [Getting Started with the RDKit in Python](https://www.rdkit.org/docs/GettingStartedInPython.html) by Greg Landrum   
- Package: [DeepChem](https://deepchem.io/) project aims to create high quality, open source tools for drug discovery, materials science, quantum chemistry, and biology.
    - [Tutorials and Codes:](https://deepchem.io/docs/notebooks/index.html) These tutorials show off various aspects or capabilities of DeepChem. They can be run interactively in Jupyter (IPython) notebook.  
- [OCHEM:](https://ochem.eu/home/show.do) The Online Chemical Modeling Environment is a web-based platform that aims to automate and simplify the typical steps required for QSAR modeling. The platform consists of two major subsystems: the database of experimental measurements and the modeling framework. A user-contributed database contains a set of tools for easy input, search and modification of thousands of records. OCHEM was created  by Iurii Sushko.   
- Blog: [Mol2vec:](https://github.com/samoturk/mol2vec) An unsupervised machine learning approach to learn vector   representations of molecular substructures  
- Package: [Chemprop](https://github.com/chemprop/chemprop)
- Package: [DGL-LifeSci](https://lifesci.dgl.ai/index.html) is a python package for applying graph neural networks to various tasks in chemistry and biology, on top of PyTorch and DGL.  
    - Code: [Property Prediction](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/moleculenet)  
- Package: [Mordred Descriptor:](https://github.com/mordred-descriptor/mordred) a molecular descriptor calculator  
- Knowledge Graph: [Drug Repurposing Knowledge Graph (DRKG)](https://github.com/gnn4dr/DRKG/) 
- Platform: [Cytoscape](https://cytoscape.org/)  is an open source bioinformatics software platform for visualizing molecular interaction networks and integrating with gene expression profiles and other state data.    

### <a name="Online-Softwares"></a>Online Softwares   
- Blog: [Chemprop — Machine Learning for Molecular Property Prediction](http://chemprop.csail.mit.edu) This website can be used to predict molecular properties using a Message Passing Neural Network (MPNN).
- Blog: [JSME Molecule Editor:](https://peter-ertl.com/jsme/) JSME is a free molecule editor written in JavaScript. JSME is a direct successor of the JME Molecule Editor applet. JSME supports drawing and editing of molecules. JSME was created  by Peter Ertl and Bruno Bienfait.  
Here are some softwares to calculate molecular descriptors and fingerprints online:   
- Blog: [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) 
- Blog: [RDkit Descriptors Generator](http://www.pirika.com/Program/2019/RDKit18.html) by Hiroshi Yamamoto   

### <a name="Draw-Molecule"></a>Softwares (Draw a Molecule)   
 - Blog: [Marvin JS](https://marvinjs-demo.chemaxon.com/latest/demo.html) provides quick and convenient ways to draw and modify standard and advanced chemical structures. It's seamlessly integrated into third-party web-based applications, and runs smoothly on all major browsers. 
 - Blog: [MolView](https://molview.org/)  is an intuitive, Open-Source web-application to make science and education more awesome! MolView is mainly intended as web-based data visualization platform.
 - Blog: [Draw Structue](https://pubchem.ncbi.nlm.nih.gov/#draw=true) by [PubChem](https://pubchem.ncbi.nlm.nih.gov/)
 - [Simulations:](https://phet.colorado.edu/en/simulations/filter?subjects=chemistry&sort=alpha&view=grid) [Build an Atom](https://phet.colorado.edu/sims/html/build-an-atom/latest/build-an-atom_en.html), [Build a Molecule](https://phet.colorado.edu/sims/html/build-a-molecule/latest/build-a-molecule_en.html), [Molecule Shapes: Basics](https://phet.colorado.edu/en/simulation/molecule-shapes-basics), [Molecule Shapes](https://phet.colorado.edu/en/simulation/molecule-shapes), [Isotopes and Atomic Mass](https://phet.colorado.edu/en/simulation/isotopes-and-atomic-mass), [Molecule Polarity](https://phet.colorado.edu/en/simulation/molecule-polarity)    
