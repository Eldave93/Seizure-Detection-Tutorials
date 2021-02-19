# Seizure Detection Tutorials

This series of notebooks, which are currently under development, demonstrate the application of signal processing and machine learning classification to epileptic seizure detection.

Currently four open-source datasets are used: 

1. The Epileptologie Database<sup>18</sup> 

2. UPenn and Mayo Clinic's Seizure Detection Challenge<sup>21</sup> 

3. CHB-MIT Scalp EEG Database<sup>19</sup>

4. NEDC TUH EEG Seizure corpus<sup>26</sup>

Other databases exist but have their limitations:

- The European Epilepsy Database (epilepsy-database.eu)
    - Big, Well documented
    - €3000/6000 3 year licence

- IEEG (https://www.ieeg.org/)
    - Open-source
    - Hard to navigate
    
- LONDI (https://ida.loni.usc.edu)
    - Has updated projects
    - Permission for access
    
**Topics covered**

- Biosignal Feature Extraction 
- Feature Pre-Processing
- Supervised Classification
- Model Evaluation
- Hyperparameter Tuning (Gridsearch, Random Search, Bayesian Optimization)
- Ensemble Learning
- Dimensionality Reduction
- Batch Learning
- Multilayer Perceptrons
- Convolutional Neural Networks
- Recurrent Neural Networks

I'd like to give a massive thank you to the open source community for all the hard work that is put into the Python language, interactive computing resourses, and packages used to make this project a reality. Also thank you to Hvass Laboratories (http://www.hvass-labs.org/) whos Tutorials on TensorFlow were the inspiration for creating my own tutorials.

**Progress Update**
- 19/02/2021: I've finally come to the realisation that I don't have a much time as I'd like to work on these, so as a number of people have emailed me about the code to create the dataframes I use I've uploaded my draft version now. I'd love to spend more time tidying it up and improving it, and maybe I'll get chance to in the future, but right now consider it more of a work in progress (like the rest of the tutorials are). If you plan on using it for more than just the tutorials, just go though it carefully - there are likely mistakes due to the lack of time. If your able to fix them then great, let me know. Also please cite my work if you find things useful in here, I'm still pretty early career and it would help a lot (and justify me spending more time working on them in the future). If you like this tutorial then you may be interested in some of the lecture materials I am creating for a masters course "Machine Learning in Python" at Edinburgh University. I'll be aiming to make the materials open source when possible (like https://open.ed.ac.uk/introduction-to-data-science/) and will leave a message on here when available.
- 30/10/2020: I ended up finding a job which started at the end of August! I joined the University of Edinburgh as a University Teacher in Mathematical Sciences Computing (https://www.maths.ed.ac.uk/school-of-mathematics/people/a-z?person=855)! So my work on this project has slowed right down so I can focus on teaching and developing courses there (one of which has gone open source: https://open.ed.ac.uk/introduction-to-data-science/). My aim is to get back to working on this some time next year to polish it up and see if I can tie it to my teaching at Edinburgh. I still need to submit my research papers to journals so more code which is connected to these tutorials will come out when that happens.

## Getting Started

Due to the size of these notebooks you may need to use [nbviewer](https://nbviewer.jupyter.org/) to view the notebooks. To do this all you need to do is copy the url for the notebook (e.g. https://github.com/Eldave93/Seizure-Detection-Tutorials/blob/master/01.%20Overview%20of%20Datasets.ipynb) into the URL bar on the [nbviewer](https://nbviewer.jupyter.org/) website.

### Prerequisites

The easiest way of interacting with these notebooks is to use [Google Colaboratory](https://colab.research.google.com).

*"Colaboratory allows you to use and share Jupyter notebooks with others without having to download, install, or run anything on your own computer other than a browser"* (https://research.google.com/colaboratory/faq.html).

I recommend Google Colaboratory mostly because of the size of the RAM available and the access to GPU's via the cloud. When working on the later notebooks, which use the TensorFlow package, the notebook will be ran a tonne faster! You can open the notebook in Colaboratory by clicking on the "Open in Colab" button at the top of the notebook.

Another option is to use [Binder](https://mybinder.org/). Binder is a open-source cloud deployment for Jupyter notebooks (see https://mybinder.readthedocs.io/en/latest/ for details). Although completely free it does have relatively limited computational resources, with a maximum of 2GB of RAM. This means some of the later notebooks which use larger datasets will likely not fit into memory.

If you want to interact with the notebooks locally on your machine then I advise you download Anaconda with Python 3 (See http://docs.anaconda.com/anaconda/install/) to get you started. Anaconda makes it easy to create a virtual environment in which to install and manage Python packages. It also provides an easy interface in which to launch the Jupyter Notebook interface (the Anaconda Navigator).

### Installing Packages

The notebooks start with a method to automatically install the required packages and data if using Google Colab. If working on these locally I encourage you to use a virtual environment. All you need to do is create a new environment and launch the Jupyter Notebook application from that environment (See http://docs.anaconda.com/anaconda/navigator/ for more information). 

If you want to install the packages manually from command prompt (or Anaconda prompt) then install the following packages below.

- matplotlib
- pandas 
- numpy 
- scipy 
- scikit-learn
- umap-learn 
- imblearn 
- seaborn 
- mlxtend 
- mne 
- PyWavelets
- Tensorflow

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
