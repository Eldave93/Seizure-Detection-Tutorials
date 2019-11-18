# Seizure Detection Tutorials

This series of notebooks demonstrate the application of signal processing and machine learning classification to epileptic seizure detection.

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
- Evolutionary Algorithms
- Automated Machine Learning
- Batch Learning
- Multilayer Perceptrons
- Convolutional Neural Networks
- Recurrent Neural Networks

I'd like to give a massive thank you to the open source community for all the hard work that is put into all the python language, interactive computing resourses, and packages used to make this project a reality. I am continually learning about coding and one day, when I feel more competent, I hope to contibute to such great work. Also thank you to Hvass Laboratories (http://www.hvass-labs.org/) whos Tutorials on TensorFlow were the inspiration for creating my own tutorials.

**Progress Updates**
- 21/03/19: This is an ongoing project which I work on alongside my PhD. Although many of the notebooks have had their first drafts finished, it may take time to make them all available as I tidy them up. Please be patient as I update this repository over the coming months.
- 18/11/19: A number of the notebooks after "Model Evaluation and Hyperparameter Tuning" rely on a script I'm putting together to extract features from windowed data commonly used for seizure detection. As I'm still tidying this up and adding to it as part of one of my thesis chapters, I am yet to upload the notebooks that use this. I aim for this and all tutorial notebooks to be uploaded before the submission of my thesis which (hopefully) will be before summer 2020.

## Getting Started

Due to the size of these notebooks you may need to use [nbviewer](https://nbviewer.jupyter.org/) to view the notebooks. To do this all you need to do is copy the url for the notebook (e.g. https://github.com/Eldave93/Seizure_Detection_Tutorials/blob/master/Feature_Extraction_01_Epileptologie.ipynb) into the URL bar on the [nbviewer](https://nbviewer.jupyter.org/) website.

### Prerequisites

The easiest way of interacting with these notebooks is to use [Google Colaboratory](https://colab.research.google.com).

*"Colaboratory allows you to use and share Jupyter notebooks with others without having to download, install, or run anything on your own computer other than a browser"*<sup>25</sup>.

I recommend Google Colaboratory mostly because of the size of the RAM available and the access to GPU's via the cloud. When working on the later notebooks, which use the TensorFlow package, the notebook will be ran a tonne faster! You can open the notebook in Colaboratory by clicking on the "Open in Colab" button at the top of the notebook.

Another option is to use [Binder](https://mybinder.org/). Binder is a open-source cloud deployment for Jupyter notebooks (see https://mybinder.readthedocs.io/en/latest/ for details). Although completely free it does have relatively limited computational resources, with a maximum of 2GB of RAM. This means some of the later notebooks which use larger datasets will likely not fit into memory.

If you want to interact with the notebooks locally on your machine then I advise you download Anaconda<sup>20</sup> with Python 3 (See http://docs.anaconda.com/anaconda/install/) to get you started. Anaconda makes it easy to create a virtual environment in which to install and manage Python packages. It also provides an easy interface in which to launch the Jupyter Notebook interface (the Anaconda Navigator).

- http://docs.anaconda.com/anaconda/install/
- http://docs.anaconda.com/anaconda/navigator/

### Installing Packages

The notebooks start with a method to automatically install the required packages into an environment. If working on these locally I encourage you to use a virtual environment. All you need to do is create a new environment and launch the Jupyter Notebook application from that environment (See http://docs.anaconda.com/anaconda/navigator/ for more information). 

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

## Background

### EEG and Epilpesy

Epilepsy is the tendency to have unprovoked and recurrent seizures. Epileptic seizures are often accompanied by an alteration of consciousness, symptomatic of abnormal, excessive, or synchronized neuronal discharges which are either widespread or localized in nature<sup>1</sup> <sup>2</sup>. There are over 40 types of epilepsy<sup>3</sup> and over 40 different types of seizure; of which individuals may experience several<sup>4</sup>. Clinical manifestations of epilepsy are dependent on several factors; such as the particular epilepsy syndrome, patients age, the brain area that generates seizures, and if discharges remain local or propagate to other brain areas<sup>1</sup>. Whilst all seizures result from an increase in cellular excitability, the mechanisms of synchronization differ between seizures, broadly categorising them as focal or generalized epilepsies. Local seizure initiation may be a result of a burst of action potentials coupled with depolarization shifts and hyper synchronization of neighbouring neurons, with seizure propagation to other brain areas a result of concomitant loss of surround inhibition of connected neurons<sup>1</sup>. Although historically atonic, tonic, clonic, tonic-clonic, myoclonic, or absence seizures were thought to be “primarily generalized” in nature, there is an increasing acceptance that these still originate in local ictogenenic microcircuits which then propagate to other areas<sup>5</sup> <sup>6</sup>; representative of a larger shift towards viewing epilepsy as a dysfunction of neuronal networks than single sources<sup>7</sup>. 

The diagnosis of epilepsy relies on the identification of clinical features specific to a particular epilepsy syndrome. Electroencephalography (EEG), magnetic resonance imaging (MRI) reports and verbal descriptions of seizures are the most commonly available information to neurologists; with hospital records, seizure diaries, and videos of patient events desirable but not always available<sup>8</sup>. In clinic scalp EEG is commonly used as it provides an un-invasive, easy, and inexpensive method to characterise the mean electrical activity generated by the synchronous firing of open field neurons at a high temporal resolution. Typically, in the UK national health service (NHS), patients have an approximately 30-minute scalp EEG assessment, during which the patient may be asked to hyperventilate or exposed to photic stimulation to provoke a seizure. If a diagnosis is suspected, but not gained, a patient may then have a longer EEG assessment. Human experts, trained to qualitatively assess EEG records for epilepsy, will look at the EEG record to identify the presence and type of epilepsy, assessing the data based on a number of aspects. The spatial and temporal information is used to report seizures when the EEG appears to have seizure-like oscillations over a long duration and a number of channels. The pattern of EEG needs to be clearly different from the background activity, with consideration given to the difference of awake and asleep background EEG. The appearance of an epileptic event comparitive to artefacts or rhythms is also required to avoid falsely classify artefactual activity<sup>9</sup>. However, manual review of EEG is time consuming, expensive, and prone to error<sup>9</sup>. Indeed, it has been found below 80\% of events were similarly identified between two or more experts on a previously marked EEG record<sup>10</sup>. Indeed in general in developed countries, such as the UK, misdiagnosis rates are estimated to be between 20-30 percent, and consequently costly to the health service<sup>11</sup>.

The limitations of scalp EEG no doubt factor into these misclassifications. Scalp EEG has limited spatial sensitivity, as the signal needs to propagate through several layers of non-neural tissue, and therefore require larger brain areas to have synchronous activity. Scalp EEG is also often contaminated with artefacts, which represent noise caused by sources other than the brain such as by ambient electromagnetic interference, eye blinks, and muscle movements. Due to these limitations, intra-cranial EEG is therefore more often used for pre-surgical analysis to determine brain regions for surgical resection, as it is less effected by artefacts and has better spatial sensitivity<sup>9</sup>.

EEG data is first prepared for feature extraction by appropriately sampling and referencing the raw signal. EEG sampling needs to abide by the Nyquist criterion, therefore appropriate sampling rates need to be selected during data acquisition to ensure aliasing does not affect the signal of interest. Re-referencing can then be conducted to emphasise differences in electrical activity between electrodes. Electrodes are often referenced to a common source to remove unspecific brain activity by representing the electrical potential between an active electrode of interest and a relatively inactive reference, that is never-the-less still affected by global voltage changes and collected against the signal ground. Referencing can be done using a physical reference electrode placed on the earlobe, using any electrode during recording and later re-referencing electrodes to the average output of all electrodes, or by measuring potential between two active electrodes (bipolar recording). The combination of active electrodes with a reference and a ground creates a “channel”, and the general configuration of these are called a “montage”<sup>12</sup>. The most recognised re-reference method for visual seizure detection uses the linked ears or mastoids, which can be used to show the spike and wave pattern in seizures at a large amplitude<sup>13</sup>, but can introduce some bias. Other common re-reference methods remove the average of all the electrodes, with high density EEG, and configuring electrodes to be referenced to neighbouring ones (bipolar referencing)<sup>9</sup>.

As part of pre-processing, or during feature extraction, data can be filtered to remove frequencies at or above the Nyquist frequency, which will be influenced by aliasing. Noise can also be suppressed by attenuating frequencies of interest, such as between 0.1 Hz and 30Hz, which encompass most of the frequencies relevant for cognitive neuroscience. Suppressing high frequencies removes the effects of some artefacts and line noise, and the suppression of frequencies below 0.1 Hz reduces the effects of slow voltage shifts caused by skin potentials. However, if the signal and noise oscillate at similar frequencies, then reduction of noise can be difficult without distorting the signal of interest<sup>14</sup>. Strategies for dealing with artefacts can also be used, such as by removing epochs or channels with excessive artefacts, removal using separation methods, or training a system to identify and cope with common artefacts<sup>15</sup><sup>16</sup>.

More general pre-processing methods include dealing with missing values, with strategies including removal, mean/median imputation, or using the most frequent value, and encoding nominal features using methods such as one-hot encoding<sup>17</sup>.

### Machine Learning

**TODO**
- supplement this with introduction material from other machine learning books
- Follow up the challenges in the notebooks and how we are addressing them. For example moving away from some of the smaller datasets, how they remove artefacts ect.

*"Machine learning is the science of programming computers so they can learn from data"*<sup>22</sup>. 

Machine learning is great for problems that would require a lot of tuning by hand or a lot of rules, changing environments, and large data. This is because ML techiques are flexible and are not 'hard coded' so can adapt as well as able to discover patterns un-noticed previously<sup>22</sup>. Different types of machine learning systems exist based on:
- whether they have human supervision 
- whether they can learn incrementally
- whether they compare new data to known data or detect patterns and build a predictive model

At the start of this series of tutorials we will focus on Supervised learning, where a model is trained on a set of data so that it learns how to classify it using a set of labels. This model can then be used to predict the classification of a new set of data given a set of features. We then look at unsupervised learning, primarily in the context of dimensionality reduction, where the model attempts to group data not based on class labels but on their similarity or differences to other data; typically in the form of a clustering algorithm<sup>22</sup>.

There are a number of challenges to overcome when developing a machine learning system that we will come back to during the series (taken from<sup>22</sup>):
- Insufficient Quantity of Training Data
    - It has been shown<sup>23,24</sup> that even the most basic of machine learning pipelines can perfrom well on a complex task if given the right amount of data 
- Nonrepresentative Training Data
    - For a model to generalise to new data well, the data it was trained on has to be representative. This can be effected by the quantity of data, due to small samples having larger chance of noise, and flawed sampling methodology, influenced by sampling bias.
- Poor Data Quality
    - As is often the case with real world data, it is filled with errors, outliers and noise which will impact the ability of the system to detect patterns. Decisions then need to be made how to address these, for example do you ignore or fill in missing values.
- Irrelevant Features
    - Feature engineering is a critical part of a machine learning project as it ensures relevent features are used to train a model on. As well as creating the features, they can be futher selected for their usefulness, or extracted by combing features to create more useful ones.
- Overfitting the Training Data
    - Models can overgeneralize from the data it was trained on, sometimes focusing on patterns that only occour in thie training data. The model is too complex for the data and needs to be simplified, often by chaging its hyperparmeters.
- Underfitting the Training Data
    - Occours when the model is too simple and requires a more complex model, better features, or reduced model constraints.

## Sources

1. Giourou,  E.,  Stavropoulou-Deli,  A.,  Giannakopoulou,  A.,Kostopoulos, G. K., & Koutroumanidis, M. (2015). In-troduction to Epilepsy and Related Brain Disorders. InN. S. Voros & C. P. Antonopoulos (Eds.),Cyberphys-ical systems for epilepsy and related brain disorders:Multi-parametric monitoring and analysis for diagno-sis and optimal disease management(Chap. 2, pp. 11–38). doi:10.1007/978-3-319-20049-1

2. Krumholz, A., Wiebe, S., Gronseth, G., Shinnar, S., Levisohn, P., Ting, T., . . . French, J. (2007). Evaluating an Apparent Unprovoked First Seizure in Adults (An Evidence-Based Review). Neurology, 69(21), 1996– 2007. doi:10.1212/01.wnl.0000285084.93652.43

3. Berg,  A.  T.,  Berkovic,  S.  F.,  Brodie,  M.  J.,  Buchhalter,  J.,Cross, J. H., Van Emde Boas, W., . . .  Scheffer, I. E.(2010). Revised terminology and concepts for organi-zation of seizures and epilepsies: Report of the ILAECommission on Classification and Terminology, 2005-2009.Epilepsia,51(4), 676–685. doi:10.1111/j.1528-1167.2010.02522.x

4. Blume, W. T., Lüders, H. O., Mizrahi, E., Tassinari, C., VanEmde Boas, W., & Engel J., J. (2001). Glossary of de-scriptive  terminology  for  ictal  semiology:  Report  ofthe  ILAE  Task  Force  on  classification  and  terminol-ogy.Epilepsia,42(9), 1212–1218. doi:10.1046/j.1528-1157.2001.22001.x

5. Paz,  J.  T.,  &  Huguenard,  J.  R.  (2014).  Optogenetics  andepilepsy: Past, present and future.Epilepsy Currents,15(1), 34–38. doi:10.5698/1535-7597-15.1.34

6. Holmes,  M.  D.,  Brown,  M.,  &  Tucker,  D.  M.  (2004).  Are"generalized" seizures truly generalized? Evidence oflocalized mesial frontal and frontopolar discharges inabsence.Epilepsia,45(12), 1568–1579. doi:10.1111/j.0013-9580.2004.23204.x

7. Spencer, S. (2002). Neural Networks in human epilepsy: ev-idence  of  and  implications  for  treatment.Epilepsia,43(3), 219–227

8. Bidwell2015

9. Varsavsky, A., Mareels, I., & Cook, M. (2011). EEG Generation and Measurement. InEpileptic seizures and theeeg: Measurement, models, detection and prediction(Chap. 2, p. 337). doi:doi:10.1201/b10459-3

10. Wilson,  S.  B.,  Scheuer,  M.  L.,  Plummer,  C.,  Young,  B.,& Pacia, S. (2003). Seizure detection: Correlation ofhuman  experts.Clinical Neurophysiology,114(11),2156–2164. doi:10.1016/S1388-2457(03)00212-8

11. NICE

12. Teplan2002

13. LopesdaSilva2005

14. Luck, S. J. (2014). Basics of Fourier Analysis and Filtering. In An introduction to the event related potential technique (2nd, Chap. 7, pp. 219–248)

15. Gotman, J., Ives, J., & Gloor, P. (1981). Frequency content of EEG and EMG at seizure onset: Possibility of removal of EMG artefact by digital filtering. Electroencephalography and Clinical Neurophysiology, 52(6), 626–639. doi:10.1016/0013-4694(81)91437-1

16. Osorio, I., Frei, F. G., & Wilkinson, S. B. (1998). Realtime automated detection and quantitative analysis of seizures and short-term prediction of clinical onset. Epilepsia, 39(6), 615–627.

17. Raschka, S., & Mirjalili, V. (2017). Python Machine Learning (Second). Packt Publishing.

18. http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3

19. https://www.physionet.org/pn6/chbmit/

20. https://www.anaconda.com/download/

21. https://www.kaggle.com/c/seizure-detection

22. Géron, A. (2017). Hands-on machine learning with Scikit-Learn and TensorFlow: concepts, tools, and techniques to build intelligent systems. " O'Reilly Media, Inc.".

23. Banko, M., & Brill, E. (2001, July). Scaling to very very large corpora for natural language disambiguation. In Proceedings of the 39th annual meeting on association for computational linguistics (pp. 26-33). Association for Computational Linguistics.

24. Halevy, A., Norvig, P., & Pereira, F. (2009). The unreasonable effectiveness of data. IEEE Intelligent Systems, 24(2), 8-12.

25. https://research.google.com/colaboratory/faq.html

26. https://www.isip.piconepress.com/projects/tuh_eeg/index.shtml
