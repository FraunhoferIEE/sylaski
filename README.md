# Sylas-KI

<div style="text-align: center;">
  <img src="https://github.com/FraunhoferIEE/sylaski/blob/main/pictures/Sylaski_project_pic.png" alt="Sylas-KI project logo" style="width:90%;">
</div>

## Idea

Sylas-KI stands for "Synthetische Lastzeitreihen für Energiesystemanalysen mit Verfahren der Künstlichen Intelligenz / synthetic power profiles for energy system analysis using AI". It is a project funded by the German ministry for research and education ([BMBF](https://www.bmbf.de/bmbf/de/home/home_node.html)) that aims to use power consumption data to train machine learning models in order to generate synthetic power profiles.

### Data

If you are just interested in synthetic power profiles here you can find consumption profiles for a complete year ([hybridWGAN](https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/IBWVJC&version=DRAFT) and [DDPM](https://data.goettingen-research-online.de/dataset.xhtml?persistentId=doi:10.25625/IBWVJC&version=DRAFT)) as a CSV-file. The models were trained on ~550 private households of the [OpenMeter](https://appstore.logarithmo.de/app/openmeterplatform/v1/demo/page-datenuebersicht?lang=EN) platform.

## Generator for synthetic power profiles

A hybridWGAN and DDPM model were the most successful models developed to create synthetic energy consumption profiles. We created two Jupyter notebooks for users to easily generate new time series data with each trained model.

## In depth - Train the models with your own data

### Pre-processing of power consumption data

Time series data usually contains outliers, level-shifts, missing values... - this makes it hard to use this data for machine learning applications. In order to generate continous "anomaly free" time series data from energy consumption data we applied some data cleaning to generate a currated data set.

### Clustering of power profiles

If you have data from multiple different consumers one is often interested to distinguish between different consumer types. For this we developed a frame work that can cluster different consumers in order to seperate them.

### Training hybridWGAN

The hybrid WGAN......

### Training DDPM
**TLDR:**
Denoising Diffusion Probabilistic Models (DDPM) are model architectures that are capable of synthesizing real data by learning a so called "denoising process". This process is applied to a randomly initialized sample ultimately resulting in an authentic generated data sample.



During the SylasKI project DDPMs were one of the main architectures that were investigated. The implementation of models, necessary utilities and experiments can be found under `syndatagenerators/models/ddpm`.
An exemplary notebook were a model is initialized, trained and sampled can be found under [example notebook](https://github.com/FraunhoferIEE/sylaski/blob/main/syndatagenerators/models/ddpm/training_example.ipynb)

### Metrics, Analysis, Visualization

In order to evaluate the quality of synthetic power profiles we established a framework with Metrics that can be used to describe time series data and compare them.

## Project partners

* [Fraunhofer IEE](https://www.iee.fraunhofer.de/)
* [University Kassel](https://www.uni-kassel.de/uni/)
* [University Göttingen](https://www.uni-goettingen.de/)
* [Green Excellence](https://www.green-excellence.de/home-en/)
