
Geomagnetic Induced Currents (GIC)
=====

**Alireza Motazedian**


**Disclaimer**: This file contains a web search for personal use. In case of using the materials in our works, we need to take care of plagiarism issues first. 


## Table of contents

- [0. Jargon](#0-jargon) 
- [1. Data problem-solving strategies](#1-data-problem-solving-strategies)   
- [2. Introduction](#2-introduction)  
    - [2.1. Solar wind: What is it and how does it affect Earth?](#21-solar-wind-what-is-it-and-how-does-it-affect-earth)	
    - [2.2. Geomagnetic Induced Currents (GIC)](#22-geomagnetic-induced-currents-gic)	
    - [2.3. Concept of UT](#23-concept-of-ut)  
    - [2.4. Dst index ](#24-dst-index) 
    - [2.5. Normalization](#25-normalization) 	
- [3. Website](#3-website)
    - [3.1. Measure instruments](#31-measure-instruments)  
        - [3.1.1. Magnetometer](#311-magnetometer)  
        - [3.1.2. Solar Wind Electron Proton Alpha Monitor](#312-solar-wind-electron-proton-alpha-monitor)  
    - [3.2. SuperMAG](#32-supermag)	
        - [3.2.1 SuperMAG definition](#321-supermag-definition)	
        - [3.2.2. Subtract Baseline](#322-subtract-baseline)	
        - [3.2.3. High Fidelity and Low Fidelity](#323-high-fidelity-and-low-fidelity)	
    - [3.3. Solar wind](#33-solar-wind)	
- [4. Datasets descriptions](#4-datasets-descriptions)	
    - [4.1. Solar wind (description)](#41-solar-wind-description)	
    - [4.2. SuperMAG (description)](#42-supermag-description)
    - [4.3. The magnitude of variables](#43-the-magnitude-of-variables)
    - [4.4. Feature scaling vs. normalization](#44-feature-scaling-vs-normalization)    	
- [5. Machine Learning](#5-machine-learning)	
    - [5.1. Time Series Analysis](#51-time-series-analysis)	
    - [5.2. Kriging Techniques](#52-kriging-techniques)	
    - [5.3. ADAM](#53-adam)	
    - [5.4. Filling missing values](#54-filling-missing-values)
        - [5.4.1. Mean](#541-mean)    
        - [5.4.2. Median](#542-median)  
        - [5.4.3. Mode](#543-mode)  
        - [5.4.4. Constant number or string](#544-constant-number-or-string)  
        - [5.4.5. Miss Forest](#545-miss-forest)  
        - [5.4.6. Mice Forest](#546-mice-forest)  
        - [5.4.7. KNN Imputation](#547-knn-imputation)  
        - [5.4.8. Bayesian imputation](#548-bayesian-imputation) 
        - [5.4.9. Multiple imputation](#549-multiple-imputation)
        - [5.4.10. Expectation-maximization](#5410-expectation-maximization)
        - [5.4.11. stochastic regression imputation](#5411-stochastic-regression-imputation) 
    - [5.5. Principal component analysis (PCA)](#55-principal-component-analysis-pca)
    - [5.6. LSTM](#56-lstm)
    - [5.7. CNN](#57-cnn)
    - [5.8. SVM](#58-svm)
    - [5.9. RF](#59-rf)                                   
- [6. Papers](#6-papers)	
    - [6.1. PILM: A Survey on Problems, Methods and Applications](#61-pilm-a-survey-on-problems-methods-and-applications)	
    - [6.2. a real-time GMD monitoring system](#62-a-real-time-gmd-monitoring-system)  
- [7. Questions](#7-questions) 
    - [7.1.1. Question 1](#71-question-1)  
    - [7.1.2. Question 2](#72-question-2) 
- [8. Code optimization ](#8-code-optimization) 
    - [8.1. Pure Python to Scientific Python](#81-pure-python-to-scientific-phyton)  
    


# 0. Jargon
01. **PIML**: Physics-Informed Machine Learning 
02. **PINN**: Physics-Informed Neutral Network
03. **GIC**: Geomagnetic Induced Current
04. **ACE**: The Advanced Composition Explorer (ACE) satellite
05. **DSCOVR**: Deep Space Climate Observatory (DSCOVR) spacecraft
06. **MAG**: Magnetometer
07. **IMF**: Interplanetary Magnetic Field
08. **PDE**: Partial Differential Equation
09. **ODE**: Ordinary Differential Equation
10. **SDE**: Stochastic Differential Equation
11. **GCN**: Graph Convolutional Network
12. **FDM**: Finite Difference Method 
13. **FVM**: Finite Volume Method
14. **SGD**: Stochastic Gradient Descent
15. **CNN**: Convolutional Neural Network
16. **NSF**: National Science Foundation
17. **RNN**: Recurrent Neural Networks 
18. **GRU**: Gated Recurrent Unit 
19. **MLP**: Multi-Layer Perceptron
20. **NTK**: Neural Tangeting Kernel 
21. **TFC**: Theory of Functional Connections
22. **HNN**: Hamiltonian Neural Networks
23. **VAE**: Variational Auto-Encoders
24. **GMD**: Geomagnetic Disturbances
25. **CME**: Coronal Mass Ejection
26. **GDV**: Geographic Data View
27. **POD**: Proper Orthogonal Decomposition
28. **FNO**: Fourier Neural Operator 
29. **FFT**: Fast Fourier Transformation
30. **GEM**: Geospace Environment Modeling
31. **MSE**: Mean Square Error
33. **PCA**: Principal component analysis 
33. **RCN**: Research Coordination Network
34. **SVM**: support vector machines 
35. **ACF**: autocorrelation function 
36. **SWPC**: Space Weather Prediction Center 
37. **NOAA**: National Oceanic and Atmospheric Administration 
38. **DASI**: Distributed Arrays of Small Instruments
39. **PGMD**: Pseudo-Geographic Mosaic Display
40. **RELU**: REctified Linear Unit
41. **RMSE**: Root Mean Square Error
42. **LSTM**: Long-Short Term Memory network 
43. **ADAM**: Adaptive Moment Estimation
44. **IAGA**: International Association of Geomagnetism and Aeronomy
45. **RTSW**: Real-Time Solar Wind
46. **SWEPAM**: Solar Wind Electron Proton Alpha Monitor 


# 1. Data problem-solving strategies  
 
![Figure_0](./images_GIC/Figure_0.png)  
*Figure_0*
 
- **Understanding the Problem**: This is the most important step, as it determines the success of the entire project. **Problem**: predicting solar wind peaks to prepare for and mitigate the impact of solar storms. It would be important to further understand what constitutes a 'peak' in this context (e.g., is it a certain threshold value? Or a relative increase?). It would also be useful to understand the prediction horizon (how far in advance do we need to predict these peaks?). Are there any specific performance metrics that matter most (e.g., sensitivity to false negatives vs false positives)?

- **Data Gathering**: Since you already have the dataset, this step is less of a concern. However, it's worth ensuring you have all the relevant data fields necessary for making the prediction. If additional data could improve the prediction (e.g., other space weather parameters, historical solar storm records), it might be worth gathering that.

- **Data Cleaning**: This step is crucial in your case since you've mentioned that about 12% of the data is missing. Since you've filled the missing data with the median, it's important to check if this approach is not introducing any bias in the data. You might want to consider other imputation methods (like using a model to predict missing values based on other variables) if the median imputation isn't working well. Additionally, dealing with noise in the response variable will also fall under this step. Techniques such as smoothing might be useful in this case.

- **Feature Engineering**: Given the time-series nature of your problem, feature engineering could play a big role. This could involve creating lagged versions of your variables (i.e., the value of the variable at previous time steps), rolling statistics (like rolling mean or rolling standard deviation), and time-based features like hour of the day, day of the week, etc. If your dataset spans multiple solar cycles, cycle-based features could also be useful.

- **Machine Learning/Deep Learning**: In this case, since the goal is to predict peaks in a time series, you might want to look into models designed for time series forecasting (like ARIMA, state space models, LSTM for deep learning) or change point detection models. If the 'peaks' can be well-defined, this could also potentially be treated as a classification problem (predicting whether a time point is a 'peak' or not), in which case classification algorithms would be useful.

- **Maintenance**: Maintenance is important for models that are deployed and used over a long time. In this particular case, since we're still at the problem-solving stage, we don't need to worry about maintenance yet. However, if you plan to use this model for ongoing solar storm prediction, you'll need to think about how to maintain it - this might involve regular retraining with new data, monitoring its predictions over time, and adjusting the model as necessary.

# 2. Introduction

## 2.1. Solar wind: What is it and how does it affect Earth?
 
Solar wind is a stream of charged particles, mainly protons and electrons, flowing from the Sun into space. It is a continuous flow of high-speed particles that carries the Sun's magnetic field throughout the solar system. The solar wind is created by the Sun's hot outer atmosphere, known as the corona, which has temperatures reaching millions of degrees Celsius. The solar wind affects Earth in several ways:

1.	**Aurora Formation**: The solar wind plays a crucial role in the formation of auroras, also known as the northern lights (aurora borealis) in the Northern Hemisphere and the southern lights (aurora australis) in the Southern Hemisphere. When solar wind particles interact with Earth's magnetosphere, they create a beautiful display of colorful lights in the polar regions.

2.	**Magnetic Field Interaction**: The solar wind carries the Sun's magnetic field, which interacts with Earth's magnetosphere. This interaction can cause various effects, including geomagnetic storms and disturbances in Earth's magnetic field. Geomagnetic storms can lead to disruptions in satellite communications, power grids, and navigation systems.

3.	**Space Weather**: The solar wind and its variations contribute to space weather, which refers to the dynamic conditions in the space environment surrounding Earth. Space weather can impact satellite operations, spacecraft trajectories, and astronaut safety during spacewalks or extended missions in space. Understanding and predicting solar wind behavior is crucial for space weather forecasting and protecting space-based assets.

4.	**Radiation Hazard**: The solar wind carries energetic particles that can pose a radiation hazard to astronauts and satellites in space. Exposure to high-energy particles from the solar wind can damage DNA and increase the risk of cancer and other health effects in humans. It is necessary to consider radiation protection measures for space missions beyond Earth's protective magnetosphere.

In summary, the solar wind, as a continuous flow of charged particles from the Sun, affects Earth's magnetic field, causes aurora formations, contributes to space weather phenomena, and presents radiation hazards for space activities. Understanding the behavior of the solar wind is essential for studying and mitigating its impacts on Earth and space-based systems.

![Figure_01](./images_GIC/Figure_01.png)  
*Figure_01*  

## 2.2. Geomagnetic Induced Currents (GIC)

Geomagnetic Induced Currents (GICs) are created as a result of the interaction between the Earth's magnetic field and varying external magnetic fields caused by geomagnetic disturbances, particularly during geomagnetic storms. The process by which GICs are created can be explained as follows:

1.	**Solar Activity**: Geomagnetic storms and GICs are primarily driven by solar activity. The Sun constantly emits a stream of charged particles called the solar wind. Occasionally, solar eruptions like coronal mass ejections (CMEs) occur, which release a large amount of plasma and magnetic fields into space.

2.	**Interaction with the Earth**: When the solar wind and its embedded magnetic fields reach the Earth, they interact with the Earth's magnetic field, which surrounds our planet and extends into space. The Earth's magnetic field is generated by the motion of molten iron within the planet's core.

3.	**Magnetic Field Disturbances**: The interaction between the solar wind and the Earth's magnetic field can cause variations in the Earth's magnetic field. These variations are typically stronger during geomagnetic storms triggered by intense solar activity.

4.	**Induced Electric Fields**: The changing magnetic field induces electric fields in the Earth's conductive crust and near-surface materials. This occurs due to Faraday's law of electromagnetic induction, which states that a changing magnetic field induces an electric field in a conductor.

5.	**Conductive Materials**: The induced electric fields drive electric currents through conductive materials on or near the Earth's surface. Common conductive materials include power transmission lines, pipelines, and other long metallic structures.

6.	**Path of Least Resistance**: The induced currents flow through the path of least resistance, following conductive paths such as power grids and pipelines. These currents can circulate through extensive networks, potentially affecting multiple regions.

The magnitude of GICs depends on various factors, including the intensity and duration of the geomagnetic disturbance, the characteristics of the Earth's magnetic field at the location, the conductivity of the materials, and the configuration of the infrastructure.

Monitoring and predicting geomagnetic disturbances are essential for assessing the potential risk of GICs and implementing measures to mitigate their effects on critical infrastructure.

## 2.3. Concept of UT
The magnetometer operates on Coordinated Universal Time (UT) and has a lag of 5 hours. 
This means that local midnight in Ottawa occurs at 05:00 UT. The lag is important because it helps align the magnetometer readings with specific time references and allows for consistent analysis and comparison of the data.  

## 2.4. Dst index  
In the context of Geomagnetic Disturbances (GMDs), the Dst (disturbance storm time) index is a measure used to characterize the size and intensity of a geomagnetic storm. GMDs occur when the Earth's magnetic field experiences significant fluctuations due to solar activity, such as solar flares and coronal mass ejections.

The Dst index provides an estimation of the globally averaged change in the horizontal component of the Earth's magnetic field at the magnetic equator. It is computed based on measurements from a network of magnetometer stations and is typically calculated once per hour. The index is reported in near-real-time and represents the magnitude of the geomagnetic storm.

During a geomagnetic storm, the Dst index shows negative values, indicating a decrease in the strength of the Earth's magnetic field. The magnitude of the negative excursion in the Dst index corresponds to the severity of the storm. Larger negative values indicate more intense geomagnetic disturbances.

The Dst index is widely used to monitor and track the progression of geomagnetic storms. It serves as a standard measurement for quantifying the strength and duration of these disturbances. By analyzing the Dst index, scientists and space weather forecasters can assess the impact of geomagnetic storms on various systems, including power grids, satellite communications, and navigation systems.

Additionally, the Dst index is employed in studies related to field-aligned currents in the magnetosphere and their connection to intense currents in the auroral ionosphere. Field-aligned currents are currents that flow along the Earth's magnetic field lines in the magnetosphere, and their behavior is linked to the Dst index.  

Here are some images that illustrate the Dst index:  

![Figure_26](./images_GIC/Figure_26.png)  
*Figure_26: This image shows the Earth's magnetic field in the absence of a geomagnetic storm. The field is strongest at the poles and weakest at the equator.*     

![Figure_27](./images_GIC/Figure_27.png)  
*Figure_27: This image shows the Earth's magnetic field during a geomagnetic storm. The ring current is shown as a red doughnut-shaped current. The ring current causes the Earth's magnetic field to be weaker at the equator.*     

![Figure_28](./images_GIC/Figure_28.png)  
*Figure_28: This image shows the Dst index over time. The Dst index is negative during the geomagnetic storm and then returns to normal after the storm.*     

Overall, the Dst index plays a crucial role in understanding, monitoring, and characterizing the effects of geomagnetic storms, including their association with field-aligned currents and their impact on various technological systems.  

## 2.5. Normalization  
Normalization typically means to scale a variable to have a values between 0 and 1, while retaining the original distribution of the values.

The formula for normalization is:

normalized = (value − min) / (max − min)

Where:
value is the original value
min is the minimum value of the feature
max is the maximum value of the feature

All in all, Normalization is a technique to bring different features to a similar scale, which can be very useful in various data processing tasks, especially in machine learning where certain algorithms are sensitive to the magnitude of features. It's easy to perform normalization in pandas with just a few lines of code.

# 3. Website  

## 3.1. Measure instruments:  
 
### 3.1.1. Magnetometer  
MAG is an abbreviation often used for a Magnetometer, an instrument that measures the strength and, in some cases, the direction of the magnetic field. Magnetometers can be used in various applications ranging from geology to space studies.

In the context of space and solar physics, a MAG (Magnetometer) on a spacecraft is used to measure the interplanetary magnetic field (IMF) and the magnetic field surrounding the spacecraft. For instance, magnetometers have been instrumental components of many space missions, like Voyager, Mars Global Surveyor, and Advanced Composition Explorer (ACE), among others. They provide data critical for understanding space weather and the Sun-Earth connection.

In the solar wind, the MAG can measure the intensity and direction of the IMF carried by the solar wind from the Sun. Changes in this magnetic field can significantly affect Earth's magnetosphere, leading to phenomena like geomagnetic storms. The MAG data, combined with other instrument measurements (like solar wind plasma parameters), offer a comprehensive understanding of the space environment and help in forecasting space weather.

In a broader context, magnetometers are also used on Earth for various purposes, such as detecting underground minerals, archaeological studies, and even smartphone applications (like digital compasses).

Please note that the specifics of what the MAG measures can vary depending on the exact mission and instrument specifications.  

### 3.1.2. Solar Wind Electron Proton Alpha Monitor    
The Solar Wind Electron Proton Alpha Monitor (SWEPAM) is an instrument that was aboard the Advanced Composition Explorer (ACE) spacecraft, which was launched in 1997. SWEPAM was designed to provide continuous, accurate measurements of solar wind plasma characteristics, including the speed, density, and temperature of electrons, protons, and alpha particles.

SWEPAM consists of two separate sensors: SWEPAM-I for ions (protons and alpha particles) and SWEPAM-E for electrons.

SWEPAM-I (Ion sensor): This sensor measures the solar wind ion velocity distribution function from which properties like speed, density, and temperature of protons and alpha particles are derived. It operates in the energy range of 260eV to 36keV.

SWEPAM-E (Electron sensor): This sensor measures the solar wind electron distribution function from which properties like speed, density, and temperature of electrons are derived. It operates in the energy range of 1 to 1240eV.

The data collected by SWEPAM is critical for understanding the behavior of the solar wind, which plays a key role in space weather. Understanding the solar wind's characteristics can help in predicting space weather phenomena and its potential effects on Earth's magnetosphere, satellites, power grids, and other technologies.  

## 3.2. SuperMAG

### 3.2.1 SuperMAG definition
SuperMAG is a worldwide collaboration of organizations and agencies that provide ground-based magnetometer measurements. These instruments are used to monitor variations in Earth's magnetic field, providing valuable data for understanding various geomagnetic phenomena like auroras, solar storms, and the interaction between solar wind and Earth's magnetosphere.

### 3.2.2. Subtract Baseline
In the context of SuperMAG data, "Subtract Baseline" could refer to the process of removing the baseline or background magnetic field level from the observed data. This helps to isolate the variations or fluctuations in the magnetic field that are of interest.

In other words, the 'baseline' represents the average or normal state of the magnetic field, and by subtracting this from the actual measurements, scientists can better see and analyze the deviations or anomalies in the field. This could be particularly important for identifying and studying events like magnetic storms or substorms.
In short, "Subtract Baseline" is a data preprocessing step that helps isolate the signal of interest (in this case, magnetic variations) from the overall data.

### 3.2.3. High Fidelity and Low Fidelity
The terms "High Fidelity" and "Low Fidelity" in the context of data typically refer to the level of detail, accuracy, and quality of the data.

High Fidelity data is generally more accurate and has more detailed information. It often comes from higher quality or more precise instruments and measurements. It has a higher resolution, meaning it has more data points over a given interval, which can give a more detailed picture of changes over time.

Low Fidelity data, on the other hand, is generally less detailed and may be less accurate. It has a lower resolution, meaning it has fewer data points over the same interval. This might mean that some detail is lost, but it can also make the data easier to handle and process, particularly in large quantities or over large timescales.

SuperMAG service includes data with two different temporal resolutions, 1-min, and 1-sec. The latter is a subset of the former as not all stations provide 1-sec data. 
The 1-min data and all derived products can be accessed by selecting the 'Low Fidelity' option under Indices, Data, Polar Plots, Movies, and Products. 
The 1-sec data and all derived products can be accessed by selecting the 'High Fidelity' option under Data and Polar Plots.  

## 3.3. Solar wind

# 4. Datasets descriptions 

## 4.1. Solar wind (description)
Here is a description for each column in "Solar wind Omniweb" dataset:

⮚	**year**: The year in which the observation was made.

⮚	**day**: The day of the year when the observation was made, typically ranging from 1 to 365 (or 366 in leap years).

⮚	**hour**: The hour of the day (in a 24-hour format) when the observation was made.

⮚	**minute**: The minute of the hour when the observation was made.

⮚	**Field magnitude average**: This is the average magnitude of the Interplanetary Magnetic Field (IMF) over the given period. 
The Interplanetary Magnetic Field (IMF) is a crucial aspect of our solar system, extending from the Sun into interplanetary space. It's carried out into space by the solar wind, a stream of charged particles emitted from the Sun's upper atmosphere. The IMF varies and has complex structures based on the Sun's rotation and solar activities like solar flares and sunspots.
According to the University of Maryland's Space Weather Prediction Center, the IMF value typically ranges from about 1 to 20 nanoTesla (nT). 

Two main types of IMF exist:  
✔	**The Parker Spiral**: This type of IMF is named after solar astrophysicist Eugene Parker. It is a spiral-shaped magnetic field resulting from the rotation of the Sun. The Sun rotates faster at its equator than at its poles, causing the magnetic field lines to take on a spiral shape, much like the water from a spinning garden hose.  

![Figure_02](./images_GIC/Figure_02.png)  
*Figure_02*  

![Figure_03](./images_GIC/Figure_03.png)  
*Figure_03*  

![Figure_04](./images_GIC/Figure_04.png)   
*Figure_04* 

✔	**The Interplanetary Shock**: This is a rapid change in the IMF caused by a significant solar event like a coronal mass ejection (CME). A CME is a massive burst of solar wind and magnetic fields rising above the solar corona or being released into space. These shocks can travel through space and affect planets' magnetic fields, including Earth's.

![Figure_05](./images_GIC/Figure_05.png)  
*Figure_05: Interplanetary shock wave S f developed in the solar wind as a result of a chromospheric flare or a coronal mass ejection on the Sun S, and force lines of the interplanetary magnetic field B sw. Spacecraft are schematically shown in the neighborhood of the Lagrange point L 1 and the Earth's bow shock S b and in the magnetosheath between S b and the magnetopause m which is the boundary of the magnetosphere M (shown in section with an image of the Earth's magnetic field); broken curve corresponds to the Earth's (E) orbit.*

The "field magnitude average" metric is a measure of the IMF's average strength over a specific period. It incorporates the average of the IMF's three components: the north-south component, the east-west component, and the radial component (the component that points towards or away from the Sun). 

The IMF and its influence are not static but subject to change based on solar activity. Therefore, scientists employ various tools and techniques, including satellite observations and mathematical models, to track and forecast changes in the IMF. Understanding these changes is pivotal for our comprehension of the Sun-Earth connection, which can influence space weather and potentially disrupt technologies that our society heavily relies upon.

Here are some additional details about the field magnitude average metric:  
●	The units of the field magnitude average metric are nanotesla (nT).  
●	The average IMF strength is about 5nT.  
●	The strongest IMF storms can have field strengths of up to 100nT.  
●	The field magnitude average metric is typically calculated over a period of days or weeks.

The term "heliosphere" is used to denote the vast bubble-like region surrounding the Sun, dominated by the solar wind and its associated IMF. This region acts as a protective shield for the planets within our solar system against cosmic radiation. When we refer to the IMF as the "Heliospheric Magnetic Field," we emphasize its role and reach throughout the heliosphere.

![Figure_06](./images_GIC/Figure_06.png)  
*Figure_06*  

![Figure_07](./images_GIC/Figure_07.png)  
*Figure_07*  

⮚	**Bx**, **By**, **Bz**: These are components of the IMF in the Geocentric Solar Magnetospheric (GSM) coordinate system. 
These components of the IMF typically vary between about -10 and +10 nT. This information is based on the GSM coordinate system used in space weather studies. 
![Figure_08](./images_GIC/Figure_08.png)  
*Figure_08*

The geocentric coordinate system is not a planar coordinate system based on a map projection. It is a geographic coordinate system in which the earth is modeled as a sphere or spheroid in a right-handed XYZ (3D Cartesian) system measured from the center of the earth.


●	**Bx**: This component represents the IMF along the Earth-Sun direction. It indicates the strength and direction of the magnetic field aligned with the line connecting the Earth and the Sun. It can be thought of as the "north-south" component of the IMF.

●	**By**: This component represents the IMF perpendicular to the Earth-Sun line but within the ecliptic plane. The ecliptic plane is the plane defined by the Earth's orbit around the Sun. By indicates the strength and direction of the magnetic field in the east-west direction within this plane.

●	**Bz**: This component represents the IMF perpendicular to the ecliptic plane. It indicates the strength and direction of the magnetic field pointing either northward or southward. A positive Bz value indicates a northward-directed magnetic field, while a negative Bz value indicates a southward-directed magnetic field.

![Figure_09](./images_GIC/Figure_09.png)  
*Figure_09*

⮚	**Speed**: The speed of the solar wind, usually measured in kilometers per second.

●	The speed of the solar wind is usually measured in kilometers per second (km/s).  
●	The speed of the solar wind is influenced by the Sun's magnetic field, the solar activity, and the distance from the Sun.      
●	CMEs can cause the speed of the solar wind to increase significantly.
●	According to NASA, solar wind speeds typically range between 250 and 800 kilometers per second (km/s), but can occasionally exceed 1000 km/s during strong solar storms.

⮚	**Vx**, **Vy**, **Vz**: These are the components of the solar wind velocity in the GSM coordinate system.
●	**Vx**: This component represents the velocity of the solar wind in the direction from the Earth towards the Sun. This is along the X-axis of the GSM system. A negative Vx value would typically indicate solar wind moving from the Sun towards the Earth.

●	**Vy**: This component represents the velocity of the solar wind in the direction perpendicular to the Earth-Sun line, within the plane of the Earth's orbit around the Sun (the ecliptic plane). This is along the Y-axis of the GSM system.

●	**Vz**: This component represents the velocity of the solar wind in the direction perpendicular to the ecliptic plane, essentially northward or southward relative to the Earth-Sun line. This is along the Z-axis of the GSM system.

⮚	**Proton density**: The density of protons in the solar wind.

●	The proton density of the solar wind is usually measured in protons per cubic centimeter (p/cc).    
●	The proton density is influenced by the Sun's magnetic field, the solar activity, and the distance from the Sun.  
●	CMEs can cause the proton density of the solar wind to increase significantly.  
●	The proton density in the solar wind typically falls between 1 and 10 protons per cubic centimeter. This information is referenced from the University of Maryland's Space Weather Prediction Center.

⮚	**Proton temperature**: The temperature of the solar wind protons.

●	The proton temperature of the solar wind is usually measured in Kelvin (K).   
●	The proton temperature is influenced by the Sun's magnetic field, the solar activity, and the distance from the Sun.  
●	CMEs can cause the proton temperature of the solar wind to increase significantly.  
●	The temperature of solar wind protons can vary quite a bit, but it's typically between 10,000 and 2,000,000 degrees Kelvin according to NASA's studies on solar wind. 

⮚	**Flow pressure**: The dynamic pressure of the solar wind.

●	The flow pressure of the solar wind is usually measured in nanopascals (nPa).  
●	The flow pressure is influenced by the density and speed of the solar wind.  
●	CMEs can cause the flow pressure of the solar wind to increase significantly.  
●	The dynamic pressure of the solar wind can be in the range of 1 to 10 nPa. This can be referenced from a paper by Richardson and Cane in the Journal of Geophysical Research.  


⮚	**Electric field**: 
The solar wind's electric field is a critical parameter in understanding space weather and geomagnetic activities. It arises due to the motion of the charged particles in the solar wind across the Interplanetary Magnetic Field (IMF). This motion of charged particles creates an electric field, which is perpendicular to both the solar wind velocity and the IMF direction. Visualize the solar wind velocity vector (V) and the IMF vector (B) within this 3D space. The electric field vector (E) will be perpendicular to both V and B. In reality, these vectors may not align perfectly with the axes of the GSM system, but it's important to understand that E, V, and B are mutually perpendicular in the frame of the solar wind.

●	The electric field in the solar wind is usually measured in volts per meter (V/m).      
●	The electric field is influenced by the solar wind speed and the IMF.  
●	CMEs can cause the electric field in the solar wind to increase significantly.  
●	This could range quite broadly depending on solar wind speed and IMF, typically between 0 and 10 mV/m according to space weather prediction models.

![Figure_10](./images_GIC/Figure_10.png)  
*Figure_10*  

![Figure_11](./images_GIC/Figure_11.png)  
*Figure_11*  

![Figure_12](./images_GIC/Figure_12.png)  
*Figure_12*  

⮚	**SYM/H**:
The SYM/H index is a very important tool in the study of geomagnetism and space weather, specifically for monitoring and studying geomagnetic storms. This index is derived from magnetic field measurements taken at several locations around the Earth, and it provides a measure of the changes in the Earth's magnetic field in response to solar activity.

Here's how it works:
Geomagnetic storms are global disturbances in the Earth's magnetic field caused by changes in the solar wind. These disturbances can cause the Earth's magnetic field to fluctuate, and the SYM/H index is designed to measure these fluctuations.

✔	**Symmetric part of the disturbance**: When a geomagnetic storm occurs, it causes disturbances in the Earth's magnetic field that can be roughly divided into two parts: a symmetric part and an asymmetric part. The symmetric part represents the average global effect of the storm, while the asymmetric part represents localized effects. The SYM/H index specifically measures the symmetric part of the disturbance in the horizontal plane at the Earth's surface.

✔	**Measurement in the horizontal plane**: The SYM/H index focuses on changes in the horizontal component of the Earth's magnetic field. This is important because it's the horizontal component that primarily interacts with the Earth's surface and atmosphere, causing the effects we associate with geomagnetic storms.

✔	**Monitoring and studying geomagnetic storms**: The SYM/H index provides a way to quantify the intensity of a geomagnetic storm. By tracking changes in the SYM/H index, researchers can monitor the progress of a storm, measure its peak intensity, and study its effects. A larger change in the SYM/H index corresponds to a stronger storm.

●	The SYM/H index is usually measured in nanotesla (nT).  
●	The SYM/H index is a measure of the symmetric part of the disturbance magnetic field in the horizontal plane at the Earth's surface.  
●	The disturbance magnetic field is the magnetic field that is caused by the interaction of the solar wind with the Earth's magnetic field.  
●	Geomagnetic storms are caused by large solar storms, such as coronal mass ejections (CMEs).  
●	The SYM/H index is a useful tool for space weather forecasting.  
●	This index can range from around -500 to +500 nanoTesla during strong geomagnetic storms as per NOAA's space weather scale for geomagnetic storms

## 4.2. SuperMAG (description)
Here is a description for each column in "SuperMAG" dataset:

⮚ **Date_UTC**:"Date_UTC" in the SuperMAG dataset represents each observation's timestamp, standardized to Coordinated Universal Time (UTC), an international time standard. Regardless of the geographical location or local time of data collection, using UTC ensures accurate comparison and correlation of global data.

The timestamp typically includes year, month, day, hour, minute, and possibly second of the observation in a format like "YYYY-MM-DD HH:MM:SS". An example could be "2023-07-05 14:30:00", indicating a measurement made at 2:30 pm on July 5, 2023, in UTC time.

The presence of a UTC timestamp is critical for time series analysis, tracking temporal changes, and correlating events across various datasets or locations.

⮚ **IAGA**: The International Association of Geomagnetism and Aeronomy (IAGA) promotes the study of geomagnetism and aeronomy, key fields for understanding Earth's magnetic field and its interactions with solar and cosmic radiation.

In a SuperMAG dataset, an IAGA code is a unique identifier for each magnetometer station across the globe. These stations, equipped with magnetometers, measure the strength and direction of the magnetic field at specific locations.

Each station's unique IAGA code (typically a three-letter code) helps identify its contributed data, pinpoint the station's location, and facilitate cross-referencing with other databases. This aids in understanding global geomagnetic phenomena, like geomagnetic storms, which can have varying impacts at different Earth locations.

⮚ **GEOLON (Geographic Longitude)**: The Geographic Longitude, also known as GEOLON in data sets, represents the east-west position of a point on the Earth's surface. It's the angular distance east or west of the Prime Meridian, a line of longitude at 0 degrees that runs through Greenwich, London.

Longitude is measured in degrees, ranging from -180 (180 degrees west) to +180 (180 degrees east). The Prime Meridian (0 degrees longitude) serves as the reference point for these measurements.

In the context of the SuperMAG dataset, the GEOLON value would refer to the longitude of the location of each magnetometer station. This gives the east-west position of the station on the Earth's surface.

For example, a magnetometer station in New York would have a GEOLON value around -74 (since New York is approximately 74 degrees west of the Prime Meridian), while a station in Tokyo would have a GEOLON value around +140 (since Tokyo is approximately 140 degrees east of the Prime Meridian).

Knowing the longitude of the magnetometer station is crucial for a variety of geophysical analyses. It helps in correlating the geomagnetic data with its geographic location, which is essential when examining global phenomena like geomagnetic storms and their effects on different parts of the world.

![Figure_13](./images_GIC/Figure_13.png)  
*Figure_13*: Longitude lines are drawn between the North Pole and the South Pole. (A) The prime meridian (0°) divides earth into two halves of 180°. (B) Longitude is measured in degrees from 0° to 180° east or west of the prime meridian. | 

![Figure_14](./images_GIC/Figure_14.png)  
*Figure_14*: (A) East and west longitude meeting at 180˚ meridian. (B) The 180˚ meridian is on the opposite side of the globe from the prime meridian.

⮚ **GEOLAT (Geographic Latitude)**: Geographic Latitude, or GEOLAT in many datasets, represents the north-south position of a point on the Earth's surface. It's the angular distance from the equator to that point, north or south.

Latitude is measured in degrees, with the equator representing 0 degrees, the North Pole +90 degrees, and the South Pole -90 degrees.

In the context of the SuperMAG dataset, the GEOLAT value refers to the latitude of each magnetometer station's location. This provides the north-south position of the station on the Earth's surface.

For instance, a magnetometer station in Sydney, Australia would have a GEOLAT value of approximately -34 (as Sydney is about 34 degrees south of the equator), while a station in Oslo, Norway would have a GEOLAT value of around +60 (as Oslo is about 60 degrees north of the equator).

![Figure_15](./images_GIC/Figure_15.png)  
*Figure_15*

![Figure_16](./images_GIC/Figure_16.png)  
*Figure_16*

⮚ **MAGON (Magnetic Longitude)**: The magnetic longitude, also known as MAGLON in many datasets, is similar to geographic longitude, but it's based on the Earth's magnetic field rather than the surface geography. It refers to the east-west position of a point relative to the Earth's magnetic field.

The geomagnetic coordinates are based on a geomagnetic model (like the International Geomagnetic Reference Field, IGRF) which represents the Earth's magnetic field. In this model, the "prime meridian" is not the geographic prime meridian that goes through Greenwich, but the meridian that goes through the magnetic north pole.

The magnetic north pole does not align perfectly with the geographic North Pole, and it even moves over time (a phenomenon called secular variation). So, a magnetometer's magnetic longitude can be quite different from its geographic longitude.  
Magnetic longitude ranges from 0 to 360 degrees. The same reference can be used.

![Figure_17](./images_GIC/Figure_17.png)  
*Figure_17*

⮚ **MAGLAT (Magnetic Latitude)**: This is the latitude of the magnetometer station in geomagnetic coordinates.

Magnetic Latitude, often abbreviated as MAGLAT in datasets, is similar to geographic latitude, but it's based on the Earth's magnetic field rather than the Earth's surface geography.

Geographic latitude denotes the north-south position of a point on the Earth's surface, measured as an angle from the equator (0 degrees) to the North (+90 degrees) or South (-90 degrees) poles.

In contrast, Magnetic Latitude refers to the north-south position of a point relative to the Earth's magnetic field. In this coordinate system, 0 degrees refers to the magnetic equator (the line around the Earth halfway between the magnetic north and south poles), +90 degrees refers to the magnetic North Pole, and -90 degrees refers to the magnetic South Pole.

The Earth's magnetic poles do not perfectly align with the geographic poles, and they even move over time due to changes in the Earth's core (a phenomenon known as geomagnetic secular variation). Therefore, a location's magnetic latitude can be different from its geographic latitude.  
Magnetic latitude also ranges from -90 degrees at the magnetic south pole to +90 degrees at the magnetic north pole. The same reference can be used. 

![Figure_18](./images_GIC/Figure_18.png)  
*Figure_18*  

⮚ **MLT (Magnetic Local Time)**: Magnetic Local Time (MLT) is a measure of time based on the position of a location with respect to the Sun, but with reference to the Earth's magnetic field rather than its geographical features. In essence, it's solar time, but tied to magnetic, not geographic, coordinates.

Just like geographic local time, where noon is defined as when the Sun is at its highest point in the sky, in Magnetic Local Time, magnetic noon is when the Sun is in line with the magnetic meridian of the location. This is the line running from magnetic north to magnetic south through that point.

Because the Earth's magnetic field is not perfectly aligned with its rotation axis, and because the magnetic poles wander over time, the MLT for a given location will not usually match the geographic local time. It's also worth noting that MLT varies as the Earth rotates, just like geographical local time.

The concept of MLT is particularly important in the field of space weather and geomagnetism, because many phenomena related to the Earth's magnetic field and its interaction with the solar wind have a strong dependence on MLT. For example, the occurrence and strength of auroras and geomagnetic disturbances can vary significantly with MLT. In the context of the SuperMAG dataset, the MLT would be the local time at each magnetometer station, in terms of magnetic coordinates.  
Magnetic Local Time ranges from 0 to 24 hours, like conventional time.

⮚ **MCOLAT (Magnetic Co-latitude)**: Magnetic Co-latitude, often referred to as MCOLAT in various datasets, is a way of defining a location's position with respect to the Earth's magnetic field. More specifically, it is calculated as 90 degrees minus the magnetic latitude of a given point.

While magnetic latitude measures the angle between the location and the magnetic equator (ranging from -90 degrees at the magnetic South Pole to +90 degrees at the magnetic North Pole), magnetic co-latitude is the complementary angle measuring from the magnetic North Pole.

This means that the magnetic co-latitude of a point is the angle from that point to the magnetic North Pole along a line of longitude, with the magnetic North Pole itself being at 0 degrees, and the magnetic equator being at 90 degrees.

This way of measuring location is particularly useful in spherical coordinates and certain areas of study related to the Earth's magnetic field, such as magnetospheric physics and space weather analysis. In the SuperMAG dataset, the MCOLAT for each magnetometer station would allow scientists to understand the station's position relative to the Earth's magnetic North Pole, which could be relevant when studying certain magnetospheric phenomena.  
The magnetic co-latitude ranges from 0 to 180 degrees, with 0 degrees at the magnetic north pole and 180 degrees at the magnetic south pole.

⮚ **IGRF_DECL (International Geomagnetic Reference Field Declination)**: The International Geomagnetic Reference Field (IGRF) Declination, or IGRF_DECL, refers to the angle between the magnetic north and true (geographic) north at a specific location based on the IGRF model.

Geographic north is a constant that refers to the North Pole, the point where Earth's axis of rotation intersects the surface in the northern hemisphere. On the other hand, magnetic north is the direction that a compass needle points, and it's determined by the Earth's magnetic field. Due to the tilted and dynamic nature of Earth's magnetic field, magnetic north doesn't align perfectly with geographic north, and this discrepancy is what we call magnetic declination.

Magnetic declination varies both with location on the Earth's surface and over time, as the Earth's magnetic field changes. At some locations, the declination angle can be quite significant, and if not accounted for, can result in substantial navigation errors.

The International Geomagnetic Reference Field (IGRF) is a mathematical model of Earth's magnetic field produced by an international collaboration of scientists. It's used for precise navigation, mineral exploration, and some types of scientific research. The IGRF model provides an accurate estimate of the Earth's magnetic field and its declination at any location. The declination according to the IGRF model is represented as IGRF_DECL in the SuperMAG dataset.

⮚ **SZA (Solar Zenith Angle)**: The Solar Zenith Angle (SZA) is a measure of the Sun's position in the sky relative to a particular location on Earth. Specifically, it is the angle between the line that points straight up from that location (the line perpendicular to the Earth's surface, also known as the zenith) and the line from that location to the Sun.

When the Sun is directly overhead at noon (the Sun is at the zenith), the SZA is 0 degrees. As the Sun moves across the sky towards the horizon, the SZA increases, reaching 90 degrees when the Sun is on the horizon. During twilight hours, the SZA is greater than 90 degrees.

The SZA is critical for understanding and calculating the amount of solar radiation reaching a particular location on Earth's surface. The larger the SZA, the longer the path of the Sun's rays through the Earth's atmosphere, which results in more scattering and absorption of sunlight and less solar radiation reaching the Earth's surface.

In the context of SuperMAG or other geophysical datasets, the SZA can be an important parameter for understanding variations in ionospheric and magnetospheric processes that are driven by solar radiation, including auroral activity and ionospheric conductivity.

⮚  **dbn_nez, dbe_nez, dbz_nez**: The variables dbn_nez, dbe_nez, and dbz_nez in a dataset like SuperMAG represent changes in the components of the magnetic field in the North, East, and Down directions respectively.
They typically ranges from -100 to +100 nanoTesla (nT) during quiet conditions, but can exceed these values during geomagnetic storms.  

- **dbn_nez**: This represents the change (delta, denoted by 'db') in the northward component of the magnetic field.

- **dbe_nez**: This represents the change in the eastward component of the magnetic field.

- **dbz_nez**: This represents the change in the downward (or vertical) component of the magnetic field.

These changes are calculated over a specific time interval, such as from one observation time to the next.

The North, East, Down (NED) or NEZ coordinate system is a geographical system often used in geodesy and navigation, among other fields. It's a frame of reference attached to the Earth's surface, with the 'North' axis pointing towards the geographic North Pole, the 'East' axis pointing towards the geographic East (perpendicular to North and in the same horizontal plane), and the 'Down' axis pointing vertically downward.

In the context of a magnetometer station in the SuperMAG network, these measurements would reflect the variations in the local magnetic field due to various sources, such as the Earth's core, the ionosphere, and the magnetosphere, as well as the solar wind and the interplanetary magnetic field (IMF). Such measurements are key for studying the dynamics of Earth's magnetic field and space weather phenomena.

![Figure_19](./images_GIC/Figure_19.png)  
*Figure_19*

⮚ **dbn_geo, dbe_geo, dbz_geo**: The variables dbn_geo, dbe_geo, and dbz_geo represent the changes in the northward, eastward, and downward (or vertical) components of the Earth's magnetic field, as measured in geographic coordinates.
They typically ranges from -100 to +100 nanoTesla (nT) during quiet conditions, but can exceed these values during geomagnetic storms.  

Let's break down each term:

- **dbn_geo**: This represents the change in the northward component of the magnetic field in geographic coordinates. If you were standing at the location of the magnetometer and facing geographic north, this measurement would represent how much the magnetic field has changed in that direction.

- **dbe_geo**: This represents the change in the eastward component of the magnetic field in geographic coordinates. If you were standing at the location of the magnetometer and facing geographic east, this measurement would represent how much the magnetic field has changed in that direction.

- **dbz_geo**: This represents the change in the downward (vertical) component of the magnetic field in geographic coordinates. This would measure how much the magnetic field has changed in a direction going straight down into the Earth at the location of the magnetometer.

The term "delta" (represented by 'db') in each of these variables represents a change. So these variables aren't giving you the strength of the magnetic field in each direction, but rather how much that strength has changed over a certain period of time, such as from one measurement to the next.

The 'geo' part of these variables indicates that these measurements are made in the geographic coordinate system, which is based on the Earth's shape and orientation in space, rather than in a magnetic coordinate system, which would be based on the Earth's magnetic field.  

⮚ **decl**: The magnetic declination varies greatly across the globe, but it's typically between -30 and +30 degrees.  

## 4.3. The magnitude of variables
**It is important to know the magnitude of variables in datasets like solar wind and supermag. What's the reason?**    
- **Feature Scaling**: Feature scaling is a method used to normalize the range of independent variables or features of data. In other words, it brings all the variables to the same range. Machine learning algorithms perform computations on input data, and if the range of values is not standardized, it could negatively affect the learning process, causing the algorithm to take longer to train or leading to less accurate models. For example, in the case of the gradient descent algorithm (used in linear regression, logistic regression, neural networks), feature scaling can speed up the algorithm's convergence.

- **Understanding Data**: Recognizing the magnitudes of your variables can provide profound insights into your dataset. This understanding encompasses both statistical and data-analytic aspects, necessitating comprehension of the magnitudes, distributions, and relationships of the variables. For instance, knowing that the solar wind speed typically lies between 250-800 km/s can guide the process of data analysis in multiple ways. It can help identify what constitutes normal conditions versus extreme space weather events, acting as a useful reference for detecting outliers. This knowledge can also aid in the selection of appropriate statistical methods or data transformations, enhancing the interpretability of results.

- **Outlier Detection**: An outlier is a data point that differs significantly from other observations. They might be due to variability in the data or may indicate experimental errors. Knowing the typical magnitude and distribution of your data can help identify these outliers. For example, a single measurement of solar wind speed that is significantly higher than the typical range might be an error or an unusually extreme event.

- **Normalization**: When dealing with variables of different units and scales (like solar wind speed in km/s and density in particles/cm^3), high-magnitude variables might dominate the model's behavior. Normalization helps ensure that each variable contributes approximately proportionately to the final prediction. For some algorithms, not scaling the data can lead to suboptimal results or longer training times.

- **Improving Accuracy**: Variables with large magnitudes can dominate or skew machine learning models because they have a larger absolute effect on the model's predictions. This might not reflect their true importance in the real-world system being modeled. By scaling features to a similar range, you can prevent any single feature from dominating the model just because of its scale, thereby improving model accuracy.  

- **To understand the physical processes involved**: It is about interpreting the data in the context of underlying real-world phenomena. The magnitudes of the variables aren't just numbers - they represent physical quantities tied to real-world processes. In the supermag data example, understanding the magnitudes is a key part of understanding the interactions of magnetic field lines. It's about connecting the data to the physical world and using it to infer the dynamics of the system we are studying.  

## 4.4. Feature scaling vs. normalization
They are two related concepts used in machine learning and data analysis for preparing the data, but they are not exactly the same thing.

- **Feature Scaling** is a general term that refers to the process of transforming your data so that it fits within a specific scale, like 0-100 or 0-1. You use feature scaling when you're dealing with data that has varying scales and you want to bring all features to the same level of magnitudes. This prevents a feature with a large scale from dominating other features when you apply machine learning algorithms. There are several ways to perform feature scaling, including Min-Max scaling, Standardization (Z-score normalization), and Robust scaling.

    - **Min-Max Scaling**: It rescales the feature to a fixed range, usually 0 to 1. It uses the minimum and maximum values of the feature.

    - **Standardization (Z-score Normalization)**: It standardizes features by subtracting the mean (centering the data around 0) and scaling to unit variance. Each value is subtracted from the mean of the feature and then divided by the standard deviation of the feature. This method is widely used for normalization in many machine learning algorithms (e.g., support vector machines, logistic regression, and neural networks).

    - **Robust Scaling**: This method removes the median from the data and scales the data according to the Interquartile Range (IQR). It is robust to outliers.

**Normalization** is a specific type of feature scaling that changes the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information. Normalization typically means rescales the values into a range of [0,1].

It's important to note that whether to use feature scaling and/or normalization depends on your specific use case and the machine learning algorithm you're using. Some algorithms (like linear regression, logistic regression, and neural networks) can benefit from normalization or feature scaling, while others (like decision trees and random forests) don't require it.  

# 5. Machine Learning  

## 5.1. Time Series Analysis

## 5.2. Kriging Techniques
Kriging, also known as Gaussian process regression, is a statistical method used for interpolation and spatial prediction in various fields, including geostatistics, soil science, geology, and public health. It is named after the South African mining engineer Danie Krige, who played a key role in its development. 

Kriging involves a multistep process that includes exploratory statistical analysis of the data, variogram modeling, creating the surface, and optionally exploring a variance surface. The method is particularly appropriate when there is a spatially correlated distance or directional bias in the data. 

The key idea behind kriging is to estimate the value of a variable at unsampled locations based on a limited set of sampled data points. It leverages the spatial correlation structure of the data, assuming that nearby locations are more similar than distant ones. Kriging aims to provide the best linear unbiased prediction (BLUP) at the unsampled locations, taking into account the spatial autocorrelation. 

The method is based on the concept of a Gaussian process, where the values at different locations are considered as random variables following a multivariate normal distribution. By estimating the spatial autocovariance or semivariogram from the data, kriging allows for the interpolation of values at any unobserved location within the study area. 

Kriging provides several advantages, including the ability to incorporate spatial dependence, quantify uncertainty through prediction variances, and produce smooth surfaces. It can be applied to various types of data, such as continuous variables (e.g., pollutant concentrations) or discrete variables (e.g., presence/absence of a disease). The choice of kriging variant depends on the specific characteristics of the data and the desired spatial predictions. In practice, kriging is implemented through specialized software packages and programming languages like R or SAS. 

Overall, kriging is a powerful geostatistical technique that enables spatial interpolation and prediction by incorporating spatial autocorrelation and providing estimates with uncertainty measures. It has wide-ranging applications in fields that require spatial data analysis and prediction.

## 5.3. ADAM
ADAM, short for Adaptive Moment Estimation, is a popular optimization algorithm used in machine learning and deep learning for training models. It's an extension to stochastic gradient descent, which is a commonly used optimization method for training neural networks.
ADAM offers several advantages over basic stochastic gradient descent:

Efficient computation: ADAM only requires first-order gradients (derivatives), and the computation requirements are invariant to diagonal rescaling of the gradients.

Memory requirement: ADAM has a minimal memory requirement as it only needs to keep track of past gradients.

Invariance to the scale of the gradients: ADAM performs well with problems that have noisy and/or sparse gradients.

Parameter updates are invariant to rescaling: Each parameter's update rule in ADAM is essentially independent, which makes it perform well with objectives that have a non-uniform curvature.

Appropriate for non-stationary objectives: Non-stationary objectives are ones where the optimal solution changes over time. ADAM performs well in these situations.

Suitable for problems with very noisy or infrequent gradients: ADAM is often used in reinforcement learning and in training generative adversarial networks where the gradients can be very noisy.

Hyperparameters have intuitive interpretations and typically require little tuning: The default values for ADAM's hyperparameters often perform well, so it's a good choice when you want to quickly train a model without having to do extensive hyperparameter tuning.

ADAM works by maintaining a moving (exponentially decaying) average of past gradients and using this information to adapt the learning rate for each weight in the model individually. This adaptive learning rate approach makes it particularly effective when dealing with sparse data and/or large-scale problems.

However, despite these benefits, it's worth noting that there are also cases where ADAM might not work as well as other methods, such as RMSProp, SGD with momentum, or others. The choice of optimizer often depends on the specific characteristics of the problem at hand.

## 5.4. Filling missing values
### 5.4.1. Mean: 
Mean Imputation is a method for handling missing data where the missing values in a dataset are replaced with the mean (average) of the observed values. The mean is calculated by summing all the observed values and dividing by the count of these values.

This technique is straightforward, fast, and easy to understand, making it a popular choice for initial imputations. However, it comes with certain limitations. It assumes that the data are Missing Completely at Random (MCAR), which means that the probability of a value being missing is independent of both observed and unobserved data. If this assumption does not hold true, mean imputation could lead to biased estimates.

Moreover, while the mean provides a central value, it may not be a good representative of the data if the distribution is skewed or if there are significant outliers, as it does not adequately capture these aspects of data distribution.

Lastly, mean imputation doesn't consider possible correlations between variables, meaning it ignores any potential relationships that could help provide a more accurate imputation of the missing values. This might oversimplify the imputed data and underestimate the variability in the data, reducing the efficiency of subsequent analyses.

### 5.4.2. Median:
Median Imputation is an approach for addressing missing data where the missing values are replaced with the median of the observed values. The median is the middle value in a sorted list of numbers, dividing the distribution into two equal halves. If the count of numbers is odd, the median is the middle number. If it's even, the median is the average of the two middle numbers.

This technique is especially useful when the data distribution is skewed or contains outliers, as the median is less sensitive to such anomalies compared to the mean. It is a simple and robust method to estimate missing values.

Like mean imputation, median imputation is a univariate method, meaning it considers only the distribution of the variable being imputed and ignores potential correlations with other variables. Furthermore, it operates under the assumption that the data are Missing Completely at Random (MCAR). If this is not the case, median imputation could lead to biased estimates, just like mean imputation.  

### 5.4.3. Mode:
The mode is the most frequent value in a column, and it is a less common method for filling missing values. The mode is a good choice if the data is categorical, but it is not a good choice if the data is continuous.

**Advantages**:  
- Simple to implement  
- Easy to understand
- Does not require any assumptions about the distribution of the data

**Disadvantages**:
- Only works for categorical data (it is **not** a good choice if the data is continuous.)
- Can introduce bias into the data if the mode is not representative of the entire data set

### 5.4.4. Constant number or string: 
This method involves filling in the missing values with a constant number or string, such as 0 or "unknown". This method is simple to implement, but it can be inaccurate if the constant value does not represent the true value of the missing data.

### 5.4.5. Miss Forest:
MissForest is a non-parametric method that uses the Random Forest algorithm for missing data imputation. It can handle different types of variables, both continuous and categorical. The algorithm treats each feature with missing values as the dependent variable (target) and the others as independent variables (predictors), and then trains a Random Forest model to predict the missing values.

The process involves:
- 1. Replacing missing values initially with a simple imputation method like mean or mode.
- 2. Iteratively applying the Random Forest model for each variable with missing data until the imputed values don't significantly change between iterations or a set stopping criterion is met.  

While MissForest often outperforms other methods in imputing missing data, it may struggle with high-dimensional data or when there's a high percentage of missing data. Also, due to its iterative nature, it can be computationally intensive and time-consuming for large datasets. The suitability of MissForest depends on the nature of the data and the pattern of missing values. Always consider these factors and explore your data thoroughly before choosing an imputation strategy.

### 5.4.6. Mice Forest:

"Mice Forest" is a Python package that implements the Multiple Imputation by Chained Equations (MICE) algorithm in conjunction with Random Forests. This powerful combination leverages the flexibility of MICE and the predictive power of Random Forests to handle missing data in a variety of datasets, both categorical and numerical.

The MICE algorithm imputes missing values iteratively. In each iteration, a regression model, in this case, a Random Forest, predicts the missing values for a particular variable, using the observed values of the other variables. The predicted values then replace the missing values in the dataset. This process is repeated several times, each time using a different model or set of predictor variables.

"Mice Forest" enriches this process by creating multiple copies of the dataset, each filled with a different imputation method. Common methods include mean imputation, mode imputation, and Random Forest imputation. The results from each imputed dataset are then combined to form a final, imputed dataset.

This multiple imputation approach can help to reduce bias that can occur with single imputation methods, creating a good approximation of the original distribution of the data.

However, there are some important considerations to keep in mind:

- 1. Computation: Mice Forest can be computationally intensive, especially for large datasets.

- 2. Interpretability: As a machine learning model, Random Forest is often seen as a "black box" algorithm. It might be difficult to explain why certain imputations were made.

- 3. Assumptions: The MICE algorithm assumes that the data are Missing At Random (MAR), meaning the probability a value is missing can be related to other observed data. If this is not the case, the imputed values may be biased.

In summary, Mice Forest brings together the robustness of Multiple Imputations and the predictive strength of Random Forests. Despite its potential computational expense and the challenge of interpretability, it offers an effective solution for handling various types of missing data in diverse datasets. However, it's essential to carefully consider the nature of your missing data and understand the assumptions underlying this method to avoid bias in your analysis.

### 5.4.7. KNN Imputation:
KNN Imputation is a non-parametric method for filling missing values. KNN stands for "k-nearest neighbors," and it is a machine learning algorithm that is used to find similar observations in a data set.

KNN Imputation works by first finding the k most similar observations to the observation with the missing value. The k most similar observations are the observations that have the smallest distances to the observation with the missing value. The distances between observations are usually calculated using a Euclidean distance metric.

Once the k most similar observations have been found, the values of the k most similar observations are used to impute the missing value. The most common way to do this is to simply average the values of the k most similar observations.

KNN Imputation is a non-parametric method, which means that it does not make any assumptions about the distribution of the data. This makes KNN Imputation a relatively robust method for filling missing values.

However, KNN Imputation can be computationally expensive, especially for large data sets. Additionally, KNN Imputation can be sensitive to the choice of k. If k is too small, then the imputed values may be too noisy. If k is too large, then the imputed values may be too smooth.

Overall, KNN Imputation is a powerful and robust method for filling missing values. However, it is important to be aware of the limitations of KNN Imputation before using it.

**Advantages**  
- It is a non-parametric method, which means that it does not make any assumptions about the distribution of the data.  
- It is relatively robust to outliers and noise in the data.  
- It can be used to impute missing values in both categorical and continuous data.

**Disadvantages**  
- It can be computationally expensive, especially for large data sets.
- It can be sensitive to the choice of k.
- It can introduce bias into the data if the missing values are not missing completely at random (MCAR).

**Assumption**  
KNN assumes that the dataset has a metric space structure, which may not be the case for all datasets.

![Figure_20](./images_GIC/Figure_20.png)   
*Figure_20: Euclidean (metric) distance **vs** Manhattan (taxi-cab) distance*

![Figure_21](./images_GIC/Figure_21.png)  
*Figure_21: Classification: Majority vote (Mode)*  

![Figure_22](./images_GIC/Figure_22.png)    
*Figure_22: Regression: Mean*  
  
### 5.4.8. Bayesian imputation
Bayesian imputation refers to a probabilistic method used to estimate missing values in a dataset by employing Bayesian principles. Unlike deterministic imputation methods, which fill missing data points with a single fixed value, Bayesian imputation takes into account the uncertainty associated with imputing missing values and often provides multiple plausible imputed datasets.

Here's a high-level overview of Bayesian imputation:

Prior Distribution: Bayesian methods rely on prior beliefs (prior distributions) about the parameters of interest. In the context of imputation, a prior distribution can reflect our beliefs about the process generating the missing values.

Likelihood Function: The likelihood represents the probability of observing the data given some parameters. For imputation, the likelihood would describe the probability of observing the available data under different assumptions about the missing values and the parameters governing their distribution.

Posterior Distribution: By combining the likelihood with the prior using Bayes' theorem, we get the posterior distribution. This posterior distribution gives updated beliefs about the parameters of interest after observing the data.

Imputation Process:

Randomly draw parameter values from the posterior distribution.
Use these parameter values to impute the missing data points.
This process is repeated many times to generate multiple complete datasets. Each dataset will have slightly different imputations based on the random draws from the posterior, reflecting the uncertainty about the missing values.
Analysis: After obtaining multiple imputed datasets, an analysis (e.g., regression) can be performed on each dataset. The results are then combined to produce a single set of estimates that account for imputation uncertainty.

Bayesian imputation offers several benefits:

Uncertainty Handling: By generating multiple imputed datasets, Bayesian imputation provides a way to account for the uncertainty associated with filling in missing data.

Flexibility: It can incorporate complex models and prior information.

Rich Outputs: Instead of a single value, it offers a distribution of plausible values for the missing data, allowing for richer downstream analyses.

However, Bayesian imputation can be computationally intensive due to the need to sample from the posterior distribution multiple times. Additionally, the choice of prior can sometimes be subjective, although non-informative priors can be used when little is known beforehand.

In practice, Bayesian imputation is often performed using software like WinBUGS, JAGS, or Stan, which facilitate Bayesian modeling and inference.

### 5.4.9. Multiple imputation  

### 5.4.10. Expectation-maximization  

### 5.4.11. stochastic regression imputation  

## 5.5. Principal component analysis (PCA)
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of correlated variables into a set of uncorrelated variables called principal components. The principal components are ordered in terms of their variance, with the first principal component having the highest variance and the last principal component having the lowest variance.

PCA is a widely used technique for dimensionality reduction, which is the process of reducing the number of variables in a dataset while preserving as much information as possible. PCA can be used to reduce the dimensionality of a dataset for a variety of purposes, such as:

- **Improving the interpretability of the data**. When a dataset has a large number of variables, it can be difficult to understand the relationships between the variables. PCA can be used to reduce the dimensionality of the dataset, making it easier to visualize and understand the relationships between the variables.
- **Improving the performance of machine learning algorithms**. Many machine learning algorithms are more efficient when they are trained on datasets with a smaller number of variables. PCA can be used to reduce the dimensionality of a dataset, making it easier for machine learning algorithms to learn the relationships between the variables.  

The idea behind PCA is to find a new set of variables that capture the most variation in the original dataset. This is done by finding a set of orthogonal vectors, called principal components, that are aligned with the directions of maximum variance in the dataset. The principal components are then used to create a new dataset, called the principal component space, that is a lower-dimensional representation of the original dataset.

The number of principal components that are used to create the principal component space is determined by the user. The user can choose to use as many or as few principal components as they want, depending on the desired level of dimensionality reduction.

PCA is a powerful tool for dimensionality reduction and data analysis. It is a widely used technique in a variety of fields, including machine learning, statistics, and data mining.

- **Here is an example of how PCA can be used**.

    - **Original dataset**: This is a dataset of 1000 points in 2D space. The points are correlated, meaning that they tend to cluster together.  
![Figure_23](./images_GIC/Figure_23.png)    
*Figure_23*

    - **Principal components**: PCA finds two principal components that capture the most variation in the original dataset. The first principal component is shown in blue, and the second principal component is shown in red.  
![Figure_24](./images_GIC/Figure_24.png)    
*Figure_24*

    - **Principal component space**: The principal component space is a new representation of the original dataset in 2D space. The points in the principal component space are uncorrelated, meaning that they do not cluster together.  
![Figure_25](./images_GIC/Figure_25.png)    
*Figure_25*

As you can see, PCA has transformed the original dataset of 1000 points in 2D space into a new dataset of 2 points in 2D space. The new dataset is a lower-dimensional representation of the original dataset, but it still captures most of the information in the original dataset. 

## 5.6. LSTM     
LSTM stands for Long Short-Term Memory. It is a type of recurrent neural network (RNN) that is well-suited for sequential data. Sequential data is data that is arranged in a sequence, such as time series data, natural language, and speech.

LSTM networks are able to learn long-term dependencies in sequential data. This means that they can remember information from previous steps in the sequence, and use that information to make predictions about future steps. This is important for tasks such as machine translation, speech recognition, and time series forecasting.

LSTM networks work by using a memory cell to store information about previous steps in the sequence. The memory cell is a special type of neural network unit that has three gates: the input gate, the forget gate, and the output gate.

- **Input gate**: The input gate controls how much information is added to the memory cell. The input gate is a sigmoid function that takes as input the current input and the previous state of the memory cell. The output of the input gate is a vector of numbers between 0 and 1. The numbers represent how much of the current input and the previous state of the memory cell should be added to the memory cell.  

- **Forget gate**: The forget gate controls how much information is removed from the memory cell. The forget gate is also a sigmoid function that takes as input the current input and the previous state of the memory cell. The output of the forget gate is a vector of numbers between 0 and 1. The numbers represent how much of the previous state of the memory cell should be forgotten.

- **Output gate**: The output gate controls how much information is output from the memory cell. The output gate is also a sigmoid function that takes as input the current input and the previous state of the memory cell. The output of the output gate is a vector of numbers between 0 and 1. The numbers represent how much of the current state of the memory cell should be output.

The LSTM network uses these gates to control the flow of information in the memory cell. This allows the network to learn long-term dependencies in sequential data.

LSTM networks have been shown to be very effective for a variety of tasks, including:

- **Machine translation**: LSTM networks have been used to translate text from one language to another. For example, LSTM networks have been used to translate English to French, German, and Spanish.  

- **Speech recognition**: LSTM networks have been used to recognize speech. For example, LSTM networks have been used to recognize words and phrases in English, French, and German.  

- **Time series forecasting**: LSTM networks have been used to forecast time series data. For example, LSTM networks have been used to forecast stock prices, weather patterns, and energy consumption.

LSTM networks are a powerful tool for machine learning. They are able to learn long-term dependencies in sequential data, which makes them well-suited for a variety of tasks.  

**Pros of LSTM**:

- **Long-Term Dependencies**: LSTMs are explicitly designed to avoid the long-term dependency problem that traditional RNNs face. They can remember and recall information for long periods of time, which is extremely useful in many applications.

- **Handling of Time Series**: LSTMs are excellent at learning from sequences of data, making them particularly useful for time series analysis, natural language processing, and other sequential tasks.

- **Flexibility**: LSTMs can handle sequences of variable lengths and are not required to have their inputs and outputs be synchronized.

**Cons of LSTM**:

- **Computational Complexity**: LSTMs can be quite complex and computationally intensive, leading to longer training times. This can be a problem when dealing with large datasets or when resources are limited.

- **Need for Large Datasets**: LSTMs typically require large amounts of data to train effectively without overfitting. For smaller datasets, simpler models may be more effective.

- **Complexity and Difficulty to Train**: The complex structure of LSTMs makes them more difficult to understand and train. The presence of various gates and the cell state can make it difficult to interpret what the model has learned.

- **Risk of Overfitting**: Like any complex machine learning model, LSTMs can be prone to overfitting, especially when used with small datasets. Techniques like regularization, dropout, and early stopping are often needed to prevent this.

 - **Difficulty Handling Longer Sequences**: While LSTMs are designed to handle the vanishing gradient problem better than traditional RNNs, they can still struggle with it when sequences are very long. Recently, transformers, a different architecture, have been found to handle such long sequences more effectively.

## 5.7. CNN  

## 5.8. SVM  

## 5.9. RF  


# 6. Papers

## 6.1. PILM: A Survey on Problems, Methods and Applications
 Introduction
- Traditional machine learning algorithms have limitations in incorporating prior knowledge or constraints about physical systems. 
Traditional machine learning algorithms are designed to learn patterns and relationships from data without any prior knowledge or constraints about the physical system being modeled. These algorithms are often based on statistical methods that aim to optimize a specific objective function, such as minimizing prediction error or maximizing accuracy. However, in many real-world applications, there is often prior knowledge or constraints about the physical system that can be leveraged to improve the accuracy and efficiency of machine learning models. For example, in predicting GIC, there is prior knowledge about the Earth's magnetic field and its interactions with solar wind and other factors that cause GIC. Traditional machine learning algorithms have limitations in incorporating this type of prior knowledge or constraints into their models. They may not be able to capture complex relationships between variables or account for physical laws and principles that govern the behavior of the system being modeled. This is where Physics-Informed Machine Learning (PIML) comes in. PIML is a paradigm that seeks to construct models that make use of both empirical data and prior physical knowledge to enhance performance on tasks that involve a physical mechanism. By incorporating physical prior knowledge into machine learning models, PIML can overcome the limitations of traditional machine learning algorithms and improve accuracy and efficiency in tasks such as predicting GIC.

- Physics-informed machine learning (PIML) is a new approach that seeks to integrate empirical data and physical prior knowledge into machine learning models. 

- The paper presents a survey of recent developments in PIML and discusses its potential applications. 

Problem Formulation: 
- PIML addresses fundamental problems in physics-informed machine learning. 

- Representation methodology for physical knowledge, approach for integrating physical knowledge into machine learning models, and practical problems that PIML resolves are discussed. 

- The paper proposes a theoretical framework for machine learning problems with physical constraints based on probabilistic graphical models using latent variables to represent the real state of a system that satisfies physical prior constraints. 

- The paper also discusses the challenges associated with incorporating physical prior knowledge into machine learning models such as the need for domain expertise, availability of data, and computational complexity. 

Possible Ways towards PIML: 
- The incorporation of physical prior knowledge can be achieved through modifications to one or more components of the machine learning model. 

- Training data, model architecture, loss functions, optimization algorithms, and inference can all be modified to incorporate physical prior knowledge. 

- The paper discusses several approaches for incorporating physical prior knowledge into machine learning models such as hybrid modeling, probabilistic graphical models, and feature engineering. 

- The paper also provides examples of how different types of physical prior knowledge can be incorporated into machine learning models such as conservation laws, symmetries, and constitutive relations. 

Applications of PIML: 
- PIML has been successfully applied in various fields such as fluid dynamics, materials science, and robotics. 

- By incorporating physical prior knowledge into machine learning models, improved performance and better alignment with practical problems that adhere to the laws of physics can be achieved. 

- The paper provides examples of successful applications of PIML such as predicting fluid flow behavior in complex geometries, designing new materials with desired properties, and controlling robotic systems. 

Conclusion: 
- PIML is a promising new paradigm for solving physical problems using machine learning. 

- By integrating empirical data and physical prior knowledge into machine learning models, it is possible to overcome the limitations of traditional machine learning algorithms and achieve improved accuracy and efficiency. 

- The paper concludes by discussing future directions for research in PIML such as developing more efficient algorithms for incorporating physical prior knowledge, exploring new types of physical prior knowledge


## 6.2. a real-time GMD monitoring system
A real-time GMD (Geomagnetic Disturbance) monitoring system is designed to track and assess geomagnetic events and their potential impact on power grids and electrical systems. Geomagnetic disturbances occur when the Earth's magnetic field is disrupted by solar flares or other space weather phenomena. These disturbances can induce currents in power grids, known as Geomagnetically Induced Currents (GICs), which can pose risks to the stability and reliability of electrical infrastructure.

The purpose of a real-time GMD monitoring system is to provide continuous monitoring and assessment of geomagnetic activity, allowing operators to detect and respond to potential GIC-related issues promptly. Such a system typically involves the following components:

Magnetic Field Sensors: The monitoring system incorporates a network of magnetometers or magnetic field sensors strategically placed in different locations. These sensors measure the strength and variation of the Earth's magnetic field in real-time.

Data Acquisition: The sensors capture magnetic field data at regular intervals and transmit it to a central data acquisition system. This system collects and processes the data for analysis.

Data Analysis and Processing: Real-time data analysis algorithms are employed to assess the geomagnetic conditions and identify any anomalies or disturbances. These algorithms analyze the data to detect rapid changes or abnormal variations in the magnetic field associated with geomagnetic disturbances.

Alerting and Notification: When significant geomagnetic disturbances are detected, the monitoring system generates alerts and notifications to inform operators or system administrators. These alerts provide timely information about the potential risks and allow for proactive measures to mitigate any adverse effects on the power grid.

Integration with Power Grid Systems: The real-time GMD monitoring system is often integrated with the control and monitoring infrastructure of the power grid. This integration allows operators to correlate the geomagnetic data with other operational parameters, such as current flows and system stability, to assess the potential impact of GICs on the grid components.

The implementation of a real-time GMD monitoring system is essential for grid operators and utility companies to understand and manage the potential risks associated with geomagnetic disturbances. By continuously monitoring the Earth's magnetic field and promptly detecting changes, the system enables proactive responses, such as adjusting power grid operations, implementing protective measures, or isolating vulnerable components, to mitigate the impact of GICs on electrical systems.  

# 7. Questions  

## 7.1. Question 1:  
**How do various solar wind conditions (e.g., IMF components, speed, density, level of turbulence) and different large-scale drivers control the coupling efficiency and the energy/mass transfer from the solar wind to the magnetosphere?**   

The Sun's continuous emission of charged particles, known as the solar wind, interacts with Earth's magnetosphere, the region around our planet dominated by its magnetic field. This interaction is an intricate process, dependent on various properties of the solar wind and controlled by different large-scale drivers.  

- **Interplanetary Magnetic Field (IMF) Components**: The direction and strength of the IMF, particularly the southward component (Bz), strongly influence the level of energy transfer from the solar wind to the magnetosphere. When Bz is southward, it opposes Earth's magnetic field, which allows for reconnection at the dayside magnetopause (the boundary separating the magnetosphere and the solar wind). This reconnection opens the magnetosphere to the solar wind, allowing energy, momentum, and mass to enter. 
    
![Figure_29](./images_GIC/Figure_29.png)   
*Figure_29: A magnetosphere for northward IMF with 3 reconnection sites. Solid lines indicate the magnetic field and the arrowheads indicate the direction of the flow. Reconnection sites are indicated by X.*

- **Solar Wind Speed**: The faster the solar wind, the greater the dynamic pressure on the magnetosphere, compressing it on the dayside and extending the tail on the nightside. High-speed streams can cause geomagnetic storms and enhance energy input into the magnetosphere. 

![Figure_30](./images_GIC/Figure_30.png)   
*Figure_30: Flow of plasma energy around Earth's magnetosphere. Solar energy absorbed through the magnetopause circulates in the magnetosphere and becomes energy that generates the radiation belt and auroras.*

- **Solar Wind Density**: The solar wind density also affects solar wind-magnetosphere coupling. A denser solar wind increases the dynamic pressure on the magnetosphere than a less dense solar wind. This is because a denser solar wind will have more particles, which can interact with the magnetosphere and transfer energy and momentum. 

![Figure_31](./images_GIC/Figure_31.png)   
*Figure_31: Solar wind entry regions in the high-latitude terrestrial magnetosphere as detected by Cluster.* 

- **Solar Wind Turbulence**: Turbulence in the solar wind can also affect solar wind-magnetosphere coupling. Turbulence can increase the rate of magnetic reconnection, which is the process by which the magnetic field lines of the solar wind and the magnetosphere are reconnected. This can allow more energy and momentum to be transferred from the solar wind to the magnetosphere.  

![Figure_32](./images_GIC/Figure_32.png)   
*Figure_32: Sketch of turbulent wave packages in Jupiter's magnetosphere. The interaction of counter-propagating Alfvén wave packages generates a turbulent cascade. Waves packages are reflected at the ionosphere of Jupiter or other density gradients in the magnetosphere.*

- **Large-Scale Drivers**: Solar events like Coronal Mass Ejections (CMEs) or co-rotating interaction regions (CIRs) carry enhanced solar wind parameters (higher speed, density, and stronger magnetic fields). These can lead to intense geomagnetic storms by dramatically increasing the energy input to the magnetosphere.    

⮚ **Coronal mass ejections (CMEs)**: CMEs are large clouds of plasma and magnetic field that are ejected from the Sun. CMEs can have a significant impact on the magnetosphere, causing large geomagnetic storms.  

![Figure_33](./images_GIC/Figure_33.png)   
*Figure_33*  

⮚ **Geomagnetic storms**: Geomagnetic storms are large disturbances in the Earth's magnetosphere that are caused by CMEs. Geomagnetic storms can cause power outages, disrupt communications, and damage satellites.  

⮚ **Substorms**: Substorms are smaller disturbances in the Earth's magnetosphere that occur on a regular basis. Substorms can cause auroras and can also disrupt communications and satellites.  

![Figure_34](./images_GIC/Figure_34.png)   
*Figure_34: The main magnetospheric features of the substorm growth phase.*  

These factors combine to control the efficiency of the energy and mass transfer from the solar wind to the magnetosphere. The more efficient this transfer, the more likely it is to cause geomagnetic disturbances, such as substorms and geomagnetic storms, that can impact our technology-dependent society. Predicting these disturbances and understanding the factors that affect them are major goals of space weather research.

## 7.2. Question 2:   
**How do solar wind conditions control the occurrence frequency and location of different magnetospheric plasma waves?**  

The occurrence frequency and location of magnetospheric plasma waves are significantly influenced by solar wind conditions. The solar wind, a stream of charged particles flowing out from the Sun, interacts with the Earth's magnetosphere, creating different types of plasma waves.

- Solar Wind Speed and Density: Higher solar wind speeds and densities increase the energy and dynamic pressure exerted on the Earth's magnetosphere. This can enhance the excitation of magnetospheric plasma waves, including Ultra Low Frequency (ULF) waves, Very Low Frequency (VLF) waves, and others. The location of these waves depends on how the pressure distorts the magnetosphere, which can affect the plasma's distribution and motion within it.

- Interplanetary Magnetic Field (IMF): The orientation of the IMF plays a crucial role in determining where and when magnetospheric plasma waves occur. A southward-directed IMF can open magnetic reconnection sites on the day-side magnetopause, allowing solar wind particles to enter the magnetosphere and alter plasma conditions. This can stimulate the production of waves, such as magnetosonic waves or whistler waves, at specific locations in the magnetosphere where conditions are ripe. Conversely, a northward IMF orientation generally lessens wave activity.

- Solar Wind Turbulence: The level of turbulence or fluctuation in the solar wind can create instabilities that generate magnetospheric plasma waves. These can occur at various locations in the magnetosphere, particularly at boundaries where plasma conditions change rapidly, such as the magnetopause and the plasmapause.

- Large-Scale Solar Events: Events such as Coronal Mass Ejections (CMEs) and solar flares can dramatically change solar wind conditions. These sudden increases in solar wind energy input can create strong disturbances in the magnetosphere, exciting various plasma waves and causing them to propagate to different regions. This includes phenomena like magnetospheric chorus waves, which are enhanced during geomagnetic storms driven by solar events.

The interaction between solar wind and the Earth's magnetosphere is a complex process involving many factors. The occurrence of different types of magnetospheric plasma waves depends on a combination of these factors, and research is ongoing to better understand these interactions. 

## 7.3. Question 3:
**What is the differences between Gaussian distributions and lognormal laws?**  

The Gaussian (also known as Normal) distribution and the Lognormal distribution are both important statistical distributions, but they differ significantly in their properties and applications. Here are some of the key differences:

1. Shape and Symmetry:

Gaussian Distribution: This distribution is symmetric and its graph forms a bell curve, with the mean, median, and mode all equal and located at the center of the distribution.

Lognormal Distribution: This distribution is not symmetric; it is skewed to the right (positive skewness), with a longer tail towards the right side. The mean, median, and mode are not equal and the mode is less than the median, which in turn is less than the mean.  

![Figure_37](./images_GIC/Figure_37.png)   
*Figure_37: Normal distribution: The mean, median, and mode are all equal in the normal distribution and other symmetric distributions. *

![Figure_38](./images_GIC/Figure_38.png)   
*Figure_38: Lognormal (Right skewed): The mean is greater than the median. The mean overestimates the most common values in a positively skewed distribution.*

![Figure_39](./images_GIC/Figure_39.png)   
*Figure_39: Lognormal (Left skewed): The mean is less than the median. The mean underestimates the most common values in a negatively skewed distribution.*

2. Domain:

Gaussian Distribution: The Gaussian distribution is defined over the entire real line, i.e., from negative infinity to positive infinity.

Lognormal Distribution: The Lognormal distribution is only defined for positive values, i.e., from zero to positive infinity. It cannot take negative values.

3. Underlying Assumptions:

Gaussian Distribution: This distribution assumes that the data being modeled has been generated by a process that has additive effects and the overall effect is due to the sum of individual effects.

Lognormal Distribution: This distribution assumes that the data being modeled has been generated by a multiplicative process, and the overall effect is due to the product of individual effects.

4. Applications:

Gaussian Distribution: It is used extensively in statistics and natural sciences, and is the foundation of many statistical procedures. It's often used to model phenomena such as IQ scores, heartbeats, etc.

Lognormal Distribution: It is often used to model data that cannot go below zero but can become very large, such as populations, prices of goods, and information in bits.  

# 8. Code optimization  

## 8.1. Pure Python to Scientific Phyton  
Python is a powerful language that's easy to learn and use, but it can be slow when dealing with large datasets or complex mathematical operations. This is where scientific Python packages like NumPy, SciPy, and pandas can help.

- **Vectorization with NumPy**: Pure Python loops can be slow. NumPy, a fundamental package for scientific computing in Python, provides support for arrays. These arrays can be multi-dimensional and can do element-wise operations, which is much faster than looping over elements.  
For example, instead of using a for loop to iterate over a list of numbers and add them together, you can use the numpy.sum() function to vectorize the operation and add all of the numbers together in a single step. 

![Figure_35](./images_GIC/Figure_35.png)   
*Figure_35: Use vectorized operations instead of loops.*

- **Data Manipulation with pandas**: pandas provides high-performance, easy-to-use data structures like DataFrames and operations for manipulating numerical tables and time-series data.  

![Figure_36](./images_GIC/Figure_36.png)   
*Figure_36: Use DataFrames instead of loops.*

- **Scientific Computing with SciPy**: SciPy is built on NumPy and provides more utilities for optimization, linear algebra, integration, interpolation, special functions, FFT, signal and image processing, ODE solvers, and more.

- **Parallel Computing with joblib or Dask**: These libraries allow for easy parallelization of tasks, which can significantly speed up processing times.

- **Just-In-Time Compilation with Numba**: Numba is a just-in-time compiler for Python that's best suited for mathematical functions, like those used in machine learning and data analysis. With a few annotations, array-oriented and math-heavy Python code can be just-in-time optimized to achieve performance similar to C, C++ and Fortran, without having to switch languages or Python interpreters.

- **Efficient Array Computing with JAX**: JAX is like NumPy, but with powerful transformations like automatic differentiation, vectorization, parallelization, and just-in-time compilation to GPU/TPU.

- **Profiling**: The other answer suggests using a profiler to find bottlenecks in the code. A profiler can help you to identify the parts of your code that are taking the most time to execute. Once you have identified the bottlenecks, you can focus your optimization efforts on those areas.  



