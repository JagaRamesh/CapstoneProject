# Clustering Bay Area Cities on Housing Price

![page1](/images/page1.png)
### Objective :
The objective of the project is to cluster bayarea zipcodes using time series data of housing price for the last 10 years. 

### Approach :
1. Gather the last 10 year Median Home value for all major bayarea zip codes. Use clustering algorithm Kmeans to group the zipcodes with similar housing market price.
2. To minimize noise and to accurately group the areas, apply Bootstrapping on time series data, and repeat the Kmeans for 1000 iterations. Identify the groups based on the consistency in clustering together in repeated iterations.


### Code :
Program code can be found [here](https://github.com/JagaRamesh/CapstoneProject/tree/master/Code)

