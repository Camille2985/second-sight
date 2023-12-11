# Second Sight
### UCB W210 Capstone Project 
#### Camille Church, Amanda Teschko, Cameron Wright, Lisa Wu
<hr />

### Motivation
<p>
Our project was motivated by the strong need for affordable solutions for those with visual impairment. In the US alone, 20 million people are affected by visual impairments that make reading and processing text difficult. Of the 20 million affected in the US, 27% live below the poverty level. While text-reading tools exist on the market, many come with high price tags.
</p>

<hr />

### Architecture
  <img width="543" alt="Screenshot 2023-12-11 at 6 54 24 PM" src="https://github.com/Camille2985/second-sight/assets/36643562/9a1482e2-d033-44b4-848b-f949f5b12a05">
<br />
<li> The overall process begins with a developer applying the Kubernetes manifests to our EKS cluster to build the infrastructure for the training environment.</li>
<li> Next, in step 2, a developer checks their code into Github, which builds a docker image, pushes it to Amazon’s Elastic Container registry, and reapplies the manifest with the latest image via a GitHub action.</li>
<li> In step 3, the developer triggers a run of the model’s training, which Cameron will cover in more detail. At the end of the training, the model object is saved as a .joblib file and stored in an S3 bucket.</li>
<li> Lastly, the file containing the model object is retrieved and used in conjunction with Apple’s Vision SDK to make predictions within the mobile application that is deployed to the Apple App Store.</li>

