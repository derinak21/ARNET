# ARNET
Animal Recognition Network

This is a project for IBM Z Datathon.
[Link to our presentation](https://github.com/derinak21/ARNET/Presentation.pdf)

# Team: Enemy of Syntax
Derin Ak
Ratnali Shwati Sreekeessoon	
Ayham Al-Saffar	
Aditya Tandon	

# Problem Statement: 
Preventing logging and oil companies from redeveloping important areas such as the Amazon rainforest is a great struggle for governmental and non-governmental organizations for wildlife conservation. If these areas are redeveloped, many endemic species could be forever removed from the earth due to habitat loss. One of the main reasons behind this struggle is that it is often quite difficult to prove and catalogue the existence of these endemic species. The central problem that we address is the cataloguing of these endemic species. 

# Proposed Solution: 
The proposed solution is to train an animal image recognition network to be able to quickly prove whether an animal only photographed a few times is in fact a new species or not. We plan to take a trained animal image classifier network and fine tune it with triplet loss function to act as a Siamese face recognition type network. Furthermore, this network would also allow people to quickly add new species to the database as new species are discovered. Thus, this would lead to cataloguing and differentiating between hundreds of species as they are being discovered. 

# Target Audience: 
Governmental and nongovernmental organizations that are concerned about wildlife conservation. These organizations can use the proposed solution to differentiate between different of known and also newly discovered species and take right actions to protect them. 

# Datasets: 
We are planning to use CIFAR100, ImageNet and Kaggle animal image datasets for this proposed solution. For the future scope of this project, more advanced networks can be developed using Snapshot Serengeti dataset for mammalian species and even INaturalist dataset for plants since cataloguing and endemic plants for conservation. 
