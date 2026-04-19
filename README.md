# Hyperparameter tuning using bio-inspired algorithms in GCN, GAN and GraphSAGE models for link prediction in a PPI network 

## Overview

This project is made to compare different biologically inspired algorithms like Genetic Algorithm, Particle Swarm Optimization, Ant Colony Optimization, Artificial Bee Colony and classic ones like Grid Search, Simulated Annealing, Hill Climbing and Random Search as well as Optuma and Bayesian Optimization in hyperparameter optimization in Graph Convolutional Network, Generative Adversarial Network and GraphSAGE models for link prediction in a SNAPS protein-protein interaction network. 

## 🧬 Protein-Protein Interaction Network: PP-Pathways
This repository explores the <a href="https://snap.stanford.edu/biodata/datasets/10000/10000-PP-Pathways.html">PP-Pathways dataset from the Stanford SNAP BioData collection</a>. It represents a large-scale protein-protein interaction (PPI) network derived from pathway databases.

<ul>
<li>Nodes: 21,554 proteins</li>
<li>Edges: 342,338 interactions</li>
<li>Data Type: Undirected, unweighted graph</li>
<li>Source: Pathway-based protein associations</li>
<li>Format: Edge list (.csv) with each row representing a protein-protein interaction</li>
</ul>

### Graph visualization for the PPI network
<img src="ppi-visualization.png" alt="Protein Graph" width="500"/>
This image was done using <a href="https://cytoscape.org/">Cytoscape</a>.

## Results
The following results are obtained after 10 epochs of training in each GCN and GAN model.


<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Algorithm</th>
      <th>F1</th>
      <th>AUC</th>
      <th>Loss / Avg Loss</th>
      <th>NDCG</th>
      <th>Hidden Channels</th>
      <th>Learning Rate</th>
      <th># Layers</th>
      <th>Dropout</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr><td rowspan="11">GCN</td><td>None</td><td>0.8378</td><td>0.9097</td><td>1.3414</td><td>0.9910</td><td>256</td><td>0.01</td><td>3</td><td>0</td><td>81s</td></tr>
    <tr><td>GA</td><td>0.8506</td><td>0.9125</td><td>1.2876</td><td>0.9913</td><td>73</td><td>0.0122</td><td>3</td><td>0.4</td><td>560s</td></tr>
    <tr><td>PSO</td><td>0.8506</td><td>0.779</td><td>1.392</td><td>0.968</td><td>107</td><td>0.01339</td><td>3</td><td>0.55</td><td>716s</td></tr>
    <tr><td>ABC</td><td>0.8496</td><td>0.8831</td><td>1.4161</td><td>0.9885</td><td>106</td><td>0.00691</td><td>3</td><td>0.15</td><td>868s</td></tr>
    <tr><td>Simulated Annealing</td><td>0.8435</td><td>0.7841</td><td>1.3751</td><td>0.9685</td><td>160</td><td>0.00974</td><td>3</td><td>0.23</td><td>798s</td></tr>
    <tr><td>Hill Climbing</td><td>0.843</td><td>0.9109</td><td>1.3705</td><td>0.9914</td><td>244</td><td>0.01102</td><td>3</td><td>0.66</td><td>550s</td></tr>
    <tr><td>Random Search</td><td>0.8493</td><td>0.9147</td><td>1.2535</td><td>0.992</td><td>38</td><td>0.01378</td><td>3</td><td>0.18</td><td>740s</td></tr>
    <tr><td>ACO</td><td>0.8419</td><td>0.9145</td><td>1.2698</td><td>0.9918</td><td>224</td><td>0.00215</td><td>3</td><td>0.7</td><td>330s</td></tr>
    <tr><td>Bayesian Search</td><td>0.8504</td><td>0.9154</td><td>1.2674</td><td>0.9911</td><td>80</td><td>0.00785</td><td>4</td><td>0.1</td><td>622s</td></tr>
    <tr><td>Grid Search</td><td>0.8497</td><td>0.9151</td><td>1.2031</td><td>0.992</td><td>64</td><td>0.01</td><td>3</td><td>0.3</td><td>553s</td></tr>
    <tr><td>Optuna</td><td>0.8512</td><td>0.9166</td><td>1.2522</td><td>0.992</td><td>277</td><td>0.00153</td><td>4</td><td>0</td><td>1342s</td></tr>
  </tbody>
  <tbody>
    <tr><td rowspan="11">GAN</td><td>None</td><td>0.7337</td><td>0.7528</td><td>0.1522</td><td>0.9699</td><td>256</td><td>1e-4</td><td>-</td><td>0.3</td><td>81s</td></tr>
    <tr><td>GA</td><td>0.7538</td><td>0.7772</td><td>0.0088</td><td>0.9723</td><td>459</td><td>0.0019</td><td>-</td><td>0.33</td><td>539s</td></tr>
    <tr><td>PSO</td><td>0.7571</td><td>0.7781</td><td>0.0147</td><td>0.9706</td><td>512</td><td>0.002</td><td>-</td><td>0.31</td><td>922s</td></tr>
    <tr><td>ABC</td><td>0.7545</td><td>0.7773</td><td>0.0421</td><td>0.9724</td><td>480</td><td>0.00136</td><td>-</td><td>0.22</td><td>392s</td></tr>
    <tr><td>Simulated Annealing</td><td>0.7584</td><td>0.7583</td><td>-0.026</td><td>0.9699</td><td>412</td><td>0.002</td><td>-</td><td>0.1</td><td>1092s</td></tr>
    <tr><td>Hill Climbing</td><td>0.7559</td><td>0.7773</td><td>0.0015</td><td>0.9726</td><td>393</td><td>0.002</td><td>-</td><td>0.4</td><td>707s</td></tr>
    <tr><td>Random Search</td><td>0.7541</td><td>0.7752</td><td>0.0432</td><td>0.9725</td><td>367</td><td>0.00166</td><td>-</td><td>0.2</td><td>516s</td></tr>
    <tr><td>ACO</td><td>0.7167</td><td>0.7353</td><td>0.017</td><td>0.9649</td><td>384</td><td>0.00003</td><td>-</td><td>0.3</td><td>809s</td></tr>
    <tr><td>Bayesian Search</td><td>0.7505</td><td>0.7743</td><td>0.2257</td><td>0.9725</td><td>144</td><td>0.00062</td><td>-</td><td>0.35</td><td>595s</td></tr>
    <tr><td>Grid Search</td><td>0.7428</td><td>0.763</td><td>0.0554</td><td>0.9712</td><td>512</td><td>0.0001</td><td>-</td><td>0</td><td>726s</td></tr>
    <tr><td>Optuna</td><td>0.756</td><td>0.7817</td><td>0.2407</td><td>0.9738</td><td>373</td><td>0.00068</td><td>-</td><td>0.51</td><td>1224s</td></tr>
  </tbody>
</table>



## Quickstart

### Prerequisites
- Python 3.8+
- [scikit-learn](https://scikit-learn.org/stable/)
- [PyTorch](https://pytorch.org/get-started/locally/) 
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)
- [NiaPy](https://niapy.org/en/stable/index.html#niapy-s-documentation)

### Installation
```bash
git clone https://github.com/milagjurovska/PPI-link-prediction-with-optimized-gcn-and-gan.git
cd PPI-link-prediction-with-optimized-gcn-and-gan
```
The results can be displayed in the Jupyter Notebook provided, however if you want to run the code in Python only, there is a results.py file.
