# LEGO-Cryst: Local Environment Geometry-Oriented Crystal Generator

LEGO-Cryst (Local Environment Geometry-Oriented Crystal Generator) is designed for the rapid generation of crystal structures for complex materials characterized by well-defined local structural building units (SBUs). By leveraging a small set of known material examples, LEGO-Cryst learns target crystal structures by exploiting crystal symmetry. A generative model navigates the vast combinatorial space of symmetry operations to produce numerous trial structures, which are then optimized to ensure each atom satisfies the desired local atomic environment.

The LEGO-Cryst workflow is illustrated below using sp²-hybridized carbon structures as an example:

1.  **Training Data Collection and Augmentation:**
    *   **1.1 Data Selection:** Curate a dataset of known structures (e.g., 140 sp² carbon structures from the SACADA database).
    *   **1.2 Crystal Representation:** Describe each structure using an irreducible crystal representation, encoding space group number, cell parameters, and Wyckoff site information (typically resulting in a ~40-column feature vector).
    *   **1.3 Symmetry-Based Data Augmentation:** Generate alternative representations based on group-subgroup relationships to maximize the utility of crystal symmetry. This allows the model to learn from diverse symmetry perspectives and ensures sufficient data for training.

2.  **Featurization:**
    *   Represent each crystal in the augmented dataset (e.g., ~60,000 rows × 40 columns) using a mix of continuous (cell parameters, Wyckoff coordinates) and discrete features (space group number, Wyckoff indices).
    *   Process these features using techniques like one-hot encoding and Gaussian mixture models to enhance model learning.

3.  **Generative Model Training:**
    *   Use the transformed feature table to train various generative models, such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), or Transformers.

4.  **Sampling and Optimization:**
    *   Employ the trained models to generate a large number of novel crystal structure candidates.
    *   Optimize each generated structure using L-BFGS or Adam optimizers to match a reference local environment, described by radial and Fourier distribution functions.

5.  **Database Construction:**
    *   Compile the generated and validated structures into a database for further analysis (e.g., machine learning or DFT energy, density, eletronic band structures).

By utilizing different SBUs as training sources, LEGO-Cryst can rapidly generate high-quality crystal candidates for designing complex materials, including metal-organic frameworks (MOFs) and battery materials. Combining materials science domain knowledge with advanced AI methodologies, LEGO-Cryst aims to advance AI-driven crystal structure generation, paving the way for more efficient materials discovery and design.