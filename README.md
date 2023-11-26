# Decision-Tree
Decision trees, including algorithms like ID3, C4.5, and CART, are powerful tools for classification and regression tasks. ID3, developed by Ross Quinlan, uses Information Gain to select the best attribute at each node, recursively dividing the data until homogenous subsets are achieved. C4.5, an improvement over ID3, addresses its limitations by using Information Gain Ratio, allowing for better handling of continuous attributes and reducing sensitivity to irrelevant features. CART (Classification and Regression Trees) is a versatile algorithm that can be used for both classification and regression tasks. It employs Gini impurity for classification, aiming to minimize impurity in each split, and mean squared error for regression. These decision tree algorithms are widely employed in machine learning due to their interpretability and effectiveness in capturing complex relationships within data.

## C4.5
This Python code implements a decision tree model using the C4.5 algorithm to classify Fisher's Iris dataset. The script begins by importing necessary libraries and loading the Iris dataset, converting numerical target values to corresponding class names. The loaded dataset is displayed, showcasing its structure.

Shannon entropy is calculated as a measure of dataset impurity. The script then defines functions for splitting datasets, calculating information gain, and constructing the decision tree based on the best features. The resulting tree is visualized using a provided tree plotting module.

To demonstrate the decision tree's functionality, an example is provided for classifying an Iris instance. Users can replace the example instance with actual feature values. The predicted class is then printed. This code provides a clear and practical implementation of the C4.5 decision tree algorithm for classification tasks, especially useful for understanding and visualizing decision tree structures.

## CART
The script begins by importing necessary libraries, including pandas, numpy, and seaborn. It then loads the "tips" dataset using seaborn and displays a few examples to provide insights into the dataset's structure.

Next, a class Node is defined to represent a node in the decision tree. This class encapsulates information about the splitting feature, threshold, prediction value for a leaf, and references to the left and right subtrees.

The script includes a function to calculate the Gini index, a measure of impurity used in the CART algorithm. Another function is defined to split the data based on a given feature and threshold value.

The find_best_split function identifies the best feature and threshold for splitting the data to minimize the Gini index. This is a crucial step in building an effective decision tree.

A recursive function, build_tree, constructs the decision tree based on the identified splits. The process continues until a specified depth is reached or a node becomes pure (contains only one class).

To make predictions, the script provides functions predict_one for a single observation and predict for a set of observations using the constructed decision tree.

Finally, the script demonstrates the entire process by loading the tips dataset, preparing the data, building a decision tree with a maximum depth of 3, and making predictions for the examples in the dataset.

In summary, the script offers a clear and practical implementation of a CART decision tree for regression tasks, particularly applied to the "tip" column in the Tips dataset. The provided functions and structure facilitate understanding and application of the CART algorithm.
