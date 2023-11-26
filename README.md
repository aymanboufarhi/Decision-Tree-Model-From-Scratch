# Decision-Tree
Decision trees, including algorithms like ID3, C4.5, and CART, are powerful tools for classification and regression tasks. ID3, developed by Ross Quinlan, uses Information Gain to select the best attribute at each node, recursively dividing the data until homogenous subsets are achieved. C4.5, an improvement over ID3, addresses its limitations by using Information Gain Ratio, allowing for better handling of continuous attributes and reducing sensitivity to irrelevant features. CART (Classification and Regression Trees) is a versatile algorithm that can be used for both classification and regression tasks. It employs Gini impurity for classification, aiming to minimize impurity in each split, and mean squared error for regression. These decision tree algorithms are widely employed in machine learning due to their interpretability and effectiveness in capturing complex relationships within data.

## Implementation from Scratch: C4.5 and CART Decision Trees
In the provided code examples, both the C4.5 and CART decision tree algorithms have been implemented from scratch, showcasing a hands-on approach to understanding and constructing these fundamental machine learning models.

## C4.5 Decision Tree Implementation
The C4.5 algorithm, introduced by Ross Quinlan, is a classic decision tree algorithm designed for classification tasks. In this implementation, Python is used to create a decision tree for classifying instances in Fisher's Iris dataset. The code takes a step-by-step approach, starting with dataset loading, entropy calculation, and recursive tree construction based on information gain.

## CART Decision Tree Implementation
The CART (Classification and Regression Trees) algorithm, known for its versatility in handling both classification and regression tasks, is also implemented from scratch in the provided code. Focused on regression, the script approximates the "tip" column in the Tips dataset using a CART decision tree model. Key components include Gini index calculation, data splitting, and recursive tree building.

## Hands-On Learning Experience
By building these decision tree models from scratch, the code offers a valuable learning experience. It allows users to delve into the inner workings of these algorithms, gaining insights into the mechanics of decision tree construction, information gain, and impurity measures. This hands-on approach fosters a deeper understanding of the decision-making processes fundamental to machine learning algorithms.

In conclusion, the provided code exemplifies the implementation of C4.5 and CART decision trees without relying on external libraries, promoting a richer understanding of these algorithms and their application in real-world scenarios.
