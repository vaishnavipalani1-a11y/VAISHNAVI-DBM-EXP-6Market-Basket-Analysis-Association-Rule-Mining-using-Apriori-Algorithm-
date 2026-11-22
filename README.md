# Experiment 6 : Market Basket Analysis (Association Rule Mining using Apriori Algorithm)

## Aim
To perform Market Basket Analysis using the Apriori algorithm to identify frequent itemsets and generate association rules from transaction data for decision-making in retail and business analytics.

## Dataset
A sample retail transaction dataset containing **10 transactions** across **4 grocery items**: `Bread`, `Butter`, `Milk`, and `Eggs`.

| Transaction ID | Items Purchased |
| :---: | :--- |
| **T1** | Milk, Bread, Eggs |
| **T2** | Milk, Bread |
| **T3** | Milk, Eggs |
| **T4** | Bread, Butter |
| **T5** | Milk, Bread, Butter, Eggs |
| **T6** | Bread, Eggs |
| **T7** | Milk, Butter |
| **T8** | Milk, Bread, Butter |
| **T9** | Bread, Butter, Eggs |
| **T10** | Milk, Bread, Butter, Eggs |

*The transaction list is transformed into a One-Hot Encoded binary DataFrame prior to mining.*

## Algorithm / Steps Followed
1. Import required Python libraries (`pandas`, `mlxtend`).
2. Create the sample grocery transaction dataset.
3. Convert the dataset into a one-hot encoded binary matrix using `TransactionEncoder`.
4. Apply the `apriori()` algorithm (`min_support=0.3`) to identify frequent itemsets.
5. Generate association rules from frequent itemsets using `association_rules()` with `lift >= 1.0`.
6. Filter strong rules based on thresholds (`confidence >= 0.6` and `lift > 1`).
7. Interpret the rules to uncover cross-selling and store layout insights.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Handling** | `TransactionEncoder()` | Converts a list of transactions into a one-hot encoded binary matrix |
| **Frequent Pattern Mining** | `apriori()` | Finds frequent itemsets meeting the minimum support threshold |
| **Association Rule Generation** | `association_rules()` | Derives association rules using metrics such as support, confidence, and lift |
| **Filtering** | Conditional filtering (`rules[...]`) | Selects rules meeting strong confidence and lift thresholds |
| **Visualization / Display** | `print()` / `DataFrame.head()` | Displays encoded transactions, itemsets, and rules in tabular format |

## Program & Output
## Input 1
```
# Market Basket Analysis (Association Rule Mining)

# Step 1: Import Required Libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 2: Create Sample Transaction Dataset
dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Butter'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Bread', 'Butter', 'Eggs']
]

# Step 3: Convert Dataset into One-Hot Encoding
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)

df = pd.DataFrame(te_array, columns=te.columns_)

print("----- Transaction Dataset (One-Hot Encoded) -----")
print(df.head())
```
## Ouput 1
```
----- Transaction Dataset (One-Hot Encoded) -----
   Bread  Butter   Eggs   Milk
0   True   False   True   True
1   True   False  False   True
2  False   False   True   True
3   True    True  False  False
4   True    True   True   True
```
## Input 2
```
# Step 4: Apply Apriori Algorithm
frequent_itemsets = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\n----- Frequent Itemsets -----")
print(frequent_itemsets)
```
## Output 2
```
----- Frequent Itemsets -----
    support                          itemsets
0       0.8                frozenset({Bread})
1       0.6               frozenset({Butter})
2       0.6                 frozenset({Eggs})
3       0.7                 frozenset({Milk})
4       0.5        frozenset({Bread, Butter})
5       0.5          frozenset({Eggs, Bread})
6       0.5          frozenset({Bread, Milk})
7       0.3         frozenset({Eggs, Butter})
8       0.4         frozenset({Butter, Milk})
9       0.4           frozenset({Eggs, Milk})
10      0.3  frozenset({Eggs, Bread, Butter})
11      0.3  frozenset({Bread, Butter, Milk})
12      0.3    frozenset({Eggs, Bread, Milk})
```
## Input 3
```
# Step 5: Generate Association Rules
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

print("\n----- Association Rules -----")
print(rules[['antecedents', 'consequents',
             'support', 'confidence', 'lift']])

```
## Output 3
```
----- Association Rules -----
                   antecedents                 consequents  support  \
0           frozenset({Bread})         frozenset({Butter})      0.5   
1          frozenset({Butter})          frozenset({Bread})      0.5   
2            frozenset({Eggs})          frozenset({Bread})      0.5   
3           frozenset({Bread})           frozenset({Eggs})      0.5   
4     frozenset({Eggs, Bread})         frozenset({Butter})      0.3   
5    frozenset({Eggs, Butter})          frozenset({Bread})      0.3   
6   frozenset({Bread, Butter})           frozenset({Eggs})      0.3   
7            frozenset({Eggs})  frozenset({Bread, Butter})      0.3   
8           frozenset({Bread})   frozenset({Eggs, Butter})      0.3   
9          frozenset({Butter})    frozenset({Eggs, Bread})      0.3   
10    frozenset({Bread, Milk})         frozenset({Butter})      0.3   
11         frozenset({Butter})    frozenset({Bread, Milk})      0.3   
12    frozenset({Bread, Milk})           frozenset({Eggs})      0.3   
13           frozenset({Eggs})    frozenset({Bread, Milk})      0.3   

    confidence      lift  
0     0.625000  1.041667  
1     0.833333  1.041667  
2     0.833333  1.041667  
3     0.625000  1.041667  
4     0.600000  1.000000  
5     1.000000  1.250000  
6     0.600000  1.000000  
7     0.500000  1.000000  
8     0.375000  1.250000  
9     0.500000  1.000000  
10    0.600000  1.000000  
11    0.500000  1.000000  
12    0.600000  1.000000  
13    0.500000  1.000000  
```
## Input 4
```

# Step 6: Sort and Filter Strong Rules
strong_rules = rules[
    (rules['confidence'] >= 0.6) &
    (rules['lift'] > 1)
]

print("\n----- Strong Association Rules -----")
print(strong_rules[['antecedents', 'consequents',
                    'support', 'confidence', 'lift']])

```
## Output 4
```
----- Strong Association Rules -----
                 antecedents          consequents  support  confidence  \
0         frozenset({Bread})  frozenset({Butter})      0.5    0.625000   
1        frozenset({Butter})   frozenset({Bread})      0.5    0.833333   
2          frozenset({Eggs})   frozenset({Bread})      0.5    0.833333   
3         frozenset({Bread})    frozenset({Eggs})      0.5    0.625000   
5  frozenset({Eggs, Butter})   frozenset({Bread})      0.3    1.000000   

       lift  
0  1.041667  
1  1.041667  
2  1.041667  
3  1.041667  
5  1.250000  
```

## Result

The Apriori algorithm successfully discovered frequent itemsets and strong association rules.
Items like Bread, Butter, and Eggs often appear together, revealing useful buying pattern for cross-selling and store layout optimization.
