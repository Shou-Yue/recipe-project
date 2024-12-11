# Recipe Rating Prediction Based on Meat Inclusion

Authors: Kevin Huang & Shoutai Yue

## Overview

This data science project is the end of year projects for DSC80 at UCSD. It is focused on predicting the recipe's rating based on relevant info.

## Introduction

In this project, we examined the dataset from [food.com](food.com). The link to download the dataset can be found [here](https://drive.google.com/file/d/1kIbMz6jlhleiZ9_3QthmUnifoSds_2EI/view). The dataset was originally used to build a recipe recommendatoin system and contains two tables:

1. `RAW_recipes.csv` contains all recipes
2. `RAW_interactions.csv` contains all reviews and ratings submitted for recipes in `RAW_interactions.csv`

The dataset contains meticulously designed recipes that are for all possiblescenarios, ranging across all cuisines and difficulty, and using all different kinds of ingredients. **Therefore, it comes to our interest to study the factors that determine rating of a recipe** and ultimately determine the preference of the general public.

With all being said, we will first start by cleaning the dataset and conduct some exploratory analysis.

Then we will try to access the missingness mechanism of the `review` column.

**Finally, we will create a model that can predict a recipe's rating based on relevant info.**

The first dataset, `recipe`, that is coming from `RAW_recipe.csv` contains 83782 rows and 12 columns, each row is a unique recipe. The columns are:

| **Column**       | **Description**                                                                                                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`           | Recipe name                                                                                                                                                                                         |
| `id`             | Recipe ID                                                                                                                                                                                           |
| `minutes`        | Minutes to prepare recipe                                                                                                                                                                           |
| `contributor_id` | User ID who submitted this recipe                                                                                                                                                                   |
| `submitted`      | Date recipe was submitted                                                                                                                                                                           |
| `tags`           | Food.com tags for recipe                                                                                                                                                                            |
| `nutrition`      | Nutrition information in the form `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`; PDV stands for “percentage of daily value” |
| `n_steps`        | Number of steps in recipe                                                                                                                                                                           |
| `steps`          | Text for recipe steps, in order                                                                                                                                                                     |
| `description`    | User-provided description                                                                                                                                                                           |
| `ingredients`    | Ingredients of the recipe                                                                                                                                                                           |
| `n_ingredients`  | Number of ingredients for the recipe                                                                                                                                                                |

The second dataset, `interactions`, contains 731927 rows and 5 columns, each row correspond to a review to a recipe in `recipe`. The columns are:

| **Column**  | **Description**     |
| ----------- | ------------------- |
| `user_id`   | User ID             |
| `recipe_id` | Recipe ID           |
| `date`      | Date of interaction |
| `rating`    | Rating given        |
| `review`    | Review text         |

## Data Cleaning and Exploratory Analysis

When cleaning the data, we used the following steps:

1. Left merge `recipe` and `interactions`, drop the one row in `recipe` where name is `NaN`.

2. Check the data type for resulting columns:

   - This step allows us to evaluate following cleaning operations
   - | **Column**       | **Dtype** |
     | ---------------- | --------- |
     | `name`           | object    |
     | `id`             | int64     |
     | `minutes`        | int64     |
     | `contributor_id` | int64     |
     | `submitted`      | object    |
     | `tags`           | object    |
     | `nutrition`      | object    |
     | `n_steps`        | int64     |
     | `steps`          | object    |
     | `description`    | object    |
     | `ingredients`    | object    |
     | `n_ingredients`  | int64     |
     | `user_id`        | float64   |
     | `recipe_id`      | float64   |
     | `date`           | object    |
     | `rating`         | float64   |
     | `review`         | object    |

3. Replace `0` in `rating` with `np.NaN`.
   - Since valid `rating` is an integer from 1 being the lowest to 5 being the highest. Then `0` is then representing missing values. So we replace `0` with `np.NaN` to avoid bias in future analysis
4. Convert the `date` and `submitted` column from `str` to `pd.DateTime`.
5. Convert the `nutritions`, `ingredients`, `tags` column from `str` to `list`.
6. Drop the row where `user_id` is `NaN`. That recipe has no review.
7. Add column `avg_rating` containing the avegrage rating for the recipe in that row.
   - Since a recipe gets many reviews, we want to know its average rating.

### Result

The final df contains the following columns:

| **Column**       | **Dtype**      |
| ---------------- | -------------- |
| `name`           | object         |
| `id`             | int64          |
| `minutes`        | int64          |
| `contributor_id` | int64          |
| `submitted`      | datetime64[ns] |
| `tags`           | object         |
| `nutrition`      | object         |
| `n_steps`        | int64          |
| `steps`          | object         |
| `description`    | object         |
| `ingredients`    | object         |
| `n_ingredients`  | int64          |
| `user_id`        | float64        |
| `recipe_id`      | float64        |
| `date`           | datetime64[ns] |
| `rating`         | float64        |
| `review`         | object         |
| `avg_rating`     | float64        |

Shown below is part of the cleaned table with some of the most relevant columns:

| name                               |     id | minutes | contributor_id | submitted           | n_steps | n_ingredients |     user_id | recipe_id | date                | rating | avg_rating |
| :--------------------------------- | -----: | ------: | -------------: | :------------------ | ------: | ------------: | ----------: | --------: | :------------------ | -----: | ---------: |
| 1 brownies in the world best ever  | 333281 |      40 |         985201 | 2008-10-27 00:00:00 |      10 |             9 |      386585 |    333281 | 2008-11-19 00:00:00 |      4 |          4 |
| 1 in canada chocolate chip cookies | 453467 |      45 |        1848091 | 2011-04-11 00:00:00 |      12 |            11 |      424680 |    453467 | 2012-01-26 00:00:00 |      5 |          5 |
| 412 broccoli casserole             | 306168 |      40 |          50969 | 2008-05-30 00:00:00 |       6 |             9 |       29782 |    306168 | 2008-12-31 00:00:00 |      5 |          5 |
| 412 broccoli casserole             | 306168 |      40 |          50969 | 2008-05-30 00:00:00 |       6 |             9 |     1196280 |    306168 | 2009-04-13 00:00:00 |      5 |          5 |
| 412 broccoli casserole             | 306168 |      40 |          50969 | 2008-05-30 00:00:00 |       6 |             9 |      768828 |    306168 | 2013-08-02 00:00:00 |      5 |          5 |

### Univariate Analysis

First, we want to explore the distribution of `rating`. We want to see if `rating` is biased. If people only rate if they have a strong opinion. Then the histogram will be skewed.
We discovered that `rating` is significantly skewed to the left, in fact, the number of 5-star ratings is more than the rest of the ratings combined. We may consider that `rating` might be biased because most people coming back to review only when they like the recipe a lot.

<iframe
    src = "assets/univariate_1.html"
    width = "800"
    height = "600"
    frameborder = "0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

### Bivariate Analysis

We try to explore if there is a relationship between **Inclusion of the Meat Tag in Tags** and **Average Rating**. We use create-kde from lecture. And we found that **the dish having meat doesn't seem to affect its rating**.

<iframe
    src = "assets/bivariate_1.html"
    width = "800"
    height = "600"
    frameborder = "0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

### Interesting Aggregates

Finally, we decide to pivot by `rating`, and see the average `minutes`, `num_steps`, and `n_ingredients` for each category in `rating` (1 to 5). The table is shown below.

| rating | minutes | n_steps | n_ingredients |
| -----: | ------: | ------: | ------------: |
|      1 | 38.7173 | 10.3264 |       8.80106 |
|      2 |  39.631 | 10.3975 |       9.12531 |
|      3 | 38.0882 | 9.73812 |        9.0955 |
|      4 | 36.4743 | 9.43446 |       8.99958 |
|      5 | 36.3649 | 9.71074 |       8.95111 |

## Missingness Mechanism

The three columns in the cleaned dataframe that have non-trivial number missing entries are `rating`, `review`, `description`

### NMAR Analysis

We think none of the three columns are **NMAR**.
We can argue that `description` is **MAR** depending on `contributor_id`, if a recipe is submitted by a frequent creator then it's more
We can also argue that `review` is **MAR** dependent on `rating`, because if a person either feel too positive or too negative about the recipe might not give a review.
Finally, we think `rating` may be **MAR** depending on other columns, which we examine below.

### Missingness Dependency

Since we are trying to predict `rating`, we want to access the missingness dependency of `rating`.
First, we decide to test if `rating` is missing conditionally on `n_steps`, the number of steps to make the recipe, and `minutes`, the number of minutes it takes to make the recipe.

> n-steps and rating

We suspect that when people complete dishes that takes too many steps to make they might not give a rating.

**Null Hypothesis**: The missingness of ratings does not depend on the number steps.

**Alternative Hypothesis**: The missingness of ratings does depend on the number of steps to make the dish.

**Test Statistic**: the difference in mean of `n_steps` when `rating` is missing and when `rating` is present.

**Significance Level**: 0.01

We run the permutation test by shuffling the missingness of rating 1000 times. The distribution of test statistic and the observed statistic are shown below.

<iframe
    src = "assets/mar_n_steps.html"
    width = "800"
    height = "600"
    frameborder = "0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

We found p=0.0<0.01, so we reject the null hypothesis. We then conclude that `rating` is indeed **MAR** conditional on `n_steps`.

> minutes and rating

We suspect that dishes that takes too many minutes to make may lead to users to not give a rating.

**Null Hypothesis**: The missingness of ratings does not depends on the number of minutes to make the dish.

**Alternative Hypothesis**: The missingness of ratings does depends on the number of minutes to make the dish.

**Test Statistic**: difference in means of `minutes` when `rating` is missing and when `rating` is present.

**Significance Level**: 0.01

We run the permutation test by shuffling the missingness of `rating` 1000 times. The distribution of test statistic and the observed statistic are shown below.

<iframe
    src="assets/mar_minutes.html"
    width="800"
    height="600"
    frameborder="0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

We found p=0.112>0.01, we fail to reject the null hypothesis. So we conclude that `rating` is not **MAR** depending on `minutes`.

## Hypothesis Testing

As mentioned in the introduction, we want to predict the rating of a recipe. We suspect that have `"meat"` in their tag have a higher `rating` on average then those that are not.

To investigate this question, we run a **permutaiton test** where we shuffled a boolean column created by checking the presence of `meat` in `tags`. The hypothesis and statistics are as follows:

**Null Hypothesis:** People rate meat and non-meat tagged dishes the same <br>

**Alternative Hypothesis:** People rate meat tagged dishes lower than they rate non-meat tagged dishes <br>

**Test Statistic:** Difference in mean between ratings of non-meat dishes and meat dishes

We run a 10000-simulation permutaiton test in order to get the empirical distribution of the test statistics under the null hypothesis.

We then get p=0.0<0.05, so we reject the null hypothesis. We conclude that people rate dishes with meat in their tag lower than dishes without meat in their tag.

## Framing a Predicition

After reviewing our results from above, we intend to **Predict Rating of a Recipe**, a multi-class classification problem, as ratings can take values of any integer from 1-5, which would make rating an catagorical, ordinal variable.

We chose `rating` as our response variable as it is the single statistic that users will prioritize when looking through recipes online. Additionally, most previous analysis were done with `rating` as one of the variables, thus, we have the most information at the time regarding which columns will affect `rating`, making it an appropriate variable to predict.

Because we previously analyzed `minutes` and `is_meat` and concluded that they had significant corrolation with the `ratings` column, we will use these columns to create features for our baseline model in the following section.

To evaluate our model's effectiveness, we will be reporting both the model's accuracy, as well as the model's f1 score. 
1. **Accuracy:** The accuracy will allow us to analyze overall how our model is doing in terms of creating predictions based on given parameters, in this case being `minute` and `is_meat`
2. **F1-Score:** Although accuracy will be a direct and effective measure of the effectiveness of our model, it does not allow insights into important aspects of the model, such as fairness for each groups in our data, as well as analyze if our model is creating excessive false positive/negative predictions.

## Baseline Model

In our baseline model, we utilized a **Random Forest Classifier** with two parameters that we transfromed and process according to the process below:
1. `minutes`: The minutes column, which, as mentioned before contains the time necessary to create the recipe, this is a continuous, numerical variable in our dataframe. Thus, to accurately determine the effects this has on rating, we used `StandardScaler()` to standardize the data, allowing us to interpret if the specific row has above, below, or about average time consumed.
2. `is_meat`: As seen previously, people are likely to rate meat and non-meat tagged dishes differently, thus making this nominal, true/false variable effective when predicting rating. We kept all values in their according groups, and transformed booleans into integers with 1 corresponding to `True` and 0 responding to `False`.

We did not tune any hyperparameters and used the default arguments for the Random Forest Classifier, but we did split our data into a training set, containing 80% of the data, and a test set, containing 20% of the data, allowing us to analyze if our model is over/underfitting the data. This was accomplished by utilizing `train_test_split` from the sklearn library.

The statistics produced by our baseline model is as follows:
- **Accuracy:** `0.776`
- **F1-Score - Averaged:** `0.175`
- **F1-Score - By Category:** `0.  , 0.  , 0.  , 0.  , 0.87`

Strangely, the accuracy came out rather high at almost 0.78, meaning that our baseline model classifies around 80% of recipes effectively, but the f1 score was extremely low at 0.17, taking a glance at each f1 score for different columns, it can be seen that the model was much more effective at predicting recipes with ratings of 5, and rather low for everythin else.

To analyze why, we generated the list of unique values from our prediction, and found that our models only outputted either 4 or 5 for any given recipe, this is likely due to the uneven distribution of points in the dataset. After analyzing, we saw that almost 95% of all data had ratings of either 4 or 5. We will address this in the final model.

At the time being, our model is not very useful, it has a high accracy, but it is not much better than a constant model that always outputs 5 for the prediction. To make it more effective, we need to classify recipes with lower ratings better.

## Final Model

In our final model, we included many more parameters, and utilized **GridSearchCV** to find optimal hyperparameters for our model. The list of features and how they were treated is as follows:
1. `date` and `submitted`: These were columns containing datetime objects, we used these parameters in consideration that users interpretation of recipes might vary as time goes on, people today likely do not have the same preferences for food now compared to a few years ago. Or perhaps they rated seaonal dishes differently based on time of year, such as the tendency to enjoy ice-cream in summer rather than winter.
   - To account for these differences, we defined a function to extract year, month, day, and day of the week, and standardized them.
2. `description` and `review`: These columns contained the description of the recipe, and the written review from ysers, both of which are text columns, we will combine these columns and use TfIdf in order to find words that corrolate with higher/lower ratings.
3. `ingredients`: This contains a list of strings containing ingredients for the recipe, and similarly to before, we will use TfIdf to find ingredients that are likely to be corrolated with higher/lower ratings
4. `nutrition`: This column contains a list corresponding to the nutritional value in the recipe, however, unlike before we cannot treat them as a collective, as entry in a specific cell of the nutritional value of the recipe is a different category, as a result, this is a multi-label categorical feature, we used `MultiLabelBinizer` in order to create bins for each of these values, and interpreted how each value affected rating.
5. `is_meat` and `minutes`: Because these were used in our baseline model, and proved relatively effective in predicting the rating, we treat them the same way as before, and include them in our model.
6. `n_steps` and `n_ingredients`: These columns contained the number of steps, and the number of ingredients in the recipe, accordingly. As these go up, complexity of the recipe will also increaase, likely having a negative effect on rating. We simply applied standardscaler to these categories.

Additionally to adding these parameters, we utilized the argument class_weight = `balanced` in our `RandomForestClassifier`. This will allow us to account for the uneven distribution that was seen in our previous model.

Finally, we will run a GridSearch on our model to find optimal hyperparameters, recursively searching the number of estimators, the max depth, and the minimum split arguments. Which turned out to be 20, 5, and 200 respectively.

In our final model, our statistics were as follows:
- **Accuracy:** `0.754`
- **F1-Score - Averaged:** `0.329`

Comparing with our baseline model, it can be seen that the model of our accuracy had not improved at all, and infact had gone down a little, but our f1 score had almost doubled.

This means that our fixed model was able to predict reviews with lower rating more accuratly than before at the cost of creating some false classification in groups with 4 or 5 as the rating, most likely due to the inclusion of the argument `balanced = True` when creating our pipeline.

## Fairness Analysis

For our fairness analysis, we revisit the idea of **meat** inclusion in the tags of each recipe. We will split the recipes into two groups, ones with **meat** in the tag, and the other without. We can use this as the portion of meat-tagged recipes is high enough in our dataset as shown below.
- count of `is_meat == False`: 155771
- count of `is_meat == True`: 41534
This high count means that any patterns exhibited by our model when predicting this group is likely accurate and not due to chance alone.

Additionally, since we had 5 categories that ratings can be, integers from 1-5, we will treat these values and classify them into a binary column for which precision/recall values will have more significant meaning. This column will contain 1 for recipes with 4 or 5 as the rating, and 0 for recipes with 1-3 and will be named `highly_rated`.

Currently, the f1 score of our model uses the `macro` setting, which represents the mean f1 score of each label individually, thus, evaluating precision/recall parity using this metric will give similar results as evaluating accuracy parity as having a false positive in one category means having a false negative in another. 

We will be using precision parity to measure model fairness, as we believe that having false positives, low rated recipes being classified as high, will disappoint a user more. As they are likely to be unsatisfied with the recipe after putting time and effort into making it, rather than seeing a good recipe falsely classified and ignoring it.

To predict if our model is fair for recipes with/without meat, we conduct a permutation test using the hypothesis pair below:
1. **Null Hypothesis:** Our model is fair, and its precision is the same for recipes with or without `meat` being contained in the tag.
2. **Alternative Hypothesis:** Our model is unfair, and its precision is higher for recipes with `meat` being contained in the tag.
3. **Test Statistic:** Difference in precision between rows with `is_meat == True` and `is_meat == False`
4. **Significance Level:** 0.01

After running the permutation test through randomly shuffling the `is_meat` column, we get a p-value of 0.117, since this is greater than our set significance level at 0.01, we do not reject the null hypothesis, and we conclude that our model is fair in the sense that creates similar predicted precision for `meat` and `non-meat` dishes.
