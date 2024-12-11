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
4. Add column `avg_rating` containing the avegrage rating fot the recipe in that row.
   - Since a recipe gets many reviews, we want to know its average rating.

### Result

The final df contains the following columns:

| **#** | **Column**       | **Dtype** |
| ----- | ---------------- | --------- |
| 0     | `name`           | object    |
| 1     | `id`             | int64     |
| 2     | `minutes`        | int64     |
| 3     | `contributor_id` | int64     |
| 4     | `submitted`      | object    |
| 5     | `tags`           | object    |
| 6     | `nutrition`      | object    |
| 7     | `n_steps`        | int64     |
| 8     | `steps`          | object    |
| 9     | `description`    | object    |
| 10    | `ingredients`    | object    |
| 11    | `n_ingredients`  | int64     |
| 12    | `user_id`        | float64   |
| 13    | `recipe_id`      | float64   |
| 14    | `date`           | object    |
| 15    | `rating`         | float64   |
| 16    | `review`         | object    |

Shown below is part of the cleaned table with some of the most relevant columns:

| **name**                           | **id** | **minutes** | **contributor_id** | **submitted** | **n_steps** | **n_ingredients** | **user_id** | **recipe_id** | **date**   | **rating** | **avg_rating** |
| ---------------------------------- | ------ | ----------- | ------------------ | ------------- | ----------- | ----------------- | ----------- | ------------- | ---------- | ---------- | -------------- |
| 1 brownies in the world best ever  | 333281 | 40          | 985201             | 2008-10-27    | 10          | 9                 | 386585      | 333281        | 2008-11-19 | 4          | 4              |
| 1 in canada chocolate chip cookies | 453467 | 45          | 1848091            | 2011-04-11    | 12          | 11                | 424680      | 453467        | 2012-01-26 | 5          | 5              |
| 412 broccoli casserole             | 306168 | 40          | 50969              | 2008-05-30    | 6           | 9                 | 29782       | 306168        | 2008-12-31 | 5          | 5              |
| 412 broccoli casserole             | 306168 | 40          | 50969              | 2008-05-30    | 6           | 9                 | 1.19628e+06 | 306168        | 2009-04-13 | 5          | 5              |
| 412 broccoli casserole             | 306168 | 40          | 50969              | 2008-05-30    | 6           | 9                 | 768828      | 306168        | 2013-08-02 | 5          | 5              |

### Univariate Analysis

First, we want to explore the distribution of `rating`. We want to see if `rating` is biased. If people only rate if they have a strong opinion. Then the histogram will be skewed.
We discovered that `rating` is significantly skewed to the left, in fact, the number of 5-star ratings is more than the rest of the ratings combined. We may consider that `rating` might be biased because most people coming back to review only when they like the recipe a lot.

<iframe
    src = "assets/univariate_1.html"
    width = "800"
    height = "600"
    frameborder = "0"
></iframe>

### Bivariate Analysis

We try to explore if there is a relationship between **Inclusion of the Meat Tag in Tags** and **Average Rating**. We use create-kde from lecture. And we found that **the dish having meat doesn't seem to affect its rating**.

<iframe
    src = "assets/bivariate_1.html"
    width = "800"
    height = "600"
    frameborder = "0"
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
We can argue that `description` is **MAR** (missing by design) depending on `contributor_id`, if a recipe is submitted by a frequent creator then it's more
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

## Baseline Model

## Final Model

## Fairness Analysis
