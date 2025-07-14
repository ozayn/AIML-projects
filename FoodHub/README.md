Performed EDA using NumPy, Pandas, and Seaborn to uncover demand patterns in cuisines and restaurants, and provided business recommendations.

Please refer to [code/notebook.ipynb](code/notebook.ipynb) to view the code.

# FoodHub Orders – EDA

Analyzing ~2K restaurant orders to uncover patterns in customer behavior, delivery dynamics, and revenue.  
**Azin Faghihi | November 2024**  
Part of the Great Learning ML & AI program

---

## Project Summary

We explored restaurant order data from FoodHub to help the company understand demand trends and improve customer experience.  
Insights were drawn from univariate and multivariate analysis across cuisines, delivery metrics, cost, ratings, and time.

---

## Problem

FoodHub wants to optimize restaurant operations and customer satisfaction using data from previous orders.

---

## Approach

- Explored 1,898 rows, 9 columns – no nulls, some missing ratings marked as "Not given"
- Analyzed variables:
  - Day of week (weekday/weekend)
  - Food prep time, delivery time
  - Ratings, cost
  - Cuisine, customer, restaurant IDs
- Conducted:
  - Univariate analysis (distributions and patterns)
  - Multivariate analysis (relationships between prep time, cost, rating, etc.)

---

## Key Findings

### 🥘 Cuisine
- Italian: longest prep time (median), Thai: most variable prep time  
- French: highest order cost, Vietnamese: lowest  
- American cuisine most popular on both weekdays and weekends  

### ⏱ Time
- Prep time: 20–35 mins (avg 27)  
- Delivery time: 15–33 mins (avg 24), longer on weekdays  
- ~10.5% of orders take > 60 mins total (prep + delivery)

### 💬 Ratings
- 40% of orders are unrated  
- Ratings increase when delivery/prep times decrease  
- Higher-priced orders → higher ratings  
- Japanese and Italian cuisines have higher proportions of unrated orders  

### 💰 Revenue
- 25% commission on orders > $20, 15% for $5–$20  
- Total revenue: $6,166.30  
- 72% of revenue comes from weekend orders  

---

## Recommendations

**For customers**  
- Encourage more reviews to earn discounts (esp. for cuisines with high unrated orders)

**For restaurants**  
- Improve prep time for Italian/Thai cuisines  
- Reduce weekday delivery times  
- Hire more help during busy weekends

---

## Tools

- Python (Pandas, Matplotlib, Seaborn)  
- Jupyter Notebook

---

## Note

This project was created as part of Great Learning’s ML & AI program.  
**Proprietary content – do not distribute.**
